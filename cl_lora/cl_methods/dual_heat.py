"""DualHeatCLMethod — Inibição lateral + EWC per-neuron para Continual Learning.

Baseado no DualHeat original (dual_heat_module.py) adaptado para o pipeline
PEFT/LoRA deste projeto.

O EWC hooka a saída combinada do módulo PEFT. Como os pesos base ficam
congelados, isso escala simetricamente os gradientes de A e B sem reimplementar
o forward nativo do PEFT.

Algoritmo (por passo de treino):
  1. base = W_base(x) + b_base           (pré-treinado, congelado)
  2. delta = scaling · B(A(x))            (adaptador LoRA)
  3. z = base + delta                     (saída combinada)
  4. output = z / (1 + γ·mean_others)     (inibição lateral divisiva, opcional)
  5. fast_heat = max(0, α·|output| + (1-α)·fast_heat − δ)
  6. slow_heat += (|output| − slow_heat) / min(n, W)  (capped incremental mean)
  7. backward: grad(output) /= (1 + β·slow_heat)
     → via chain rule, tanto B quanto A recebem gradiente escalado

Referência:
    dual_heat_module.py — v3: Inibição lateral + decay ativo + pós-inibição + EWC
    + slow heat com memória limitada (forgetting)
    dual_heat_LoRA_module.py — proteção simétrica dos adapters

Integração no pipeline:
    pre_train()  → substitui forward dos módulos LoRA (EWC + lateral inhibition)
    aux_loss()   → None (DualHeat não usa loss aditivo)
    post_train() → captura estado final para persistência
    save/load()  → persiste/restaura heat state entre tasks
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import CLMethod

logger = logging.getLogger("cl_lora.cl_methods.dual_heat")


def _iter_lora_modules(
    lora_model: nn.Module,
    active_adapter: str = "default",
):
    """Yield (module_name, mod, lora_A_weight, lora_B_weight, out_features)
    for every active LoRA module in the PEFT model.

    Yields modules where both lora_A and lora_B have the given adapter.
    """
    from peft.tuners.lora import Linear as LoraLinear

    for name, mod in lora_model.named_modules():
        if not isinstance(mod, LoraLinear):
            continue
        a_dict = getattr(mod, "lora_A", None)
        b_dict = getattr(mod, "lora_B", None)
        if a_dict is None or b_dict is None:
            continue
        if active_adapter not in a_dict or active_adapter not in b_dict:
            continue
        A_w = a_dict[active_adapter].weight  # (r, in_features)
        B_w = b_dict[active_adapter].weight  # (out_features, r)
        yield name, mod, A_w, B_w, B_w.shape[0]


def _make_default_hyperparams() -> Dict[str, Any]:
    """Default DualHeat hyperparameters matching the original implementation."""
    return {
        "fast_decay": 0.93,           # α — EMA decay for fast heat
        "fast_strength": 2.0,         # γ — lateral inhibition strength
        "fast_decay_rate": 0.04,      # δ — active decay per step
        "slow_strength": 2.0,         # β — EWC regularization strength
        "slow_window": None,          # memory window (None = infinite)
        "lateral_inhibition": True,   # enable lateral inhibition
    }


class _DualHeatModule(nn.Module):
    """Per-neuron heat tracking for one LoRA module.

    Maintains fast_heat (short-term) and slow_heat (long-term) for each
    output neuron.  Because hooks fire on different GPU replicas under
    DataParallel, heat state is stored in a *per-device* dict so each
    GPU replica gets its own independent buffers.
    """

    def __init__(
        self,
        out_features: int,
        fast_decay: float = 0.93,
        fast_strength: float = 2.0,
        fast_decay_rate: float = 0.04,
        slow_strength: float = 2.0,
        slow_window: Optional[int] = None,
        lateral_inhibition: bool = True,
    ):
        super().__init__()
        self.out_features = out_features
        self.fast_decay = fast_decay
        self.fast_strength = fast_strength
        self.fast_decay_rate = fast_decay_rate
        self.slow_strength = slow_strength
        self.slow_window = slow_window
        self.lateral_inhibition = lateral_inhibition

        # Heat state keyed by device string:  str -> Dict[str, Tensor]
        self._per_device: Dict[str, Dict[str, torch.Tensor]] = {}

    # ── Per-device heat state access ──────────────────────────────

    def load_state_snapshot(self, state: Dict[str, torch.Tensor]) -> None:
        """Load heat state from a checkpoint onto CPU.

        The tensors will be moved to the correct device on the next
        forward call via _get_or_restore().
        """
        if not state:
            return
        # Store as "cpu_fp32" base key; _get_or_restore will check this prefix
        # when creating new entries if no exact match exists.
        self._loaded_cpu_state = {
            "fast_heat": state["fast_heat"].float().clone(),
            "slow_heat": state["slow_heat"].float().clone(),
            "slow_n": state["slow_n"].long().clone(),
            "step": state["step"].long().clone() if "step" in state else torch.zeros((), dtype=torch.long),
        }

    def _get_or_restore(self, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Get per-device state, initializing from loaded CPU state if available."""
        key = str(device)
        if key in self._per_device:
            return self._per_device[key]

        loaded = getattr(self, "_loaded_cpu_state", None)
        if loaded is not None:
            # Heat is canonical fp32 state; computation dtype is a boundary concern.
            d = {}
            for k, v in loaded.items():
                if k in ("fast_heat", "slow_heat"):
                    d[k] = v.to(device=device, dtype=torch.float32)
                else:
                    d[k] = v.to(device=device)
            self._per_device[key] = d
            self._loaded_cpu_state = None
        else:
            self._per_device[key] = {
                "fast_heat": torch.zeros(self.out_features, device=device, dtype=torch.float32),
                "slow_heat": torch.zeros(self.out_features, device=device, dtype=torch.float32),
                "slow_n": torch.zeros((), dtype=torch.long, device=device),
                "step": torch.zeros((), dtype=torch.long, device=device),
            }
        return self._per_device[key]

    @torch.no_grad()
    def update_heat(self, output: torch.Tensor) -> None:
        """Update fast_heat and slow_heat from the current output."""
        state = self._get_or_restore(output.device, dtype=output.dtype)

        reduce_dims = tuple(range(output.dim() - 1))
        post_mag = output.detach().float().abs().mean(dim=reduce_dims)

        # Fast: EMA + decay ativo
        state["fast_heat"].mul_(self.fast_decay).add_(
            (1.0 - self.fast_decay) * post_mag, alpha=1.0
        ).sub_(self.fast_decay_rate).clamp_(min=0.0)

        # Slow: capped incremental mean
        state["slow_n"] += 1
        n_true = state["slow_n"].item()
        n_eff = min(n_true, self.slow_window) if self.slow_window is not None else n_true
        state["slow_heat"].add_((post_mag - state["slow_heat"]) / float(n_eff))
        state["step"] += 1

    def get_ewc_scale(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return per-neuron EWC scale: 1 / (1 + beta * slow_heat)."""
        if self.slow_strength <= 0.0:
            return torch.ones(self.out_features, device=device, dtype=dtype)
        state = self._get_or_restore(device, dtype=dtype)
        return (1.0 / (1.0 + self.slow_strength * state["slow_heat"])).to(dtype=dtype)

    def get_state_snapshot(self) -> Dict[str, torch.Tensor]:
        """Return merged heat state for checkpointing.

        Takes the first available device's state (all should be equivalent
        after single-device training; under DataParallel, any device works).
        """
        if not self._per_device:
            loaded = getattr(self, "_loaded_cpu_state", None)
            if loaded is None:
                return {}
            return {k: v.cpu().clone() for k, v in loaded.items()}
        first = next(iter(self._per_device.values()))
        return {
            "fast_heat": first["fast_heat"].cpu().clone(),
            "slow_heat": first["slow_heat"].cpu().clone(),
            "slow_n": first["slow_n"].cpu().clone(),
            "step": first["step"].cpu().clone(),
        }

    def extra_repr(self) -> str:
        nd = len(self._per_device)
        return (
            f"out={self.out_features}, α={self.fast_decay}, γ={self.fast_strength}, "
            f"δ={self.fast_decay_rate}, β={self.slow_strength}, devices={nd}"
        )


class DualHeatCLMethod(CLMethod):
    """DualHeat: lateral inhibition + EWC per-neuron for Continual Learning.

    Operates as a CLMethod hooking into the PEFT LoRA training pipeline.
    """

    name = "dual_heat"

    def __init__(
        self,
        *,
        fast_decay: float = 0.93,
        fast_strength: float = 2.0,
        fast_decay_rate: float = 0.04,
        slow_strength: float = 2.0,
        slow_window: Optional[int] = None,
        lateral_inhibition: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            fast_decay=fast_decay,
            fast_strength=fast_strength,
            fast_decay_rate=fast_decay_rate,
            slow_strength=slow_strength,
            slow_window=slow_window,
            lateral_inhibition=lateral_inhibition,
            **kwargs,
        )
        self.fast_decay = float(fast_decay)
        self.fast_strength = float(fast_strength)
        self.fast_decay_rate = float(fast_decay_rate)
        self.slow_strength = float(slow_strength)
        self.slow_window = slow_window
        self.lateral_inhibition = bool(lateral_inhibition)

        self._dual_modules: Dict[str, _DualHeatModule] = {}
        self._orig_forwards: List[callable] = []
        self._patched_modules: List[nn.Module] = []
        self._active_adapter: str = "default"

    # ─── Hook lifecycle ────────────────────────────────────────────────

    def _make_ewc_hook_on_output(self, name: str):
        """Create a backward hook for the combined PEFT module output.

        The frozen base branch receives no parameter gradients, while the chain
        rule applies the scale to both lora_B and lora_A. Hooking the native
        output preserves PEFT's adapter, dtype, variant, and merge semantics.

        Grad shape: (..., out_features) — scale broadcasts over last dim.
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            dh_mod = self._dual_modules.get(name)
            if dh_mod is None:
                return grad
            scale = dh_mod.get_ewc_scale(grad.device, dtype=grad.dtype)
            return grad * scale  # broadcast over (..., out_features)

        return hook

    def _make_patched_forward(
        self, name: str, out_features: int, dh_mod, mod: nn.Module, original_forward
    ):
        """Create a patched forward method for a LoRA module.

        This wraps the native PEFT forward to:
          1. Preserve PEFT's adapter selection, variants, and dtype handling
          2. Register the EWC hook on the combined output
          3. Apply lateral inhibition on the full output
          4. Track heat magnitudes
        """
        def patched_forward(x, *args, **kwargs):
            # Delegate PEFT details (ModuleDicts, casts, variants, mixed adapters,
            # merged/disabled states) to the supported native implementation.
            result = original_forward(x, *args, **kwargs)
            if dh_mod.slow_strength > 0.0 and result.requires_grad:
                result.register_hook(self._make_ewc_hook_on_output(name))

            # ── Lateral inhibition ──────────────────────────────────────
            if dh_mod.lateral_inhibition and dh_mod.fast_strength > 0.0 \
                    and out_features > 1 and mod.training:
                state = dh_mod._get_or_restore(result.device, dtype=result.dtype)
                fh = state["fast_heat"]
                sum_h = fh.sum()
                mean_others = (sum_h - fh) / float(out_features - 1)
                scale = (1.0 + dh_mod.fast_strength * mean_others).to(result.dtype)
                result = result / scale

            # ── Heat tracking ───────────────────────────────────────────
            if mod.training:
                dh_mod.update_heat(result)

            return result

        return patched_forward

    def _register_patched_forward(self, lora_model: nn.Module) -> None:
        """Replace each LoRA module's forward with a patched version.

        The patched forward includes:
          - Native PEFT forwarding with an EWC hook on the combined output
          - Lateral inhibition on the full output
          - Heat magnitude tracking
        """
        self._remove_patched_forward()

        self._dual_modules = {}
        self._orig_forwards: List[callable] = []
        self._patched_modules: List[nn.Module] = []

        for name, mod, A_w, B_w, out_features in _iter_lora_modules(
            lora_model, active_adapter=self._active_adapter
        ):
            dh_mod = _DualHeatModule(
                out_features=out_features,
                fast_decay=self.fast_decay,
                fast_strength=self.fast_strength,
                fast_decay_rate=self.fast_decay_rate,
                slow_strength=self.slow_strength,
                slow_window=self.slow_window,
                lateral_inhibition=self.lateral_inhibition,
            )
            self._dual_modules[name] = dh_mod

            # Save original forward and replace with patched version
            original_forward = mod.forward
            self._orig_forwards.append(original_forward)
            self._patched_modules.append(mod)
            mod.forward = self._make_patched_forward(
                name, out_features, dh_mod, mod, original_forward
            )

        if not self._dual_modules:
            raise RuntimeError(
                f"DualHeat matched zero LoRA modules for adapter {self._active_adapter!r}"
            )

        logger.info("DualHeat patched forwards registered: %d modules", len(self._dual_modules))

    def _remove_patched_forward(self) -> None:
        """Restore original forward methods on all patched modules."""
        for mod, orig_fwd in zip(self._patched_modules, self._orig_forwards):
            mod.forward = orig_fwd
        self._orig_forwards.clear()
        self._patched_modules.clear()

    # ─── CLMethod interface ────────────────────────────────────────────

    def pre_train(
        self,
        lora_model: torch.nn.Module,
        *,
        stage_idx: int,
        retain_tasks: Optional[List[Any]],
    ) -> None:
        """Patch LoRA module forward methods with DualHeat logic."""
        active = getattr(lora_model, "active_adapter", "default")
        if isinstance(active, (list, tuple)):
            active = active[0] if active else "default"
        self._active_adapter = str(active)

        self._register_patched_forward(lora_model)
        logger.info(
            "DualHeat pre_train: stage=%d retain_tasks=%s modules=%d",
            stage_idx,
            bool(retain_tasks),
            len(self._dual_modules),
        )

        # Restore heat state after registration
        self.restore_heat_state(lora_model)

    def aux_loss(self, lora_model: torch.nn.Module) -> Optional[torch.Tensor]:
        """DualHeat does not use an additive loss term."""
        return None

    def post_train(
        self,
        lora_model: torch.nn.Module,
        *,
        tokenizer: Any,
        train_dataset: Any,
        device: torch.device,
        stage_idx: int,
        task_name: str,
    ) -> None:
        """Capture heat state after training completes (pre-merge)."""
        heat_info = {}
        for name, dh_mod in self._dual_modules.items():
            snapshot = dh_mod.get_state_snapshot()
            if snapshot:
                heat_info[name] = snapshot

        logger.info(
            "DualHeat post_train: stage=%d task=%s modules=%d",
            stage_idx,
            task_name,
            len(heat_info),
        )
        self._train_heat_snapshot = heat_info
        self._loaded_heat_state = heat_info
        self._remove_patched_forward()

    def save(self, state_dir: str) -> None:
        """Persist DualHeat state between tasks."""
        os.makedirs(state_dir, exist_ok=True)
        module_state = {}
        for name, dh_mod in self._dual_modules.items():
            snapshot = dh_mod.get_state_snapshot()
            if snapshot:
                module_state[name] = snapshot

        snapshot = getattr(self, "_train_heat_snapshot", {})
        for name, info in snapshot.items():
            if name not in module_state:
                module_state[name] = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in info.items()}

        payload = {
            "hyperparams": {
                "fast_decay": self.fast_decay,
                "fast_strength": self.fast_strength,
                "fast_decay_rate": self.fast_decay_rate,
                "slow_strength": self.slow_strength,
                "slow_window": self.slow_window,
                "lateral_inhibition": self.lateral_inhibition,
            },
            "module_state": module_state,
        }
        torch.save(payload, os.path.join(state_dir, "dual_heat_state.pt"))
        logger.info("DualHeat state saved: modules=%d", len(module_state))

    def load(self, state_dir: str) -> None:
        """Restore DualHeat state from a previous stage."""
        path = os.path.join(state_dir, "dual_heat_state.pt")
        if not os.path.exists(path):
            logger.info("DualHeat state not found at %s (fresh start)", path)
            return

        payload = torch.load(path, map_location="cpu", weights_only=True)
        hp = payload.get("hyperparams", {})
        self.fast_decay = float(hp.get("fast_decay", self.fast_decay))
        self.fast_strength = float(hp.get("fast_strength", self.fast_strength))
        self.fast_decay_rate = float(hp.get("fast_decay_rate", self.fast_decay_rate))
        self.slow_strength = float(hp.get("slow_strength", self.slow_strength))
        self.slow_window = hp.get("slow_window", self.slow_window)
        self.lateral_inhibition = bool(hp.get("lateral_inhibition", self.lateral_inhibition))

        self._loaded_heat_state = payload.get("module_state", {})
        logger.info("DualHeat state loaded: %d modules from %s", len(self._loaded_heat_state), path)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fast_decay": float(self.fast_decay),
            "fast_strength": float(self.fast_strength),
            "fast_decay_rate": float(self.fast_decay_rate),
            "slow_strength": float(self.slow_strength),
            "slow_window": self.slow_window,
            "lateral_inhibition": bool(self.lateral_inhibition),
            "num_modules": len(self._dual_modules),
        }

    # ─── Extended interface ───────────────────────────────────────────

    def restore_heat_state(self, lora_model: nn.Module) -> None:
        """Apply loaded heat state to live _DualHeatModule instances."""
        loaded = getattr(self, "_loaded_heat_state", None)
        if loaded is None:
            return
        for name, dh_mod in self._dual_modules.items():
            state = loaded.get(name)
            if state is None:
                continue
            dh_mod.load_state_snapshot(state)
        self._loaded_heat_state = None
        logger.info("DualHeat heat state restored: %d modules", len(self._dual_modules))
