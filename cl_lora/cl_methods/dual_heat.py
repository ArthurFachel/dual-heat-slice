"""DualHeatCLMethod — Inibição lateral + EWC per-neuron para Continual Learning.

Baseado no DualHeat original (dual_heat_module.py) adaptado para o pipeline
PEFT/LoRA deste projeto.

CORREÇÃO: O EWC hooka o tensor delta (contribuição LoRA), NÃO lora_B.weight.
Isso garante proteção simétrica sobre A e B, equivalente ao MLP original.

Algoritmo (por passo de treino):
  1. base = W_base(x) + b_base           (pré-treinado, congelado)
  2. delta = scaling · B(A(x))            (adaptador LoRA)
  3. z = base + delta                     (saída combinada)
  4. output = z / (1 + γ·mean_others)     (inibição lateral divisiva, opcional)
  5. fast_heat = max(0, α·|output| + (1-α)·fast_heat − δ)
  6. slow_heat += (|output| − slow_heat) / min(n, W)  (capped incremental mean)
  7. backward: grad(delta) /= (1 + β·slow_heat)
     → via chain rule, tanto B quanto A recebem gradiente escalado

Referência:
    dual_heat_module.py — v3: Inibição lateral + decay ativo + pós-inibição + EWC
    + slow heat com memória limitada (forgetting)
    dual_heat_LoRA_module.py — Hook no tensor delta (proteção simétrica)

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
            "fast_heat": state["fast_heat"].clone(),
            "slow_heat": state["slow_heat"].clone(),
            "slow_n": state["slow_n"].clone(),
            "step": state["step"].clone() if "step" in state else torch.zeros((), dtype=torch.long),
        }

    def _get_or_restore(self, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Get per-device state, initializing from loaded CPU state if available."""
        key = f"{str(device)}_{dtype}"
        if key in self._per_device:
            return self._per_device[key]

        loaded = getattr(self, "_loaded_cpu_state", None)
        if loaded is not None:
            # Transfer loaded CPU tensors to target device/dtype
            d = {}
            for k, v in loaded.items():
                if k in ("fast_heat", "slow_heat"):
                    d[k] = v.to(device=device, dtype=dtype)
                else:
                    d[k] = v.to(device=device)
            self._per_device[key] = d
            self._loaded_cpu_state = None
        else:
            self._per_device[key] = {
                "fast_heat": torch.zeros(self.out_features, device=device, dtype=dtype),
                "slow_heat": torch.zeros(self.out_features, device=device, dtype=dtype),
                "slow_n": torch.zeros((), device=device),
                "step": torch.zeros((), dtype=torch.long, device=device),
            }
        return self._per_device[key]

    @torch.no_grad()
    def update_heat(self, output: torch.Tensor) -> None:
        """Update fast_heat and slow_heat from the current output."""
        state = self._get_or_restore(output.device, dtype=output.dtype)

        reduce_dims = tuple(range(output.dim() - 1))
        post_mag = output.detach().abs().mean(dim=reduce_dims)

        # Fast: EMA + decay ativo
        state["fast_heat"].mul_(self.fast_decay).add_(
            (1.0 - self.fast_decay) * post_mag, alpha=1.0
        ).sub_(self.fast_decay_rate).clamp_(min=0.0)

        # Slow: capped incremental mean
        n_true = state["slow_n"].item()
        n_eff = min(n_true, self.slow_window) if self.slow_window is not None else n_true
        state["slow_heat"].add_((post_mag - state["slow_heat"]) / float(n_eff))
        state["slow_n"] += 1
        state["step"] += 1

    def get_ewc_scale(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return per-neuron EWC scale: 1 / (1 + beta * slow_heat)."""
        if self.slow_strength <= 0.0:
            return torch.ones(self.out_features, device=device, dtype=dtype)
        state = self._get_or_restore(device, dtype=dtype)
        return 1.0 / (1.0 + self.slow_strength * state["slow_heat"])

    def get_state_snapshot(self) -> Dict[str, torch.Tensor]:
        """Return merged heat state for checkpointing.

        Takes the first available device's state (all should be equivalent
        after single-device training; under DataParallel, any device works).
        """
        if not self._per_device:
            return {}
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

    def _make_ewc_hook_on_delta(self, name: str):
        """Create a backward hook for the delta tensor (NOT lora_B.weight).

        This hook is registered on the delta tensor during the forward pass.
        It scales the gradient per output neuron, and via the chain rule
        both lora_B and lora_A receive the scaled gradient — equivalent to
        the original DualHeat hook on self.weight.

        Grad shape: (..., out_features) — scale broadcasts over last dim.
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            dh_mod = self._dual_modules.get(name)
            if dh_mod is None:
                return grad
            scale = dh_mod.get_ewc_scale(grad.device, dtype=grad.dtype)
            return grad * scale  # broadcast over (..., out_features)

        return hook

    def _make_patched_forward(self, name: str, out_features: int, dh_mod):
        """Create a patched forward method for a LoRA module.

        This replaces the PEFT module's forward to:
          1. Compute delta = scaling * B(A(x)) for each active adapter
          2. Register EWC hook on delta (not on lora_B.weight)
          3. Apply lateral inhibition on the full output (base + delta)
          4. Track heat magnitudes
        """
        def patched_forward(mod_self, x, *args, **kwargs):
            # ── Base layer forward ──────────────────────────────────────
            # mod_self is the PEFT Linear (the module whose forward we patch)
            result = mod_self.base_layer(x, *args, **kwargs)

            # ── LoRA adapters with EWC hook on delta ───────────────────
            for adapter in mod_self.active_adapters:
                if adapter not in mod_self.lora_A:
                    continue
                lora_A = mod_self.lora_A[adapter]
                lora_B = mod_self.lora_B[adapter]
                dropout = mod_self.lora_dropout.get(adapter, lambda x: x)
                scaling = mod_self.scaling[adapter]

                x_for_lora = dropout(x)
                lora_A_out = lora_A(x_for_lora)
                delta = lora_B(lora_A_out) * scaling

                # Register EWC hook on delta (NOT on lora_B.weight)
                if dh_mod.slow_strength > 0.0 and delta.requires_grad:
                    delta.register_hook(self._make_ewc_hook_on_delta(name))

                result = result + delta

            # ── Lateral inhibition ──────────────────────────────────────
            if dh_mod.lateral_inhibition and dh_mod.fast_strength > 0.0 \
                    and out_features > 1 and mod_self.training:
                state = dh_mod._get_or_restore(result.device, dtype=result.dtype)
                fh = state["fast_heat"]
                sum_h = fh.sum()
                mean_others = (sum_h - fh) / float(out_features - 1)
                scale = 1.0 + dh_mod.fast_strength * mean_others
                result = result / scale

            # ── Heat tracking ───────────────────────────────────────────
            if mod_self.training:
                dh_mod.update_heat(result)

            return result

        return patched_forward

    def _register_patched_forward(self, lora_model: nn.Module) -> None:
        """Replace each LoRA module's forward with a patched version.

        The patched forward includes:
          - EWC hook on the delta tensor (protects A and B)
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
            self._orig_forwards.append(mod.forward)
            self._patched_modules.append(mod)
            mod.forward = self._make_patched_forward(name, out_features, dh_mod)

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
