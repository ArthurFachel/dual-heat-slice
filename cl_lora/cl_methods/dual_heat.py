"""
DualHeatCLMethod — Inibição lateral + EWC per-neuron para Continual Learning.

Baseado no DualHeat original (dual_heat_module.py) adaptado para o pipeline
PEFT/LoRA deste projeto.

Algoritmo (por passo de treino):
  1. z = Wx + b                          (pré-ativação)
  2. output = z / (1 + γ·mean_others)    (inibição lateral divisiva, opcional)
  3. fast_heat = max(0, α·|output| + (1-α)·fast_heat − δ)
  4. slow_heat += (|output| − slow_heat) / min(n, W)  (capped incremental mean)
  5. grad /= (1 + β·slow_heat)           (EWC gradient hook em lora_B.weight)

Referência:
    dual_heat_module.py — v3: Inibição lateral + decay ativo + pós-inibição + EWC
    + slow heat com memória limitada (forgetting)

Integração no pipeline:
    pre_train()  → registra hooks forward + backward nos módulos LoRA
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
    """Container for one LoRA module's heat tracking state.

    Each instance manages fast_heat, slow_heat, and slow_n for one PEFT
    LoRA linear module.  Registered as a child of the CLMethod so it is
    discoverable for save/load.
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

        # Fast heat (pós-inibição, com decay ativo)
        self.register_buffer("fast_heat", torch.zeros(out_features))
        # Slow heat (média amostral com janela limitada opcional)
        self.register_buffer("slow_heat", torch.zeros(out_features))
        self.register_buffer("slow_n", torch.ones(1, dtype=torch.long))
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_heat(self, output: torch.Tensor) -> None:
        """Update fast_heat and slow_heat from the current output.

        Called from the forward hook on each training step.
        """
        # |output| médio sobre batch (e seq, se 3D)
        # output shape: (..., out_features)
        reduce_dims = tuple(range(output.dim() - 1))
        post_mag = output.detach().abs().mean(dim=reduce_dims)  # (out_features,)

        # Ensure buffers are on the same device as post_mag
        target_device = post_mag.device
        if self.fast_heat.device != target_device:
            self.fast_heat = self.fast_heat.to(target_device)
            self.slow_heat = self.slow_heat.to(target_device)
            self.slow_n = self.slow_n.to(target_device)
            self._step = self._step.to(target_device)

        # Fast: EMA + decay ativo
        self.fast_heat.mul_(self.fast_decay).add_(
            post_mag, alpha=1.0 - self.fast_decay
        ).sub_(self.fast_decay_rate).clamp_(min=0.0)

        # Slow: capped incremental mean
        n_true = self.slow_n.item()
        n_eff = min(n_true, self.slow_window) if self.slow_window is not None else n_true
        self.slow_heat.add_((post_mag - self.slow_heat) / float(n_eff))
        self.slow_n += 1
        self._step += 1

    def get_ewc_scale(self) -> torch.Tensor:
        """Return per-neuron EWC scale: 1 / (1 + beta * slow_heat)."""
        if self.slow_strength <= 0.0:
            return torch.ones(self.out_features, device=self.slow_heat.device)
        return 1.0 / (1.0 + self.slow_strength * self.slow_heat)

    def extra_repr(self) -> str:
        w = self.slow_window if self.slow_window is not None else "∞"
        return (
            f"out={self.out_features}, α={self.fast_decay}, γ={self.fast_strength}, "
            f"δ={self.fast_decay_rate}, β={self.slow_strength}, slow_window={w}"
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

        # _dual_modules: dict[module_name -> _DualHeatModule]
        self._dual_modules: Dict[str, _DualHeatModule] = {}
        # Saved forward_hook_handle references for removal between stages
        self._fwd_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._bwd_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Store active adapter name from pre_train for reuse in hooks
        self._active_adapter: str = "default"

    # ─── Hook lifecycle ────────────────────────────────────────────────

    def _make_ewc_hook(self, name: str, B_w: nn.Parameter):
        """Create a backward hook that scales lora_B's gradient per output neuron."""

        def hook(grad: torch.Tensor) -> torch.Tensor:
            dh_mod = self._dual_modules.get(name)
            if dh_mod is None:
                return grad
            scale = dh_mod.get_ewc_scale()  # (out_features,)
            # Move scale to grad's device if needed
            if scale.device != grad.device:
                scale = scale.to(grad.device)
            # grad shape: (out_features, r)
            return grad * scale.view(-1, 1).to(dtype=grad.dtype, device=grad.device)

        return hook

    def _make_forward_hook(self, name: str, out_features: int):
        """Create a forward hook that applies lateral inhibition and tracks heat.

        This hook is registered on the output of the LoRA module's forward pass.
        It:
          1. (Optional) applies lateral inhibition to the combined output
          2. Tracks |output| magnitudes for fast_heat/slow_heat updates
        """

        def hook(module, inputs, output):
            dh_mod = self._dual_modules.get(name)
            if dh_mod is None:
                return output

            # Ensure heat buffers are on the same device as module output
            target_device = output.device
            if dh_mod.fast_heat.device != target_device:
                dh_mod.fast_heat = dh_mod.fast_heat.to(target_device)
                dh_mod.slow_heat = dh_mod.slow_heat.to(target_device)
                dh_mod.slow_n = dh_mod.slow_n.to(target_device)
                dh_mod._step = dh_mod._step.to(target_device)

            output_to_track = output

            # Lateral inhibition: output /= (1 + gamma * mean_others)
            if dh_mod.lateral_inhibition and dh_mod.fast_strength > 0.0 and out_features > 1 and module.training:
                with torch.no_grad():
                    fast_heat = dh_mod.fast_heat.to(output.device, output.dtype)
                    sum_h = fast_heat.sum()
                    mean_others = (sum_h - fast_heat) / float(out_features - 1)
                    scale = 1.0 + dh_mod.fast_strength * mean_others  # (out_features,)
                    output = output / scale
                output_to_track = output

            # Track heat (only in training mode)
            if module.training:
                dh_mod.update_heat(output_to_track)

            return output

        return hook

    def _register_hooks(self, lora_model: nn.Module) -> None:
        """Register forward and backward hooks on all active LoRA modules."""
        # Remove any existing hooks first
        self._remove_hooks()

        self._dual_modules = {}
        self._fwd_handles = []
        self._bwd_handles = []

        for name, mod, A_w, B_w, out_features in _iter_lora_modules(
            lora_model, active_adapter=self._active_adapter
        ):
            # Create heat tracking state for this module
            dh_mod = _DualHeatModule(
                out_features=out_features,
                fast_decay=self.fast_decay,
                fast_strength=self.fast_strength,
                fast_decay_rate=self.fast_decay_rate,
                slow_strength=self.slow_strength,
                slow_window=self.slow_window,
                lateral_inhibition=self.lateral_inhibition,
            )
            # Move to model device
            try:
                dev = next(lora_model.parameters()).device
            except StopIteration:
                dev = torch.device("cpu")
            dh_mod = dh_mod.to(dev)

            # Store in dict (CLMethod is not an nn.Module, so no add_module)
            self._dual_modules[name] = dh_mod

            # Forward hook (tracks heat, optionally applies lateral inhibition)
            fwd_hook = self._make_forward_hook(name, out_features)
            self._fwd_handles.append(mod.register_forward_hook(fwd_hook))

            # Backward hook on lora_B.weight for EWC scaling
            bwd_hook = self._make_ewc_hook(name, B_w)
            self._bwd_handles.append(B_w.register_hook(bwd_hook))

        logger.info(
            "DualHeat hooks registered: %d modules", len(self._dual_modules)
        )

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._fwd_handles:
            h.remove()
        self._fwd_handles.clear()
        for h in self._bwd_handles:
            h.remove()
        self._bwd_handles.clear()

    # ─── CLMethod interface ────────────────────────────────────────────

    def pre_train(
        self,
        lora_model: torch.nn.Module,
        *,
        stage_idx: int,
        retain_tasks: Optional[List[Any]],
    ) -> None:
        """Register DualHeat hooks before training begins."""
        # Detect the active adapter name from the PEFT model
        active = getattr(lora_model, "active_adapter", "default")
        if isinstance(active, (list, tuple)):
            active = active[0] if active else "default"
        self._active_adapter = str(active)

        self._register_hooks(lora_model)
        logger.info(
            "DualHeat pre_train: stage=%d retain_tasks=%s modules=%d",
            stage_idx,
            bool(retain_tasks),
            len(self._dual_modules),
        )

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
        heat_info = {
            name: {
                "slow_heat": dh_mod.slow_heat.detach().cpu().clone(),
                "fast_heat": dh_mod.fast_heat.detach().cpu().clone(),
                "slow_n": dh_mod.slow_n.detach().cpu().clone().item(),
                "step": dh_mod._step.detach().cpu().clone().item(),
            }
            for name, dh_mod in self._dual_modules.items()
        }
        logger.info(
            "DualHeat post_train: stage=%d task=%s modules=%d",
            stage_idx,
            task_name,
            len(heat_info),
        )
        self._train_heat_snapshot = heat_info

        # Remove hooks — next stage will re-register
        self._remove_hooks()

    def save(self, state_dir: str) -> None:
        """Persist DualHeat state between tasks.

        Saves:
          - Hyperparameters
          - All dual module heat buffers (fast_heat, slow_heat, slow_n)
          - Number of training steps accumulated
        """
        os.makedirs(state_dir, exist_ok=True)
        module_state = {}
        for name, dh_mod in self._dual_modules.items():
            module_state[name] = {
                "fast_heat": dh_mod.fast_heat.detach().cpu().clone(),
                "slow_heat": dh_mod.slow_heat.detach().cpu().clone(),
                "slow_n": dh_mod.slow_n.detach().cpu().clone(),
                "_step": dh_mod._step.detach().cpu().clone(),
            }

        # Also include the snapshot from post_train (in case save is called
        # after hooks are removed)
        snapshot = getattr(self, "_train_heat_snapshot", {})
        if snapshot:
            for name, info in snapshot.items():
                if name not in module_state:
                    module_state[name] = info

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
        logger.info(
            "DualHeat state saved: modules=%d", len(module_state)
        )

    def load(self, state_dir: str) -> None:
        """Restore DualHeat state from a previous stage."""
        path = os.path.join(state_dir, "dual_heat_state.pt")
        if not os.path.exists(path):
            logger.info("DualHeat state not found at %s (fresh start)", path)
            return

        payload = torch.load(path, map_location="cpu", weights_only=False)
        hp = payload.get("hyperparams", {})
        self.fast_decay = float(hp.get("fast_decay", self.fast_decay))
        self.fast_strength = float(hp.get("fast_strength", self.fast_strength))
        self.fast_decay_rate = float(hp.get("fast_decay_rate", self.fast_decay_rate))
        self.slow_strength = float(hp.get("slow_strength", self.slow_strength))
        self.slow_window = hp.get("slow_window", self.slow_window)
        self.lateral_inhibition = bool(hp.get("lateral_inhibition", self.lateral_inhibition))

        module_state = payload.get("module_state", {})
        # We cannot rebuild _dual_modules here because we don't have the
        # model structure yet.  The state will be re-applied in pre_train()
        # when hooks are registered.
        self._loaded_module_state = module_state
        logger.info(
            "DualHeat state loaded: %d modules from %s",
            len(module_state),
            path,
        )

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

    # ─── Extended interface for pipeline ───────────────────────────────

    def restore_heat_state(self, lora_model: nn.Module) -> None:
        """Apply loaded heat state to live _DualHeatModule instances.

        Called from pre_train AFTER hooks are registered, so that heat
        tracking from previous tasks carries forward.
        """
        loaded = getattr(self, "_loaded_module_state", None)
        if loaded is None:
            return
        for name, dh_mod in self._dual_modules.items():
            state = loaded.get(name)
            if state is None:
                continue
            with torch.no_grad():
                dh_mod.fast_heat.copy_(state["fast_heat"].to(dh_mod.fast_heat.device))
                dh_mod.slow_heat.copy_(state["slow_heat"].to(dh_mod.slow_heat.device))
                dh_mod.slow_n.copy_(state["slow_n"].to(dh_mod.slow_n.device))
                if "_step" in state:
                    dh_mod._step.copy_(state["_step"].to(dh_mod._step.device))
        # Clear the loaded state so it isn't applied twice
        self._loaded_module_state = None
        logger.info("DualHeat heat state restored: %d modules", len(self._dual_modules))


# Patch pre_train to restore heat state after registration.
# This is done here so DualHeatCLMethod remains self-contained.
_original_pre_train = DualHeatCLMethod.pre_train

def _patched_pre_train(self, lora_model, *, stage_idx, retain_tasks):
    _original_pre_train(self, lora_model, stage_idx=stage_idx, retain_tasks=retain_tasks)
    self.restore_heat_state(lora_model)

DualHeatCLMethod.pre_train = _patched_pre_train
