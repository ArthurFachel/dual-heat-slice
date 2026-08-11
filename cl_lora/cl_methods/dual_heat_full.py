"""DualHeatFullCLMethod — DualHeat sem LoRA para fine-tuning completo.

Aplica inibicao lateral + EWC per-neuron diretamente nos pesos dos
modelos base (nn.Linear), sem usar PEFT/LoRA.

Algoritmo (por passo de treino):
  1. z = Wx + b                            (pre-ativacao)
  2. output = z / (1 + gamma * mean_others) (inibicao lateral divisiva)
  3. fast_heat = max(0, alpha*|output| + (1-alpha)*fast_heat - delta)
  4. slow_heat += (|output| - slow_heat) / min(n, W)  (capped incremental mean)
  5. backward: grad(W) /= (1 + beta * slow_heat)
     grad(b) /= (1 + beta * slow_heat)

Diferenca do DualHeatCLMethod (LoRA):
  - Opera em nn.Linear direto, nao em adapters PEFT
  - Hook no gradiente do peso W (nao no tensor delta do LoRA)
  - Nao precisa de scaling de LoRA
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import CLMethod

logger = logging.getLogger("cl_lora.cl_methods.dual_heat_full")


# ── Target module patterns para Qwen 2.5 0.5B ────────────────────────
# Queries, Keys, Values, Output, Gate, Up, Down projections
TARGET_MODULE_PATTERNS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def _is_target_module(name: str, patterns: List[str] | None = None) -> bool:
    """Check if a module name matches any target pattern."""
    if patterns is None:
        patterns = TARGET_MODULE_PATTERNS
    return any(p in name for p in patterns)


def _iter_linear_layers(
    model: nn.Module,
    patterns: List[str] | None = None,
) -> List[Tuple[str, nn.Linear]]:
    """Yield (module_name, linear_module) for every target nn.Linear.

    Returns a list so we can index into it deterministically.
    Only yields modules whose name matches a target pattern.
    """
    results: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not _is_target_module(name, patterns):
            continue
        results.append((name, mod))
    return results


def _make_default_hyperparams() -> Dict[str, Any]:
    """Default DualHeat hyperparameters para full fine-tuning.

    Valores ligeiramente diferentes do LoRA (full FT precisa de
    menos regularizacao para nao emperrar o treino).
    """
    return {
        "fast_decay": 0.93,           # alpha — EMA decay for fast heat
        "fast_strength": 1.5,         # gamma — lateral inhibition strength (menor que LoRA)
        "fast_decay_rate": 0.04,      # delta — active decay per step
        "slow_strength": 1.5,         # beta — EWC regularization strength (menor que LoRA)
        "slow_window": None,          # memory window (None = infinite)
        "lateral_inhibition": True,   # enable lateral inhibition
    }


class _DualHeatFullModule(nn.Module):
    """Per-neuron heat tracking for one nn.Linear layer.

    Mantem fast_heat (curto prazo) e slow_heat (longo prazo) para
    cada neuronio de saida. Suporta multiplos dispositivos via
    dicionario per-device.

    Identico ao _DualHeatModule da versao LoRA, mas sem escalonamento
    de LoRA.
    """

    def __init__(
        self,
        out_features: int,
        fast_decay: float = 0.93,
        fast_strength: float = 1.5,
        fast_decay_rate: float = 0.04,
        slow_strength: float = 1.5,
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

        # Heat state keyed by device string: str -> Dict[str, Tensor]
        self._per_device: Dict[str, Dict[str, torch.Tensor]] = {}

    def load_state_snapshot(self, state: Dict[str, torch.Tensor]) -> None:
        """Load heat state from a checkpoint onto CPU."""
        if not state:
            return
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
        """Return merged heat state for checkpointing."""
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
            f"out={self.out_features}, alpha={self.fast_decay}, gamma={self.fast_strength}, "
            f"delta={self.fast_decay_rate}, beta={self.slow_strength}, devices={nd}"
        )


class DualHeatFullCLMethod(CLMethod):
    """DualHeat para full fine-tuning (sem LoRA).

    Operacoes:
      - Inibicao lateral divisiva na saida de cada nn.Linear
      - EWC per-neuron via backward hook no gradiente do peso
      - Rastreio de magnitude (fast/slow heat)

    Uso:
        model = AutoModelForCausalLM.from_pretrained(...)
        cl_method = DualHeatFullCLMethod(slow_strength=2.0)
        cl_method.pre_train(model, stage_idx=1, retain_tasks=None)
        # ... treino ...
        cl_method.post_train(model, ...)
    """

    name = "dual_heat_full"

    def __init__(
        self,
        *,
        fast_decay: float = 0.93,
        fast_strength: float = 1.5,
        fast_decay_rate: float = 0.04,
        slow_strength: float = 1.5,
        slow_window: Optional[int] = None,
        lateral_inhibition: bool = True,
        target_modules: Optional[List[str]] = None,
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
        self.target_module_patterns = target_modules or TARGET_MODULE_PATTERNS

        self._dual_modules: Dict[str, _DualHeatFullModule] = {}
        self._orig_forwards: List[callable] = []
        self._patched_modules: List[nn.Module] = []
        self._ewc_hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ── Hook lifecycle ────────────────────────────────────────────

    def _make_ewc_hook_on_weight(self, name: str):
        """Create a backward hook for the weight tensor (and bias).

        Escala o gradiente por neuronio de saida.
        grad.shape: (out_features, in_features) — escala na dim 0 (out).
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            dh_mod = self._dual_modules.get(name)
            if dh_mod is None:
                return grad
            scale = dh_mod.get_ewc_scale(grad.device, dtype=grad.dtype)
            # grad tem shape (out_features, in_features)
            return grad * scale.view(-1, 1)  # broadcast sobre in_features
        return hook

    def _make_ewc_hook_on_bias(self, name: str):
        """Create a backward hook for the bias tensor."""
        def hook(grad: torch.Tensor) -> torch.Tensor:
            dh_mod = self._dual_modules.get(name)
            if dh_mod is None:
                return grad
            scale = dh_mod.get_ewc_scale(grad.device, dtype=grad.dtype)
            return grad * scale
        return hook

    def _make_patched_forward(self, name: str, out_features: int, dh_mod, mod: nn.Linear):
        """Create a patched forward method for a nn.Linear layer.

        Inclui:
          1. Computa z = Wx + b (normal)
          2. Aplica inibicao lateral na saida
          3. Registra hook EWC no gradiente do peso
          4. Atualiza heat tracking
        """
        def patched_forward(x):
            # Forward normal
            result = nn.functional.linear(x, mod.weight, mod.bias)

            # Lateral inhibition
            if dh_mod.lateral_inhibition and dh_mod.fast_strength > 0.0 \
                    and out_features > 1 and mod.training:
                state = dh_mod._get_or_restore(result.device, dtype=result.dtype)
                fh = state["fast_heat"]
                sum_h = fh.sum()
                mean_others = (sum_h - fh) / float(out_features - 1)
                scale = 1.0 + dh_mod.fast_strength * mean_others
                result = result / scale

            # Heat tracking
            if mod.training:
                dh_mod.update_heat(result)

            return result

        return patched_forward

    def _register_patched_forward(self, model: nn.Module) -> None:
        """Replace each target nn.Linear's forward with a patched version.

        Patched forward inclui:
          - Inibicao lateral na saida
          - Rastreio de magnitude (heat tracking)
        
        Alem disso, registra backward hooks nos pesos para EWC.
        """
        self._remove_patched_forward()

        self._dual_modules = {}
        self._orig_forwards = []
        self._patched_modules = []
        self._ewc_hooks = []

        target_layers = _iter_linear_layers(model, self.target_module_patterns)
        logger.info(
            "DualHeatFull: found %d target linear layers",
            len(target_layers),
        )

        for name, mod in target_layers:
            dh_mod = _DualHeatFullModule(
                out_features=mod.out_features,
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
            mod.forward = self._make_patched_forward(name, mod.out_features, dh_mod, mod)

            # Register backward hooks on weight (and bias) for EWC
            handle_w = mod.weight.register_hook(self._make_ewc_hook_on_weight(name))
            self._ewc_hooks.append(handle_w)
            if mod.bias is not None:
                handle_b = mod.bias.register_hook(self._make_ewc_hook_on_bias(name))
                self._ewc_hooks.append(handle_b)

        logger.info(
            "DualHeatFull patched forwards + EWC hooks registered: %d modules",
            len(self._dual_modules),
        )

    def _remove_patched_forward(self) -> None:
        """Restore original forward methods and remove EWC hooks."""
        for mod, orig_fwd in zip(self._patched_modules, self._orig_forwards):
            mod.forward = orig_fwd
        self._orig_forwards.clear()
        self._patched_modules.clear()

        for handle in self._ewc_hooks:
            handle.remove()
        self._ewc_hooks.clear()

    # ── CLMethod interface ────────────────────────────────────────

    def pre_train(
        self,
        model: torch.nn.Module,
        *,
        stage_idx: int,
        retain_tasks: Optional[List[Any]],
    ) -> None:
        """Patch nn.Linear forward methods with DualHeat logic.

        Args:
            model: O modelo completo (nao PEFT). Deve ter target nn.Linear layers.
        """
        self._register_patched_forward(model)
        logger.info(
            "DualHeatFull pre_train: stage=%d retain_tasks=%s modules=%d",
            stage_idx,
            bool(retain_tasks),
            len(self._dual_modules),
        )

        # Restore heat state after registration
        self.restore_heat_state(model)

    def aux_loss(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        """DualHeatFull nao usa loss aditiva."""
        return None

    def post_train(
        self,
        model: torch.nn.Module,
        *,
        tokenizer: Any,
        train_dataset: Any,
        device: torch.device,
        stage_idx: int,
        task_name: str,
    ) -> None:
        """Capture heat state after training completes."""
        heat_info = {}
        for name, dh_mod in self._dual_modules.items():
            snapshot = dh_mod.get_state_snapshot()
            if snapshot:
                heat_info[name] = snapshot

        logger.info(
            "DualHeatFull post_train: stage=%d task=%s modules=%d",
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
                module_state[name] = {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in info.items()
                }

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
        torch.save(payload, os.path.join(state_dir, "dual_heat_full_state.pt"))
        logger.info("DualHeatFull state saved: modules=%d", len(module_state))

    def load(self, state_dir: str) -> None:
        """Restore DualHeat state from a previous stage."""
        path = os.path.join(state_dir, "dual_heat_full_state.pt")
        if not os.path.exists(path):
            logger.info("DualHeatFull state not found at %s (fresh start)", path)
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
        logger.info(
            "DualHeatFull state loaded: %d modules from %s",
            len(self._loaded_heat_state), path,
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
            "target_patterns": list(self.target_module_patterns),
        }

    # ── Extended interface ────────────────────────────────────────

    def restore_heat_state(self, model: nn.Module) -> None:
        """Apply loaded heat state to live _DualHeatFullModule instances."""
        loaded = getattr(self, "_loaded_heat_state", None)
        if loaded is None:
            return
        for name, dh_mod in self._dual_modules.items():
            state = loaded.get(name)
            if state is None:
                continue
            dh_mod.load_state_snapshot(state)
        self._loaded_heat_state = None
        logger.info(
            "DualHeatFull heat state restored: %d modules",
            len(self._dual_modules),
        )
