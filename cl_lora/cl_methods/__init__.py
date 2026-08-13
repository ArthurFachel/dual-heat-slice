"""Continual-learning training methods (composable with any LoRA init).

Public entry points:
  - REGISTRY: name -> CLMethod class
  - build_cl_method(name, **kwargs): factory used by orchestrator/train.
"""
from __future__ import annotations

from typing import Any, Dict, Type

from .base import CLMethod
from .dual_heat import DualHeatCLMethod
from .dual_heat_full import DualHeatFullCLMethod
from .o_lora import OLoRAMethod
from .vanilla import VanillaCLMethod
from .ewc import EWCMethod
from .replay import ReplayMethod


REGISTRY: Dict[str, Type[CLMethod]] = {
    "vanilla": VanillaCLMethod,
    "o_lora": OLoRAMethod,
    "dual_heat": DualHeatCLMethod,
    "dual_heat_full": DualHeatFullCLMethod,
    "activation_protection": DualHeatCLMethod,
    "sensitivity_protection": DualHeatCLMethod,
    "lateral_inhibition": DualHeatCLMethod,
    "ewc": EWCMethod,
    "replay": ReplayMethod,
}


def build_cl_method(name: str, **kwargs: Any) -> CLMethod:
    """Instantiate a CL method by registry name. Unknown kwargs are ignored
    by methods that don't accept them, so a single argparse namespace can be
    forwarded to any method without per-method dispatch in callers."""
    key = (name or "vanilla").lower()
    if key == "activation_protection":
        kwargs["importance"] = "activation"
        kwargs["lateral_inhibition"] = False
    elif key == "sensitivity_protection":
        kwargs["importance"] = "sensitivity"
        kwargs["lateral_inhibition"] = False
    elif key == "lateral_inhibition":
        kwargs["slow_strength"] = 0.0
        kwargs["lateral_inhibition"] = True
    if key == "full_finetune":
        # Backward-compatible direct factory access for the separate full
        # pipeline, without advertising it to the LoRA orchestrator registry.
        from .full_finetune import FullFineTuneMethod

        cls = FullFineTuneMethod
    elif key in REGISTRY:
        cls = REGISTRY[key]
    else:
        raise ValueError(
            f"Unknown CL method: {name!r}. Available: {sorted(REGISTRY.keys())}"
        )
    accepted = _accepted_kwargs(cls)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    instance = cls(**filtered)
    if key in {"activation_protection", "sensitivity_protection", "lateral_inhibition"}:
        instance.name = key
    return instance


def _accepted_kwargs(cls: Type[CLMethod]) -> set:
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set()
    return {p for p in sig.parameters if p not in {"self", "args", "kwargs"}}


__all__ = [
    "CLMethod",
    "DualHeatCLMethod",
    "OLoRAMethod",
    "REGISTRY",
    "VanillaCLMethod",
    "build_cl_method",
]
