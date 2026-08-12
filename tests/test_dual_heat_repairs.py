from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from cl_lora.cl_methods.dual_heat import DualHeatCLMethod, _DualHeatModule
from cl_lora.cl_methods.dual_heat_full import DualHeatFullCLMethod, _DualHeatFullModule


class _TinyFullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.self_attn.q_proj(x)


def _tiny_lora_model(*, base_dtype=torch.float16):
    from peft import LoraConfig, get_peft_model

    model = nn.Sequential(nn.Linear(4, 3, bias=False, dtype=base_dtype))
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["0"], bias="none")
    return get_peft_model(model, config)


@pytest.mark.parametrize("tracker_cls", [_DualHeatModule, _DualHeatFullModule])
def test_first_heat_update_is_finite(tracker_cls):
    tracker = tracker_cls(out_features=3)

    tracker.update_heat(torch.tensor([[1.0, 2.0, 3.0]]))

    state = tracker.get_state_snapshot()
    assert torch.isfinite(state["fast_heat"]).all()
    assert torch.isfinite(state["slow_heat"]).all()
    assert state["slow_n"].item() == 1
    assert torch.isfinite(tracker.get_ewc_scale(torch.device("cpu"))).all()


@pytest.mark.parametrize("tracker_cls", [_DualHeatModule, _DualHeatFullModule])
def test_heat_state_is_one_fp32_state_per_device(tracker_cls):
    tracker = tracker_cls(out_features=3, slow_strength=2.0)
    tracker.update_heat(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16))

    scale = tracker.get_ewc_scale(torch.device("cpu"), dtype=torch.float32)

    assert list(tracker._per_device) == ["cpu"]
    state = tracker._per_device["cpu"]
    assert state["fast_heat"].dtype == torch.float32
    assert state["slow_heat"].dtype == torch.float32
    assert state["slow_n"].dtype == torch.long
    assert state["step"].dtype == torch.long
    torch.testing.assert_close(scale, 1.0 / (1.0 + 2.0 * state["slow_heat"]))


@pytest.mark.parametrize("tracker_cls", [_DualHeatModule, _DualHeatFullModule])
def test_loaded_heat_is_canonicalized_to_fp32(tracker_cls):
    tracker = tracker_cls(out_features=2)
    tracker.load_state_snapshot(
        {
            "fast_heat": torch.ones(2, dtype=torch.float16),
            "slow_heat": torch.ones(2, dtype=torch.float16),
            "slow_n": torch.tensor(1.0),
            "step": torch.tensor(1),
        }
    )

    state = tracker.get_state_snapshot()

    assert state["fast_heat"].dtype == torch.float32
    assert state["slow_heat"].dtype == torch.float32


def test_lora_forward_preserves_peft_moduledict_and_dtype_behavior():
    torch.manual_seed(0)
    model = _tiny_lora_model()
    method = DualHeatCLMethod()
    method.pre_train(model, stage_idx=1, retain_tasks=None)
    lora_layer = next(mod for mod in model.modules() if hasattr(mod, "lora_A"))

    assert isinstance(lora_layer.lora_dropout, nn.ModuleDict)
    result = model(torch.ones(2, 4, dtype=torch.float16))

    assert result.dtype == torch.float16
    assert torch.isfinite(result).all()


def _finish_stage(method, model, stage_idx):
    model(torch.ones(2, 4)).sum().backward()
    method.post_train(
        model,
        tokenizer=None,
        train_dataset=None,
        device=torch.device("cpu"),
        stage_idx=stage_idx,
        task_name=f"task-{stage_idx}",
    )


def test_full_heat_survives_uninterrupted_stage_transition():
    model = _TinyFullModel()
    method = DualHeatFullCLMethod(lateral_inhibition=False)
    method.pre_train(model, stage_idx=1, retain_tasks=None)
    _finish_stage(method, model, 1)
    expected = method._train_heat_snapshot["self_attn.q_proj"]["slow_heat"].clone()

    method.pre_train(model, stage_idx=2, retain_tasks=["task-1"])
    restored = method._dual_modules["self_attn.q_proj"].get_state_snapshot()

    torch.testing.assert_close(restored["slow_heat"], expected)


def test_full_uninterrupted_and_resumed_state_are_equivalent(tmp_path):
    model = _TinyFullModel()
    uninterrupted = DualHeatFullCLMethod(lateral_inhibition=False)
    uninterrupted.pre_train(model, stage_idx=1, retain_tasks=None)
    _finish_stage(uninterrupted, model, 1)
    uninterrupted.save(str(tmp_path))
    uninterrupted.pre_train(model, stage_idx=2, retain_tasks=["task-1"])

    resumed = DualHeatFullCLMethod(lateral_inhibition=False)
    resumed.load(str(tmp_path))
    resumed.pre_train(_TinyFullModel(), stage_idx=2, retain_tasks=["task-1"])

    left = uninterrupted._dual_modules["self_attn.q_proj"].get_state_snapshot()
    right = resumed._dual_modules["self_attn.q_proj"].get_state_snapshot()
    for key in left:
        torch.testing.assert_close(left[key], right[key])


def test_lora_heat_survives_uninterrupted_stage_transition():
    model = _tiny_lora_model(base_dtype=torch.float32)
    method = DualHeatCLMethod(lateral_inhibition=False)
    method.pre_train(model, stage_idx=1, retain_tasks=None)
    _finish_stage(method, model, 1)
    name = next(iter(method._train_heat_snapshot))
    expected = method._train_heat_snapshot[name]["slow_heat"].clone()

    method.pre_train(model, stage_idx=2, retain_tasks=["task-1"])

    torch.testing.assert_close(
        method._dual_modules[name].get_state_snapshot()["slow_heat"], expected
    )


def test_lora_fails_loudly_when_no_modules_match():
    method = DualHeatCLMethod()
    with pytest.raises(RuntimeError, match="zero LoRA modules"):
        method.pre_train(nn.Linear(2, 2), stage_idx=1, retain_tasks=None)


def test_full_fails_loudly_when_no_modules_match():
    method = DualHeatFullCLMethod()
    with pytest.raises(RuntimeError, match="zero target modules"):
        method.pre_train(nn.Linear(2, 2), stage_idx=1, retain_tasks=None)
