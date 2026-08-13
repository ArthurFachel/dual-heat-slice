from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from cl_lora.cl_methods import REGISTRY, build_cl_method
from cl_lora.cl_methods.dual_heat import DualHeatCLMethod
from cl_lora.cl_methods.ewc import EWCMethod
from cl_lora.cl_methods.replay import ReplayMethod
from cl_lora.train import _prepare_cl_train_dataset


def _tiny_lora(dtype: torch.dtype = torch.float32):
    base = nn.Sequential(nn.Linear(3, 2, bias=False)).to(dtype=dtype)
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["0"], bias="none")
    model = get_peft_model(base, config)
    layer = model.base_model.model[0]
    with torch.no_grad():
        layer.lora_A["default"].weight.fill_(0.25)
        layer.lora_B["default"].weight.fill_(0.5)
    return model, layer


def test_dual_heat_preserves_native_fp16_and_disabled_adapter_forward():
    model, _layer = _tiny_lora(torch.float16)
    model.train()
    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float16)
    native_enabled = model(x).detach()
    with model.disable_adapter():
        native_disabled = model(x).detach()

    method = DualHeatCLMethod(lateral_inhibition=False, slow_strength=0.0)
    method.pre_train(model, stage_idx=1, retain_tasks=None)

    actual_enabled = model(x).detach()
    with model.disable_adapter():
        actual_disabled = model(x).detach()

    assert actual_enabled.dtype == torch.float16
    torch.testing.assert_close(actual_enabled, native_enabled)
    torch.testing.assert_close(actual_disabled, native_disabled)


def test_sensitivity_observes_post_inhibition_gradient_before_protection():
    model, layer = _tiny_lora()
    model.train()
    method = DualHeatCLMethod(
        importance="sensitivity",
        fast_strength=2.0,
        slow_strength=1.0,
        fast_decay_rate=0.0,
    )
    method.pre_train(model, stage_idx=1, retain_tasks=None)
    tracker = next(iter(method._dual_modules.values()))
    state = tracker._get_or_restore(torch.device("cpu"), torch.float32)
    state["fast_heat"].copy_(torch.tensor([0.0, 1.0]))
    state["slow_heat"].fill_(1.0)

    x = torch.tensor([[1.0, 0.0, 0.0]])
    base = layer.base_layer(x).detach()
    # The configured fast heat gives inhibition scales [3, 1].
    with torch.no_grad():
        raw_delta = (
            layer.lora_B["default"](layer.lora_A["default"](x))
            * layer.scaling["default"]
        )
        expected_signal = (raw_delta / torch.tensor([3.0, 1.0])).abs().squeeze(0)
    model.zero_grad()
    (model(x) - base).sum().backward()

    torch.testing.assert_close(state["slow_heat"], expected_signal)


def test_ablation_aliases_override_conflicting_generic_kwargs():
    activation = build_cl_method(
        "activation_protection", importance="sensitivity", lateral_inhibition=True
    )
    sensitivity = build_cl_method(
        "sensitivity_protection", importance="activation", lateral_inhibition=True
    )
    inhibition = build_cl_method(
        "lateral_inhibition", slow_strength=9.0, lateral_inhibition=False
    )

    assert activation.importance == "activation"
    assert activation.lateral_inhibition is False
    assert sensitivity.importance == "sensitivity"
    assert sensitivity.lateral_inhibition is False
    assert inhibition.slow_strength == 0.0
    assert inhibition.lateral_inhibition is True


def test_train_pipeline_calls_cl_dataset_preparation_before_tokenization():
    class MarkerMethod:
        def prepare_train_dataset(self, dataset):
            return dataset + ["replay"]

    seen = []

    def tokenize(dataset, tokenizer, max_length):
        seen.append(list(dataset))
        return dataset

    result = _prepare_cl_train_dataset(
        ["current"], MarkerMethod(), tokenizer=None, max_length=8, tokenize_fn=tokenize
    )

    assert result == ["current", "replay"]
    assert seen == [["current", "replay"]]


def test_train_pipeline_keeps_raw_dataset_for_replay_post_train():
    raw = [{"text": "current", "target": "answer"}]
    prepared = raw + [{"text": "replay", "target": "old"}]

    class MarkerMethod:
        def prepare_train_dataset(self, dataset):
            assert dataset is raw
            return prepared

    tokenized, original = _prepare_cl_train_dataset(
        raw,
        MarkerMethod(),
        tokenizer=None,
        max_length=8,
        tokenize_fn=lambda dataset, **_: [row["text"] for row in dataset],
        return_original=True,
    )

    assert tokenized == ["current", "replay"]
    assert original is raw


def test_replay_reservoir_num_seen_persists_across_save_load(tmp_path):
    method = ReplayMethod(replay_size=2, seed=7)
    model = nn.Linear(1, 1)
    method.post_train(
        model,
        tokenizer=None,
        train_dataset=[{"text": str(i)} for i in range(5)],
        device=torch.device("cpu"),
        stage_idx=1,
        task_name="one",
    )
    assert method.num_seen == 5
    method.save(str(tmp_path))

    resumed = ReplayMethod(replay_size=2, seed=7)
    resumed.load(str(tmp_path))
    assert resumed.num_seen == 5
    resumed.post_train(
        model,
        tokenizer=None,
        train_dataset=[{"text": "next"}],
        device=torch.device("cpu"),
        stage_idx=2,
        task_name="two",
    )
    assert resumed.num_seen == 6


def test_ewc_normalizes_fisher_and_consolidates_same_parameter_coordinates():
    model = nn.Linear(1, 1, bias=False)
    method = EWCMethod(lambda_ewc=1.0)

    method.pre_train(model, stage_idx=1, retain_tasks=None)
    for _ in range(2):
        model(torch.ones(1, 1)).sum().backward()
        model.zero_grad()
    method.post_train(
        model,
        tokenizer=None,
        train_dataset=None,
        device=torch.device("cpu"),
        stage_idx=1,
        task_name="one",
    )
    first_fisher = method._importance["weight"].clone()
    assert method._fisher_samples == 2
    torch.testing.assert_close(first_fisher, torch.ones_like(first_fisher))

    method.pre_train(model, stage_idx=2, retain_tasks=["one"])
    model(torch.ones(1, 1)).sum().backward()
    model.zero_grad()
    method.post_train(
        model,
        tokenizer=None,
        train_dataset=None,
        device=torch.device("cpu"),
        stage_idx=2,
        task_name="two",
    )

    assert len(method._anchors) == 1
    torch.testing.assert_close(method._importance["weight"], first_fisher + 1.0)


def test_lora_registry_excludes_full_finetune():
    assert "full_finetune" not in REGISTRY
