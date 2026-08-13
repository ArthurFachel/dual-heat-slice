from __future__ import annotations

import torch
import torch.nn as nn

from cl_lora.cl_methods import build_cl_method
from cl_lora.cl_methods.ewc import EWCMethod
from cl_lora.cl_methods.replay import ReplayMethod


def test_requested_methods_are_registered():
    names = {
        "full_finetune",
        "vanilla",
        "activation_protection",
        "sensitivity_protection",
        "lateral_inhibition",
        "replay",
        "ewc",
    }
    for name in names:
        assert build_cl_method(name).name == name


def test_ewc_snapshots_and_aux_loss_protect_previous_parameters():
    model = nn.Linear(3, 2)
    method = EWCMethod(lambda_ewc=2.0)
    method.post_train(model, tokenizer=None, train_dataset=None, device=torch.device("cpu"), stage_idx=1, task_name="a")
    with torch.no_grad():
        model.weight.add_(1.0)
    method.pre_train(model, stage_idx=2, retain_tasks=["a"])
    loss = method.aux_loss(model)
    assert loss is not None
    assert loss.item() > 0


def test_replay_method_combines_previous_examples():
    method = ReplayMethod(replay_size=2)
    current = [{"text": "current"}]
    previous = [{"text": "old-1"}, {"text": "old-2"}, {"text": "old-3"}]
    combined = method.mix_examples(current, [previous])
    assert len(combined) == 3
    assert any(row["text"] == "old-1" for row in combined)
