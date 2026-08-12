from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from cl_lora.metrics import compute_cl_metrics
from cl_lora.orchestrator_full import _collect_env_info
from cl_lora.train import load_base_model
from cl_lora.train_full import _build_training_arguments


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_full_stage_schema_produces_metrics():
    stages = [
        {"stage": 1, "trained_task": "A", "seen_tasks": {"A": {"score": 0.8}}, "general": {}},
        {"stage": 2, "trained_task": "B", "seen_tasks": {"A": {"score": 0.5}, "B": {"score": 0.9}},
         "general": {"gp_mean": 0.4, "ip_mean": 0.3}},
    ]
    summary = compute_cl_metrics(stages, ["A", "B"])
    assert summary["metrics"] == pytest.approx(
        {"AP": 0.85, "FP": 0.7, "Forget": 0.15, "GP": 0.4, "IP": 0.3}
    )


def test_full_training_arguments_honor_precision_and_cli_values(tmp_path):
    with patch("transformers.training_args.is_torch_bf16_gpu_available", return_value=True):
        args = _build_training_arguments(
            output_path=tmp_path,
            learning_rate=3e-4,
            num_train_epochs=1.5,
            warmup_ratio=0.2,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=7,
            logging_steps=9,
            eval_steps=11,
            seed=13,
            use_bf16=True,
        )
    assert args.learning_rate == 3e-4
    assert args.num_train_epochs == 1.5
    assert args.per_device_train_batch_size == 3
    assert args.gradient_accumulation_steps == 7
    assert args.warmup_ratio == 0.2
    assert args.bf16 is True and args.fp16 is False
    assert args.remove_unused_columns is False


def test_train_model_loader_does_not_pass_device_map_to_trainer_model():
    fake_model = type("M", (), {"config": type("C", (), {"_attn_implementation": "eager"})()})()
    with patch("cl_lora.train.AutoModelForCausalLM.from_pretrained", return_value=fake_model) as load:
        load_base_model("fake", torch_dtype=torch.float32)
    assert "device_map" not in load.call_args.kwargs


def test_full_orchestrator_forwards_training_cli_controls():
    tree = ast.parse((ROOT / "cl_lora/orchestrator_full.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "train_on_task_full")
    forwarded = {kw.arg for kw in call.keywords}
    assert {"learning_rate", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "max_seq_length", "use_bf16"} <= forwarded


def test_full_orchestrator_exposes_canonical_general_eval_policy():
    source = (ROOT / "cl_lora/orchestrator_full.py").read_text()
    assert '"--general-eval-strategy"' in source
    assert 'choices=["every_stage", "first_and_last", "final_only"]' in source


def test_lora_orchestrator_carries_gradvac_state_across_stages_and_resume():
    source = (ROOT / "cl_lora/orchestrator.py").read_text()
    assert "slice_gradvac_state: Dict[str, float] = {}" in source
    assert 'partial.get("slice_gradvac_state", {})' in source
    assert "slice_gradvac_state=slice_gradvac_state" in source
    assert '"slice_gradvac_state": slice_gradvac_state' in source


def test_cpu_environment_reports_zero_cuda_devices():
    with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.device_count", return_value=99):
        assert _collect_env_info()["cuda"]["device_count"] == 0
