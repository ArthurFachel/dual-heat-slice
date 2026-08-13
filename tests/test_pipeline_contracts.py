from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from cl_lora import eval as eval_module
from cl_lora import eval_standalone, orchestrator_full, qwen_experiment, qwen_full_experiment
from cl_lora.metrics import compute_cl_metrics
from cl_lora.qwen_tasks import QwenTask, build_qwen_dataset


ROOT = Path(__file__).resolve().parents[1]


class WordTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(1, len(text.split()) + 1)), "attention_mask": [1] * len(text.split())}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = " ".join(f"{m['role']} {m['content']}" for m in messages)
        return body + (" assistant" if add_generation_prompt else "")


def test_completion_tokenization_always_retains_a_supervised_token():
    dataset = Dataset.from_dict({
        "text": ["one two three four answer"],
        "prompt": ["one two three four"],
        "target": ["answer"],
    })
    for module in (qwen_experiment, qwen_full_experiment):
        row = module.tokenize_dataset(dataset, WordTokenizer(), max_length=4)[0]
        assert row["prompt_length"] < len(row["input_ids"])
        assert len(row["input_ids"]) <= 4


def test_qwen_tokenization_marks_prompt_boundary_for_completion_only_loss():
    dataset = Dataset.from_dict({"text": ["prompt words answer"], "prompt": ["prompt words"], "target": ["answer"]})
    tokenized = qwen_experiment.tokenize_dataset(dataset, WordTokenizer(), max_length=16)
    assert tokenized[0]["prompt_length"] == 2
    assert tokenized[0]["input_ids"] == [1, 2, 1]


def test_qwen_task_split_deduplicates_before_partitioning():
    task = QwenTask("duplicate", "sentiment", [("same", "positive"), ("same", "positive"), ("other", "negative")])
    train, evaluation = build_qwen_dataset(task, seed=3, eval_split=0.5)
    train_prompts = set(train["prompt"])
    eval_prompts = set(evaluation["prompt"])
    assert train_prompts.isdisjoint(eval_prompts)
    assert len(train) + len(evaluation) == 2


def test_compare_all_preserves_shared_kwargs_and_method_overrides():
    calls = []
    with patch.object(qwen_experiment, "run_experiment", side_effect=lambda **kw: calls.append(kw) or {"AP": 1, "FP": 1, "Forget": 0}):
        qwen_experiment.compare_all_methods(slow_strength=7.0, replay_size=9)
    assert all(call["slow_strength"] == 7.0 for call in calls)
    replay = next(call for call in calls if call["method"] == "replay")
    assert replay["replay_size"] == 128


def _capture_dtype(module, cuda_available, bf16_supported=False):
    fake = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    with patch("torch.cuda.is_available", return_value=cuda_available), \
         patch("torch.cuda.is_bf16_supported", return_value=bf16_supported), \
         patch.object(module.AutoModelForCausalLM, "from_pretrained", return_value=fake) as loader:
        module.load_qwen_model("fake")
    return loader.call_args.kwargs["torch_dtype"]


def test_qwen_model_dtype_is_fp32_on_cpu_and_bf16_only_when_supported():
    for module in (qwen_experiment, qwen_full_experiment):
        assert _capture_dtype(module, False) is torch.float32
        assert _capture_dtype(module, True, False) is torch.float16
        assert _capture_dtype(module, True, True) is torch.bfloat16


def test_eval_runtime_dtype_is_fp32_on_cpu_and_bf16_only_when_supported():
    with patch("torch.cuda.is_available", return_value=False):
        assert eval_module._resolve_eval_runtime() == ("cpu", "float32")
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=False):
        assert eval_module._resolve_eval_runtime() == ("cuda", "float16")
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=True):
        assert eval_module._resolve_eval_runtime() == ("cuda", "bfloat16")


def test_metrics_reject_incomplete_matrix_and_stage_gaps():
    incomplete = [{"stage": 1, "trained_task": "A", "seen_tasks": {"A": {"score": 0.8}}}]
    with pytest.raises(ValueError, match="incomplete"):
        compute_cl_metrics(incomplete, ["A", "B"])
    gaps = [
        {"stage": 1, "trained_task": "A", "seen_tasks": {"A": {"score": 0.8}}},
        {"stage": 3, "trained_task": "B", "seen_tasks": {"A": {"score": 0.7}, "B": {"score": 0.9}}},
    ]
    with pytest.raises(ValueError, match="contiguous"):
        compute_cl_metrics(gaps, ["A", "B"])


def test_metrics_reject_missing_required_score_cells():
    stages = [
        {"stage": 1, "trained_task": "A", "seen_tasks": {"A": {"score": 0.8}}},
        {"stage": 2, "trained_task": "B", "seen_tasks": {"B": {"score": 0.9}}},
    ]
    with pytest.raises(ValueError, match="missing required score cells"):
        compute_cl_metrics(stages, ["A", "B"])


def test_diagonal_final_metrics_require_only_diagonal_and_final_row():
    stages = [
        {"stage": 1, "trained_task": "A", "seen_tasks": {"A": {"score": 0.8}}},
        {"stage": 2, "trained_task": "B", "seen_tasks": {"B": {"score": 0.9}}},
        {"stage": 3, "trained_task": "C", "seen_tasks": {
            "A": {"score": 0.6}, "B": {"score": 0.7}, "C": {"score": 0.85}
        }},
    ]
    summary = compute_cl_metrics(stages, ["A", "B", "C"], coverage_mode="diagonal_final")
    assert summary["coverage"]["complete"] is True


def test_completed_resume_skips_canonical_metrics_for_noncanonical_modes(tmp_path):
    sequence = SimpleNamespace(tasks=[SimpleNamespace(name="A")], description="test")
    partial = tmp_path / "run" / "stage_records.partial.json"
    partial.parent.mkdir(parents=True)
    partial.write_text(json.dumps({
        "sequence": "S", "task_order": ["A"],
        "stage_records": [{"stage": 1, "trained_task": "A", "seen_tasks": {}}],
    }))
    with patch.object(orchestrator_full, "get_sequence", return_value=sequence), \
         patch.object(orchestrator_full, "compute_cl_metrics") as metrics:
        result = orchestrator_full.run_sequence(
            "S", "fake", tmp_path / "run", tmp_path / "train", [], 1, 2, 2, 2,
            True, False, True, 0.0, train_only=True,
        )
    metrics.assert_not_called()
    assert result["summary"] is None


def test_training_precision_uses_bf16_only_when_runtime_supports_it():
    for module in (qwen_experiment, qwen_full_experiment):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            assert module._training_precision() == {"bf16": False, "fp16": True, "use_cpu": False}
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            assert module._training_precision() == {"bf16": True, "fp16": False, "use_cpu": False}
        with patch("torch.cuda.is_available", return_value=False):
            assert module._training_precision() == {"bf16": False, "fp16": False, "use_cpu": True}


def test_ewc_reuses_one_persistent_adapter_coordinate_system():
    base = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    config = qwen_experiment.LoraConfig(
        r=2, lora_alpha=2, target_modules=["0"], bias="none"
    )
    with patch.object(qwen_experiment, "get_peft_model", wraps=qwen_experiment.get_peft_model) as wrap:
        first = qwen_experiment._prepare_stage_model(base, config, "ewc")
        second = qwen_experiment._prepare_stage_model(first, config, "ewc")
    assert second is first
    assert wrap.call_count == 1


def test_metrics_include_coverage_and_document_final_task_forgetting():
    stages = [
        {"stage": 1, "trained_task": "A", "seen_tasks": {"A": {"score": 0.8}}},
        {"stage": 2, "trained_task": "B", "seen_tasks": {"A": {"score": 0.5}, "B": {"score": 0.9}}, "general": {"gp_mean": 0.4, "ip_mean": 0.3}},
    ]
    summary = compute_cl_metrics(stages, ["A", "B"])
    assert summary["per_task_forgetting"]["B"] == 0.0
    assert summary["coverage"] == {"expected_stages": 2, "observed_stages": 2, "score_cells_expected": 3, "score_cells_observed": 3, "complete": True, "missing_score_cells": [], "mode": "full_matrix"}
    assert summary["forgetting_definition"]["includes_final_task"] is True


def test_quick_eval_is_explicitly_noncanonical():
    seen = {"A": {"score": None, "perplexity": 2.0}}
    with patch.object(eval_module, "evaluate_seen_tasks_perplexity", return_value=seen), \
         patch.object(torch, "compile", side_effect=RuntimeError):
        result = eval_module.evaluate_all(object(), object(), [], "unused", quick_eval=True)
    assert result["metrics_eligible"] is False
    assert result["evaluation_mode"] == "quick_perplexity"


def test_standalone_summary_rejects_noncanonical_quick_eval(tmp_path):
    stage = tmp_path / "stages" / "stage_01_A"
    stage.mkdir(parents=True)
    (stage / "stage_record.json").write_text(json.dumps({
        "stage": 1, "trained_task": "A", "seen_tasks": {"A": {"score": None}},
        "metrics_eligible": False, "evaluation_mode": "quick_perplexity",
    }))
    with pytest.raises(ValueError, match="non-canonical"):
        eval_standalone.recompute_run_summary(tmp_path)


def test_full_orchestrator_cli_exposes_pipeline_behavior_options():
    source = (ROOT / "cl_lora/orchestrator_full.py").read_text()
    for option in ("--train-only", "--keep-all-checkpoints", "--seen-eval-strategy"):
        assert option in source


class FakeSaveable:
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeMethod:
    name = "vanilla"
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
    def load(self, path):
        pass
    def metadata(self):
        return {}


def test_full_orchestrator_train_only_skips_eval_and_keeps_latest_checkpoint(tmp_path):
    tasks = [SimpleNamespace(name="A"), SimpleNamespace(name="B")]
    sequence = SimpleNamespace(tasks=tasks, description="test")
    model = FakeSaveable()
    tokenizer = FakeSaveable()
    with patch.object(orchestrator_full, "get_sequence", return_value=sequence), \
         patch.object(orchestrator_full, "build_tokenizer", return_value=tokenizer), \
         patch.object(orchestrator_full, "load_base_model", return_value=model), \
         patch.object(orchestrator_full, "build_cl_method", return_value=FakeMethod()), \
         patch.object(orchestrator_full, "train_on_task_full", side_effect=lambda **kw: (model, {})), \
         patch.object(orchestrator_full, "evaluate_all") as evaluate:
        result = orchestrator_full.run_sequence(
            "S", "fake", tmp_path / "run", tmp_path / "train", [], 1, 2, 2, 2,
            False, True, False, 0.0, train_only=True, keep_all_checkpoints=False,
        )
    evaluate.assert_not_called()
    assert result["summary"] is None
    assert not (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "final_model").is_dir()
    checkpoints = list((tmp_path / "run" / "checkpoints").glob("stage_*/model"))
    assert [p.parent.name for p in checkpoints] == ["stage_02_B"]


def test_sequence_metrics_script_discovers_nested_runs(tmp_path):
    run = tmp_path / "sequence" / "method" / "stages" / "stage_01_A"
    run.mkdir(parents=True)
    (run / "stage_record.json").write_text("{}")
    fake_python = tmp_path / "python"
    log = tmp_path / "calls"
    fake_python.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\n")
    fake_python.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / "scripts/compute_sequence_metrics.sh")], env={"RUNS_ROOT": str(tmp_path), "PYTHON_BIN": str(fake_python)}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert str(tmp_path / "sequence" / "method") in log.read_text()
