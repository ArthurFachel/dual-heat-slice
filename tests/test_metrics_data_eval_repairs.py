from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from cl_lora.eval import _evaluate_task_with_generation
from cl_lora.load_dataset import _format_superni_instance, _split_dataset
from cl_lora.metrics import compute_cl_metrics
from cl_lora.train import CompletionOnlyDataCollator


class TinyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 9
    padding_side = "right"

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": [1] * len(texts.split())}
        return {"input_ids": torch.tensor([[1, 2] for _ in texts]), "attention_mask": torch.ones((len(texts), 2), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "alternate"


class TinyModel:
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        return torch.tensor([[1, 2, 7] for _ in range(batch)])


def test_forgetting_rejects_unpaired_task_differences():
    stages = [
        {"seen_tasks": {"A": {"score": 0.8}}},
        {"seen_tasks": {"B": {"score": 0.9}}},
    ]
    with pytest.raises(ValueError, match="missing required score cells"):
        compute_cl_metrics(stages, ["A", "B"])


def test_superni_formatter_preserves_all_references():
    row = _format_superni_instance({"input": "x", "output": ["first", "alternate"]}, "answer")
    assert row["target"] == "first"
    assert row["references"] == ["first", "alternate"]


def test_generation_scores_best_valid_reference():
    dataset = Dataset.from_list([{"prompt": "p", "target": "first", "references": ["first", "alternate"]}])
    result = _evaluate_task_with_generation(TinyModel(), TinyTokenizer(), dataset, 1, 2, 16, "exact_match")
    assert result["exact_match"] == 1.0


def test_singleton_split_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        _split_dataset(Dataset.from_list([{"x": 1}]), eval_size=1, seed=42)


def test_completion_collator_masks_prompt_and_padding_tokens():
    tokenizer = TinyTokenizer()
    collator = CompletionOnlyDataCollator(tokenizer)
    batch = collator([
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "prompt_length": 2},
        {"input_ids": [5, 6], "attention_mask": [1, 1], "prompt_length": 1},
    ])
    assert batch["labels"].tolist() == [[-100, -100, 3, 4], [-100, 6, -100, -100]]
