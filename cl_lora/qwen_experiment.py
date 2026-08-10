"""
qwen_experiment.py — Continual Learning experiment with Qwen 0.5B + DualHeat.

Runs a 4-task CL sequence on a small Qwen 0.5B Instruct model with LoRA
adapters, evaluating after each task to measure catastrophic forgetting.

Usage:
  # Baseline (vanilla LoRA fine-tuning)
  python -m cl_lora.qwen_experiment --method vanilla --model Qwen/Qwen2.5-0.5B-Instruct

  # DualHeat
  python -m cl_lora.qwen_experiment --method dual_heat

  # DualHeat + small EWC
  python -m cl_lora.qwen_experiment --method dual_heat --slow-strength 5.0

  # Disable lateral inhibition (EWC only)
  python -m cl_lora.qwen_experiment --method dual_heat --no-lateral-inhibition

  # O-LoRA comparison
  python -m cl_lora.qwen_experiment --method o_lora --o-lora-lambda 0.5

  # All methods comparison (run sequentially)
  python -m cl_lora.qwen_experiment --compare-all
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset as HFDataset

# ── DualHeat is loaded as a CL Method ─────────────────────────────────────
try:
    from .cl_methods import build_cl_method
    from .repro import set_global_seed
    from .train import _CLAuxLossTrainer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cl_lora.cl_methods import build_cl_method
    from cl_lora.repro import set_global_seed
    from cl_lora.train import _CLAuxLossTrainer

from .qwen_tasks import QWEN_CL_TASKS, QwenTask, build_qwen_dataset, make_prompt

logger = logging.getLogger("cl_lora.qwen_experiment")


# ── Model loading ─────────────────────────────────────────────────────────

QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_qwen_model(
    model_name: str = QWEN_MODEL,
    torch_dtype: torch.dtype = None,
    device_map: str = None,
):
    """Load Qwen model with auto-detected dtype.

    Pascal GPUs (GTX 1080 Ti, Titan Xp) don't support bf16 natively.
    Auto-selects fp16 on Pascal, bf16 on Volta+.  No device_map by default
    (HF Trainer manages device placement for DDP).
    """
    if torch_dtype is None:
        import torch.cuda as cu
        if cu.is_available():
            cc = cu.get_device_capability(0)
            major = cc[0]
        else:
            major = 99
        if major < 7:
            torch_dtype = torch.float16
            print(f"[load_qwen_model] GPU CC={major}.x → using float16")
        else:
            torch_dtype = torch.bfloat16
            print(f"[load_qwen_model] GPU CC={major}.x → using bfloat16")

    kwargs = dict(
        torch_dtype=torch_dtype,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        import flash_attn  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    # Move to GPU 0 explicitly to prevent Trainer from wrapping in DataParallel
    # (DataParallel causes device conflicts with DualHeat's registered hooks)
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda:0")
    return model


def load_qwen_tokenizer(model_name: str = QWEN_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# ── LoRA setup ────────────────────────────────────────────────────────────

def build_lora_config(r: int = 16, lora_alpha: int = 8):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ── Tokenization ──────────────────────────────────────────────────────────

def tokenize_dataset(dataset: HFDataset, tokenizer, max_length: int = 128):
    return dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length),
        remove_columns=[c for c in dataset.column_names if c != "target"],
    )


# ── Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_task(
    model,
    tokenizer,
    task: QwenTask,
    eval_dataset: HFDataset,
    max_new_tokens: int = 8,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate model accuracy on one task.

    Uses the eval split's 'prompt' and 'target' columns (set by
    build_qwen_dataset).  Accuracy = exact match of the generated
    label (lowercased, stripped) against the ground-truth label.
    """
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0

    prompts = list(eval_dataset["prompt"])
    targets = list(eval_dataset["target"])
    indices = list(range(len(prompts)))

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_prompts = [prompts[i] for i in batch_idx]
        batch_targets = [targets[i] for i in batch_idx]

        # Left padding for generation
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        tokenizer.padding_side = prev_side

        # Use fp16 autocast if available and model dtype is fp16
        model_dtype = next(model.parameters()).dtype
        autocast_dtype = model_dtype if model_dtype in (torch.float16, torch.bfloat16) else None

        with torch.amp.autocast("cuda", enabled=autocast_dtype is not None, dtype=autocast_dtype):
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = encoded["input_ids"].shape[1]
        for i, target in enumerate(batch_targets):
            pred_ids = outputs[i, input_len:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip().lower()
            correct += int(prediction == target.lower())
            total += 1

    accuracy = correct / max(1, total)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_all_tasks(
    model,
    tokenizer,
    tasks: List[QwenTask],
    eval_datasets: List[HFDataset],
) -> Dict[str, float]:
    """Evaluate model on all tasks. Returns dict of task_name -> accuracy."""
    results = {}
    for i, task in enumerate(tasks):
        result = evaluate_task(model, tokenizer, task, eval_datasets[i])
        results[task.name] = result["accuracy"]
    return results


# ── Training loop ─────────────────────────────────────────────────────────

def train_one_task(
    model,
    tokenizer,
    task: QwenTask,
    cl_method,
    stage_idx: int,
    lora_config,
    output_dir: str,
    seed: int = 42,
    learning_rate: float = 5e-5,
    num_epochs: float = 3.0,
    per_device_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    max_seq_length: int = 128,
    logging_steps: int = 5,
) -> Any:
    """Train a LoRA adapter on one task using HF Trainer."""
    set_global_seed(seed)

    # Build dataset
    train_dataset, eval_dataset = build_qwen_dataset(task, seed=seed, tokenizer=tokenizer)
    train_tok = tokenize_dataset(train_dataset, tokenizer, max_length=max_seq_length)
    eval_tok = tokenize_dataset(eval_dataset, tokenizer, max_length=max_seq_length)

    # Build PEFT model
    peft_model = get_peft_model(model, lora_config)
    active_adapter = "default"

    # CL method pre-training hook
    cl_method.pre_train(peft_model, stage_idx=stage_idx, retain_tasks=None)

    # Training args
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training args — use fp16 on Pascal (CC < 7.0), bf16 on Volta+
    import torch.cuda as cu
    if cu.is_available():
        gpu_major = cu.get_device_capability(0)[0]
        use_fp16 = gpu_major < 7
    else:
        use_fp16 = False

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=logging_steps,
        eval_strategy="no",
        save_strategy="no",
        bf16=not use_fp16,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=True,
        seed=seed,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = _CLAuxLossTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        _cl_aux_loss_fn=cl_method.aux_loss,
    )

    trainer.train()

    # Post-training hook
    try:
        cl_device = next(peft_model.parameters()).device
    except StopIteration:
        cl_device = torch.device("cuda")
    cl_method.post_train(
        peft_model,
        tokenizer=tokenizer,
        train_dataset=train_tok,
        device=cl_device,
        stage_idx=stage_idx,
        task_name=task.name,
    )

    # Merge adapter into base model
    merged_model = peft_model.merge_and_unload()

    return merged_model


# ── Full experiment ───────────────────────────────────────────────────────

def run_experiment(
    model_name: str = QWEN_MODEL,
    method: str = "vanilla",
    seed: int = 42,
    output_dir: str = "results/qwen_experiment",
    learning_rate: float = 5e-5,
    num_epochs: float = 3.0,
    per_device_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    lora_rank: int = 16,
    lora_alpha: int = 8,
    slow_strength: float = 2.0,
    fast_decay: float = 0.93,
    fast_strength: float = 2.0,
    fast_decay_rate: float = 0.04,
    slow_window: Optional[int] = None,
    lateral_inhibition: bool = True,
) -> Dict[str, Any]:
    """Run a full CL experiment on the 4 Qwen tasks.

    Returns a dict with evaluation matrices, metrics, and forgetting.
    """
    run_name = f"{method}_seed{seed}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(seed)

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    model = load_qwen_model(model_name)
    tokenizer = load_qwen_tokenizer(model_name)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build LoRA config
    lora_cfg = build_lora_config(r=lora_rank, lora_alpha=lora_alpha)
    print(f"LoRA rank={lora_rank}, alpha={lora_alpha}")

    # CL method
    cl_kwargs = {}
    if method == "dual_heat":
        cl_kwargs = {
            "fast_decay": fast_decay,
            "fast_strength": fast_strength,
            "fast_decay_rate": fast_decay_rate,
            "slow_strength": slow_strength,
            "slow_window": slow_window,
            "lateral_inhibition": lateral_inhibition,
        }
    elif method == "o_lora":
        cl_kwargs = {"lambda_orth": 0.5}

    cl_method = build_cl_method(method, **cl_kwargs)
    print(f"CL method: {method} | {cl_method.metadata()}")

    tasks = QWEN_CL_TASKS
    n_tasks = len(tasks)

    # Build eval datasets once
    eval_datasets = [build_qwen_dataset(t, seed=seed, tokenizer=tokenizer)[1] for t in tasks]

    # ── Evaluation matrix ──────────────────────────────────────────
    # results[i][j] = accuracy on task j after training through task i
    results_matrix = [[None] * n_tasks for _ in range(n_tasks)]

    # Evaluate before any training
    print(f"\n--- Initial evaluation (before training) ---")
    pre_scores = evaluate_all_tasks(model, tokenizer, tasks, eval_datasets)
    print(f"  Pre-training: {pre_scores}")

    # ── Task loop ──────────────────────────────────────────────────
    for stage_idx in range(n_tasks):
        task = tasks[stage_idx]
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx + 1}/{n_tasks}: {task.name} (domain: {task.domain})")

        # Train
        stage_dir = run_dir / f"stage_{stage_idx + 1:02d}"
        model = train_one_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            cl_method=cl_method,
            stage_idx=stage_idx + 1,
            lora_config=lora_cfg,
            output_dir=str(stage_dir),
            seed=seed,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # Save CL method state for next stage
        cl_method.save(str(run_dir / "cl_state"))

        # Evaluate on ALL tasks
        print(f"\n  Evaluating after {task.name}...")
        scores = evaluate_all_tasks(model, tokenizer, tasks, eval_datasets)
        for j, t in enumerate(tasks):
            results_matrix[stage_idx][j] = scores[t.name]

        # Print row
        row_str = " | ".join(f"{t.name}: {scores[t.name]:.3f}" for t in tasks)
        print(f"  Results: {row_str}")

    # ── Compute metrics ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS MATRIX")
    print(f"{'':<15}", end="")
    for t in tasks:
        print(f"{t.name:<15}", end="")
    print()
    for i in range(n_tasks):
        print(f"After {tasks[i].name:<8}", end="")
        for j in range(n_tasks):
            val = results_matrix[i][j]
            if val is not None:
                print(f"{val:<15.3f}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()

    # AP = diagonal (score right after training each task)
    ap = sum(results_matrix[i][i] for i in range(n_tasks)) / n_tasks

    # FP = final stage scores (all tasks after last training)
    fp = sum(results_matrix[-1][j] for j in range(n_tasks)) / n_tasks

    # Per-task forgetting
    per_task_forgetting = {}
    for j in range(n_tasks):
        diag = results_matrix[j][j]
        final = results_matrix[-1][j]
        if diag is not None and final is not None:
            per_task_forgetting[tasks[j].name] = diag - final
        else:
            per_task_forgetting[tasks[j].name] = None

    avg_forgetting = sum(v for v in per_task_forgetting.values() if v is not None) / max(1, sum(1 for v in per_task_forgetting.values() if v is not None))

    metrics = {
        "model": model_name,
        "method": method,
        "cl_hyperparams": cl_kwargs,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "epochs_per_task": num_epochs,
        "batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "seed": seed,
        "num_tasks": n_tasks,
        "results_matrix": results_matrix,
        "average_accuracy_ap": ap,
        "final_performance_fp": fp,
        "avg_forgetting": avg_forgetting,
        "per_task_forgetting": per_task_forgetting,
    }

    # Save results
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
                   for k, v in metrics.items()}, f, indent=2, default=str)

    print(f"\nMetrics saved to {run_dir / 'metrics.json'}")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Method:               {method}")
    print(f"  Average Accuracy (AP): {ap:.4f}")
    print(f"  Final Performance (FP): {fp:.4f}")
    print(f"  Average Forgetting:     {avg_forgetting:.4f}")
    print(f"  Per-task forgetting:    {per_task_forgetting}")

    return metrics


def compare_all_methods(
    model_name: str = QWEN_MODEL,
    seed: int = 42,
    output_dir: str = "results/qwen_experiment",
):
    """Run all methods sequentially and print comparison table."""
    methods = [
        {"method": "vanilla", "label": "Vanilla LoRA (baseline)", "kwargs": {}},
        {"method": "o_lora", "label": "O-LoRA", "kwargs": {"lambda_orth": 0.5}},
        {"method": "dual_heat", "label": "DualHeat", "kwargs": {}},
        {"method": "dual_heat", "label": "DualHeat (EWC only, no lateral inhibition)", "kwargs": {"lateral_inhibition": False}},
        {"method": "dual_heat", "label": "DualHeat (strong EWC, beta=5.0)", "kwargs": {"slow_strength": 5.0}},
    ]

    results = []
    for cfg in methods:
        print(f"\n\n{'#'*60}")
        print(f"# Running: {cfg['label']}")
        print(f"{'#'*60}")
        kwargs = cfg["kwargs"]
        metrics = run_experiment(
            model_name=model_name,
            method=cfg["method"],
            seed=seed,
            output_dir=output_dir,
            **kwargs,
        )
        results.append({"label": cfg["label"], **metrics})

    # Print comparison
    print(f"\n\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<45} {'AP':<10} {'FP':<10} {'Forget':<10}")
    print("-" * 70)
    for r in results:
        label = r["label"]
        ap = r.get("average_accuracy_ap", 0)
        fp = r.get("final_performance_fp", 0)
        forget = r.get("avg_forgetting", 0)
        print(f"{label:<45} {ap:<10.4f} {fp:<10.4f} {forget:<10.4f}")
    print("-" * 70)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen 0.5B Continual Learning experiment")
    parser.add_argument("--model", default=QWEN_MODEL,
                        help=f"Model name (default: {QWEN_MODEL})")
    parser.add_argument("--method", choices=["vanilla", "o_lora", "dual_heat"], default="dual_heat",
                        help="CL method (default: dual_heat)")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run all methods sequentially and compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/qwen_experiment")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=float, default=3.0, help="Epochs per task")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha")

    # DualHeat-specific
    parser.add_argument("--fast-decay", type=float, default=0.93)
    parser.add_argument("--fast-strength", type=float, default=2.0)
    parser.add_argument("--fast-decay-rate", type=float, default=0.04)
    parser.add_argument("--slow-strength", type=float, default=2.0)
    parser.add_argument("--slow-window", type=int, default=None)
    parser.add_argument("--no-lateral-inhibition", action="store_true")

    # O-LoRA specific
    parser.add_argument("--o-lora-lambda", type=float, default=0.5)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.compare_all:
        compare_all_methods(
            model_name=args.model,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        run_experiment(
            model_name=args.model,
            method=args.method,
            seed=args.seed,
            output_dir=args.output_dir,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            slow_strength=args.slow_strength,
            fast_decay=args.fast_decay,
            fast_strength=args.fast_strength,
            fast_decay_rate=args.fast_decay_rate,
            slow_window=args.slow_window,
            lateral_inhibition=not args.no_lateral_inhibition,
        )


if __name__ == "__main__":
    main()
