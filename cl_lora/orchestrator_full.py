"""orchestrator_full.py — Full fine-tune orchestrator for NI sequences (no LoRA).

Executa uma sequencia de tarefas (ex: NI-Seq-G2) em full fine-tuning,
usando DualHeatFullCLMethod para protecao contra forgetting.

Uso:
    python -m cl_lora.orchestrator_full \\
        --sequence NI-Seq-G2 --run-name dh_full_g2 \\
        --method dual_heat_full --slow-strength 2.0

Compara com:
    python -m cl_lora.orchestrator_full \\
        --sequence NI-Seq-G2 --run-name vanilla_full_g2 \\
        --method vanilla
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from importlib import metadata as importlib_metadata
except Exception:
    import importlib_metadata

try:
    from .cl_methods import REGISTRY as CL_METHOD_REGISTRY, build_cl_method
    from .eval import evaluate_all
    from .metrics import compute_cl_metrics
    from .repro import set_global_seed
    from .task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS, get_sequence
    from .train_full import (
        HF_TOKEN, MODEL_NAME,
        build_tokenizer, load_base_model,
        train_on_task_full,
    )
except ImportError:
    from cl_methods import REGISTRY as CL_METHOD_REGISTRY, build_cl_method
    from eval import evaluate_all
    from metrics import compute_cl_metrics
    from repro import set_global_seed
    from task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS, get_sequence
    from train_full import (
        HF_TOKEN, MODEL_NAME,
        build_tokenizer, load_base_model,
        train_on_task_full,
    )


logger = logging.getLogger("cl_lora.orchestrator_full")


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def _safe_model_dir_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", str(model_name)).strip("_")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pkg_version(dist_name: str) -> str | None:
    try:
        return str(importlib_metadata.version(dist_name))
    except Exception:
        return None


def _collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "packages": {
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "accelerate": _pkg_version("accelerate"),
            "datasets": _pkg_version("datasets"),
        },
    }
    try:
        import torch
        info["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        pass
    return info


def _collect_model_info(model) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "class": model.__class__.__name__,
        "name_or_path": getattr(getattr(model, "config", None), "_name_or_path", None)
        or getattr(model, "name_or_path", None),
    }
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "to_dict"):
        try:
            out["config"] = cfg.to_dict()
        except Exception:
            out["config"] = str(cfg)
    return out


def _collect_tokenizer_info(tokenizer) -> Dict[str, Any]:
    return {
        "class": tokenizer.__class__.__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
    }


def run_sequence(
    sequence_name: str,
    model_name: str,
    run_output_dir: Path,
    train_output_dir: Path,
    general_eval_keys: List[str],
    seed: int,
    eval_size: int,
    task_eval_samples: int,
    task_eval_max_new_tokens: int,
    quick_eval: bool,
    save_final_model: bool,
    resume: bool,
    warmup_ratio: float,
    use_bf16: bool = False,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 3.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_seq_length: int = 256,
    # No LoRA params — full fine-tuning
    keep_all_checkpoints: bool = False,
    general_eval_strategy: str = "every_stage",
    seen_eval_strategy: str = "full_matrix",
    train_only: bool = False,
    cl_method_name: str = "vanilla",
    cl_method_kwargs: Dict[str, Any] | None = None,
    orchestrator_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run a sequence of tasks with full fine-tuning (no LoRA).

    Args:
        sequence_name: Name of the task sequence (e.g. 'NI-Seq-G2')
        model_name: HuggingFace model name or path
        cl_method_name: 'vanilla' (no protection) or 'dual_heat_full'
        use_bf16: Use bfloat16 if available (Pascal GPUs nao suportam, use fp16)
    """
    set_global_seed(seed)
    run_output_dir = run_output_dir.resolve()
    train_output_dir = train_output_dir.resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)
    sequence = get_sequence(sequence_name)
    task_order = [task.name for task in sequence.tasks]

    resolved_cfg: Dict[str, Any] = {
        "sequence": sequence_name,
        "description": sequence.description,
        "task_order": task_order,
        "general_eval_keys": general_eval_keys,
        "seed": seed,
        "quick_eval": bool(quick_eval),
        "eval_size": int(eval_size),
        "task_eval_samples": int(task_eval_samples),
        "task_eval_max_new_tokens": int(task_eval_max_new_tokens),
        "cl_method": str(cl_method_name),
        "cl_method_kwargs": dict(cl_method_kwargs or {}),
        "training_type": "full_finetune",
        "use_bf16": bool(use_bf16),
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_seq_length": int(max_seq_length),
        "warmup_ratio": float(warmup_ratio),
    }

    run_cfg_payload: Dict[str, Any] = {
        "orchestrator": orchestrator_config or {},
        "env": _collect_env_info(),
        "resolved": resolved_cfg,
        "model": None,
        "tokenizer": None,
        "notes": {
            "hf_token_present": bool(HF_TOKEN),
            "hf_token_redacted": True,
        },
    }
    _write_json(run_output_dir / "run_config.json", run_cfg_payload)

    partial_path = run_output_dir / "stage_records.partial.json"
    checkpoint_root = run_output_dir / "checkpoints"

    if partial_path.exists() and not resume:
        raise ValueError(
            f"Found existing partial state at {partial_path}. "
            "Use --resume to continue this run or choose a different --run-name."
        )

    stage_records: List[Dict[str, Any]] = []
    start_stage = 1
    seen_tasks = []

    if resume and partial_path.exists():
        partial = _read_json(partial_path)
        if partial.get("sequence") != sequence_name:
            raise ValueError("Resume failed: sequence mismatch.")
        if partial.get("task_order") != task_order:
            raise ValueError("Resume failed: task order mismatch.")

        stage_records = partial.get("stage_records", [])
        completed = len(stage_records)
        start_stage = completed + 1
        seen_tasks = sequence.tasks[:completed]

        if completed >= len(sequence.tasks):
            summary = compute_cl_metrics(stage_records=stage_records, task_order=task_order)
            final_payload = {
                "sequence": sequence_name,
                "description": sequence.description,
                "task_order": task_order,
                "general_eval_keys": general_eval_keys,
                "stage_records": stage_records,
                "summary": summary,
            }
            _write_json(run_output_dir / "results_matrix.json", summary["results_matrix"])
            _write_json(run_output_dir / "metrics.json", summary["metrics"])
            _write_json(run_output_dir / "run_summary.json", final_payload)
            return final_payload

        if completed > 0:
            model_ckpt = checkpoint_root / f"stage_{completed:02d}_{_safe_name(sequence.tasks[completed - 1].name)}" / "model"
            if not model_ckpt.exists():
                raise FileNotFoundError(
                    f"Resume failed: model checkpoint not found at {model_ckpt}."
                )
            tokenizer = build_tokenizer(model_name=str(model_ckpt), hf_token=HF_TOKEN)
            model = load_base_model(str(model_ckpt), hf_token=HF_TOKEN)
        else:
            tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
            model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)
    else:
        tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
        model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)

    run_cfg_payload["model"] = _collect_model_info(model)
    run_cfg_payload["tokenizer"] = _collect_tokenizer_info(tokenizer)
    _write_json(run_output_dir / "run_config.json", run_cfg_payload)

    # Build the CL method object
    cl_state_dir = run_output_dir / "cl_state"
    cl_state_dir.mkdir(parents=True, exist_ok=True)
    cl_method = build_cl_method(cl_method_name, **(cl_method_kwargs or {}))
    if start_stage > 1:
        cl_method.load(str(cl_state_dir))
    print(f"CL method: {cl_method.name} | state dir: {cl_state_dir}")

    total_params_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params_before:,} | Trainable: {trainable_before:,} ({100*trainable_before/total_params_before:.1f}%)")

    for idx in range(start_stage, len(sequence.tasks) + 1):
        task = sequence.tasks[idx - 1]
        task_name = task.name
        safe_task_name = _safe_name(task_name)

        stage_train_dir = train_output_dir / sequence_name / run_output_dir.name / f"stage_{idx:02d}_{safe_task_name}"
        stage_eval_dir = run_output_dir / "stages" / f"stage_{idx:02d}_{safe_task_name}"

        print(f"\n=== Stage {idx}/{len(sequence.tasks)} | Training task: {task_name} ===")
        retain_tasks = list(sequence.tasks[:idx - 1]) if idx > 1 else None

        model_checkpoint_dir = checkpoint_root / f"stage_{idx:02d}_{safe_task_name}" / "model"

        model, train_report = train_on_task_full(
            model=model,
            tokenizer=tokenizer,
            task=task,
            output_dir=str(stage_train_dir),
            eval_size=eval_size,
            seed=seed,
            retain_tasks=retain_tasks,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            use_bf16=use_bf16,
            cl_method=cl_method,
            stage_idx=idx,
        )

        cl_method.save(str(cl_state_dir))
        print(f"  CL-method state saved: {cl_state_dir} | {cl_method.metadata()}")

        seen_tasks.append(task)
        is_final_stage = (idx == len(sequence.tasks))

        # Decide which seen tasks to evaluate
        if seen_eval_strategy == "diagonal_final" and not is_final_stage:
            eval_seen = [task]
            print(f"  Seen-task eval: diagonal only ({task_name})")
        else:
            eval_seen = list(seen_tasks)

        # Decide whether to run general (GP/IP) evaluation
        run_general = True
        if general_eval_strategy == "first_and_last":
            run_general = (idx == 1 or is_final_stage)
        elif general_eval_strategy == "final_only":
            run_general = is_final_stage

        # Evaluate (seen tasks + general tasks, handled internally by evaluate_all)
        print(f"  Evaluating on {len(eval_seen)} seen tasks...")
        evaluation = evaluate_all(
            model,
            tokenizer,
            eval_seen,
            output_dir=str(stage_eval_dir),
            general_eval_task_keys=general_eval_keys,
            eval_size=eval_size,
            task_eval_samples=task_eval_samples,
            task_eval_max_new_tokens=task_eval_max_new_tokens,
            quick_eval=quick_eval,
            skip_general_eval=not run_general,
            seed=seed,
        )
        seen_scores = evaluation.get("seen_tasks", {})
        general_scores = evaluation.get("general", {"gp": {}, "ip": {}, "gp_mean": None, "ip_mean": None, "mode": "skipped"})
        print(f"  Seen-task scores: {seen_scores}")
        if general_scores.get("gp") or general_scores.get("ip"):
            print(f"  General-task scores: GP={general_scores.get('gp_mean', 'N/A'):.4f} IP={general_scores.get('ip_mean', 'N/A'):.4f}")

        stage_record = {
            "stage": idx,
            "trained_task": task_name,
            "train_report": train_report,
            "seen_tasks": seen_scores,
            "general": general_scores,
        }
        stage_records.append(stage_record)

        # Save partial progress
        _write_json(partial_path, {
            "sequence": sequence_name,
            "description": sequence.description,
            "task_order": task_order,
            "stage_records": stage_records,
        })
        print(f"  Partial state saved ({idx}/{len(sequence.tasks)} stages)")

        # Save model checkpoint
        model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_checkpoint_dir))
        tokenizer.save_pretrained(str(model_checkpoint_dir))
        print(f"  Model checkpoint saved: {model_checkpoint_dir}")

    # Compute final metrics
    summary = compute_cl_metrics(stage_records=stage_records, task_order=task_order)

    final_payload = {
        "sequence": sequence_name,
        "description": sequence.description,
        "task_order": task_order,
        "general_eval_keys": general_eval_keys,
        "stage_records": stage_records,
        "summary": summary,
    }

    _write_json(run_output_dir / "results_matrix.json", summary["results_matrix"])
    _write_json(run_output_dir / "metrics.json", summary["metrics"])
    _write_json(run_output_dir / "run_summary.json", final_payload)

    # Cleanup partial state
    if partial_path.exists():
        partial_path.unlink()

    if partial_path.exists():
        partial_path.unlink()

    # Print summary
    print(f"\n{'='*60}")
    print(f"RUN COMPLETE: {run_output_dir.name}")
    print(f"  Sequence: {sequence_name}")
    print(f"  Method: {cl_method_name}")
    print(f"  Type: Full fine-tune (no LoRA)")
    print(f"  Tasks: {len(sequence.tasks)}")

    metrics = summary.get("metrics", {})

    def _fmt(val, decimals=4):
        """Format metric value — float gets decimals, str/Nones pass through."""
        if isinstance(val, (int, float)):
            return f"{val:.{decimals}f}"
        return str(val)

    print(f"  Avg performance (AP): {_fmt(metrics.get('AP', 'N/A'))}")
    print(f"  Final performance:    {_fmt(metrics.get('FP', 'N/A'))}")
    print(f"  Avg forgetting:       {_fmt(metrics.get('Forget', 'N/A'))}")
    print(f"  General performance:  {_fmt(metrics.get('GP', 'N/A'))}")
    print(f"  In-context perf.:     {_fmt(metrics.get('IP', 'N/A'))}")

    print(f"\nResults saved to {run_output_dir}")
    return final_payload


def main():
    parser = argparse.ArgumentParser(
        description="Full fine-tune CL orchestrator for NI sequences (no LoRA)"
    )

    # Core
    parser.add_argument("--sequence", default="NI-Seq-G2",
                        help="Task sequence name (default: NI-Seq-G2)")
    parser.add_argument("--run-name", default=None,
                        help="Run name (default: {method}_full_seed{seed})")
    parser.add_argument("--model", default=None,
                        help="Model name or path (default: from train_full.py)")
    parser.add_argument("--method", default="vanilla",
                        choices=["vanilla", "dual_heat_full"],
                        help="CL method (default: vanilla)")
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--use-bf16", action="store_true")

    # DualHeat hyperparams
    parser.add_argument("--slow-strength", type=float, default=1.5)
    parser.add_argument("--fast-decay", type=float, default=0.93)
    parser.add_argument("--fast-strength", type=float, default=1.5)
    parser.add_argument("--fast-decay-rate", type=float, default=0.04)
    parser.add_argument("--slow-window", type=int, default=None)
    parser.add_argument("--no-lateral-inhibition", action="store_false",
                        dest="lateral_inhibition", default=True)

    # Eval
    parser.add_argument("--task-eval-samples", type=int, default=50)
    parser.add_argument("--task-eval-max-new-tokens", type=int, default=50)
    parser.add_argument("--quick-eval", action="store_true")
    parser.add_argument("--general-eval-keys", nargs="*", default=[])
    parser.add_argument(
        "--general-eval-strategy",
        choices=["every_stage", "first_and_last", "final_only"],
        default="every_stage",
    )

    # Resume / output
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument("--run-dir", default="results_full",
                        help="Root output directory")
    parser.add_argument("--train-dir", default="training_output_full",
                        help="Training output directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    if args.run_name is None:
        args.run_name = f"{args.method}_full_seed{args.seed}"

    model_name = args.model or MODEL_NAME

    cl_kwargs = {}
    if args.method == "dual_heat_full":
        cl_kwargs = {
            "slow_strength": args.slow_strength,
            "fast_decay": args.fast_decay,
            "fast_strength": args.fast_strength,
            "fast_decay_rate": args.fast_decay_rate,
            "slow_window": args.slow_window,
            "lateral_inhibition": args.lateral_inhibition,
        }

    run_output_dir = Path(args.run_dir) / args.run_name
    train_output_dir = Path(args.train_dir)

    run_sequence(
        sequence_name=args.sequence,
        model_name=model_name,
        run_output_dir=run_output_dir,
        train_output_dir=train_output_dir,
        general_eval_keys=args.general_eval_keys or [],
        seed=args.seed,
        eval_size=args.eval_size,
        task_eval_samples=args.task_eval_samples,
        task_eval_max_new_tokens=args.task_eval_max_new_tokens,
        quick_eval=args.quick_eval,
        save_final_model=args.save_final_model,
        resume=args.resume,
        warmup_ratio=args.warmup_ratio,
        use_bf16=args.use_bf16,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        general_eval_strategy=args.general_eval_strategy,
        cl_method_name=args.method,
        cl_method_kwargs=cl_kwargs,
    )


if __name__ == "__main__":
    main()
