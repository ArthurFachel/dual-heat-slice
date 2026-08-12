"""train_full.py — Full fine-tuning training loop (no PEFT/LoRA).

Adaptado de train.py, mas sem PEFT: treina todos os parametros do modelo
diretamente. Suporta CL methods que operam em nn.Linear (DualHeatFullCLMethod).

Diferencas de train.py:
  - Nao usa get_peft_model / merge_and_unload
  - Nao ha lora_config, rank, lora_alpha
  - CL method hooka direto nos pesos do modelo
  - train_on_task_full retorna o modelo com pesos atualizados (in-place)
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import accelerate
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from .cl_methods import CLMethod, VanillaCLMethod
    from .load_dataset import (
        CompletionOnlyDataCollator,
        configure_prompt_tokenizer,
        load_training_dataset,
    )
    from .repro import set_global_seed
except ImportError:
    from cl_methods import CLMethod, VanillaCLMethod
    from load_dataset import (
        CompletionOnlyDataCollator,
        configure_prompt_tokenizer,
        load_training_dataset,
    )
    from repro import set_global_seed


def _patch_accelerate_unwrap_model_compat() -> None:
    unwrap = accelerate.Accelerator.unwrap_model
    params = inspect.signature(unwrap).parameters
    if "keep_torch_compile" in params:
        return

    @functools.wraps(unwrap)
    def _wrapped(self, model, *args, keep_torch_compile=None, **kwargs):
        return unwrap(self, model, *args, **kwargs)

    accelerate.Accelerator.unwrap_model = _wrapped


_patch_accelerate_unwrap_model_compat()

load_dotenv()

#MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = "roneneldan/TinyStories-33M"
# MODEL_NAME = "facebook/opt-350m"
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HUGGING_TOKEN")


def build_tokenizer(model_name: str = MODEL_NAME, hf_token: str | None = HF_TOKEN):
    local = Path(model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, local_files_only=local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    configure_prompt_tokenizer(tokenizer)
    return tokenizer


def load_base_model(
    model_name: str = MODEL_NAME,
    hf_token: str | None = HF_TOKEN,
    torch_dtype: torch.dtype = torch.float32,
    device_map: str | None = None,
):
    """Load model in fp32 — Trainer gerencia fp16 via gradient scaler.

    Para modelos pequenos como TinyStories-33M, fp32 cabe com folga
    e o gradient scaler evita NaN no treino em half precision.
    """
    local = Path(model_name).is_dir()
    kwargs: dict = dict(
        torch_dtype=torch_dtype,
        token=hf_token,
        local_files_only=local,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        import flash_attn  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    attn_used = getattr(model.config, "_attn_implementation", None)
    print(f"[load_base_model] attn_implementation={attn_used}")
    return model


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize(ex):
        encoded = tokenizer(ex["text"], truncation=True, max_length=max_length)
        prompt = tokenizer(ex["prompt"], truncation=True, max_length=max_length)
        encoded["prompt_length"] = min(len(prompt["input_ids"]), len(encoded["input_ids"]))
        return encoded
    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
    )


def _build_training_arguments(
    *, output_path: Path, learning_rate: float, num_train_epochs: float,
    warmup_ratio: float, per_device_train_batch_size: int,
    per_device_eval_batch_size: int, gradient_accumulation_steps: int,
    logging_steps: int, eval_steps: int, seed: int, use_bf16: bool,
) -> TrainingArguments:
    """Build the effective Trainer configuration from caller controls."""
    return TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_strategy="no",
        warmup_ratio=warmup_ratio,
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_num_workers=2,
        report_to="none",
        # prompt_length is consumed by CompletionOnlyDataCollator, not model.forward.
        # Keep it so Trainer cannot strip the completion boundary before collation.
        remove_unused_columns=False,
        seed=seed,
        ddp_find_unused_parameters=False,
    )


class _CLAuxLossTrainer(Trainer):
    """Trainer that adds a CL-method auxiliary loss term on every step."""

    def __init__(self, *args, _cl_aux_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cl_aux_loss_fn = _cl_aux_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        loss, outputs = result
        if self._cl_aux_loss_fn is not None:
            aux = self._cl_aux_loss_fn(model)
            if aux is not None:
                loss = loss + aux.to(dtype=loss.dtype, device=loss.device)
        return (loss, outputs) if return_outputs else loss


def train_on_task_full(
    model,
    tokenizer,
    task,
    output_dir: str,
    retain_tasks=None,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 3.0,
    warmup_ratio: float = 0.01,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: int = 500,
    max_seq_length: int = 256,
    eval_size: int = 200,
    seed: int = 42,
    use_bf16: bool = False,
    save_model: bool = True,
    cl_method: CLMethod | None = None,
    stage_idx: int = 1,
) -> Tuple[Any, Dict[str, Any]]:
    """Train all model parameters on one task (no LoRA).

    Returns:
        (model_after_stage, training_report)
    """
    set_global_seed(seed)
    train_dataset, eval_dataset = load_training_dataset(task=task, eval_size=eval_size, seed=seed)
    train_dataset = _tokenize_dataset(train_dataset, tokenizer=tokenizer, max_length=max_seq_length)
    eval_dataset = _tokenize_dataset(eval_dataset, tokenizer=tokenizer, max_length=max_seq_length)

    # CL-method pre-training hook (fires BEFORE training)
    cl_method = cl_method or VanillaCLMethod()
    cl_method.pre_train(
        model,
        stage_idx=int(stage_idx),
        retain_tasks=retain_tasks,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = _build_training_arguments(
        output_path=output_path,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        seed=seed,
        use_bf16=use_bf16,
    )

    data_collator = CompletionOnlyDataCollator(tokenizer)

    trainer = _CLAuxLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        _cl_aux_loss_fn=cl_method.aux_loss,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    # CL-method post-training hook (e.g. snapshot heat state).
    # Runs after training, before saving.
    try:
        cl_device = next(model.parameters()).device
    except StopIteration:
        cl_device = torch.device("cuda")
    cl_method.post_train(
        model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        device=cl_device,
        stage_idx=int(stage_idx),
        task_name=getattr(task, "name", str(task)),
    )

    if save_model:
        model.save_pretrained(str(output_path / "model"))
        tokenizer.save_pretrained(str(output_path / "model"))

    trainer.save_state()

    def _to_serializable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): _to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_serializable(v) for v in value]
        return str(value)

    model_cfg = getattr(model, "config", None)
    model_name_or_path = getattr(model_cfg, "_name_or_path", None) if model_cfg is not None else None
    if model_name_or_path is None:
        model_name_or_path = getattr(model, "name_or_path", None)

    report = {
        "task_name": getattr(task, "name", str(task)),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "warmup_ratio": float(warmup_ratio),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "per_device_eval_batch_size": int(per_device_eval_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_seq_length": int(max_seq_length),
        "seed": int(seed),
        "stage_idx": int(stage_idx),
        "model_name": model_name_or_path,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }

    report_serialized = _to_serializable(report)
    report_path = output_path / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report_serialized, fp, indent=2)

    return model, report
