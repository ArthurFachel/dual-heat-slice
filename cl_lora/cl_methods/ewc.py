"""Classical diagonal EWC for sequential NLP stages."""
from __future__ import annotations
import os
from typing import Any, Dict
import torch
from .base import CLMethod

class EWCMethod(CLMethod):
    name = "ewc"
    def __init__(self, *, lambda_ewc: float = 10.0, **kwargs: Any) -> None:
        super().__init__(lambda_ewc=lambda_ewc, **kwargs)
        self.lambda_ewc = float(lambda_ewc)
        self._snapshots: Dict[str, torch.Tensor] = {}
        self._importance: Dict[str, torch.Tensor] = {}
        self._anchors = []
        self._active_anchors = []
        self._current_fisher: Dict[str, torch.Tensor] = {}
        self._fisher_counts: Dict[str, int] = {}
        self._fisher_samples = 0
        self._fisher_handles = []

    def pre_train(self, lora_model, *, stage_idx, retain_tasks):
        for handle in self._fisher_handles:
            handle.remove()
        self._fisher_handles = []
        self._current_fisher = {}
        self._fisher_counts = {}
        self._fisher_samples = 0
        device = next(lora_model.parameters()).device
        self._active_anchors = [
            (
                {n: v.to(device) for n, v in snapshots.items()},
                {n: v.to(device) for n, v in importance.items()},
            )
            for snapshots, importance in self._anchors
        ]
        if self._snapshots:
            device = next(lora_model.parameters()).device
            self._snapshots = {n: v.to(device) for n, v in self._snapshots.items()}
            self._importance = {n: v.to(device) for n, v in self._importance.items()}
        for name, parameter in lora_model.named_parameters():
            if not parameter.requires_grad:
                continue
            self._current_fisher[name] = torch.zeros_like(parameter, device=parameter.device, dtype=torch.float32)
            self._fisher_counts[name] = 0
            self._fisher_handles.append(parameter.register_hook(self._make_fisher_hook(name)))

    def _make_fisher_hook(self, name: str):
        def hook(grad):
            if name in self._current_fisher:
                self._current_fisher[name].add_(grad.detach().float().pow(2))
                self._fisher_counts[name] += 1
                self._fisher_samples = max(self._fisher_counts.values(), default=0)
            return grad
        return hook

    def aux_loss(self, lora_model):
        if not self._anchors and not self._snapshots:
            return None
        total = None
        anchors = self._active_anchors or self._anchors or [(self._snapshots, self._importance)]
        for name, p in lora_model.named_parameters():
            for snapshots, importance in anchors:
                old = snapshots.get(name)
                imp = importance.get(name)
                if old is None or imp is None:
                    continue
                old = old.to(dtype=torch.float32)
                imp = imp.to(dtype=torch.float32)
                term = (imp * (p.float() - old).pow(2)).mean()
                total = term if total is None else total + term
        return None if total is None else self.lambda_ewc * total

    def post_train(self, lora_model, *, tokenizer, train_dataset, device, stage_idx, task_name):
        for handle in self._fisher_handles:
            handle.remove()
        self._fisher_handles = []
        current_snapshots = {
            n: p.detach().cpu().float().clone()
            for n, p in lora_model.named_parameters()
            if p.requires_grad
        }
        current_importance = {}
        for name in current_snapshots:
            current = self._current_fisher.get(name)
            if current is None:
                current = torch.ones_like(current_snapshots[name])
            current = current.detach().cpu().float()
            count = max(self._fisher_counts.get(name, 0), 1)
            current = current / float(count)
            previous = self._importance.get(name)
            if previous is not None:
                current = current + previous.detach().cpu().float()
            current_importance[name] = current
        # Online consolidation keeps one anchor in the current parameter
        # coordinates instead of accumulating references to recreated adapters.
        self._anchors = [(current_snapshots, current_importance)]
        self._snapshots = current_snapshots
        self._importance = current_importance
        self._active_anchors = []

    def save(self, state_dir):
        os.makedirs(state_dir, exist_ok=True)
        torch.save({"lambda_ewc": self.lambda_ewc, "snapshots": self._snapshots, "importance": self._importance, "anchors": self._anchors}, os.path.join(state_dir, "ewc_state.pt"))

    def load(self, state_dir):
        path = os.path.join(state_dir, "ewc_state.pt")
        if os.path.exists(path):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.lambda_ewc = float(payload.get("lambda_ewc", self.lambda_ewc))
            self._snapshots = payload.get("snapshots", {})
            self._importance = payload.get("importance", {})
            self._anchors = payload.get("anchors", [])

    def metadata(self):
        return {"name": self.name, "lambda_ewc": self.lambda_ewc, "parameters": len(self._snapshots), "tasks": len(self._anchors)}
