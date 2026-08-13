"""Reservoir replay bookkeeping for sequential text tasks."""
from __future__ import annotations
import os, random
from typing import Any
from .base import CLMethod

class ReplayMethod(CLMethod):
    name = "replay"
    def __init__(self, *, replay_size: int = 128, seed: int = 42, **kwargs: Any):
        super().__init__(replay_size=replay_size, seed=seed, **kwargs)
        self.replay_size = int(replay_size)
        self.seed = int(seed)
        self.buffer = []
        self.num_seen = 0

    def mix_examples(self, current, previous_datasets):
        old = [item for dataset in previous_datasets for item in dataset]
        rng = random.Random(self.seed + len(self.buffer))
        rng.shuffle(old)
        return list(current) + old[:self.replay_size]

    def prepare_train_dataset(self, dataset):
        if not self.buffer:
            return dataset
        from datasets import Dataset, concatenate_datasets
        replay = Dataset.from_list(self.buffer)
        return concatenate_datasets([dataset, replay])

    def post_train(self, lora_model, *, tokenizer, train_dataset, device, stage_idx, task_name):
        rows = list(train_dataset) if train_dataset is not None else []
        rng = random.Random(self.seed + stage_idx)
        for row in rows:
            self.num_seen += 1
            if len(self.buffer) < self.replay_size:
                self.buffer.append(row)
            else:
                index = rng.randrange(self.num_seen)
                if index < self.replay_size:
                    self.buffer[index] = row

    def save(self, state_dir):
        os.makedirs(state_dir, exist_ok=True)
        import torch
        torch.save({"replay_size": self.replay_size, "seed": self.seed, "buffer": self.buffer, "num_seen": self.num_seen}, os.path.join(state_dir, "replay_state.pt"))

    def load(self, state_dir):
        path = os.path.join(state_dir, "replay_state.pt")
        if os.path.exists(path):
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.replay_size = int(payload.get("replay_size", self.replay_size))
            self.seed = int(payload.get("seed", self.seed))
            self.buffer = payload.get("buffer", [])
            self.num_seen = int(payload.get("num_seen", len(self.buffer)))

    def metadata(self):
        return {"name": self.name, "replay_size": self.replay_size, "buffer_size": len(self.buffer)}
