from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import logging

logger = logging.getLogger("cl_lora.slice.cache")


@dataclass
class SliceCacheEntry:
    inits: Dict[str, Dict[str, torch.Tensor]]

    def to(self, device: torch.device) -> "SliceCacheEntry":
        for _, ab in self.inits.items():
            for k, v in ab.items():
                ab[k] = v.to(device)
        return self


def make_cache_key(payload: Dict[str, Any]) -> str:
    def _to_json_safe(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(k): _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_json_safe(v) for v in obj]
        return str(obj)

    safe_payload = _to_json_safe(payload)
    payload_str = json.dumps(safe_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def load_slice_cache(
    cache_dir: str,
    cache_key: str,
    device: Optional[torch.device] = None,
) -> Optional[SliceCacheEntry]:
    root_dir = os.path.join(cache_dir, cache_key)
    if not os.path.isdir(root_dir):
        logger.debug("Slice cache root missing: %s", root_dir)
        return None

    manifest_path = os.path.join(root_dir, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Rejecting incomplete slice cache without valid manifest: %s", root_dir)
        return None
    if manifest.get("version") != 1 or manifest.get("complete") is not True:
        logger.warning("Rejecting incomplete or incompatible slice cache: %s", root_dir)
        return None

    inits_dir = os.path.join(root_dir, "inits")
    if not os.path.isdir(inits_dir):
        return None

    expected_modules = manifest.get("modules")
    if not isinstance(expected_modules, list) or not expected_modules:
        return None
    inits: Dict[str, Dict[str, torch.Tensor]] = {}
    for key in expected_modules:
        path = os.path.join(inits_dir, f"{key}.pt")
        if not os.path.isfile(path):
            logger.warning("Rejecting partial slice cache missing %s", path)
            return None
        map_loc = device if device is not None else "cpu"
        try:
            payload = torch.load(path, map_location=map_loc, weights_only=True)
        except (OSError, RuntimeError, EOFError):
            logger.warning("Rejecting unreadable slice cache tensor: %s", path)
            return None
        if not isinstance(payload, dict) or "A" not in payload or "B" not in payload:
            return None
        inits[key] = {"A": payload["A"], "B": payload["B"]}

    if not inits:
        logger.debug("Slice cache at %s contains no inits", root_dir)
        return None

    logger.info("Loaded slice cache from %s with %d modules", root_dir, len(inits))
    return SliceCacheEntry(inits=inits)


def save_slice_cache(
    cache_dir: str,
    cache_key: str,
    entry: SliceCacheEntry,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    final_dir = os.path.join(cache_dir, cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=f".{cache_key}.", dir=cache_dir)
    try:
        inits_dir = os.path.join(staging_dir, "inits")
        os.makedirs(inits_dir)
        modules = sorted(entry.inits)
        for name in modules:
            ab = entry.inits[name]
            payload = {"A": ab["A"], "B": ab["B"]}
            torch.save(payload, os.path.join(inits_dir, f"{name}.pt"))

        if meta is not None:
            with open(os.path.join(staging_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, sort_keys=True, indent=2)
        with open(os.path.join(staging_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"version": 1, "complete": True, "modules": modules}, f,
                      sort_keys=True, indent=2)

        old_dir = None
        if os.path.exists(final_dir):
            old_dir = final_dir + ".old"
            shutil.rmtree(old_dir, ignore_errors=True)
            os.replace(final_dir, old_dir)
        try:
            os.replace(staging_dir, final_dir)
        except Exception:
            if old_dir and os.path.exists(old_dir):
                os.replace(old_dir, final_dir)
            raise
        if old_dir:
            shutil.rmtree(old_dir, ignore_errors=True)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    logger.info("Saved slice cache to %s with %d modules", final_dir, len(entry.inits))


def save_ab_stats_csv(
    cache_dir: str,
    cache_key: str,
    inits: Dict[str, Dict[str, torch.Tensor]],
) -> str:
    root_dir = os.path.join(cache_dir, cache_key)
    os.makedirs(root_dir, exist_ok=True)
    out_path = os.path.join(root_dir, "ab_stats.csv")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["module", "tensor", "shape", "mean", "var", "min", "max"])

        for module_name in sorted(inits.keys()):
            ab = inits[module_name]
            for tensor_name in ("A", "B"):
                t = ab[tensor_name].detach().float()
                writer.writerow([
                    module_name,
                    tensor_name,
                    "x".join(str(d) for d in t.shape),
                    float(t.mean().item()),
                    float(t.var(unbiased=False).item()),
                    float(t.min().item()),
                    float(t.max().item()),
                ])

    logger.info("Saved A/B summary CSV: %s", out_path)
    return out_path


def save_projection_stats_json(
    cache_dir: str,
    cache_key: str,
    projection_stats: Dict[str, Any],
) -> str:
    root_dir = os.path.join(cache_dir, cache_key)
    os.makedirs(root_dir, exist_ok=True)
    out_path = os.path.join(root_dir, "projection_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(projection_stats, f, sort_keys=True, indent=2)
    logger.info("Saved projection stats JSON: %s", out_path)
    return out_path
