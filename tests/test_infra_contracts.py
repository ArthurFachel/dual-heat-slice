from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cl_lora.slice.cache import SliceCacheEntry, load_slice_cache, save_slice_cache


def _entry() -> SliceCacheEntry:
    return SliceCacheEntry({"layer.weight": {"A": torch.ones(1, 2), "B": torch.ones(2, 1)}})


def test_slice_cache_requires_complete_manifest_and_rejects_partial(tmp_path: Path):
    save_slice_cache(str(tmp_path), "good", _entry(), meta={"payload": {"seed": 7}})
    manifest = json.loads((tmp_path / "good" / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["modules"] == ["layer.weight"]
    assert load_slice_cache(str(tmp_path), "good") is not None

    (tmp_path / "good" / "manifest.json").unlink()
    assert load_slice_cache(str(tmp_path), "good") is None


def test_gradient_and_sketch_caches_validate_full_identity(tmp_path: Path):
    from cl_lora.find_conflicting_seq import (
        _load_grad_cache, _load_sketch_cache, _save_grad_cache, _save_sketch_cache,
    )

    identity = {"model": "m@rev", "dataset": "task@v1", "seed": 3,
                "max_steps": 4, "batch_size": 2, "max_seq_length": 128}
    _save_grad_cache("task", {"w": torch.ones(2)}, str(tmp_path), identity=identity)
    assert set(_load_grad_cache("task", str(tmp_path), identity=identity)) == {"w"}
    with pytest.raises(ValueError, match="stale"):
        _load_grad_cache("task", str(tmp_path), identity={**identity, "seed": 4})

    sketch_identity = {**identity, "k": 11, "sketch_seed": 9}
    _save_sketch_cache("task", torch.ones(11), 1.0, {"w": 1.0}, str(tmp_path),
                       identity=sketch_identity)
    assert _load_sketch_cache("task", str(tmp_path), identity=sketch_identity)["sketch"].numel() == 11
    with pytest.raises(ValueError, match="stale"):
        _load_sketch_cache("task", str(tmp_path), identity={**sketch_identity, "k": 12})


def test_compression_skips_matching_sketch_without_loading_large_gradient(tmp_path, monkeypatch):
    import cl_lora.find_conflicting_seq as conflict

    grad_dir = tmp_path / "grads"
    sketch_dir = tmp_path / "sketches"
    grad_dir.mkdir()
    sketch_dir.mkdir()
    grad_path = grad_dir / "grad_task.pt"
    grad_path.write_bytes(b"large-gradient-placeholder")
    stat = grad_path.stat()
    identity = {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                "k": 11, "sketch_seed": 9}
    conflict._save_sketch_cache("task", torch.ones(11), 1.0, {}, str(sketch_dir),
                                identity=identity)
    real_load = torch.load

    def guarded_load(path, *args, **kwargs):
        if Path(path) == grad_path:
            raise AssertionError("gradient payload should not be loaded on a sketch hit")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    assert conflict.compress_grad_cache(str(grad_dir), str(sketch_dir), k=11, seed=9) == ["task"]


def test_dataset_cache_is_versioned_and_rejects_wrong_url(tmp_path: Path, monkeypatch):
    import cl_lora.load_dataset as loader

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"records": [1]}

    monkeypatch.setattr(loader, "_DATASET_CACHE_DIR", tmp_path)
    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: Response())
    assert loader._cached_fetch_json("https://example.test/data.json") == {"records": [1]}
    path = next(tmp_path.glob("*.json"))
    envelope = json.loads(path.read_text())
    assert envelope["cache_version"] == 1
    envelope["url"] = "https://evil.test/data.json"
    path.write_text(json.dumps(envelope))
    calls = []
    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: calls.append(a[0]) or Response())
    loader._cached_fetch_json("https://example.test/data.json")
    assert calls == ["https://example.test/data.json"]


def test_reproducibility_failures_are_warned_and_returned(monkeypatch):
    import cl_lora.repro as repro
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.warns(RuntimeWarning, match="NumPy seeding failed"):
        report = repro.set_global_seed(7)
    assert report["failures"]


def test_ssh_commands_do_not_disable_host_key_checking():
    source = (Path(__file__).parents[1] / "results_analysis.py").read_text()
    assert "StrictHostKeyChecking=no" not in source
    assert 'CL_LORA_LOCAL_RESULTS' in source
    assert 'CL_LORA_REMOTE_BASE' in source


def test_parallel_eval_preserves_extra_argument_boundaries():
    source = (Path(__file__).parents[1] / "scripts/parallel_eval.sh").read_text()
    assert "EXTRA_ARGS_STR" not in source
    assert "declare -p EXTRA_ARGS" in source


def test_recompute_metrics_import_has_no_side_effects_and_cli_dry_run(tmp_path: Path):
    module = importlib.import_module("recompute_metrics")
    assert hasattr(module, "main")
    run = tmp_path / "run"
    stage = run / "stages" / "stage_0"
    stage.mkdir(parents=True)
    record = {"general": {"raw": {"gp": {"x": {"acc,none": 0.5}}, "ip": {}}}}
    path = stage / "stage_record.json"
    path.write_text(json.dumps(record))
    result = subprocess.run(
        [sys.executable, "recompute_metrics.py", "--run-dir", str(run), "--dry-run"],
        cwd=Path(__file__).parents[1], text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(path.read_text()) == record
    assert "Would patch stage_0" in result.stdout

    result = subprocess.run(
        [sys.executable, "recompute_metrics.py", "--run-dir", str(run)],
        cwd=Path(__file__).parents[1], text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (stage / "stage_record.json.bak").is_file()
    assert json.loads((stage / "stage_record.json.bak").read_text()) == record
    assert json.loads(path.read_text())["general"]["gp"] == {"x": 0.5}
