from __future__ import annotations

import json
from pathlib import Path

from cl_lora.task_sequences import NI488, SuperNITask, all_superni_tasks
from results_analysis import collect_runs, load_run


def _write_run(path: Path, *, seed: int, rank: int, ap: float):
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"AP": ap, "FP": ap, "Forget": 0, "GP": None, "IP": None}))
    (path / "run_config.json").write_text(json.dumps({"orchestrator": {"cli_args": {
        "sequence": "S", "seed": seed, "rank": rank,
    }}}))


def test_superni_enumeration_ignores_trace_tasks():
    tasks = all_superni_tasks()
    assert tasks
    assert all(isinstance(task, SuperNITask) for task in tasks)


def test_ni488_maps_to_upstream_task_488():
    assert NI488.name == "task488_extract_all_alphabetical_elements_from_list_in_order"


def test_load_run_normalizes_legacy_qwen_metric_keys(tmp_path):
    run = tmp_path / "S" / "qwen"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({
        "average_accuracy_ap": 0.8, "final_performance_fp": 0.7, "avg_forgetting": 0.1,
    }))
    loaded = load_run(run)
    assert loaded["metrics"]["AP"] == 0.8
    assert loaded["metrics"]["FP"] == 0.7
    assert loaded["metrics"]["Forget"] == 0.1


def test_collect_runs_preserves_distinct_seed_and_rank_runs(tmp_path):
    _write_run(tmp_path / "S" / "method_seed1", seed=1, rank=16, ap=0.1)
    _write_run(tmp_path / "S" / "method_seed2", seed=2, rank=32, ap=0.2)
    runs = collect_runs(tmp_path)
    assert len(runs) == 2
    assert {(r["config"]["seed"], r["config"]["rank"]) for r in runs} == {(1, 16), (2, 32)}


def test_collect_runs_does_not_collapse_same_method_across_roots(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _write_run(root_a / "S" / "method", seed=1, rank=16, ap=0.1)
    _write_run(root_b / "S" / "method", seed=2, rank=16, ap=0.2)
    runs = collect_runs(root_a, root_b)
    assert len(runs) == 2


def test_qwen_sources_emit_canonical_metric_keys():
    root = Path(__file__).resolve().parents[1] / "cl_lora"
    for name in ("qwen_experiment.py", "qwen_full_experiment.py"):
        source = (root / name).read_text()
        assert '"AP": ap' in source
        assert '"FP": fp' in source
        assert '"Forget": avg_forgetting' in source
