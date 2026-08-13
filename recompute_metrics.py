"""Safely re-extract GP/IP values from stage records.

Importing this module never reads or modifies experiment artifacts. Use the CLI
with an explicit --run-dir; writes require the default backup or --no-backup.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _extract_primary_metric(task_result):
    preferred = ["acc_norm,none", "acc,none", "exact_match,none",
                 "exact_match,get-answer", "f1,none", "rougeL,none", "bleu,none"]
    for key in preferred:
        if key in task_result:
            return float(task_result[key])
    for key, value in task_result.items():
        if isinstance(value, float) and (key.startswith("exact_match,") or key.startswith("acc,")):
            return float(value)
    for key, value in task_result.items():
        if isinstance(value, float) and "stderr" not in key:
            return float(value)
    return None


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _patched_record(data: dict[str, Any]) -> dict[str, Any] | None:
    result = copy.deepcopy(data)
    general = result.get("general", {})
    raw = general.get("raw", {})
    if not raw:
        return None
    gp_scores = {task: _extract_primary_metric(result) for task, result in raw.get("gp", {}).items()}
    ip_scores = {task: _extract_primary_metric(result) for task, result in raw.get("ip", {}).items()}
    if "alpaca" in general.get("gp", {}):
        gp_scores["alpaca"] = general["gp"]["alpaca"]
        if "alpaca" in general.get("ip", {}):
            ip_scores["alpaca"] = general["ip"]["alpaca"]
    general.update(gp=gp_scores, ip=ip_scores,
                   gp_mean=_mean(gp_scores.values()), ip_mean=_mean(ip_scores.values()))
    result["general"] = general
    return result


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def recompute_stage_metrics(run_dir: Path, *, dry_run: bool, backup: bool) -> int:
    stages_dir = run_dir / "stages"
    if not stages_dir.is_dir():
        raise FileNotFoundError(f"Stages directory not found: {stages_dir}")
    changed = 0
    for stage_dir in sorted(stages_dir.iterdir()):
        record_path = stage_dir / "stage_record.json"
        if not record_path.is_file():
            continue
        data = json.loads(record_path.read_text(encoding="utf-8"))
        patched = _patched_record(data)
        if patched is None:
            print(f"No raw data in {stage_dir.name}, skipping")
            continue
        if patched == data:
            print(f"Unchanged {stage_dir.name}")
            continue
        changed += 1
        if dry_run:
            print(f"Would patch {stage_dir.name}")
            continue
        if backup:
            backup_path = record_path.with_suffix(record_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(record_path, backup_path)
        _atomic_write(record_path, patched)
        print(f"Patched {stage_dir.name}")
    return changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Explicit run directory containing stages/")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument("--no-backup", action="store_true",
                        help="Write without creating stage_record.json.bak")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    changed = recompute_stage_metrics(args.run_dir, dry_run=args.dry_run,
                                      backup=not args.no_backup)
    print(f"{changed} stage record(s) {'would change' if args.dry_run else 'changed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
