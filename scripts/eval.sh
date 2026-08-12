#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

[[ -d "$RUNS_ROOT" ]] || { echo "Runs root not found: $RUNS_ROOT" >&2; exit 1; }
mapfile -t RUN_DIRS < <(find "$RUNS_ROOT" -mindepth 1 -maxdepth 2 -type f -name run_config.json -printf '%h\n' | sort -u)
[[ ${#RUN_DIRS[@]} -gt 0 ]] || { echo "No run directories found under $RUNS_ROOT" >&2; exit 1; }

failed=0
for run_dir in "${RUN_DIRS[@]}"; do
  log_file="$run_dir/standalone_eval.log"
  echo "Evaluating $run_dir"
  if ! (cd "$REPO_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m cl_lora.eval_standalone run --run-dir "$run_dir" "$@") 2>&1 | tee "$log_file"; then
    echo "FAILED: $run_dir" >&2
    failed=1
  fi
done
exit "$failed"
