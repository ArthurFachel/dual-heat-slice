#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
EXPECTED_STAGES="${EXPECTED_STAGES:-5}"

[[ -d "$RESULTS_ROOT" ]] || { echo "Results root not found: $RESULTS_ROOT" >&2; exit 1; }
failed=0
while IFS= read -r run_dir; do
  completed=0
  while IFS= read -r log; do
    # Empty logs, Python tracebacks, and explicit error/failure markers are failures.
    if [[ -s "$log" ]] && ! grep -Eq 'Traceback \(most recent call last\)|(^|[^A-Za-z])(ERROR|FAILED|Exception):' "$log"; then
      completed=$((completed + 1))
    fi
  done < <(find "$run_dir/stages" -mindepth 2 -maxdepth 2 -name parallel_eval.log -type f 2>/dev/null | sort)
  printf '%s: %d/%d\n' "$run_dir" "$completed" "$EXPECTED_STAGES"
  [[ "$completed" -eq "$EXPECTED_STAGES" ]] || failed=1
done < <(find "$RESULTS_ROOT" -type d -name stages -printf '%h\n' | sort)
exit "$failed"
