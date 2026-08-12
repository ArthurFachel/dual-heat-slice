#!/usr/bin/env bash
set -euo pipefail

# Lean storage uses the same implemented-method matrix plus shared base cache.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
BASE_MODEL_CACHE="${BASE_MODEL_CACHE:-${REPO_ROOT}/outputs/base_models}"
exec env BASE_MODEL_CACHE="$BASE_MODEL_CACHE" bash "${SCRIPT_DIR}/test_init_x_cl_methods.sh" --base-model-cache "$BASE_MODEL_CACHE" "$@"
