#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
RANK="${RANK:-32}"
RUN_SUFFIX="${RUN_SUFFIX:-smoke}"
SEQUENCES_RAW="${SEQUENCES:-NI-Seq-G2}"
ONLY_INITS_RAW="${ONLY_INITS:-}"
ONLY_CL_RAW="${ONLY_CL:-}"
read -r -a SEQUENCES <<< "${SEQUENCES_RAW}"
read -r -a ONLY_INITS <<< "${ONLY_INITS_RAW}"
read -r -a ONLY_CL <<< "${ONLY_CL_RAW}"
INITS=(lora_vanilla loram lora_ga slice)
CL_METHODS=(vanilla o_lora dual_heat)

matches() { local x="$1"; shift; [[ $# -eq 0 ]] && return 0; local y; for y in "$@"; do [[ "$x" == "$y" ]] && return 0; done; return 1; }
init_args() { case "$1" in lora_vanilla) ;; loram|lora_ga|slice) printf '%s\n' --slice-init --slice-init-method "$1" --slice-cache-dir "${SLICE_CACHE_DIR:-slice_cache}" --slice-max-steps "${SLICE_MAX_STEPS:-8}" ;; esac; }
cl_args() { case "$1" in vanilla) printf '%s\n' --cl-method vanilla ;; o_lora) printf '%s\n' --cl-method o_lora --cl-o-lora-lambda "${O_LORA_LAMBDA:-0.5}" ;; dual_heat) printf '%s\n' --cl-method dual_heat ;; esac; }

failed=0
for sequence in "${SEQUENCES[@]}"; do
  for init in "${INITS[@]}"; do
    matches "$init" "${ONLY_INITS[@]}" || continue
    for method in "${CL_METHODS[@]}"; do
      matches "$method" "${ONLY_CL[@]}" || continue
      mapfile -t ia < <(init_args "$init")
      mapfile -t ca < <(cl_args "$method")
      run="compose_${init}_${method}_${sequence//-/_}_${RUN_SUFFIX}"
      if ! CUDA_VISIBLE_DEVICES="$GPU" python -m cl_lora.orchestrator --sequence "$sequence" --run-name "$run" --rank "$RANK" --train-only "${ia[@]}" "${ca[@]}" "$@"; then
        echo "FAILED: $sequence|$init|$method" >&2
        failed=1
        [[ "${FAIL_FAST:-1}" == 1 ]] && exit 1
      fi
    done
  done
done
exit "$failed"
