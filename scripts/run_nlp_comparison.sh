#!/usr/bin/env bash
# Run the complete NLP continual-learning comparison suite.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/nlp_comparison}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-16}"
LR="${LR:-5e-5}"

python -m cl_lora.qwen_experiment \
  --compare-all \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --lora-rank "$RANK" \
  --lora-alpha "$ALPHA" \
  --lr "$LR"

printf '\nResults written to %s\n' "$OUTPUT_DIR"
