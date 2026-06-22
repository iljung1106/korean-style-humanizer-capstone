#!/usr/bin/env bash
set -euo pipefail

cd /workspace/gemma4_style_rl_training

PYTHON="${PYTHON:-/workspace/venvs/torch-cu128-clean/bin/python3}"
MODEL="${MODEL:-/workspace/gemma4_style_rl_training/outputs/pipeline_v2/merged_for_vllm/stage05f_rewrite_final}"
OUTPUT="${OUTPUT:-/workspace/gemma4_style_rl_training/outputs/pipeline_v2/stage06b_mixed_alt_from_stage05f}"
WANDB_PROJECT="${WANDB_PROJECT:-gemma4-webnovel-style-31b-new-reward}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-pipeline_v2_stage06b_mixed_alt_grpo_200}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_GENERATIONS="${NUM_GENERATIONS:-3}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-3072}"
export WANDB_PROJECT

"${PYTHON}" -m pipeline_v2.train.stage06b_grpo_mixed_alternating \
  --model "${MODEL}" \
  --init-lora \
  --lora-last-layer-fraction 0.5 \
  --output "${OUTPUT}" \
  --max-steps "${MAX_STEPS:-200}" \
  --batch-size "${BATCH_SIZE}" \
  --num-generations "${NUM_GENERATIONS}" \
  --num-iterations "${NUM_ITERATIONS}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --report-to wandb \
  --run-name "${WANDB_RUN_NAME}"
