#!/usr/bin/env bash
set -euo pipefail

cd /workspace/gemma4_style_rl_training

PYTHON="${PYTHON:-/venv/main/bin/python3}"
MODEL="${MODEL:-/workspace/modelscope_cache/stage06b_mixed150_merged}"
OUTPUT="${OUTPUT:-/workspace/gemma4_style_rl_training/outputs/pipeline_v2/full_lora_recipe_v1/stage07c_mixed_from_stage06b_g5_iter2_150}"
WANDB_PROJECT="${WANDB_PROJECT:-gemma4-webnovel-style-31b-new-reward}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-pipeline_v2_stage07c_mixed_from_stage06b_g5_iter2_150}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
MAX_STEPS="${MAX_STEPS:-150}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-3072}"
LORA_LAST_LAYER_FRACTION="${LORA_LAST_LAYER_FRACTION:-0.5}"

export WANDB_PROJECT

"${PYTHON}" -m pipeline_v2.train.stage07c_grpo_mixed_from_stage06b \
  --model "${MODEL}" \
  --init-lora \
  --lora-last-layer-fraction "${LORA_LAST_LAYER_FRACTION}" \
  --output "${OUTPUT}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-generations "${NUM_GENERATIONS}" \
  --num-iterations "${NUM_ITERATIONS}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --report-to wandb \
  --run-name "${WANDB_RUN_NAME}"
