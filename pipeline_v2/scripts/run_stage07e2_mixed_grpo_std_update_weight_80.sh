#!/usr/bin/env bash
set -euo pipefail

cd /workspace/gemma4_style_rl_training

PYTHON="${PYTHON:-/venv/main/bin/python3}"
MODEL="${MODEL:-/workspace/modelscope_cache/stage06b_mixed150_merged}"
OUTPUT="${OUTPUT:-/workspace/gemma4_style_rl_training/outputs/pipeline_v2/full_lora_recipe_v1/stage07e2_mixed_from_stage06b_g8_std_weight_80}"
WANDB_PROJECT="${WANDB_PROJECT:-gemma4-webnovel-style-31b-new-reward}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-pipeline_v2_stage07e2_mixed_from_stage06b_g8_std_weight_80}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
MAX_STEPS="${MAX_STEPS:-80}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-3072}"
LORA_LAST_LAYER_FRACTION="${LORA_LAST_LAYER_FRACTION:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.005}"
REWARD_STD_SHAPING_POWER="${REWARD_STD_SHAPING_POWER:-0.5}"
REWARD_STD_SHAPING_FLOOR="${REWARD_STD_SHAPING_FLOOR:-0.10}"
REWARD_STD_UPDATE_WEIGHT_MIN="${REWARD_STD_UPDATE_WEIGHT_MIN:-0.35}"
REWARD_STD_UPDATE_WEIGHT_STD_LOW="${REWARD_STD_UPDATE_WEIGHT_STD_LOW:-0.05}"
REWARD_STD_UPDATE_WEIGHT_STD_HIGH="${REWARD_STD_UPDATE_WEIGHT_STD_HIGH:-0.16}"

export WANDB_PROJECT

"${PYTHON}" -m pipeline_v2.train.stage07c_grpo_mixed_from_stage06b \
  --model "${MODEL}" \
  --init-lora \
  --lora-last-layer-fraction "${LORA_LAST_LAYER_FRACTION}" \
  --output "${OUTPUT}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps 10 \
  --save-total-limit 3 \
  --batch-size "${BATCH_SIZE}" \
  --num-generations "${NUM_GENERATIONS}" \
  --num-iterations "${NUM_ITERATIONS}" \
  --learning-rate "${LEARNING_RATE}" \
  --beta "${BETA}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --reward-std-shaping-power "${REWARD_STD_SHAPING_POWER}" \
  --reward-std-shaping-floor "${REWARD_STD_SHAPING_FLOOR}" \
  --reward-std-update-weight-min "${REWARD_STD_UPDATE_WEIGHT_MIN}" \
  --reward-std-update-weight-std-low "${REWARD_STD_UPDATE_WEIGHT_STD_LOW}" \
  --reward-std-update-weight-std-high "${REWARD_STD_UPDATE_WEIGHT_STD_HIGH}" \
  --reward-std-update-weight-source raw \
  --rewrite-style-base-weight 1.10 \
  --rewrite-edit-weight 0.08 \
  --rewrite-improvement-weight 0.95 \
  --rewrite-improvement-scale 0.22 \
  --rewrite-low-edit-penalty-max 0.12 \
  --rewrite-edit-gate-min 0.35 \
  --rewrite-edit-gate-q25 0.70 \
  --rewrite-edit-gate-q50 1.0 \
  --mask-truncated-completions \
  --scale-rewards group \
  --report-to wandb \
  --run-name "${WANDB_RUN_NAME}"
