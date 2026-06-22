#!/usr/bin/env bash
set -euo pipefail

cd /workspace/gemma4_style_rl_training

PYTHON="${PYTHON:-/workspace/venvs/torch-cu128-clean/bin/python3}"
MODEL="${MODEL:-/workspace/gemma4_style_rl_training/outputs/pipeline_v2/merged_for_vllm/stage05f_rewrite_final}"
OUTPUT="${OUTPUT:-/workspace/gemma4_style_rl_training/outputs/pipeline_v2/stage06_mixed_from_stage05f}"
WANDB_PROJECT="${WANDB_PROJECT:-gemma4-webnovel-style-31b-new-reward}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-pipeline_v2_stage06_mixed_grpo}"
export WANDB_PROJECT

"${PYTHON}" -m pipeline_v2.train.stage06_grpo_mixed \
  --model "${MODEL}" \
  --init-lora \
  --lora-last-layer-fraction 0.5 \
  --output "${OUTPUT}" \
  --max-steps "${MAX_STEPS:-20}" \
  --report-to wandb \
  --run-name "${WANDB_RUN_NAME}"
