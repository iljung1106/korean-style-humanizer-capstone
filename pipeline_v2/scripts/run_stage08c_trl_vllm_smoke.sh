#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/gemma4_style_rl_training}"
MODEL="${MODEL:-/workspace/models/stage08b_merged}"
VENV="${VENV:-/workspace/venvs/trl_vllm}"
GPU="${GPU:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RUN_NAME="${RUN_NAME:-stage08c_trl_vllm_smoke_b2_g2_i1}"

source "${VENV}/bin/activate"
cd "${REPO}"

echo "[train] CUDA_VISIBLE_DEVICES=${GPU} run=${RUN_NAME}"
CUDA_VISIBLE_DEVICES="${GPU}" python pipeline_v2/train/stage08c_grpo_generate_trl_vllm.py \
  --model "${MODEL}" \
  --init-lora \
  --no-load-in-4bit \
  --load-in-16bit \
  --max-steps 2 \
  --batch-size 2 \
  --num-generations 2 \
  --num-iterations 1 \
  --grad-accum 1 \
  --max-completion-length 3072 \
  --vllm-server-host "${HOST}" \
  --vllm-server-port "${PORT}" \
  --vllm-model-impl "${VLLM_MODEL_IMPL:-vllm}" \
  --vllm-tensor-parallel-size "${TP:-1}" \
  --vllm-max-model-length "${MAX_MODEL_LEN:-8192}" \
  --report-to none \
  --run-name "${RUN_NAME}" \
  "$@"
