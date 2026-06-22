#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/workspace/models/stage08b_merged}"
VENV="${VENV:-/workspace/venvs/trl_vllm}"
GPU="${GPU:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MODEL_IMPL="${VLLM_MODEL_IMPL:-vllm}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

source "${VENV}/bin/activate"

cmd=(
  trl vllm-serve
  --model "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enable-prefix-caching True
  --vllm-model-impl "${VLLM_MODEL_IMPL}"
)

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

echo "[server] CUDA_VISIBLE_DEVICES=${GPU} ${cmd[*]}"
CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
