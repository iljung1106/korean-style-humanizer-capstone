#!/usr/bin/env bash
set -euo pipefail

cd /workspace/gemma4_style_rl_training

PYTHON="${PYTHON:-/venv/main/bin/python3}"
STAGE07C_OUTPUT="${STAGE07C_OUTPUT:-/workspace/gemma4_style_rl_training/outputs/pipeline_v2/full_lora_recipe_v1/stage07c_mixed_from_stage06b_g5_iter2_150}"
ADAPTER_PATH="${ADAPTER_PATH:-${STAGE07C_OUTPUT}/policy}"
BASE_MODEL="${BASE_MODEL:-/workspace/modelscope_cache/stage06b_mixed150_merged}"
MERGED_DIR="${MERGED_DIR:-/workspace/modelscope_cache/stage07c_mixed_from_stage06b_merged}"
EVAL_DIR="${EVAL_DIR:-/workspace/gemma4_style_rl_training/outputs/local_reports/stage07c_100_eval/eval_v2}"
EVAL_OUTPUT="${EVAL_OUTPUT:-/workspace/gemma4_style_rl_training/outputs/local_reports/stage07c_100_eval}"
GEN_DIR="${GEN_DIR:-${EVAL_OUTPUT}/generations}"
REPORT_DIR="${REPORT_DIR:-${EVAL_OUTPUT}/stage07c_human_ai_rewrite_report_om_metrics}"
LOG_DIR="${LOG_DIR:-/workspace/gemma4_style_rl_training/logs}"

HF_LORA_REPO="${HF_LORA_REPO:-}"
HF_MERGED_REPO="${HF_MERGED_REPO:-}"
HF_PRIVATE="${HF_PRIVATE:-true}"
if [[ "${HF_PRIVATE}" == "true" || "${HF_PRIVATE}" == "1" || "${HF_PRIVATE}" == "yes" ]]; then
  HF_PRIVATE_FLAG="--private"
else
  HF_PRIVATE_FLAG="--no-private"
fi

mkdir -p "${LOG_DIR}" "${EVAL_OUTPUT}"

if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "[stage07c/finalize/error] missing adapter: ${ADAPTER_PATH}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "[stage07c/finalize/error] missing base model: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_DIR}/rewrite_prompts.jsonl" ]]; then
  echo "[stage07c/finalize/error] missing eval set: ${EVAL_DIR}" >&2
  exit 1
fi

echo "[stage07c/finalize] adapter=${ADAPTER_PATH}"
echo "[stage07c/finalize] base=${BASE_MODEL}"
echo "[stage07c/finalize] merged=${MERGED_DIR}"
echo "[stage07c/finalize] eval=${EVAL_DIR}"

"${PYTHON}" pipeline_v2/eval/phase_eval/merge_lora_for_vllm.py \
  --adapter-path "${ADAPTER_PATH}" \
  --base-model "${BASE_MODEL}" \
  --output-dir "${MERGED_DIR}" \
  --dtype bfloat16 \
  --device-map auto \
  --max-shard-size 5GB \
  2>&1 | tee "${LOG_DIR}/stage07c_merge_for_vllm.log"

echo "[stage07c/finalize] vllm smoke"
"${PYTHON}" pipeline_v2/eval/phase_eval/run_generation_vllm.py \
  --eval-dir "${EVAL_DIR}" \
  --output-dir "${EVAL_OUTPUT}/vllm_smoke" \
  --model "${MERGED_DIR}" \
  --model-label stage07c_smoke \
  --chat-template gemma-4 \
  --max-seq-length 8192 \
  --max-prompt-length 4096 \
  --max-new-tokens-rewrite 512 \
  --max-new-tokens-generate 512 \
  --batch-size 1 \
  --rewrite-limit 1 \
  --skip-generate \
  --temperature 0.7 \
  --top-p 0.95 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --no-load-in-4bit \
  --no-load-in-16bit \
  --vllm-dtype bfloat16 \
  --vllm-quantization "" \
  --vllm-load-format auto \
  --vllm-kv-cache-dtype fp8 \
  --vllm-gpu-memory-utilization 0.88 \
  --vllm-max-num-seqs 4 \
  --vllm-max-num-batched-tokens 0 \
  --no-vllm-enable-prefix-caching \
  --no-vllm-enforce-eager \
  2>&1 | tee "${LOG_DIR}/stage07c_vllm_smoke.log"

echo "[stage07c/finalize] rewrite100 eval"
"${PYTHON}" pipeline_v2/eval/phase_eval/run_generation_vllm.py \
  --eval-dir "${EVAL_DIR}" \
  --output-dir "${GEN_DIR}" \
  --model "${MERGED_DIR}" \
  --model-label stage07c \
  --chat-template gemma-4 \
  --max-seq-length 8192 \
  --max-prompt-length 4096 \
  --max-new-tokens-rewrite 4096 \
  --max-new-tokens-generate 4096 \
  --batch-size 8 \
  --skip-generate \
  --temperature 0.7 \
  --top-p 0.95 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --no-load-in-4bit \
  --no-load-in-16bit \
  --vllm-dtype bfloat16 \
  --vllm-quantization "" \
  --vllm-load-format auto \
  --vllm-kv-cache-dtype fp8 \
  --vllm-gpu-memory-utilization 0.88 \
  --vllm-max-num-seqs 8 \
  --vllm-max-num-batched-tokens 0 \
  --no-vllm-enable-prefix-caching \
  --no-vllm-enforce-eager \
  2>&1 | tee "${LOG_DIR}/stage07c_rewrite100_eval.log"

echo "[stage07c/finalize] report"
"${PYTHON}" pipeline_v2/scripts/80_build_stage06b_rewrite100_report.py \
  --eval-dir "${EVAL_DIR}" \
  --generation-dir "${GEN_DIR}" \
  --output-dir "${REPORT_DIR}" \
  --prefix stage07c_rewrite100 \
  --title "Stage07C Rewrite 100 Human / AI / Rewrite 문체 지표" \
  --min-effect medium \
  2>&1 | tee "${LOG_DIR}/stage07c_rewrite100_report.log"

if [[ -n "${HF_LORA_REPO}" ]]; then
  echo "[stage07c/finalize] upload lora ${HF_LORA_REPO}"
  "${PYTHON}" pipeline_v2/scripts/upload_hf_folder.py \
    --repo-id "${HF_LORA_REPO}" \
    --folder "${ADAPTER_PATH}" \
    "${HF_PRIVATE_FLAG}" \
    --commit-message "Upload Stage07C LoRA"
fi

if [[ -n "${HF_MERGED_REPO}" ]]; then
  echo "[stage07c/finalize] upload merged ${HF_MERGED_REPO}"
  "${PYTHON}" pipeline_v2/scripts/upload_hf_folder.py \
    --repo-id "${HF_MERGED_REPO}" \
    --folder "${MERGED_DIR}" \
    "${HF_PRIVATE_FLAG}" \
    --commit-message "Upload Stage07C merged model"
fi

echo "[stage07c/finalize] done"
