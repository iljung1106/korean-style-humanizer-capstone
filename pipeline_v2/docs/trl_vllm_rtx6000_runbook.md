# TRL + vLLM Stage08C Runbook

This is an experimental alternative to the Unsloth GRPO path. It is intended for
one node with two RTX PRO 6000 GPUs:

- GPU 0: vanilla TRL `GRPOTrainer` training with LoRA.
- GPU 1: `trl vllm-serve` generation server.

This separation follows the TRL/vLLM server-mode guidance and avoids the slow
regular Transformers `generate()` path observed in the B200 Unsloth smoke tests.

## Environment

Use a fresh venv. Do not install this into the Unsloth venv.

```bash
uv venv --seed --python 3.10 /workspace/venvs/trl_vllm
source /workspace/venvs/trl_vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r /workspace/gemma4_style_rl_training/requirements-trl-vllm.txt
python -m pip check
```

The TRL docs currently state that vLLM `0.12.0` through `0.18.0` is supported.
If the resolver picks a different vLLM range, stop and check before training.
This venv intentionally uses `transformers>=4.56,<5` because current vLLM
wheels require `transformers<5`; do not reuse the Unsloth `transformers==5.5.x`
environment.

## Start vLLM Server

Run this on the inference GPU only:

```bash
source /workspace/venvs/trl_vllm/bin/activate
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model /workspace/models/stage08b_merged \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching True
```

If Gemma 4 does not load with the native vLLM implementation, retry the server
with:

```bash
  --vllm-model-impl transformers --enforce-eager
```

That fallback may be slower, but it is the first compatibility check before
abandoning vLLM.

## Smoke Train Command

Run this on the training GPU only. Keep `max_completion_length=3072` so the test
does not hide the long-generation issue.

```bash
source /workspace/venvs/trl_vllm/bin/activate
cd /workspace/gemma4_style_rl_training

CUDA_VISIBLE_DEVICES=0 python pipeline_v2/train/stage08c_grpo_generate_trl_vllm.py \
  --model /workspace/models/stage08b_merged \
  --init-lora \
  --no-load-in-4bit \
  --load-in-16bit \
  --max-steps 2 \
  --batch-size 2 \
  --num-generations 2 \
  --num-iterations 1 \
  --grad-accum 1 \
  --max-completion-length 3072 \
  --vllm-server-host 127.0.0.1 \
  --vllm-server-port 8000 \
  --vllm-model-impl vllm \
  --vllm-tensor-parallel-size 1 \
  --vllm-max-model-length 8192 \
  --report-to none \
  --run-name stage08c_trl_vllm_smoke_b2_g2_i1
```

Expected smoke evidence:

- Server GPU utilization rises during generation.
- Training log reaches reward sample writing within about two minutes after
  generation begins.
- `reward_samples.jsonl` contains at least two samples.
- `rewards/*/std` is not zero for every group.
- No OOM during logprob/backward.

Stop immediately if generation passes two minutes with zero reward samples.

## Full Stage08C Candidate

Only after the smoke works:

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline_v2/train/stage08c_grpo_generate_trl_vllm.py \
  --model /workspace/models/stage08b_merged \
  --init-lora \
  --no-load-in-4bit \
  --load-in-16bit \
  --max-steps 20 \
  --batch-size 5 \
  --num-generations 5 \
  --num-iterations 1 \
  --grad-accum 1 \
  --max-completion-length 3072 \
  --vllm-server-host 127.0.0.1 \
  --vllm-server-port 8000 \
  --vllm-model-impl vllm \
  --vllm-tensor-parallel-size 1 \
  --vllm-max-model-length 8192 \
  --report-to wandb \
  --run-name stage08c_trl_vllm_generate_from_stage08b
```

Keep `batch_size == num_generations` for GRPO group construction.

## Notes

- This script starts from the merged Stage08B model and creates a fresh LoRA.
- It does not use Unsloth and does not use 4-bit loading.
- Server mode requires separate CUDA devices. Do not run trainer and server on
  the same visible GPU.
- `vllm_server_base_url` overrides host/port if supplied.
