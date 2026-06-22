# B200 Vast Runbook

This runbook is for continuing Stage08 from the Stage08B merged checkpoint on a
single NVIDIA B200 instance.

Keep training and evaluation environments separate:

- Unsloth training: `/workspace/venvs/unsloth310`
- vLLM evaluation: `/workspace/venvs/vllm`

Do not install vLLM into the Unsloth training venv.

## Instance Template

Use a CUDA 13 / PyTorch 2.11 template when available. Blackwell GPUs need
recent CUDA/PyTorch support; avoid older CUDA 12.1/12.4 PyTorch templates even
if they start.

Prefer, in order:

1. Vast `PyTorch (Vast)` with `2.11.0-cu130-cuda-13.2-mini-py310-*`.
2. NVIDIA PyTorch container 25.03 or newer with CUDA 13 support.
3. An Unsloth Docker image that explicitly says it supports B200/Blackwell.

Avoid templates that pin old stable PyTorch wheels without Blackwell support.

The 2026-06-08 B200 Vast instance used `PyTorch (Vast)` tag
`2.11.0-cu130-cuda-13.2-mini-py310-2026-04-15`. It ships `/venv/main` with
Python 3.10, `torch 2.11.0+cu130`, CUDA 13.0 runtime packages, CUDA 13.2
toolkit, and B200 device capability `(10, 0)`.

## Environment

The last verified RTX PRO 6000 Blackwell training venv used this package family:

```text
Python 3.10
torch 2.10.0+cu128
transformers 5.5.0
trl 0.24.0
peft 0.19.1
xformers 0.0.35
unsloth 2026.5.8
```

On B200, keep the Stage08 user-space package family but do not let pip
downgrade torch. `unsloth==2026.5.8` declares `torch<2.11.0`; a normal
`pip install unsloth==2026.5.8 ...` downgraded the B200 template from
`torch 2.11.0+cu130` to `torch 2.10.0+cu128` during verification. Install or
repair torch first, then install the pinned Stage08 packages with resolver
control.

If rebuilding CUDA extensions, B200 is SM100:

```bash
export TORCH_CUDA_ARCH_LIST="10.0"
```

Do not copy RTX PRO 6000 / RTX 50 `12.0` arch settings to B200.

Recommended runtime exports:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=gemma4-webnovel-style-31b-new-reward
export LD_LIBRARY_PATH=/venv/main/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

The CUDA 13 `LD_LIBRARY_PATH` entry is required for
`bitsandbytes/libbitsandbytes_cuda130.so`; without it, import can fail with
`libnvJitLink.so.13: cannot open shared object file`.

## Build Unsloth Venv

On the Vast PyTorch 2.11/cu130 template, use `/venv/main` as the Unsloth
training environment unless a separate venv is deliberately built from the
same cu130 torch wheel. Do not install vLLM into this environment.

Repair or pin torch/xformers first:

```bash
PY=/venv/main/bin/python

$PY -m pip install --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0' 'torchvision==0.26.0'

$PY -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cu130 \
  'xformers==0.0.35'
```

If the cu130 xformers wheel lacks SM100 build coverage, build from source with
`TORCH_CUDA_ARCH_LIST="10.0"` and install that wheel. After installing
xformers 0.0.35, patch the B200 capability gate until upstream removes it:

```bash
$PY - <<'PY'
from pathlib import Path

for name in [
    "/venv/main/lib/python3.10/site-packages/xformers/ops/fmha/cutlass.py",
    "/venv/main/lib/python3.10/site-packages/xformers/ops/fmha/flash3.py",
]:
    p = Path(name)
    text = p.read_text()
    text = text.replace(
        "CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 0)",
        "CUDA_MAXIMUM_COMPUTE_CAPABILITY = (10, 0)",
    )
    p.write_text(text)
PY
```

Install Stage08 packages without letting `unsloth` resolve torch:

```bash
$PY -m pip install --no-cache-dir \
  accelerate bitsandbytes datasets sentence-transformers kiwipiepy \
  scikit-learn scipy numpy pandas joblib wandb huggingface_hub \
  jupyter ipykernel sentencepiece protobuf tyro pydantic hf_transfer \
  diffusers unsloth_zoo torchao cut_cross_entropy

$PY -m pip install --no-cache-dir --no-deps \
  'unsloth==2026.5.8' 'transformers==5.5.0' 'trl==0.24.0' 'peft==0.19.1'

$PY -m pip install --no-cache-dir 'fsspec==2025.9.0'
```

The Unsloth training venv should validate with:

```bash
export LD_LIBRARY_PATH=/venv/main/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

/venv/main/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import unsloth, transformers, trl, peft
print("imports_ok")
PY

/venv/main/bin/python -m xformers.info | egrep 'cutlass|fa2|gpu.compute|build.torch|TORCH_CUDA_ARCH_LIST'
```

Observed after the B200 gate patch on 2026-06-08:

```text
torch 2.11.0+cu130, CUDA 13.0, device capability (10, 0)
xformers 0.0.35+source build, TORCH_CUDA_ARCH_LIST=10.0
memory_efficient_attention auto/cutlass/flash direct calls: OK
GQA-style 5D call with expanded KV heads: cutlass OK, flash ERR, auto ERR
memory_efficient_attention.cutlassF-blackwell: unavailable
```

`cutlassF-blackwell: unavailable` means the named xformers Blackwell operator is
still not present. It is not by itself a pass condition. The pass condition is
actual Stage08C generation timing and nonzero GPU work after the direct
attention smoke passes. For GQA models, prefer/verify the cutlass path; do not
assume xformers auto dispatch picks a working fast path on B200.

## Build vLLM Venv

Create this separately only for evaluation/generation. Keep it out of
`/workspace/venvs/unsloth310`.

```bash
uv venv --seed --python 3.12 /workspace/venvs/vllm
uv pip install --python /workspace/venvs/vllm/bin/python vllm
```

## Download Stage08B Merged

```bash
/venv/main/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ij/gemma4-webnovel-stage08b-simpo-cpo-merged",
    repo_type="model",
    local_dir="/workspace/models/stage08b_merged",
)
PY
```

## Stage08C Smoke

Keep the 16-bit load path. The prior 4-bit merged reload path caused generation
collapse, not a useful memory fix.

```bash
cd /workspace/gemma4_style_rl_training

/venv/main/bin/python -m pipeline_v2.train.stage08c_grpo_generate_heavy_from_stage08b \
  --model /workspace/models/stage08b_merged \
  --adapter-path '' \
  --init-lora \
  --no-load-in-4bit \
  --load-in-16bit \
  --max-steps 3 \
  --batch-size 5 \
  --num-generations 5 \
  --num-iterations 2 \
  --max-completion-length 3072 \
  --report-to none
```

Pass criteria:

- Loader log shows `load_in_4bit=false` and `load_in_16bit=true`.
- `reward_std > 0`.
- `frac_reward_zero_std=0`.
- `grad_norm` is non-zero.
- No CUDA OOM in `_get_per_token_logps_and_entropies` /
  `chunked_hidden_states_selective_log_softmax`.

## Stage08C Full Run

```bash
/venv/main/bin/python -m pipeline_v2.train.stage08c_grpo_generate_heavy_from_stage08b \
  --model /workspace/models/stage08b_merged \
  --adapter-path '' \
  --init-lora \
  --no-load-in-4bit \
  --load-in-16bit \
  --max-steps 24 \
  --batch-size 5 \
  --num-generations 5 \
  --num-iterations 2 \
  --max-completion-length 3072 \
  --report-to wandb
```

The RTX PRO 6000 OOM happened after the first GRPO step while scoring old
logprobs:

```text
chunked_hidden_states_selective_log_softmax
empty_strided_cuda((5376, 262144), ..., torch.float16)
Tried to allocate 2.62 GiB with only about 2.5 GiB free
```

B200's larger VRAM should address this headroom problem without reducing
`num_generations` or `max_completion_length`.
