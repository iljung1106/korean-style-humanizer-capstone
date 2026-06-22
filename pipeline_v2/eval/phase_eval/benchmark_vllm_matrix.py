#!/usr/bin/env python3
"""Run small vLLM generation benchmarks for phase eval settings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PHASE_EVAL_ROOT = SCRIPT.parent
TRAINING_ROOT = SCRIPT.parents[3]
DEFAULT_EVAL_DIR = TRAINING_ROOT / "data" / "phase_eval_v2"
DEFAULT_OUTPUT_ROOT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "phase_eval_bench"
DEFAULT_BNB_MODEL = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
DEFAULT_NVFP4_MODEL = "nvidia/Gemma-4-31B-IT-NVFP4"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> dict[str, Any]:
    print("[command] " + " ".join(command), flush=True)
    if dry_run:
        return {"command": command, "returncode": 0, "dry_run": True}
    start = time.time()
    result = subprocess.run(command, cwd=str(cwd), check=False)
    elapsed = time.time() - start
    record = {"command": command, "returncode": result.returncode, "elapsed_sec": round(elapsed, 3)}
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return record


def variant_command(args: argparse.Namespace, variant: dict[str, Any]) -> list[str]:
    output_dir = Path(args.output_root) / variant["name"]
    command = [
        sys.executable,
        str(PHASE_EVAL_ROOT / "run_phase_eval_bundle.py"),
        "--eval-dir",
        args.eval_dir,
        "--output-dir",
        str(output_dir),
        "--model",
        variant["model"],
        "--model-label",
        variant["name"],
        "--chat-template",
        args.chat_template,
        "--max-seq-length",
        str(args.max_seq_length),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-new-tokens-rewrite",
        str(args.max_new_tokens),
        "--max-new-tokens-generate",
        str(args.max_new_tokens),
        "--generation-backend",
        "vllm",
        "--generation-batch-size",
        str(variant["batch_size"]),
        "--rewrite-limit",
        str(args.rewrite_limit),
        "--generate-limit",
        str(args.generate_limit),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-num-seqs",
        str(variant["batch_size"]),
        "--vllm-max-lora-rank",
        str(args.vllm_max_lora_rank),
        "--vllm-quantization",
        variant["quantization"],
        "--vllm-load-format",
        variant["load_format"],
        "--seed",
        str(args.seed),
        "--skip-prepare",
        "--load-in-4bit",
    ]
    if args.adapter_path:
        command.extend(["--adapter-path", args.adapter_path])
    if args.vllm_kv_cache_dtype:
        command.extend(["--vllm-kv-cache-dtype", args.vllm_kv_cache_dtype])
    if args.vllm_max_num_batched_tokens:
        command.extend(["--vllm-max-num-batched-tokens", str(args.vllm_max_num_batched_tokens)])
    if args.vllm_attention_backend:
        command.extend(["--vllm-attention-backend", args.vllm_attention_backend])
    if args.vllm_compilation_config:
        command.extend(["--vllm-compilation-config", args.vllm_compilation_config])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM phase eval generation variants.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--bnb-model", default=DEFAULT_BNB_MODEL)
    parser.add_argument("--nvfp4-model", default=DEFAULT_NVFP4_MODEL)
    parser.add_argument("--variants", default="bnb_b4,bnb_b8,nvfp4_b4,nvfp4_b8")
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--max-prompt-length", type=positive_int, default=4096)
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--rewrite-limit", type=positive_int, default=4)
    parser.add_argument("--generate-limit", type=non_negative_int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--vllm-kv-cache-dtype", default="fp8")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-lora-rank", type=positive_int, default=64)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--vllm-attention-backend", default="")
    parser.add_argument("--vllm-compilation-config", default="")
    parser.add_argument("--seed", type=int, default=260601)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = {
        "bnb_b4": {
            "name": "bnb_b4",
            "model": args.bnb_model,
            "batch_size": 4,
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
        },
        "bnb_b8": {
            "name": "bnb_b8",
            "model": args.bnb_model,
            "batch_size": 8,
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
        },
        "nvfp4_b4": {
            "name": "nvfp4_b4",
            "model": args.nvfp4_model,
            "batch_size": 4,
            "quantization": "modelopt",
            "load_format": "auto",
        },
        "nvfp4_b8": {
            "name": "nvfp4_b8",
            "model": args.nvfp4_model,
            "batch_size": 8,
            "quantization": "modelopt",
            "load_format": "auto",
        },
    }
    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(variants))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(variants)}")

    records = []
    for name in selected:
        records.append(
            run_command(
                variant_command(args, variants[name]),
                cwd=TRAINING_ROOT.parent,
                dry_run=args.dry_run,
            )
        )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"time": time.time(), "args": vars(args), "commands": records}
    (output_root / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
