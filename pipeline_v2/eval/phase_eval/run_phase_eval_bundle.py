#!/usr/bin/env python3
"""Run the phase-end eval bundle: prepare sets, generate, score, report."""

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
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "outputs" / "pipeline_v2" / "phase_eval"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete phase-end style eval bundle.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ai-novel-dir", default="")
    parser.add_argument("--ai-control-dir", default="")
    parser.add_argument("--human-novel-dir", default="")
    parser.add_argument("--generate-prompts", default="")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens-rewrite", type=int, default=4096)
    parser.add_argument("--max-new-tokens-generate", type=int, default=4096)
    parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--rewrite-limit", type=int, default=0)
    parser.add_argument("--generate-limit", type=int, default=0)
    parser.add_argument("--vllm-tokenizer", default="")
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-quantization", default="auto_bnb")
    parser.add_argument("--vllm-load-format", default="auto_bnb")
    parser.add_argument("--vllm-kv-cache-dtype", default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-max-lora-rank", type=int, default=64)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--vllm-attention-backend", default="")
    parser.add_argument("--vllm-attention-config-json", default="")
    parser.add_argument("--vllm-compilation-config", default="")
    parser.add_argument("--vllm-hf-overrides-json", default="")
    parser.add_argument("--vllm-auto-lora-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=260601)
    parser.add_argument("--rewrite-count", type=int, default=30)
    parser.add_argument("--generate-count", type=int, default=30)
    parser.add_argument("--control-count", type=int, default=30)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_dir)
    generation_dir = output_dir / "generations"
    metrics_dir = output_dir / "metrics"
    commands: list[dict[str, Any]] = []
    python = sys.executable

    if not args.skip_prepare:
        run_prepare = [
            python,
            str(PHASE_EVAL_ROOT / "prepare_eval_sets.py"),
            "--output-dir",
            str(eval_dir),
            "--seed",
            str(args.seed),
            "--rewrite-count",
            str(args.rewrite_count),
            "--generate-count",
            str(args.generate_count),
            "--control-count",
            str(args.control_count),
        ]
        if args.ai_novel_dir:
            run_prepare.extend(["--ai-novel-dir", args.ai_novel_dir])
        if args.ai_control_dir:
            run_prepare.extend(["--ai-control-dir", args.ai_control_dir])
        if args.human_novel_dir:
            run_prepare.extend(["--human-novel-dir", args.human_novel_dir])
        if args.generate_prompts:
            run_prepare.extend(["--generate-prompts", args.generate_prompts])
        commands.append(
            run_command(
                run_prepare,
                cwd=TRAINING_ROOT.parent,
                dry_run=args.dry_run,
            )
        )

    if not args.skip_generation:
        generation_script = "run_generation_vllm.py" if args.generation_backend == "vllm" else "run_generation.py"
        command = [
            python,
            str(PHASE_EVAL_ROOT / generation_script),
            "--eval-dir",
            str(eval_dir),
            "--output-dir",
            str(generation_dir),
            "--model",
            args.model,
            "--model-label",
            args.model_label,
            "--chat-template",
            args.chat_template,
            "--max-seq-length",
            str(args.max_seq_length),
            "--max-prompt-length",
            str(args.max_prompt_length),
            "--max-new-tokens-rewrite",
            str(args.max_new_tokens_rewrite),
            "--max-new-tokens-generate",
            str(args.max_new_tokens_generate),
            "--batch-size",
            str(args.generation_batch_size),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--seed",
            str(args.seed),
        ]
        if args.rewrite_limit:
            command.extend(["--rewrite-limit", str(args.rewrite_limit)])
        if args.generate_limit:
            command.extend(["--generate-limit", str(args.generate_limit)])
        if args.generation_backend == "vllm":
            command.extend(
                [
                    "--vllm-tokenizer",
                    args.vllm_tokenizer,
                    "--vllm-dtype",
                    args.vllm_dtype,
                    "--vllm-quantization",
                    args.vllm_quantization,
                    "--vllm-load-format",
                    args.vllm_load_format,
                    "--vllm-gpu-memory-utilization",
                    str(args.vllm_gpu_memory_utilization),
                    "--vllm-tensor-parallel-size",
                    str(args.vllm_tensor_parallel_size),
                    "--vllm-max-lora-rank",
                    str(args.vllm_max_lora_rank),
                    "--vllm-max-num-seqs",
                    str(args.vllm_max_num_seqs),
                    "--vllm-max-num-batched-tokens",
                    str(args.vllm_max_num_batched_tokens),
                ]
            )
            if args.vllm_kv_cache_dtype:
                command.extend(["--vllm-kv-cache-dtype", args.vllm_kv_cache_dtype])
            if args.vllm_attention_backend:
                command.extend(["--vllm-attention-backend", args.vllm_attention_backend])
            if args.vllm_attention_config_json:
                command.extend(["--vllm-attention-config-json", args.vllm_attention_config_json])
            if args.vllm_compilation_config:
                command.extend(["--vllm-compilation-config", args.vllm_compilation_config])
            if args.vllm_hf_overrides_json:
                command.extend(["--vllm-hf-overrides-json", args.vllm_hf_overrides_json])
            command.append(
                "--vllm-auto-lora-identity"
                if args.vllm_auto_lora_identity
                else "--no-vllm-auto-lora-identity"
            )
            command.append(
                "--vllm-enable-prefix-caching"
                if args.vllm_enable_prefix_caching
                else "--no-vllm-enable-prefix-caching"
            )
            command.append("--vllm-enforce-eager" if args.vllm_enforce_eager else "--no-vllm-enforce-eager")
            command.append(
                "--vllm-disable-custom-all-reduce"
                if args.vllm_disable_custom_all_reduce
                else "--no-vllm-disable-custom-all-reduce"
            )
        if args.adapter_path:
            command.extend(["--adapter-path", args.adapter_path])
        command.append("--load-in-4bit" if args.load_in_4bit else "--no-load-in-4bit")
        command.append("--load-in-16bit" if args.load_in_16bit else "--no-load-in-16bit")
        commands.append(run_command(command, cwd=TRAINING_ROOT.parent, dry_run=args.dry_run))

    if not args.skip_scoring:
        commands.append(
            run_command(
                [
                    python,
                    str(PHASE_EVAL_ROOT / "score_and_report.py"),
                    "--eval-dir",
                    str(eval_dir),
                    "--generation-dir",
                    str(generation_dir),
                    "--output-dir",
                    str(metrics_dir),
                ],
                cwd=TRAINING_ROOT.parent,
                dry_run=args.dry_run,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "time": time.time(),
        "args": vars(args),
        "eval_dir": str(eval_dir),
        "generation_dir": str(generation_dir),
        "metrics_dir": str(metrics_dir),
        "commands": commands,
    }
    (output_dir / "phase_eval_bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
