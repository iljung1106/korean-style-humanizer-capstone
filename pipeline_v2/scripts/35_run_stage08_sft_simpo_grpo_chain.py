#!/usr/bin/env python3
"""Run Stage 8: SFT anchor -> SimPO-CPO -> generate/rewrite/mixed GRPO.

This launcher does not upload. It is meant for the training instance: prepare
the generate-heavy preference dataset, build the Stage 7I-derived metric weight
preset, then run each stage sequentially with clear logs and skip markers.

GRPO stages are intentionally separated by merges:
Stage08B LoRA -> merged base -> fresh GRPO-generate LoRA -> merged base ->
fresh GRPO-rewrite LoRA -> merged base -> fresh mixed LoRA. This avoids stacking
multiple PEFT adapters across GRPO phases and keeps the reference/base state
clear. Intermediate merged bases are deleted after the next phase succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
DEFAULT_OUTPUT_ROOT = TRAINING_ROOT / "outputs" / "pipeline_v2"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def policy_exists(output: Path) -> bool:
    return (output / "policy" / "adapter_model.safetensors").exists()


def merged_model_exists(output: Path) -> bool:
    if (output / "merge_manifest.json").exists():
        return True
    if (output / "model.safetensors.index.json").exists() or (output / "pytorch_model.bin.index.json").exists():
        return True
    return any(output.glob("model-*.safetensors")) or (output / "model.safetensors").exists()


def remove_tree(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"[dry-run/remove] {path}", flush=True)
        return
    print(f"[remove] {path}", flush=True)
    shutil.rmtree(path, ignore_errors=True)


def cleanup_checkpoints(output: Path, *, keep_latest: int, dry_run: bool) -> None:
    if keep_latest < 0 or not output.exists():
        return
    checkpoints = sorted(
        [path for path in output.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for checkpoint in checkpoints[keep_latest:]:
        remove_tree(checkpoint, dry_run=dry_run)


def run_command(name: str, command: list[str], *, cwd: Path, env: dict[str, str], output: Path, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run]", " ".join(command), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{name}.command.json", {"name": name, "command": command, "time": time.time()})
    if name.startswith("stage") and policy_exists(output):
        print(f"[skip] {name}: policy already exists at {output / 'policy'}", flush=True)
        return
    if name.startswith("merge") and merged_model_exists(output):
        print(f"[skip] {name}: merged model already exists at {output}", flush=True)
        return
    log_path = output / f"{name}.log"
    print(f"[run] {name}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[run/start] {name} time={time.time()}\n")
        log.write("[command] " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        log.write(f"[run/end] {name} returncode={returncode} time={time.time()}\n")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    write_json(output / f"{name}.done.json", {"name": name, "returncode": returncode, "time": time.time()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 8 SFT/SimPO/GRPO chain.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--stage07i-adapter",
        default="",
        help="Deprecated: Stage 8 now starts from a Stage07I merged model by default.",
    )
    parser.add_argument("--model", default="auto")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="gemma4-webnovel-style-31b-new-reward")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--merged-root", default=str(DEFAULT_OUTPUT_ROOT / "stage08_merged"))
    parser.add_argument("--merge-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--cleanup-intermediate-merged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-checkpoints", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-at", choices=["prep", "sft", "simpo", "grpo_generate", "grpo_rewrite", "grpo_mixed"], default="prep")
    parser.add_argument("--skip-prep", action="store_true")
    parser.add_argument("--sft-epochs", default="1.15")
    parser.add_argument("--simpo-epochs", default="2.0")
    parser.add_argument("--grpo-steps", default="24")
    parser.add_argument("--grpo-num-generations", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    if args.stage07i_adapter:
        env["STAGE07I_ADAPTER_PATH"] = args.stage07i_adapter
    env.setdefault("WANDB_PROJECT", args.wandb_project)
    env.setdefault("WANDB_MODE", args.wandb_mode)
    cwd = TRAINING_ROOT
    py = args.python
    merged_root = Path(args.merged_root)
    merged_after_simpo = merged_root / "stage08b_simpo_cpo_merged_for_grpo_generate"
    merged_after_generate = merged_root / "stage08c_generate_merged_for_grpo_rewrite"
    merged_after_rewrite = merged_root / "stage08d_rewrite_merged_for_grpo_mixed"
    merged_final = merged_root / "stage08e_mixed_final_merged"

    prep_commands = [
        (
            "stage08_preference_mix",
            [
                py,
                "pipeline_v2/scripts/33_build_stage08_preference_mix.py",
            ],
            DEFAULT_OUTPUT_ROOT / "stage08_prep",
        ),
        (
            "stage08_dynamic_style_weights",
            [
                py,
                "pipeline_v2/scripts/34_build_stage08_dynamic_style_weights.py",
            ],
            DEFAULT_OUTPUT_ROOT / "stage08_prep",
        ),
    ]
    stage_commands = [
        (
            "stage08a_sft_anchor",
            [
                py,
                "-m",
                "pipeline_v2.train.stage08a_sft_anchor_from_stage07i",
                "--model",
                args.model,
                "--adapter-path",
                "",
                "--num-train-epochs",
                args.sft_epochs,
                "--report-to",
                args.report_to,
            ],
            DEFAULT_OUTPUT_ROOT / "stage08a_sft_anchor_from_stage07i",
        ),
        (
            "stage08b_simpo_cpo",
            [
                py,
                "-m",
                "pipeline_v2.train.stage08b_simpo_cpo_from_stage08a",
                "--model",
                args.model,
                "--num-train-epochs",
                args.simpo_epochs,
                "--report-to",
                args.report_to,
            ],
            DEFAULT_OUTPUT_ROOT / "stage08b_simpo_cpo_from_stage08a",
        ),
        (
            "stage08c_grpo_generate",
            [
                py,
                "-m",
                "pipeline_v2.train.stage08c_grpo_generate_heavy_from_stage08b",
                "--model",
                str(merged_after_simpo),
                "--adapter-path",
                "",
                "--init-lora",
                "--max-steps",
                args.grpo_steps,
                "--report-to",
                args.report_to,
            ],
            DEFAULT_OUTPUT_ROOT / "stage08c_grpo_generate_from_stage08b",
        ),
        (
            "stage08d_grpo_rewrite",
            [
                py,
                "-m",
                "pipeline_v2.train.stage08d_grpo_rewrite_from_stage08c",
                "--model",
                str(merged_after_generate),
                "--adapter-path",
                "",
                "--init-lora",
                "--max-steps",
                args.grpo_steps,
                "--report-to",
                args.report_to,
            ],
            DEFAULT_OUTPUT_ROOT / "stage08d_grpo_rewrite_from_stage08c",
        ),
        (
            "stage08e_grpo_mixed",
            [
                py,
                "-m",
                "pipeline_v2.train.stage08e_grpo_mixed_from_stage08d",
                "--model",
                str(merged_after_rewrite),
                "--adapter-path",
                "",
                "--init-lora",
                "--max-steps",
                args.grpo_steps,
                "--report-to",
                args.report_to,
            ],
            DEFAULT_OUTPUT_ROOT / "stage08e_grpo_mixed_from_stage08d",
        ),
    ]
    merge_commands = {
        "merge_after_simpo": (
            "merge_stage08b_for_grpo_generate",
            [
                py,
                "pipeline_v2/eval/phase_eval/merge_lora_for_vllm.py",
                "--adapter-path",
                str(DEFAULT_OUTPUT_ROOT / "stage08b_simpo_cpo_from_stage08a" / "policy"),
                "--output-dir",
                str(merged_after_simpo),
                "--base-model",
                args.model,
                "--dtype",
                args.merge_dtype,
            ],
            merged_after_simpo,
        ),
        "merge_after_generate": (
            "merge_stage08c_for_grpo_rewrite",
            [
                py,
                "pipeline_v2/eval/phase_eval/merge_lora_for_vllm.py",
                "--adapter-path",
                str(DEFAULT_OUTPUT_ROOT / "stage08c_grpo_generate_from_stage08b" / "policy"),
                "--output-dir",
                str(merged_after_generate),
                "--base-model",
                str(merged_after_simpo),
                "--dtype",
                args.merge_dtype,
            ],
            merged_after_generate,
        ),
        "merge_after_rewrite": (
            "merge_stage08d_for_grpo_mixed",
            [
                py,
                "pipeline_v2/eval/phase_eval/merge_lora_for_vllm.py",
                "--adapter-path",
                str(DEFAULT_OUTPUT_ROOT / "stage08d_grpo_rewrite_from_stage08c" / "policy"),
                "--output-dir",
                str(merged_after_rewrite),
                "--base-model",
                str(merged_after_generate),
                "--dtype",
                args.merge_dtype,
            ],
            merged_after_rewrite,
        ),
        "merge_final": (
            "merge_stage08e_final",
            [
                py,
                "pipeline_v2/eval/phase_eval/merge_lora_for_vllm.py",
                "--adapter-path",
                str(DEFAULT_OUTPUT_ROOT / "stage08e_grpo_mixed_from_stage08d" / "policy"),
                "--output-dir",
                str(merged_final),
                "--base-model",
                str(merged_after_rewrite),
                "--dtype",
                args.merge_dtype,
            ],
            merged_final,
        ),
    }
    if args.grpo_num_generations:
        for _name, command, _output in stage_commands:
            if "grpo" in _name:
                command.extend(["--num-generations", args.grpo_num_generations])

    order = ["prep", "sft", "simpo", "grpo_generate", "grpo_rewrite", "grpo_mixed"]
    active = set(order[order.index(args.start_at) :])
    if "prep" in active and not args.skip_prep:
        for name, command, output in prep_commands:
            run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
    if "sft" in active:
        name, command, output = stage_commands[0]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
        cleanup_checkpoints(output, keep_latest=args.keep_checkpoints, dry_run=args.dry_run)
    if "simpo" in active:
        name, command, output = stage_commands[1]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
        cleanup_checkpoints(output, keep_latest=args.keep_checkpoints, dry_run=args.dry_run)
        name, command, output = merge_commands["merge_after_simpo"]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
    if "grpo_generate" in active:
        name, command, output = stage_commands[2]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
        cleanup_checkpoints(output, keep_latest=args.keep_checkpoints, dry_run=args.dry_run)
        name, command, output = merge_commands["merge_after_generate"]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
        if args.cleanup_intermediate_merged:
            remove_tree(merged_after_simpo, dry_run=args.dry_run)
    if "grpo_rewrite" in active:
        name, command, output = stage_commands[3]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
        cleanup_checkpoints(output, keep_latest=args.keep_checkpoints, dry_run=args.dry_run)
        name, command, output = merge_commands["merge_after_rewrite"]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
        if args.cleanup_intermediate_merged:
            remove_tree(merged_after_generate, dry_run=args.dry_run)
    if "grpo_mixed" in active:
        name, command, output = stage_commands[4]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
        cleanup_checkpoints(output, keep_latest=args.keep_checkpoints, dry_run=args.dry_run)
        name, command, output = merge_commands["merge_final"]
        run_command(name, command, cwd=cwd, env=env, output=output, dry_run=args.dry_run)
        if args.cleanup_intermediate_merged:
            remove_tree(merged_after_rewrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
