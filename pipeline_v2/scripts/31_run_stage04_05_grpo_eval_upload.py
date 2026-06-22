#!/usr/bin/env python3
"""Run Stage 4/5 GRPO from an existing Stage 3 adapter, then eval and upload.

This runner is intentionally scoped to the post-Stage-3 path:

1. Stage 4 generate-only GRPO.
2. Stage 5 rewrite-only GRPO.
3. Merge final LoRA for vLLM evaluation.
4. Run phase eval and write plots/reports/samples.
5. Upload the final LoRA adapter to ModelScope.

It does not rerun Stage 1-3.
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
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
PROJECT_ROOT = TRAINING_ROOT.parent

DEFAULT_MODEL = "/workspace/modelscope_cache/unsloth/gemma-4-31B-it"
DEFAULT_GRPO_DATASET = TRAINING_ROOT / "data" / "processed" / "grpo_mixed_prompts.jsonl"
DEFAULT_OUTPUT_ROOT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage04_05_grpo_from_stage03"
DEFAULT_WANDB_PROJECT = "gemma4-webnovel-style-31b-pipeline-v2"


def print_json(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + " " + json.dumps(payload, ensure_ascii=False), flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def adapter_ready(path: Path) -> bool:
    return (
        path.exists()
        and (path / "adapter_config.json").exists()
        and ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists())
    )


def stage_done(path: Path) -> bool:
    return adapter_ready(path / "policy")


def run_command(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print_json("[run]", {"name": name, "command": command, "log": str(log_path)})
    if dry_run:
        return {"name": name, "command": command, "returncode": 0, "dry_run": True, "log": str(log_path)}

    start = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[start] " + json.dumps({"time": start, "name": name, "command": command}, ensure_ascii=False) + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        returncode = proc.wait()
        elapsed = time.time() - start
        log.write("\n[end] " + json.dumps({"returncode": returncode, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n")

    result = {
        "name": name,
        "command": command,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 3),
        "log": str(log_path),
    }
    print_json("[result]", result)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return result


def cleanup_checkpoints(run_root: Path, *, keep_latest: int) -> list[str]:
    removed: list[str] = []
    if keep_latest < 0 or not run_root.exists():
        return removed
    for stage_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        checkpoints = sorted(
            [path for path in stage_dir.glob("checkpoint-*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for checkpoint in checkpoints[keep_latest:]:
            shutil.rmtree(checkpoint, ignore_errors=True)
            removed.append(str(checkpoint))
    return removed


def write_upload_readme(
    adapter_path: Path,
    *,
    model_id: str,
    base_model: str,
    stage: str,
    run_root: Path,
    eval_dir: Path,
) -> None:
    readme = [
        "---",
        f"base_model: {base_model}",
        "library_name: peft",
        "tags:",
        "- gemma4",
        "- lora",
        "- qlora",
        "- korean-webnovel",
        "- pipeline-v2",
        "- grpo",
        "---",
        "",
        f"# {model_id}",
        "",
        "Gemma 4 31B Korean webnovel style LoRA adapter.",
        "",
        "## Training",
        "",
        "- Base recipe: pipeline_v2 Stage 1-3.",
        "- Stage 4: generate-only GRPO, 30 steps by default.",
        "- Stage 5: rewrite-only GRPO, 50 steps by default.",
        "",
        "## Local Run",
        "",
        f"- stage: `{stage}`",
        f"- run root: `{run_root}`",
        f"- eval dir: `{eval_dir}`",
    ]
    (adapter_path / "README.md").write_text("\n".join(readme), encoding="utf-8")


def upload_to_modelscope(
    *,
    adapter_path: Path,
    model_id: str,
    token: str,
    endpoint: str,
    base_model: str,
    stage: str,
    run_root: Path,
    eval_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    if not dry_run and not adapter_ready(adapter_path):
        raise FileNotFoundError(f"Adapter is not ready: {adapter_path}")
    if dry_run:
        return {"dry_run": True, "model_id": model_id, "adapter_path": str(adapter_path)}
    write_upload_readme(
        adapter_path,
        model_id=model_id,
        base_model=base_model,
        stage=stage,
        run_root=run_root,
        eval_dir=eval_dir,
    )

    from modelscope.hub.api import HubApi

    api = HubApi(endpoint=endpoint)
    token_arg = token or None
    if token:
        api.login(access_token=token, endpoint=endpoint)
    created = False
    try:
        api.create_model(
            model_id=model_id,
            visibility=5,
            license="Apache License 2.0",
            token=token_arg,
            endpoint=endpoint,
        )
        created = True
    except Exception as exc:
        print_json("[modelscope/create_skip]", {"model_id": model_id, "error": repr(exc)})

    upload_kwargs = {
        "repo_id": model_id,
        "folder_path": str(adapter_path),
        "path_in_repo": "",
        "repo_type": "model",
        "token": token_arg,
        "commit_message": f"Upload {stage} LoRA adapter",
        "ignore_patterns": [
            "checkpoint-*",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "*.tmp",
        ],
        "max_workers": 8,
    }
    try:
        result = api.upload_folder(**upload_kwargs)
    except TypeError:
        upload_kwargs.pop("max_workers", None)
        result = api.upload_folder(**upload_kwargs)
    return {"model_id": model_id, "created": created, "result": str(result)}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4/5 GRPO, eval, and ModelScope upload.")
    parser.add_argument("--stage3-adapter", required=True, help="Stage 3 policy adapter path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=str(DEFAULT_GRPO_DATASET))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--max-prompt-length", type=positive_int, default=4096)
    parser.add_argument("--max-prompt-chars", type=positive_int, default=6000)
    parser.add_argument("--max-completion-length", type=positive_int, default=3072)
    parser.add_argument("--generate-grpo-steps", type=positive_int, default=30)
    parser.add_argument("--rewrite-grpo-steps", type=positive_int, default=50)
    parser.add_argument("--grpo-batch-size", type=positive_int, default=4)
    parser.add_argument("--grpo-num-generations", type=positive_int, default=4)
    parser.add_argument("--grpo-grad-accum", type=positive_int, default=1)
    parser.add_argument("--learning-rate", type=float, default=8e-7)
    parser.add_argument("--warmup-steps", type=non_negative_int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--loss-type", default="dr_grpo")
    parser.add_argument("--num-iterations", type=positive_int, default=2)
    parser.add_argument("--scale-rewards", default="group")
    parser.add_argument("--generate-temperature", type=float, default=0.95)
    parser.add_argument("--rewrite-temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--generate-min-output-chars", type=positive_int, default=3000)
    parser.add_argument("--generate-max-output-chars", type=positive_int, default=4500)
    parser.add_argument("--short-output-penalty", type=float, default=0.20)
    parser.add_argument("--long-output-penalty", type=float, default=0.10)
    parser.add_argument("--collapse-fail-reward", type=float, default=-1.0)
    parser.add_argument("--group-diversity-bonus-max", type=float, default=0.03)
    parser.add_argument("--group-diversity-mode", default="leave_one_out")
    parser.add_argument("--rewrite-edit-weight", type=float, default=0.35)
    parser.add_argument("--rewrite-improvement-weight", type=float, default=0.30)
    parser.add_argument("--save-steps", type=positive_int, default=10)
    parser.add_argument("--save-total-limit", type=non_negative_int, default=2)
    parser.add_argument("--sample-log-text-chars", type=positive_int, default=4000)
    parser.add_argument("--seed", type=int, default=4710)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--run-name-prefix", default="pipeline_v2_final")
    parser.add_argument("--skip-stage04", action="store_true")
    parser.add_argument("--skip-stage05", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-checkpoints-keep-latest", type=int, default=1)

    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-eval-prepare", action="store_true")
    parser.add_argument("--eval-dir", default="")
    parser.add_argument("--eval-count", type=positive_int, default=30)
    parser.add_argument("--eval-output-dir", default="")
    parser.add_argument("--merged-model-dir", default="")
    parser.add_argument("--merge-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--eval-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--eval-generation-batch-size", type=positive_int, default=8)
    parser.add_argument("--eval-max-new-tokens", type=positive_int, default=3072)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-quantization", default="")
    parser.add_argument("--vllm-load-format", default="auto")
    parser.add_argument("--vllm-kv-cache-dtype", default="fp8")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--vllm-max-num-seqs", type=positive_int, default=8)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)

    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--upload-stage04", action="store_true")
    parser.add_argument("--modelscope-stage04-model-id", default="")
    parser.add_argument("--modelscope-final-model-id", default=os.environ.get("MODELSCOPE_MODEL_ID", ""))
    parser.add_argument("--modelscope-endpoint", default=os.environ.get("MODELSCOPE_ENDPOINT", "https://modelscope.ai"))
    parser.add_argument("--modelscope-token-env", default="MODELSCOPE_API_TOKEN")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def grpo_common_args(args: argparse.Namespace, output: Path, adapter: Path, *, task: str, steps: int, temperature: float) -> list[str]:
    command = [
        args.python,
        "-m",
        f"pipeline_v2.train.stage{'04_grpo_generate' if task == 'generate' else '05_grpo_rewrite'}",
        "--dataset",
        args.dataset,
        "--output",
        str(output),
        "--model",
        args.model,
        "--adapter-path",
        str(adapter),
        "--max-seq-length",
        str(args.max_seq_length),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-prompt-chars",
        str(args.max_prompt_chars),
        "--max-completion-length",
        str(args.max_completion_length),
        "--generate-min-output-chars",
        str(args.generate_min_output_chars),
        "--generate-max-output-chars",
        str(args.generate_max_output_chars),
        "--short-output-penalty",
        str(args.short_output_penalty),
        "--long-output-penalty",
        str(args.long_output_penalty),
        "--load-in-4bit",
        "--batch-size",
        str(args.grpo_batch_size),
        "--num-generations",
        str(args.grpo_num_generations),
        "--grad-accum",
        str(args.grpo_grad_accum),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-steps",
        str(args.warmup_steps),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--beta",
        str(args.beta),
        "--loss-type",
        args.loss_type,
        "--num-iterations",
        str(args.num_iterations),
        "--scale-rewards",
        args.scale_rewards,
        "--temperature",
        str(temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--no-repeat-ngram-size",
        "0",
        "--max-steps",
        str(steps),
        "--logging-steps",
        "1",
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--sample-log-every",
        "1",
        "--sample-log-max-items",
        "4",
        "--sample-log-text-chars",
        str(args.sample_log_text_chars),
        "--collapse-fail-reward",
        str(args.collapse_fail_reward),
        "--unsloth-grpo-mini-batch",
        "1",
        "--unsloth-logit-chunk-multiplier",
        "8",
        "--require-result-tags",
        "--stop-at-result-close",
        "--seed",
        str(args.seed + (3 if task == "generate" else 4)),
        "--report-to",
        args.report_to,
        "--run-name",
        f"{args.run_name_prefix}_stage{'04' if task == 'generate' else '05'}_grpo_{task}_{steps}",
    ]
    if task == "generate":
        command.extend(
            [
                "--group-diversity-bonus-max",
                str(args.group_diversity_bonus_max),
                "--group-diversity-mode",
                args.group_diversity_mode,
            ]
        )
    else:
        command.extend(
            [
                "--rewrite-edit-weight",
                str(args.rewrite_edit_weight),
                "--rewrite-improvement-weight",
                str(args.rewrite_improvement_weight),
                "--group-diversity-bonus-max",
                "0.0",
                "--group-diversity-mode",
                "none",
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    stage04 = root / "stage04_grpo_generate"
    stage05 = root / "stage05_grpo_rewrite"
    logs = root / "logs"
    merged_dir = Path(args.merged_model_dir) if args.merged_model_dir else root / "merged_for_vllm" / "stage05_final"
    eval_dir = Path(args.eval_output_dir) if args.eval_output_dir else root / "phase_eval_stage05_final"
    stage3_adapter = Path(args.stage3_adapter)

    if not args.dry_run and not adapter_ready(stage3_adapter):
        raise FileNotFoundError(f"Stage 3 adapter is not ready: {stage3_adapter}")

    root.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(TRAINING_ROOT),
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": args.wandb_mode,
        "UNSLOTH_RETURN_LOGITS": "1",
        "UNSLOTH_DISABLE_STATISTICS": "1",
        "UNSLOTH_USE_MODELSCOPE": "1",
        "TORCHDYNAMO_DISABLE": "1",
    }
    commands: list[dict[str, Any]] = []

    write_json(
        root / "stage04_05_recipe_manifest.json",
        {
            "time": time.time(),
            "stage3_adapter": str(stage3_adapter),
            "output_root": str(root),
            "model": args.model,
            "dataset": args.dataset,
            "generate_grpo_steps": args.generate_grpo_steps,
            "rewrite_grpo_steps": args.rewrite_grpo_steps,
            "args": {key: value for key, value in vars(args).items() if "token" not in key.lower()},
        },
    )

    if not args.skip_stage04:
        if args.skip_existing and stage_done(stage04):
            print_json("[skip]", {"stage": "stage04_grpo_generate", "output": str(stage04)})
        else:
            commands.append(
                run_command(
                    "stage04_grpo_generate",
                    grpo_common_args(
                        args,
                        stage04,
                        stage3_adapter,
                        task="generate",
                        steps=args.generate_grpo_steps,
                        temperature=args.generate_temperature,
                    ),
                    cwd=TRAINING_ROOT,
                    env=env,
                    log_path=logs / "stage04_grpo_generate.log",
                    dry_run=args.dry_run,
                )
            )
    if not args.dry_run and not stage_done(stage04) and not args.skip_stage04:
        raise FileNotFoundError(f"Stage 4 did not produce a policy adapter: {stage04 / 'policy'}")

    stage05_input = stage04 / "policy"
    if args.skip_stage04:
        stage05_input = stage3_adapter

    if not args.skip_stage05:
        if args.skip_existing and stage_done(stage05):
            print_json("[skip]", {"stage": "stage05_grpo_rewrite", "output": str(stage05)})
        else:
            commands.append(
                run_command(
                    "stage05_grpo_rewrite",
                    grpo_common_args(
                        args,
                        stage05,
                        stage05_input,
                        task="rewrite",
                        steps=args.rewrite_grpo_steps,
                        temperature=args.rewrite_temperature,
                    ),
                    cwd=TRAINING_ROOT,
                    env=env,
                    log_path=logs / "stage05_grpo_rewrite.log",
                    dry_run=args.dry_run,
                )
            )
    final_adapter = stage05 / "policy" if not args.skip_stage05 else stage05_input
    if not adapter_ready(final_adapter) and not args.dry_run:
        raise FileNotFoundError(f"Final adapter is not ready: {final_adapter}")

    removed = cleanup_checkpoints(root, keep_latest=args.cleanup_checkpoints_keep_latest)

    if not args.skip_eval:
        eval_model = args.model
        eval_adapter = str(final_adapter)
        if args.eval_backend == "vllm" and not args.skip_merge:
            commands.append(
                run_command(
                    "merge_stage05_for_vllm",
                    [
                        args.python,
                        str(TRAINING_ROOT / "pipeline_v2/eval/phase_eval/merge_lora_for_vllm.py"),
                        "--adapter-path",
                        str(final_adapter),
                        "--output-dir",
                        str(merged_dir),
                        "--base-model",
                        args.model,
                        "--dtype",
                        args.merge_dtype,
                        "--device-map",
                        "auto",
                    ],
                    cwd=TRAINING_ROOT,
                    env=env,
                    log_path=logs / "merge_stage05_for_vllm.log",
                    dry_run=args.dry_run,
                )
            )
            eval_model = str(merged_dir)
            eval_adapter = ""

        eval_command = [
            args.python,
            str(TRAINING_ROOT / "pipeline_v2/eval/phase_eval/run_phase_eval_bundle.py"),
            "--output-dir",
            str(eval_dir),
            "--model",
            eval_model,
            "--model-label",
            "stage05_grpo_generate_rewrite",
            "--generation-backend",
            args.eval_backend,
            "--generation-batch-size",
            str(args.eval_generation_batch_size),
            "--rewrite-count",
            str(args.eval_count),
            "--generate-count",
            str(args.eval_count),
            "--control-count",
            str(args.eval_count),
            "--max-new-tokens-rewrite",
            str(args.eval_max_new_tokens),
            "--max-new-tokens-generate",
            str(args.eval_max_new_tokens),
            "--temperature",
            "0.7",
            "--top-p",
            "0.95",
            "--top-k",
            "50",
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--seed",
            str(args.seed + 10),
        ]
        if args.eval_dir:
            eval_command.extend(["--eval-dir", args.eval_dir])
        if args.skip_eval_prepare:
            eval_command.append("--skip-prepare")
        if eval_adapter:
            eval_command.extend(["--adapter-path", eval_adapter, "--load-in-4bit"])
        else:
            eval_command.extend(["--no-load-in-4bit", "--vllm-dtype", args.vllm_dtype])
            if args.vllm_quantization:
                eval_command.extend(["--vllm-quantization", args.vllm_quantization])
            else:
                eval_command.extend(["--vllm-quantization", ""])
            eval_command.extend(
                [
                    "--vllm-load-format",
                    args.vllm_load_format,
                    "--vllm-kv-cache-dtype",
                    args.vllm_kv_cache_dtype,
                    "--vllm-gpu-memory-utilization",
                    str(args.vllm_gpu_memory_utilization),
                    "--vllm-max-num-seqs",
                    str(args.vllm_max_num_seqs),
                    "--vllm-max-num-batched-tokens",
                    str(args.vllm_max_num_batched_tokens),
                    "--no-vllm-enable-prefix-caching",
                    "--no-vllm-enforce-eager",
                ]
            )
        commands.append(
            run_command(
                "phase_eval_stage05_final",
                eval_command,
                cwd=TRAINING_ROOT,
                env=env,
                log_path=logs / "phase_eval_stage05_final.log",
                dry_run=args.dry_run,
            )
        )

    uploads: list[dict[str, Any]] = []
    if not args.skip_upload:
        token = env.get(args.modelscope_token_env, "")
        if args.upload_stage04:
            if not args.modelscope_stage04_model_id:
                raise ValueError("--modelscope-stage04-model-id is required with --upload-stage04")
            uploads.append(
                upload_to_modelscope(
                    adapter_path=stage04 / "policy",
                    model_id=args.modelscope_stage04_model_id,
                    token=token,
                    endpoint=args.modelscope_endpoint,
                    base_model=args.model,
                    stage="stage04_grpo_generate",
                    run_root=root,
                    eval_dir=eval_dir,
                    dry_run=args.dry_run,
                )
            )
        if not args.modelscope_final_model_id:
            raise ValueError("--modelscope-final-model-id or MODELSCOPE_MODEL_ID is required unless --skip-upload")
        uploads.append(
            upload_to_modelscope(
                adapter_path=final_adapter,
                model_id=args.modelscope_final_model_id,
                token=token,
                endpoint=args.modelscope_endpoint,
                base_model=args.model,
                stage="stage05_grpo_rewrite",
                run_root=root,
                eval_dir=eval_dir,
                dry_run=args.dry_run,
            )
        )

    manifest = {
        "time": time.time(),
        "output_root": str(root),
        "stage04_adapter": str(stage04 / "policy"),
        "stage05_adapter": str(stage05 / "policy"),
        "final_adapter": str(final_adapter),
        "merged_dir": str(merged_dir),
        "eval_dir": str(eval_dir),
        "commands": commands,
        "cleanup_removed": removed,
        "uploads": uploads,
        "args": {key: value for key, value in vars(args).items() if "token" not in key.lower()},
    }
    write_json(root / "stage04_05_done.json", manifest)
    print_json("[done]", manifest)


if __name__ == "__main__":
    main()
