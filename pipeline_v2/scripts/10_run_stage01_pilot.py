#!/usr/bin/env python3
"""Run the paid-instance Stage 1 pilot bundle.

The bundle is intentionally large enough to justify a full instance setup:

1. real-tokenizer mask preflight
2. Stage 1 CPT-lite/instruct-mix training
3. saved-adapter reload probe
4. fixed eval generation for baseline and trained adapter
5. standalone scoring for both eval outputs

Use `--dry-run` to print the exact commands without running them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parents[1]
TRAINING_ROOT = PIPELINE_ROOT.parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gemma4_loader import DEFAULT_BASE_MODEL
from pipeline_v2.lib.io import write_json


DEFAULT_DATASET = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_mixed_probe.jsonl"
DEFAULT_OUTPUT_ROOT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage01_pilot"


def resolve_user_path(value: str | Path, *, for_output: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == TRAINING_ROOT.name:
        return TRAINING_ROOT.parent / path
    candidate = TRAINING_ROOT / path
    if for_output or candidate.exists():
        return candidate
    repo_candidate = TRAINING_ROOT.parent / path
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def run_command(command: list[str], *, cwd: Path, dry_run: bool, log: list[dict[str, Any]]) -> None:
    record = {"command": command, "cwd": str(cwd), "start": time.time(), "dry_run": dry_run}
    print("[pilot/command] " + json.dumps(record, ensure_ascii=False), flush=True)
    if dry_run:
        record["returncode"] = 0
        record["elapsed_sec"] = 0.0
        log.append(record)
        return
    result = subprocess.run(command, cwd=str(cwd), check=False)
    record["returncode"] = int(result.returncode)
    record["elapsed_sec"] = round(time.time() - float(record["start"]), 3)
    log.append(record)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def add_bool_flag(command: list[str], name: str, value: bool) -> None:
    command.append(name if value else "--no-" + name.removeprefix("--"))


def stage01_command(args: argparse.Namespace, output: Path, *, mask_probe_only: bool) -> list[str]:
    command = [
        args.python,
        str(PIPELINE_ROOT / "train" / "stage01_cpt_lite.py"),
        "--dataset",
        str(args.dataset),
        "--output",
        str(output),
        "--model",
        args.model,
        "--chat-template",
        args.chat_template,
        "--max-seq-length",
        str(args.max_seq_length),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--lora-last-layer-fraction",
        str(args.lora_last_layer_fraction),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--max-steps",
        str(args.max_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--optim",
        args.optim,
        "--anti-slop-ul-weight",
        str(args.anti_slop_ul_weight),
        "--anti-slop-ul-lexicon",
        str(args.anti_slop_ul_lexicon),
        "--anti-slop-ul-unigram-top-k",
        str(args.anti_slop_ul_unigram_top_k),
        "--anti-slop-ul-unigram-min-lift",
        str(args.anti_slop_ul_unigram_min_lift),
        "--anti-slop-ul-bigram-top-k",
        str(args.anti_slop_ul_bigram_top_k),
        "--anti-slop-ul-bigram-min-lift",
        str(args.anti_slop_ul_bigram_min_lift),
        "--anti-slop-ul-bigram-min-weight",
        str(args.anti_slop_ul_bigram_min_weight),
        "--anti-slop-ul-trigram-top-k",
        str(args.anti_slop_ul_trigram_top_k),
        "--anti-slop-ul-trigram-min-lift",
        str(args.anti_slop_ul_trigram_min_lift),
        "--anti-slop-ul-trigram-min-weight",
        str(args.anti_slop_ul_trigram_min_weight),
        "--anti-slop-ul-start-weight-multiplier",
        str(args.anti_slop_ul_start_weight_multiplier),
        "--seed",
        str(args.seed),
        "--report-to",
        args.report_to,
        "--run-name",
        args.run_name,
        "--row-type-sampling",
        args.row_type_sampling,
        "--row-type-balance",
        args.row_type_balance,
    ]
    if args.adapter_path:
        command.extend(["--adapter-path", str(args.adapter_path)])
    if args.limit_rows > 0:
        command.extend(["--limit-rows", str(args.limit_rows)])
    if args.sampler_epoch_size > 0:
        command.extend(["--sampler-epoch-size", str(args.sampler_epoch_size)])
    if mask_probe_only:
        command.append("--mask-probe-only")
    add_bool_flag(command, "--load-in-4bit", args.load_in_4bit)
    add_bool_flag(command, "--bf16", args.bf16)
    add_bool_flag(command, "--fp16", args.fp16)
    return command


def eval_command(args: argparse.Namespace, *, adapter_path: Path | None, output: Path, model_label: str) -> list[str]:
    command = [
        args.python,
        str(PIPELINE_ROOT / "eval" / "run_fixed_eval.py"),
        "--output",
        str(output),
        "--model",
        args.model,
        "--model-label",
        model_label,
        "--chat-template",
        args.chat_template,
        "--max-seq-length",
        str(args.max_seq_length),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-new-tokens",
        str(args.eval_max_new_tokens),
        "--limit-per-dataset",
        str(args.eval_limit_per_dataset),
        "--temperature",
        str(args.eval_temperature),
        "--top-p",
        str(args.eval_top_p),
        "--top-k",
        str(args.eval_top_k),
        "--seed",
        str(args.seed),
    ]
    if adapter_path is not None:
        command.extend(["--adapter-path", str(adapter_path)])
    for eval_file in args.eval_file:
        command.extend(["--eval-file", str(eval_file)])
    add_bool_flag(command, "--load-in-4bit", args.load_in_4bit)
    add_bool_flag(command, "--stop-at-result-close", args.eval_stop_at_result_close)
    return command


def score_command(args: argparse.Namespace, *, input_path: Path, output_path: Path) -> list[str]:
    return [
        args.python,
        str(PIPELINE_ROOT / "eval" / "score_outputs.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--result-tag-policy",
        args.result_tag_policy,
    ]


def reload_probe_command(args: argparse.Namespace, *, adapter_path: Path, output: Path) -> list[str]:
    command = [
        args.python,
        str(PIPELINE_ROOT / "eval" / "probe_adapter_reload.py"),
        "--adapter-path",
        str(adapter_path),
        "--output",
        str(output),
        "--model",
        args.model,
        "--model-label",
        "stage01_reload_probe",
        "--chat-template",
        args.chat_template,
        "--max-seq-length",
        str(args.max_seq_length),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-new-tokens",
        str(args.reload_probe_max_new_tokens),
        "--limit-per-dataset",
        str(args.reload_probe_limit_per_dataset),
        "--temperature",
        "0.0",
        "--seed",
        str(args.seed),
    ]
    add_bool_flag(command, "--load-in-4bit", args.load_in_4bit)
    return command


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline_v2 Stage 1 paid-instance pilot bundle.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--max-prompt-length", type=positive_int, default=4096)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=positive_int, default=32)
    parser.add_argument("--lora-alpha", type=positive_int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-last-layer-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--grad-accum", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-7)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--max-steps", type=positive_int, default=200)
    parser.add_argument("--logging-steps", type=positive_int, default=1)
    parser.add_argument("--save-steps", type=positive_int, default=50)
    parser.add_argument("--save-total-limit", type=positive_int, default=2)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--anti-slop-ul-weight", type=float, default=0.0)
    parser.add_argument(
        "--anti-slop-ul-lexicon",
        default=str(TRAINING_ROOT / "data" / "processed" / "anti_slop_lexicon.json"),
    )
    parser.add_argument("--anti-slop-ul-unigram-top-k", type=int, default=300)
    parser.add_argument("--anti-slop-ul-unigram-min-lift", type=float, default=7.5)
    parser.add_argument("--anti-slop-ul-bigram-top-k", type=int, default=300)
    parser.add_argument("--anti-slop-ul-bigram-min-lift", type=float, default=4.0)
    parser.add_argument("--anti-slop-ul-bigram-min-weight", type=float, default=0.05)
    parser.add_argument("--anti-slop-ul-trigram-top-k", type=int, default=0)
    parser.add_argument("--anti-slop-ul-trigram-min-lift", type=float, default=0.0)
    parser.add_argument("--anti-slop-ul-trigram-min-weight", type=float, default=0.0)
    parser.add_argument("--anti-slop-ul-start-weight-multiplier", type=float, default=0.08)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="pipeline_v2_stage01_pilot")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--row-type-sampling", choices=["balanced", "shuffle"], default="balanced")
    parser.add_argument("--row-type-balance", default="raw_lm=3,continuation_sft=3,format_sft=2,general_guard=1")
    parser.add_argument("--sampler-epoch-size", type=int, default=0)
    parser.add_argument("--eval-file", action="append", default=[])
    parser.add_argument("--eval-limit-per-dataset", type=int, default=4)
    parser.add_argument("--eval-max-new-tokens", type=positive_int, default=2048)
    parser.add_argument("--eval-temperature", type=float, default=0.7)
    parser.add_argument("--eval-top-p", type=float, default=0.95)
    parser.add_argument("--eval-top-k", type=int, default=50)
    parser.add_argument("--eval-stop-at-result-close", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reload-probe-limit-per-dataset", type=int, default=1)
    parser.add_argument("--reload-probe-max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--result-tag-policy", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--skip-preflight-mask-probe", action="store_true")
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-reload-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = str(resolve_user_path(args.dataset))
    args.output_root = str(resolve_user_path(args.output_root, for_output=True))
    if args.adapter_path:
        args.adapter_path = str(resolve_user_path(args.adapter_path))
    args.eval_file = [str(resolve_user_path(path)) for path in args.eval_file]
    if args.model == "auto" and not args.adapter_path:
        args.model = DEFAULT_BASE_MODEL
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage01_output = output_root / "stage01_cpt_lite"
    final_adapter = stage01_output / "policy"
    command_log: list[dict[str, Any]] = []

    if not args.skip_preflight_mask_probe:
        run_command(stage01_command(args, output_root / "preflight_mask_probe", mask_probe_only=True), cwd=TRAINING_ROOT, dry_run=args.dry_run, log=command_log)

    run_command(stage01_command(args, stage01_output, mask_probe_only=False), cwd=TRAINING_ROOT, dry_run=args.dry_run, log=command_log)

    if not args.skip_reload_probe:
        run_command(
            reload_probe_command(args, adapter_path=final_adapter, output=output_root / "reload_probe" / "generations.jsonl"),
            cwd=TRAINING_ROOT,
            dry_run=args.dry_run,
            log=command_log,
        )

    if not args.skip_baseline_eval:
        baseline_adapter = Path(args.adapter_path) if args.adapter_path else None
        baseline_generations = output_root / "eval_baseline" / "generations.jsonl"
        run_command(
            eval_command(args, adapter_path=baseline_adapter, output=baseline_generations, model_label="baseline_start_adapter" if baseline_adapter else "baseline_base"),
            cwd=TRAINING_ROOT,
            dry_run=args.dry_run,
            log=command_log,
        )
        run_command(
            score_command(args, input_path=baseline_generations, output_path=output_root / "eval_baseline" / "scored_outputs.jsonl"),
            cwd=TRAINING_ROOT,
            dry_run=args.dry_run,
            log=command_log,
        )

    trained_generations = output_root / "eval_stage01" / "generations.jsonl"
    run_command(
        eval_command(args, adapter_path=final_adapter, output=trained_generations, model_label="stage01_adapter"),
        cwd=TRAINING_ROOT,
        dry_run=args.dry_run,
        log=command_log,
    )
    run_command(
        score_command(args, input_path=trained_generations, output_path=output_root / "eval_stage01" / "scored_outputs.jsonl"),
        cwd=TRAINING_ROOT,
        dry_run=args.dry_run,
        log=command_log,
    )

    summary = {
        "time": time.time(),
        "dry_run": args.dry_run,
        "output_root": str(output_root),
        "stage01_output": str(stage01_output),
        "final_adapter": str(final_adapter),
        "args": vars(args),
        "commands": command_log,
        "stage01_manifest": read_json_if_exists(stage01_output / "stage01_manifest.json"),
        "baseline_score_summary": read_json_if_exists(output_root / "eval_baseline" / "scored_outputs.summary.json"),
        "stage01_score_summary": read_json_if_exists(output_root / "eval_stage01" / "scored_outputs.summary.json"),
        "reload_probe_summary": read_json_if_exists(output_root / "reload_probe" / "generations.summary.json"),
    }
    write_json(output_root / "pilot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
