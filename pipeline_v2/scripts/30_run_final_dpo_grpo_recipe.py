#!/usr/bin/env python3
"""Run the final pipeline_v2 DPO/GRPO recipe sequentially.

The script is intentionally a launcher rather than a trainer. Each stage stays
standalone and writes its own manifest, logs, checkpoints, and final adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
DEFAULT_OUTPUT_ROOT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "final_dpo_grpo_recipe_v2"
DEFAULT_MODEL = "/workspace/modelscope_cache/unsloth/gemma-4-31B-it"
DEFAULT_CPT_DATASET = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_mixed_probe.jsonl"
DEFAULT_DPO_DATASET = TRAINING_ROOT / "data" / "pipeline_v2" / "simpo_curriculum" / "03_badstyle_rewrite.jsonl"
DEFAULT_GRPO_DATASET = TRAINING_ROOT / "data" / "processed" / "grpo_mixed_prompts.jsonl"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_done(path: Path) -> bool:
    return (path / "policy" / "adapter_model.safetensors").exists()


def run_stage(name: str, command: list[str], *, cwd: Path, output: Path, env: dict[str, str], dry_run: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"{name}.log"
    done_path = output / f"{name}.done.json"
    write_json(
        output / f"{name}.command.json",
        {"name": name, "command": command, "cwd": str(cwd), "time": time.time()},
    )
    if dry_run:
        print("[dry-run]", " ".join(command), flush=True)
        return
    if done_path.exists() and stage_done(output):
        print(f"[skip] {name}: done marker and policy exist", flush=True)
        return
    print(f"[stage/start] {name}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[stage/start] {name} time={time.time()}\n")
        log.write("[command] " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = proc.wait()
        log.write(f"[stage/end] {name} returncode={returncode} time={time.time()}\n")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    write_json(done_path, {"name": name, "returncode": returncode, "time": time.time()})
    print(f"[stage/done] {name}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gate_failure_budget(stage: str) -> dict[str, int]:
    """Stage-specific gate tolerance.

    Stage 1 intentionally mixes raw LM with continuation/format SFT. Stage 2 is
    still pre-GRPO and trains continuation/format rows rather than free generate
    rollouts. These stages must not show post-stop text, replacement chars, or
    post-result text, but one fixed-eval generate row missing the close tag is
    allowed because Stage 4 directly optimizes generate result-contract behavior.
    """

    if stage in {"stage01_sft_ul", "stage02_format_task_sft"}:
        return {
            "result_contract_bad_rows": 1,
            "collapse_rows": 1,
            "post_stop_rows": 0,
            "replacement_char_rows": 0,
            "post_result_rows": 0,
        }
    return {
        "result_contract_bad_rows": 0,
        "collapse_rows": 0,
        "post_stop_rows": 0,
        "replacement_char_rows": 0,
        "post_result_rows": 0,
    }


def gate_failures_from_counts(summary: dict[str, Any], *, stage: str) -> list[str]:
    failures: list[str] = []
    if int(summary.get("generation_rows") or 0) <= 0 or int(summary.get("scored_rows") or 0) <= 0:
        failures.append("empty_eval_outputs")
    budget = gate_failure_budget(stage)
    for key, allowed in budget.items():
        count = int(summary.get(key) or 0)
        if count > allowed:
            failures.append(f"{key}={count}")
    return failures


def run_eval_gate(
    name: str,
    *,
    adapter_path: Path,
    model: str,
    python: str,
    cwd: Path,
    output_root: Path,
    env: dict[str, str],
    max_seq_length: int,
    dry_run: bool,
) -> None:
    gate_dir = output_root / "eval_gates" / name
    generations = gate_dir / "generations.jsonl"
    scored = gate_dir / "scored_outputs.jsonl"
    gate_summary_path = gate_dir / "gate_summary.json"
    if gate_summary_path.exists():
        try:
            existing_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing_summary = {}
        remaining_failures = gate_failures_from_counts(existing_summary, stage=name)
        if not remaining_failures:
            existing_summary["gate_passed"] = True
            existing_summary["gate_policy"] = gate_failure_budget(name)
            if existing_summary.get("failures"):
                existing_summary["waived_failures"] = existing_summary.get("failures")
            write_json(gate_summary_path, existing_summary)
            print("[gate/skip_existing]", json.dumps(existing_summary, ensure_ascii=False), flush=True)
            return
    eval_cmd = [
        python,
        "-m",
        "pipeline_v2.eval.run_fixed_eval",
        "--output",
        str(generations),
        "--model",
        model,
        "--adapter-path",
        str(adapter_path),
        "--model-label",
        name,
        "--max-seq-length",
        str(max_seq_length),
        "--max-new-tokens",
        "3072",
        "--limit-per-dataset",
        "1",
        "--temperature",
        "0.85",
        "--top-p",
        "1.0",
        "--top-k",
        "50",
        "--repetition-penalty",
        "1.05",
        "--stop-at-result-close",
        "--stop-at-turn-token",
        "--force-eos-after-stop",
        "--grpo-compatible-prompts",
    ]
    score_cmd = [
        python,
        "-m",
        "pipeline_v2.eval.score_outputs",
        "--input",
        str(generations),
        "--output",
        str(scored),
        "--result-tag-policy",
        "auto",
    ]
    if dry_run:
        print("[dry-run/eval]", " ".join(eval_cmd), flush=True)
        print("[dry-run/score]", " ".join(score_cmd), flush=True)
        return
    run_stage(f"{name}_eval_generate", eval_cmd, cwd=cwd, output=gate_dir, env=env, dry_run=False)
    run_stage(f"{name}_eval_score", score_cmd, cwd=cwd, output=gate_dir, env=env, dry_run=False)
    generation_rows = load_jsonl(generations)
    scored_rows = load_jsonl(scored)
    post_stop = sum(1 for row in generation_rows if str(row.get("post_stop_text") or "").strip())
    replacement = sum(
        1
        for row in generation_rows
        if "\ufffd" in str(row.get("generated_text") or "") or "\ufffd" in str(row.get("raw_generated_text") or "")
    )
    contract_bad = sum(1 for row in scored_rows if row.get("require_result_tags") and not row.get("metrics.result_contract_ok"))
    post_result_bad = sum(1 for row in scored_rows if int(row.get("metrics.post_result_chars") or 0) > 0)
    collapse_bad = sum(1 for row in scored_rows if str(row.get("metrics.collapse_reason") or ""))
    score_values = [
        float(row["score"])
        for row in scored_rows
        if row.get("score") is not None and str(row.get("score")) not in {"nan", "None"}
    ]
    gate_summary = {
        "time": time.time(),
        "stage": name,
        "adapter_path": str(adapter_path),
        "generation_rows": len(generation_rows),
        "scored_rows": len(scored_rows),
        "post_stop_rows": post_stop,
        "replacement_char_rows": replacement,
        "result_contract_bad_rows": contract_bad,
        "post_result_rows": post_result_bad,
        "collapse_rows": collapse_bad,
        "score_mean": sum(score_values) / max(1, len(score_values)),
        "generations": str(generations),
        "scored": str(scored),
    }
    failures = gate_failures_from_counts(gate_summary, stage=name)
    observed_failures = []
    if not generation_rows or not scored_rows:
        observed_failures.append("empty_eval_outputs")
    if post_stop:
        observed_failures.append(f"post_stop_rows={post_stop}")
    if replacement:
        observed_failures.append(f"replacement_char_rows={replacement}")
    if contract_bad:
        observed_failures.append(f"result_contract_bad_rows={contract_bad}")
    if post_result_bad:
        observed_failures.append(f"post_result_rows={post_result_bad}")
    if collapse_bad:
        observed_failures.append(f"collapse_rows={collapse_bad}")
    gate_summary["gate_policy"] = gate_failure_budget(name)
    gate_summary["failures"] = failures
    gate_summary["observed_failures"] = observed_failures
    gate_summary["gate_passed"] = not failures
    if observed_failures and not failures:
        gate_summary["waived_failures"] = observed_failures
    write_json(gate_summary_path, gate_summary)
    print("[gate]", json.dumps(gate_summary, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError(f"Eval gate failed after {name}: {failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final pipeline_v2 DPO/GRPO recipe.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cpt-dataset", default=str(DEFAULT_CPT_DATASET))
    parser.add_argument("--dpo-dataset", default=str(DEFAULT_DPO_DATASET))
    parser.add_argument("--grpo-dataset", default=str(DEFAULT_GRPO_DATASET))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stage1-steps", type=positive_int, default=250)
    parser.add_argument("--stage2-steps", type=positive_int, default=100)
    parser.add_argument("--stage3-dpo-steps", type=positive_int, default=60)
    parser.add_argument("--generate-grpo-steps", type=positive_int, default=80)
    parser.add_argument("--rewrite-grpo-steps", type=positive_int, default=80)
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--sft-batch-size", type=positive_int, default=1)
    parser.add_argument("--sft-grad-accum", type=positive_int, default=8)
    parser.add_argument("--dpo-batch-size", type=positive_int, default=1)
    parser.add_argument("--dpo-grad-accum", type=positive_int, default=8)
    parser.add_argument("--grpo-batch-size", type=positive_int, default=4)
    parser.add_argument("--grpo-num-generations", type=positive_int, default=4)
    parser.add_argument("--grpo-grad-accum", type=positive_int, default=1)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="gemma4-webnovel-style-31b-pipeline-v2")
    parser.add_argument("--seed", type=int, default=4710)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("WANDB_PROJECT", args.wandb_project)
    env.setdefault("WANDB_MODE", "online")
    env.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    cwd = TRAINING_ROOT

    stage01 = root / "stage01_sft_ul"
    stage02 = root / "stage02_format_task_sft"
    stage03 = root / "stage03_dpo_badstyle_rewrite"
    stage04 = root / "stage04_grpo_generate"
    stage05 = root / "stage05_grpo_rewrite"

    commands: list[tuple[str, Path, list[str]]] = [
        (
            "stage01_sft_ul",
            stage01,
            [
                args.python,
                "-m",
                "pipeline_v2.train.stage01_cpt_lite",
                "--dataset",
                args.cpt_dataset,
                "--output",
                str(stage01),
                "--model",
                args.model,
                "--max-seq-length",
                str(args.max_seq_length),
                "--load-in-4bit",
                "--batch-size",
                str(args.sft_batch_size),
                "--grad-accum",
                str(args.sft_grad_accum),
                "--learning-rate",
                "8e-7",
                "--max-grad-norm",
                "0.25",
                "--max-steps",
                str(args.stage1_steps),
                "--save-steps",
                "50",
                "--save-total-limit",
                "2",
                "--row-type-sampling",
                "balanced",
                "--row-type-balance",
                "raw_lm=3,continuation_sft=3,format_sft=2",
                "--anti-slop-ul-weight",
                "0.003",
                "--anti-slop-ul-unigram-top-k",
                "200",
                "--anti-slop-ul-unigram-min-lift",
                "9.0",
                "--anti-slop-ul-bigram-top-k",
                "300",
                "--anti-slop-ul-bigram-min-lift",
                "4.0",
                "--anti-slop-ul-bigram-min-weight",
                "0.05",
                "--anti-slop-ul-trigram-top-k",
                "0",
                "--anti-slop-ul-start-weight-multiplier",
                "0.04",
                "--seed",
                str(args.seed),
                "--report-to",
                args.report_to,
                "--run-name",
                "pipeline_v2_final_stage01_sft_ul",
            ],
        ),
        (
            "stage02_format_task_sft",
            stage02,
            [
                args.python,
                "-m",
                "pipeline_v2.train.stage02_format_task_sft",
                "--dataset",
                args.cpt_dataset,
                "--output",
                str(stage02),
                "--model",
                args.model,
                "--adapter-path",
                str(stage01 / "policy"),
                "--max-seq-length",
                str(args.max_seq_length),
                "--load-in-4bit",
                "--batch-size",
                str(args.sft_batch_size),
                "--grad-accum",
                str(args.sft_grad_accum),
                "--learning-rate",
                "6e-7",
                "--max-grad-norm",
                "0.25",
                "--max-steps",
                str(args.stage2_steps),
                "--save-steps",
                "25",
                "--save-total-limit",
                "2",
                "--seed",
                str(args.seed + 1),
                "--report-to",
                args.report_to,
                "--run-name",
                "pipeline_v2_final_stage02_format_task_sft",
            ],
        ),
        (
            "stage03_dpo_badstyle_rewrite",
            stage03,
            [
                args.python,
                "-m",
                "pipeline_v2.train.stage03_simpo_curriculum",
                "--dataset",
                args.dpo_dataset,
                "--output",
                str(stage03),
                "--model",
                args.model,
                "--adapter-path",
                str(stage02 / "policy"),
                "--bucket",
                "badstyle_rejected",
                "--preference-loss",
                "dpo",
                "--dpo-loss-type",
                "sigmoid",
                "--max-seq-length",
                "4096",
                "--max-prompt-length",
                "2048",
                "--load-in-4bit",
                "--batch-size",
                str(args.dpo_batch_size),
                "--grad-accum",
                str(args.dpo_grad_accum),
                "--learning-rate",
                "3e-7",
                "--max-grad-norm",
                "0.2",
                "--beta",
                "0.001",
                "--max-steps",
                str(args.stage3_dpo_steps),
                "--save-steps",
                "20",
                "--save-total-limit",
                "2",
                "--seed",
                str(args.seed + 2),
                "--report-to",
                args.report_to,
                "--run-name",
                "pipeline_v2_final_stage03_dpo_badstyle_rewrite",
            ],
        ),
        (
            "stage04_grpo_generate",
            stage04,
            [
                args.python,
                "-m",
                "pipeline_v2.train.stage04_grpo_generate",
                "--dataset",
                args.grpo_dataset,
                "--output",
                str(stage04),
                "--model",
                args.model,
                "--adapter-path",
                str(stage03 / "policy"),
                "--max-seq-length",
                str(args.max_seq_length),
                "--max-completion-length",
                "3072",
                "--generate-min-output-chars",
                "3000",
                "--generate-max-output-chars",
                "4500",
                "--load-in-4bit",
                "--batch-size",
                str(args.grpo_batch_size),
                "--num-generations",
                str(args.grpo_num_generations),
                "--grad-accum",
                str(args.grpo_grad_accum),
                "--learning-rate",
                "8e-7",
                "--warmup-steps",
                "3",
                "--max-grad-norm",
                "0.2",
                "--beta",
                "0.005",
                "--loss-type",
                "dr_grpo",
                "--num-iterations",
                "2",
                "--scale-rewards",
                "group",
                "--temperature",
                "0.95",
                "--top-p",
                "1.0",
                "--top-k",
                "50",
                "--repetition-penalty",
                "1.05",
                "--no-repeat-ngram-size",
                "0",
                "--max-steps",
                str(args.generate_grpo_steps),
                "--save-steps",
                "20",
                "--save-total-limit",
                "2",
                "--sample-log-every",
                "1",
                "--sample-log-max-items",
                "4",
                "--sample-log-text-chars",
                "4000",
                "--group-diversity-bonus-max",
                "0.03",
                "--group-diversity-mode",
                "leave_one_out",
                "--unsloth-grpo-mini-batch",
                "1",
                "--unsloth-logit-chunk-multiplier",
                "8",
                "--require-result-tags",
                "--stop-at-result-close",
                "--seed",
                str(args.seed + 3),
                "--report-to",
                args.report_to,
                "--run-name",
                "pipeline_v2_final_stage04_grpo_generate",
            ],
        ),
        (
            "stage05_grpo_rewrite",
            stage05,
            [
                args.python,
                "-m",
                "pipeline_v2.train.stage05_grpo_rewrite",
                "--dataset",
                args.grpo_dataset,
                "--output",
                str(stage05),
                "--model",
                args.model,
                "--adapter-path",
                str(stage04 / "policy"),
                "--max-seq-length",
                str(args.max_seq_length),
                "--max-completion-length",
                "3072",
                "--load-in-4bit",
                "--batch-size",
                str(args.grpo_batch_size),
                "--num-generations",
                str(args.grpo_num_generations),
                "--grad-accum",
                str(args.grpo_grad_accum),
                "--learning-rate",
                "8e-7",
                "--warmup-steps",
                "3",
                "--max-grad-norm",
                "0.2",
                "--beta",
                "0.005",
                "--loss-type",
                "dr_grpo",
                "--num-iterations",
                "2",
                "--scale-rewards",
                "group",
                "--temperature",
                "1.0",
                "--top-p",
                "1.0",
                "--top-k",
                "50",
                "--repetition-penalty",
                "1.05",
                "--no-repeat-ngram-size",
                "0",
                "--max-steps",
                str(args.rewrite_grpo_steps),
                "--save-steps",
                "20",
                "--save-total-limit",
                "2",
                "--sample-log-every",
                "1",
                "--sample-log-max-items",
                "4",
                "--sample-log-text-chars",
                "4000",
                "--rewrite-edit-weight",
                "0.35",
                "--rewrite-improvement-weight",
                "0.30",
                "--unsloth-grpo-mini-batch",
                "1",
                "--unsloth-logit-chunk-multiplier",
                "8",
                "--require-result-tags",
                "--stop-at-result-close",
                "--seed",
                str(args.seed + 4),
                "--report-to",
                args.report_to,
                "--run-name",
                "pipeline_v2_final_stage05_grpo_rewrite",
            ],
        ),
    ]

    write_json(
        root / "recipe_manifest.json",
        {
            "time": time.time(),
            "output_root": str(root),
            "model": args.model,
            "datasets": {
                "cpt": args.cpt_dataset,
                "dpo": args.dpo_dataset,
                "grpo": args.grpo_dataset,
            },
            "stages": [name for name, _output, _command in commands],
            "args": vars(args),
        },
    )
    for name, output, command in commands:
        run_stage(name, command, cwd=cwd, output=output, env=env, dry_run=args.dry_run)
        run_eval_gate(
            name,
            adapter_path=output / "policy",
            model=args.model,
            python=args.python,
            cwd=cwd,
            output_root=root,
            env=env,
            max_seq_length=args.max_seq_length,
            dry_run=args.dry_run,
        )
    write_json(root / "recipe_done.json", {"time": time.time(), "final_adapter": str(stage05 / "policy")})
    print("[recipe/done]", str(stage05 / "policy"), flush=True)


if __name__ == "__main__":
    main()
