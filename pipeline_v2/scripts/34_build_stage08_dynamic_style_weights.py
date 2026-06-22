#!/usr/bin/env python3
"""Build a conservative metric-weight preset from the latest eval summary.

This is intentionally not per-step online reward adaptation. It reads an eval
summary with human/AI/model group means, estimates how far the model remains
from human relative to the AI-human gap, and emits bounded
metric:multiplier overrides for the next GRPO run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
DEFAULT_GENERATE_SUMMARY = (
    TRAINING_ROOT
    / "outputs"
    / "local_reports"
    / "stage07i_100_eval"
    / "stage07i"
    / "metrics"
    / "metric_summary_by_group.csv"
)
DEFAULT_OUTPUT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08_dynamic_style_weights.json"


DEFAULT_BASE_WEIGHTS = {
    "sentence_final_token_repeat_rate": 2.4,
    "sentence_length_cv": 2.8,
    "sentence_length_iqr_ratio": 2.8,
    "pos_3gram_repeat_rate": 1.4,
    "pos_4gram_diversity": 1.3,
    "pos_5gram_repeat_rate": 1.4,
    "anti_slop_density": 1.7,
    "simile_marker_per_1k_chars": 1.5,
    "simile_sentence_rate": 1.5,
    "content_modifier_repeat_occurrence_rate": 1.2,
    "modifier_repetition_mass": 1.1,
    "modifier_repeat_burst_mass": 1.0,
    "sentence_initial_token_repeat_rate": 0.8,
    "comma_per_1k_chars": 0.35,
}


DISABLED = {
    "translationese_raw",
    "content_top_10_coverage",
    "content_gini_frequency",
    "content_repeat_occurrence_rate",
    "parenthesis_pair_per_1k_chars",
}


def finite_float(value: str | None, default: float = float("nan")) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_group_means(path: Path, *, eval_task: str) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("eval_task") or row.get("\ufeffeval_task") or "") != eval_task:
                continue
            metric = str(row.get("metric") or "")
            group = str(row.get("group") or "")
            if not metric or not group:
                continue
            means.setdefault(metric, {})[group] = finite_float(row.get("mean"))
    return means


def build_weights(means: dict[str, dict[str, float]], *, model_group: str, max_multiplier: float) -> dict[str, float]:
    weights: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for metric, base_weight in DEFAULT_BASE_WEIGHTS.items():
        if metric in DISABLED:
            continue
        group_values = means.get(metric, {})
        human = finite_float(str(group_values.get("human_control")))
        ai = finite_float(str(group_values.get("ai_control")))
        model = finite_float(str(group_values.get(model_group)))
        if not all(math.isfinite(value) for value in (human, ai, model)):
            weights[metric] = base_weight
            continue
        gap = abs(ai - human)
        if gap < 1e-9:
            severity = 0.0
        else:
            closure = 1.0 - abs(model - human) / gap
            # severity 0 means mostly solved; severity 1 means no progress;
            # severity >1 means overshot or moved away from human.
            severity = clip(1.0 - closure, 0.0, 1.6)
        boost = 1.0 + 0.55 * severity
        weights[metric] = round(clip(base_weight * boost, 0.05, max_multiplier), 4)
        diagnostics[metric] = {
            "human_mean": human,
            "ai_mean": ai,
            "model_mean": model,
            "base_weight": base_weight,
            "severity": severity,
            "weight": weights[metric],
        }
    return weights, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 8 dynamic style metric overrides.")
    parser.add_argument("--generate-summary", default=str(DEFAULT_GENERATE_SUMMARY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-task", default="generate")
    parser.add_argument("--model-group", default="generate_output")
    parser.add_argument("--max-multiplier", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    means = load_group_means(Path(args.generate_summary), eval_task=args.eval_task)
    weights, diagnostics = build_weights(means, model_group=args.model_group, max_multiplier=args.max_multiplier)
    override_string = ",".join(f"{name}:{value:g}" for name, value in weights.items())
    payload = {
        "source": str(Path(args.generate_summary)),
        "eval_task": args.eval_task,
        "model_group": args.model_group,
        "style_metric_weight_overrides": override_string,
        "weights": weights,
        "diagnostics": diagnostics,
        "disabled_style_metrics": sorted(DISABLED),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(override_string)


if __name__ == "__main__":
    main()
