#!/usr/bin/env python3
"""Audit GUI-style human-vs-AI reference metrics and reward sample logs.

This script is standalone. It does not import legacy `train/`, `scripts/`, or
`eval/` modules. It consumes the existing GUI-style reference JSON and optional
reward sample logs, then reports which metrics have meaningful human/AI
separation and which are weak/noisy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parents[1]
TRAINING_ROOT = PIPELINE_ROOT.parent

DEFAULT_REFERENCE = TRAINING_ROOT / "data" / "processed" / "gui_style_reward_reference.json"
DEFAULT_DIAGNOSTICS = TRAINING_ROOT / "outputs" / "diagnostics" / "latest_vs_ai_vs_human_reward_comparison.json"
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "outputs" / "pipeline_v2" / "gui_metric_audit"


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric_family(name: str) -> str:
    if name in {"anti_slop", "translationese"}:
        return name
    if name.startswith("comma") or name.startswith("parenthesis") or name.startswith("parentheses"):
        return "punctuation"
    if name.startswith("pos_"):
        if "diversity" in name or "entropy" in name:
            return "pos_diversity_entropy"
        if "repeat" in name:
            return "pos_recurrence"
        return "pos"
    if name.startswith("content_modifier"):
        return "modifier_shape"
    if name.startswith("content_"):
        return "content_lexical"
    if name.startswith("sentence_initial") or name.startswith("sentence_final"):
        return "sentence_edge"
    if name.startswith("sentence_length"):
        return "sentence_rhythm"
    return "other"


def separation_label(abs_delta: float, ai_in_iqr: float, ai_in_q10_q90: float) -> str:
    if abs_delta >= 0.75 and ai_in_iqr <= 0.20:
        return "strong"
    if abs_delta >= 0.55 and ai_in_iqr <= 0.35:
        return "useful"
    if abs_delta >= 0.35 and ai_in_q10_q90 <= 0.75:
        return "weak_but_usable"
    if abs_delta >= 0.25:
        return "weak_or_redundant"
    return "low_signal"


def effective_weight(stats: dict[str, Any]) -> float:
    base = abs(finite_float(stats.get("weight"), 1.0))
    ai_in_iqr = finite_float(stats.get("ai_in_human_iqr_rate"), float("nan"))
    if math.isfinite(ai_in_iqr):
        base *= 0.35 + 0.65 * max(0.0, min(1.0, 1.0 - ai_in_iqr))
    return base


def audit_scalar_reference(reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_sources = dict(reference.get("metrics") or {})
    if isinstance(reference.get("anti_slop"), dict):
        metric_sources["anti_slop"] = reference["anti_slop"]
    if isinstance(reference.get("translationese"), dict):
        metric_sources["translationese"] = reference["translationese"]

    for name, stats in metric_sources.items():
        if not isinstance(stats, dict):
            continue
        human_mean = finite_float(stats.get("human_mean"))
        ai_mean = finite_float(stats.get("ai_mean"))
        delta = finite_float(stats.get("cliffs_delta_human_vs_ai"))
        ai_in_iqr = finite_float(stats.get("ai_in_human_iqr_rate"))
        ai_in_q10_q90 = finite_float(stats.get("ai_in_human_q10_q90_rate"))
        abs_delta = abs(delta) if math.isfinite(delta) else float("nan")
        gap = human_mean - ai_mean if math.isfinite(human_mean) and math.isfinite(ai_mean) else float("nan")
        rows.append(
            {
                "metric": name,
                "family": metric_family(name),
                "human_mean": human_mean,
                "ai_mean": ai_mean,
                "human_minus_ai": gap,
                "ai_direction": "ai_higher" if math.isfinite(gap) and gap < 0 else "human_higher",
                "cliffs_delta_human_vs_ai": delta,
                "abs_cliffs_delta": abs_delta,
                "ai_in_human_iqr_rate": ai_in_iqr,
                "ai_in_human_q10_q90_rate": ai_in_q10_q90,
                "weight": finite_float(stats.get("weight")),
                "effective_weight": effective_weight(stats),
                "human_q10": finite_float(stats.get("human_q10")),
                "human_q25": finite_float(stats.get("human_q25")),
                "human_q50": finite_float(stats.get("human_q50")),
                "human_q75": finite_float(stats.get("human_q75")),
                "human_q90": finite_float(stats.get("human_q90")),
                "human_n_chunks": int(finite_float(stats.get("human_n_chunks"), 0)),
                "ai_n_chunks": int(finite_float(stats.get("ai_n_chunks"), 0)),
                "signal": separation_label(abs_delta, ai_in_iqr, ai_in_q10_q90)
                if all(math.isfinite(v) for v in (abs_delta, ai_in_iqr, ai_in_q10_q90))
                else "unknown",
            }
        )
    rows.sort(key=lambda row: (finite_float(row.get("effective_weight"), 0.0), finite_float(row.get("abs_cliffs_delta"), 0.0)), reverse=True)
    return rows


def js_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return float("nan")
    midpoint = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in keys}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for key, value in a.items():
            if value <= 0.0:
                continue
            total += value * math.log(value / max(b.get(key, 0.0), 1e-12), 2)
        return total

    return math.sqrt(max(0.0, 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)))


def top_distribution_gaps(human: dict[str, float], ai: dict[str, float], limit: int = 8) -> tuple[str, str]:
    keys = set(human) | set(ai)
    human_high = sorted(((human.get(key, 0.0) - ai.get(key, 0.0), key) for key in keys), reverse=True)[:limit]
    ai_high = sorted(((ai.get(key, 0.0) - human.get(key, 0.0), key) for key in keys), reverse=True)[:limit]
    return (
        "; ".join(f"{key}:{gap:.4f}" for gap, key in human_high if gap > 0),
        "; ".join(f"{key}:{gap:.4f}" for gap, key in ai_high if gap > 0),
    )


def audit_pos_usage(reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_text, stats in sorted((reference.get("pos_ngram_usage") or {}).items(), key=lambda item: int(item[0])):
        human = {str(k): finite_float(v, 0.0) for k, v in (stats.get("human_distribution") or {}).items()}
        ai = {str(k): finite_float(v, 0.0) for k, v in (stats.get("ai_distribution") or {}).items()}
        human_high, ai_high = top_distribution_gaps(human, ai)
        rows.append(
            {
                "n": int(n_text),
                "js_distance": js_distance(human, ai),
                "weight": finite_float(stats.get("weight")),
                "human_types": int(finite_float(stats.get("human_types"), len(human))),
                "ai_types": int(finite_float(stats.get("ai_types"), len(ai))),
                "human_high_ngrams": human_high,
                "ai_high_ngrams": ai_high,
            }
        )
    return rows


def flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else str(key), nested, out)
    else:
        number = finite_float(value)
        if math.isfinite(number):
            out[prefix] = number


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize_values(values: list[float]) -> dict[str, float | int]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "negative_rate": float("nan"),
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": quantile(finite, 0.50),
        "p25": quantile(finite, 0.25),
        "p75": quantile(finite, 0.75),
        "min": min(finite),
        "max": max(finite),
        "negative_rate": sum(1 for value in finite if value < 0.0) / len(finite),
    }


def audit_reward_samples(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values: dict[str, list[float]] = {}
    reason_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    row_count = 0
    post_result_rows = 0
    clipped_rows = 0
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            row_count += 1
            task_counts[str(row.get("task") or "")] += 1
            if int(finite_float(row.get("post_result_chars"), 0)) > 0:
                post_result_rows += 1
            if bool(row.get("text_truncated")):
                clipped_rows += 1
            numeric: dict[str, float] = {}
            flatten_numeric("row.score", row.get("score"), numeric)
            flatten_numeric("row.chars", row.get("chars"), numeric)
            flatten_numeric("row.raw_chars", row.get("raw_chars"), numeric)
            flatten_numeric("row.result_text_chars", row.get("result_text_chars"), numeric)
            flatten_numeric("row.post_result_chars", row.get("post_result_chars"), numeric)
            flatten_numeric("component", row.get("component_scores") or {}, numeric)
            for key, value in numeric.items():
                values.setdefault(key, []).append(value)
            components = row.get("component_scores") or {}
            for reason_key in ("result_contract_reason", "collapse_reason"):
                reason = components.get(reason_key)
                if reason:
                    reason_counts[f"{reason_key}:{reason}"] += 1

    summaries = []
    for key, series in values.items():
        item = {"metric": key, **summarize_values(series)}
        summaries.append(item)
    summaries.sort(key=lambda row: (str(row["metric"]).count("."), str(row["metric"])))
    overview = {
        "rows": row_count,
        "task_counts": dict(task_counts),
        "post_result_rows": post_result_rows,
        "post_result_rate": post_result_rows / row_count if row_count else float("nan"),
        "text_truncated_rows": clipped_rows,
        "text_truncated_rate": clipped_rows / row_count if row_count else float("nan"),
        "reason_counts": dict(reason_counts.most_common()),
    }
    return summaries, overview


def audit_existing_diagnostics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    rows = []
    for item in payload.get("comparison") or []:
        if not isinstance(item, dict):
            continue
        latest = finite_float(item.get("latest"))
        human_mean = finite_float(item.get("human_mean"))
        ai_mean = finite_float(item.get("ai_mean"))
        rows.append(
            {
                "metric": item.get("metric"),
                "latest": latest,
                "ai_mean": ai_mean,
                "human_mean": human_mean,
                "latest_minus_ai": finite_float(item.get("latest_minus_ai")),
                "latest_minus_human": finite_float(item.get("latest_minus_human")),
                "latest_percentile_in_ai": finite_float(item.get("latest_percentile_in_ai")),
                "latest_percentile_in_human": finite_float(item.get("latest_percentile_in_human")),
                "human_p25": finite_float(item.get("human_p25")),
                "human_p50": finite_float(item.get("human_p50")),
                "human_p75": finite_float(item.get("human_p75")),
            }
        )
    return rows


def grouped_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "") for row in rows)
    return dict(counts.most_common())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GUI-style reference and reward sample metrics.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS))
    parser.add_argument("--samples", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.reference)
    reference = read_json(reference_path)

    scalar_rows = audit_scalar_reference(reference)
    pos_rows = audit_pos_usage(reference)
    diagnostics_rows = audit_existing_diagnostics(Path(args.diagnostics))
    sample_paths = [Path(path) for path in args.samples]
    sample_rows, sample_overview = audit_reward_samples(sample_paths)

    write_csv(output_dir / "scalar_metric_separation.csv", scalar_rows)
    write_csv(output_dir / "pos_ngram_usage_separation.csv", pos_rows)
    if diagnostics_rows:
        write_csv(output_dir / "existing_diagnostics_comparison.csv", diagnostics_rows)
    if sample_rows:
        write_csv(output_dir / "reward_sample_component_summary.csv", sample_rows)

    strong = [row for row in scalar_rows if row.get("signal") in {"strong", "useful"}]
    low = [row for row in scalar_rows if row.get("signal") in {"weak_or_redundant", "low_signal"}]
    summary = {
        "time": time.time(),
        "reference": str(reference_path),
        "diagnostics": str(args.diagnostics),
        "sample_paths": [str(path) for path in sample_paths],
        "scalar_metric_count": len(scalar_rows),
        "scalar_signal_counts": grouped_counts(scalar_rows, "signal"),
        "family_counts": grouped_counts(scalar_rows, "family"),
        "top_effective_metrics": scalar_rows[:12],
        "strong_or_useful_metrics": [row["metric"] for row in strong],
        "weak_or_low_metrics": [row["metric"] for row in low],
        "pos_ngram_usage": pos_rows,
        "sample_overview": sample_overview,
        "outputs": {
            "scalar_metric_separation": str(output_dir / "scalar_metric_separation.csv"),
            "pos_ngram_usage_separation": str(output_dir / "pos_ngram_usage_separation.csv"),
            "existing_diagnostics_comparison": str(output_dir / "existing_diagnostics_comparison.csv")
            if diagnostics_rows
            else "",
            "reward_sample_component_summary": str(output_dir / "reward_sample_component_summary.csv")
            if sample_rows
            else "",
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

