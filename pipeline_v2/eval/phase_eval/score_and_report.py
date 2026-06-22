#!/usr/bin/env python3
"""Score phase-end evaluation generations with raw GUI-compatible metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT = Path(__file__).resolve()
PHASE_EVAL_ROOT = SCRIPT.parent
TRAINING_ROOT = SCRIPT.parents[3]
if str(PHASE_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_EVAL_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from metrics_gui_compatible import (
    DEFAULT_REPORT_METRICS,
    Chunk,
    analyze_chunk,
    cliffs_delta,
    finite_float,
    summarize_numeric,
)
from pipeline_v2.lib.result_contract import analyze_result_contract
from pipeline_v2.lib.gui_style_scoring import (
    DEFAULT_ANTI_SLOP_LEXICON,
    DEFAULT_TRANSLATIONESE_MODEL,
    AntiSlopDensityScorer,
    TranslationeseSVMScorer,
)


DEFAULT_EVAL_DIR = TRAINING_ROOT / "data" / "phase_eval_v2"
DEFAULT_GENERATION_DIR = TRAINING_ROOT / "outputs" / "pipeline_v2" / "phase_eval" / "generations"
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "outputs" / "pipeline_v2" / "phase_eval" / "metrics"

COMPOSITE_METRIC_GROUPS: dict[str, list[str]] = {
    "punctuation": [
        "comma_per_1k_chars",
        "comma_sentence_rate",
        "commas_per_sentence",
        "parenthesis_pair_per_1k_chars",
        "parenthesis_sentence_rate",
    ],
    "sentence_rhythm": [
        "sentence_length_mean_morphs",
        "sentence_length_cv",
        "sentence_length_median_morphs",
        "sentence_length_iqr_ratio",
    ],
    "sentence_boundary": [
        "sentence_initial_token_repeat_rate",
        "sentence_initial_pos_bigram_repeat_rate",
        "sentence_final_token_repeat_rate",
        "sentence_final_pos_bigram_repeat_rate",
        "sentence_initial_token_top3_window_coverage",
        "sentence_initial_pos2_top3_window_coverage",
        "sentence_final_token_top3_window_coverage",
        "sentence_final_pos2_top3_window_coverage",
    ],
    "pos_shape": [
        "pos_3gram_diversity",
        "pos_4gram_diversity",
        "pos_5gram_diversity",
        "pos_6gram_diversity",
        "pos_3gram_repeat_rate",
        "pos_4gram_repeat_rate",
        "pos_5gram_repeat_rate",
        "pos_6gram_repeat_rate",
        "pos_3gram_window_recurrence",
        "pos_4gram_window_recurrence",
        "pos_5gram_window_recurrence",
        "pos_6gram_window_recurrence",
        "pos_3gram_entropy_norm",
        "pos_4gram_entropy_norm",
        "pos_5gram_entropy_norm",
        "pos_6gram_entropy_norm",
    ],
    "modifier_shape": [
        "modifier_per_1k_morphs",
        "modifier_hill_d2_norm",
        "modifier_repetition_mass",
        "modifier_repeat_burst_mass",
        "content_modifier_repeat_occurrence_rate",
        "content_modifier_simpson_concentration",
        "content_modifier_gini_frequency",
        "content_modifier_yule_k",
        "content_modifier_1gram_repeat_rate",
        "content_modifier_2gram_repeat_rate",
        "content_modifier_3gram_repeat_rate",
    ],
    "lexical_repetition": [
        "content_repeat_occurrence_rate",
        "content_top_10_coverage",
        "content_top_20_coverage",
        "content_max_frequency_rate",
        "content_simpson_concentration",
        "content_gini_frequency",
        "content_yule_k",
        "content_1gram_repeat_rate",
        "content_2gram_repeat_rate",
        "content_3gram_repeat_rate",
        "content_4gram_repeat_rate",
        "content_5gram_repeat_rate",
    ],
    "slop_translationese": [
        "anti_slop_density",
        "translationese_raw",
    ],
}

COMPOSITE_REPORT_METRICS = [f"composite_{name}_ai_likeness_raw" for name in COMPOSITE_METRIC_GROUPS]
AUX_REPORT_METRICS = [
    "anti_slop_density",
    "translationese_raw",
    "composite_slop_translationese_ai_likeness_raw",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "metric"


def result_text(raw_text: str, *, require_result_tags: bool) -> tuple[str, dict[str, Any]]:
    contract = analyze_result_contract(raw_text, require_result_tags=require_result_tags)
    return contract.result_text.strip(), contract.summary()


def group_text_rows(
    *,
    eval_task: str,
    group: str,
    rows: list[dict[str, Any]],
    text_key: str,
    require_result_tags: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        raw = str(row.get(text_key) or "")
        text, contract = result_text(raw, require_result_tags=require_result_tags)
        out.append(
            {
                "id": str(row.get("id") or f"{group}-{index:04d}"),
                "eval_task": eval_task,
                "group": group,
                "source_file": row.get("source_file", ""),
                "chunk_id": row.get("chunk_id") or row.get("source_chunk_id", ""),
                "raw_text": raw,
                "analysis_text": text,
                **{f"contract_{key}": value for key, value in contract.items()},
            }
        )
    return out


def build_analysis_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    eval_dir = Path(args.eval_dir)
    generation_dir = Path(args.generation_dir)
    ai_controls = read_jsonl(Path(args.ai_source_controls) if args.ai_source_controls else eval_dir / "ai_source_controls.jsonl")
    human_controls = read_jsonl(Path(args.human_controls) if args.human_controls else eval_dir / "human_controls.jsonl")
    rewrite_generations = read_jsonl(
        Path(args.rewrite_generations) if args.rewrite_generations else generation_dir / "rewrite_generations.jsonl"
    )
    generate_generations = read_jsonl(
        Path(args.generate_generations) if args.generate_generations else generation_dir / "generate_generations.jsonl"
    )

    rows: list[dict[str, Any]] = []
    rows.extend(group_text_rows(eval_task="rewrite", group="ai_source", rows=ai_controls, text_key="text"))
    rows.extend(group_text_rows(eval_task="rewrite", group="human_control", rows=human_controls, text_key="text"))
    rows.extend(
        group_text_rows(
            eval_task="rewrite",
            group="rewrite_output",
            rows=rewrite_generations,
            text_key="generated_text",
            require_result_tags=args.require_result_tags,
        )
    )
    rows.extend(group_text_rows(eval_task="generate", group="ai_control", rows=ai_controls, text_key="text"))
    rows.extend(group_text_rows(eval_task="generate", group="human_control", rows=human_controls, text_key="text"))
    rows.extend(
        group_text_rows(
            eval_task="generate",
            group="generate_output",
            rows=generate_generations,
            text_key="generated_text",
            require_result_tags=args.require_result_tags,
        )
    )
    return [row for row in rows if str(row.get("analysis_text") or "").strip()]


class _UnavailableTranslationese:
    def score(self, _text: str) -> float:
        return float("nan")


def score_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    anti_slop = AntiSlopDensityScorer(Path(args.anti_slop_lexicon))
    try:
        translationese = TranslationeseSVMScorer(Path(args.translationese_model))
    except BaseException as exc:
        print(f"[warn] translationese scorer unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        translationese = _UnavailableTranslationese()
    scored: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("analysis_text") or "")
        metrics, _surface = analyze_chunk(
            Chunk(
                group=str(row["group"]),
                source_file=str(row.get("source_file") or ""),
                chunk_id=str(row.get("chunk_id") or row.get("id") or ""),
                text=text,
            ),
            kiwi,
        )
        metrics["anti_slop_density"] = anti_slop.density(text)
        metrics["translationese_raw"] = translationese.score(text)
        out = {key: value for key, value in row.items() if key not in {"raw_text", "analysis_text"}}
        out["raw_chars"] = len(str(row.get("raw_text") or ""))
        out["analysis_chars"] = len(str(row.get("analysis_text") or ""))
        out.update(metrics)
        scored.append(out)
    add_composite_scores(scored)
    return scored


def metric_mean(rows: list[dict[str, Any]], *, eval_task: str, group: str, metric: str) -> float:
    values = [
        finite_float(row.get(metric))
        for row in rows
        if row.get("eval_task") == eval_task and row.get("group") == group
    ]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def add_composite_scores(rows: list[dict[str, Any]]) -> None:
    """Add per-sample composite AI-likeness axes.

    The axis is anchored by the task's control sets:
      0.0 = human_control mean, 1.0 = AI control/source mean.
    Values are clipped to [0, 1] for the report plot; the raw axis is also kept.
    """

    ai_group_by_task = {"rewrite": "ai_source", "generate": "ai_control"}
    for eval_task, ai_group in ai_group_by_task.items():
        baselines: dict[str, tuple[float, float]] = {}
        task_metrics = {
            metric
            for metrics in COMPOSITE_METRIC_GROUPS.values()
            for metric in metrics
        }
        for metric in task_metrics:
            human_mean = metric_mean(rows, eval_task=eval_task, group="human_control", metric=metric)
            ai_mean = metric_mean(rows, eval_task=eval_task, group=ai_group, metric=metric)
            if math.isfinite(human_mean) and math.isfinite(ai_mean) and abs(ai_mean - human_mean) > 1e-9:
                baselines[metric] = (human_mean, ai_mean)

        for row in rows:
            if row.get("eval_task") != eval_task:
                continue
            for group_name, metric_names in COMPOSITE_METRIC_GROUPS.items():
                raw_axes: list[float] = []
                clipped_axes: list[float] = []
                for metric in metric_names:
                    if metric not in baselines:
                        continue
                    value = finite_float(row.get(metric))
                    if not math.isfinite(value):
                        continue
                    human_mean, ai_mean = baselines[metric]
                    axis = (value - human_mean) / (ai_mean - human_mean)
                    raw_axes.append(axis)
                    clipped_axes.append(min(1.0, max(0.0, axis)))
                prefix = f"composite_{group_name}"
                row[f"{prefix}_metric_count"] = len(clipped_axes)
                row[f"{prefix}_ai_likeness"] = sum(clipped_axes) / len(clipped_axes) if clipped_axes else float("nan")
                row[f"{prefix}_ai_likeness_raw"] = sum(raw_axes) / len(raw_axes) if raw_axes else float("nan")


def numeric_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    excluded = {"sampled_offset"}
    for row in rows:
        for key, value in row.items():
            if key in excluded or key.startswith("contract_"):
                continue
            if key.startswith("composite_") and key.endswith("_metric_count"):
                continue
            if isinstance(value, bool):
                continue
            number = finite_float(value)
            if math.isfinite(number):
                names.add(key)
    priority = [name for name in DEFAULT_REPORT_METRICS + AUX_REPORT_METRICS + COMPOSITE_REPORT_METRICS if name in names]
    rest = sorted(name for name in names if name not in set(priority))
    return priority + rest


def summarize_by_group(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("eval_task") or ""), str(row.get("group") or ""))].append(row)
    summaries: list[dict[str, Any]] = []
    for (eval_task, group), items in sorted(grouped.items()):
        for metric in metrics:
            summary = summarize_numeric(item.get(metric) for item in items)
            if summary["count"] <= 0:
                continue
            summaries.append({"eval_task": eval_task, "group": group, "metric": metric, **summary})
    return summaries


def distribution_tests(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("eval_task") or ""), str(row.get("group") or ""))].append(row)
    pairs = {
        "rewrite": [("rewrite_output", "human_control"), ("rewrite_output", "ai_source"), ("ai_source", "human_control")],
        "generate": [("generate_output", "human_control"), ("generate_output", "ai_control"), ("ai_control", "human_control")],
    }
    out: list[dict[str, Any]] = []
    for eval_task, group_pairs in pairs.items():
        for left_group, right_group in group_pairs:
            left = grouped.get((eval_task, left_group), [])
            right = grouped.get((eval_task, right_group), [])
            if not left or not right:
                continue
            for metric in metrics:
                left_values = [finite_float(row.get(metric)) for row in left]
                right_values = [finite_float(row.get(metric)) for row in right]
                left_values = [value for value in left_values if math.isfinite(value)]
                right_values = [value for value in right_values if math.isfinite(value)]
                if len(left_values) < 2 or len(right_values) < 2:
                    continue
                out.append(
                    {
                        "eval_task": eval_task,
                        "metric": metric,
                        "left_group": left_group,
                        "right_group": right_group,
                        "left_n": len(left_values),
                        "right_n": len(right_values),
                        "left_mean": sum(left_values) / len(left_values),
                        "right_mean": sum(right_values) / len(right_values),
                        "mean_delta": sum(left_values) / len(left_values) - sum(right_values) / len(right_values),
                        "cliffs_delta_left_vs_right": cliffs_delta(left_values, right_values),
                    }
                )
    return out


def plot_metric(rows: list[dict[str, Any]], *, eval_task: str, metric: str, output_dir: Path) -> str | None:
    ordered_groups = {
        "rewrite": ["human_control", "ai_source", "rewrite_output"],
        "generate": ["human_control", "ai_control", "generate_output"],
    }[eval_task]
    data: list[list[float]] = []
    labels: list[str] = []
    for group in ordered_groups:
        values = [
            finite_float(row.get(metric))
            for row in rows
            if row.get("eval_task") == eval_task and row.get("group") == group
        ]
        values = [value for value in values if math.isfinite(value)]
        if values:
            data.append(values)
            labels.append(group)
    if len(data) < 2:
        return None

    try:
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return plot_metric_svg(data, labels, eval_task=eval_task, metric=metric, output_dir=output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{eval_task}_{safe_name(metric)}.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body in parts.get("bodies", []):
        body.set_alpha(0.45)
    ax.boxplot(data, widths=0.15, showfliers=False)
    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if metric.startswith("composite_") and metric.endswith("_ai_likeness_raw"):
        ax.axhline(0.0, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_ylabel("AI-likeness axis (0=human mean, 1=AI mean)")
    ax.set_title(f"{eval_task}: {metric}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def plot_metric_svg(
    data: list[list[float]],
    labels: list[str],
    *,
    eval_task: str,
    metric: str,
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{eval_task}_{safe_name(metric)}.svg"
    width = 820
    height = 520
    left = 80
    right = 30
    top = 58
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_values = [value for values in data for value in values]
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    def x(index: int) -> float:
        return left + (index + 0.5) * plot_w / len(data)

    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(
            w=width, h=height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="sans-serif" font-size="18" font-weight="700">{svg_escape(eval_task)}: {svg_escape(metric)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
    ]
    if metric.startswith("composite_") and metric.endswith("_ai_likeness_raw"):
        for ref_value, color, label in [(0.0, "#2ca02c", "human mean"), (1.0, "#d62728", "AI mean")]:
            if y_min <= ref_value <= y_max:
                yy = y(ref_value)
                lines.append(
                    f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.85"/>'
                )
                lines.append(
                    f'<text x="{left + plot_w - 4}" y="{yy - 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="{color}">{label}</text>'
                )

    for tick_index in range(5):
        value = y_min + (y_max - y_min) * tick_index / 4
        yy = y(value)
        lines.append(f'<line x1="{left - 5}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(
            f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444">{value:.4g}</text>'
        )

    for index, values in enumerate(data):
        xx = x(index)
        color = colors[index % len(colors)]
        ordered = sorted(values)
        q1 = percentile(ordered, 0.25)
        med = percentile(ordered, 0.50)
        q3 = percentile(ordered, 0.75)
        mean = sum(ordered) / len(ordered)
        low = min(ordered)
        high = max(ordered)
        box_w = min(90, plot_w / len(data) * 0.42)
        lines.append(f'<line x1="{xx:.2f}" y1="{y(low):.2f}" x2="{xx:.2f}" y2="{y(high):.2f}" stroke="{color}" stroke-width="2" opacity="0.75"/>')
        lines.append(
            f'<rect x="{xx - box_w / 2:.2f}" y="{y(q3):.2f}" width="{box_w:.2f}" height="{max(1.0, y(q1) - y(q3)):.2f}" fill="{color}" opacity="0.22" stroke="{color}" stroke-width="2"/>'
        )
        lines.append(f'<line x1="{xx - box_w / 2:.2f}" y1="{y(med):.2f}" x2="{xx + box_w / 2:.2f}" y2="{y(med):.2f}" stroke="#111" stroke-width="2"/>')
        lines.append(f'<circle cx="{xx:.2f}" cy="{y(mean):.2f}" r="4" fill="#111"/>')
        for point_index, value in enumerate(ordered):
            jitter = ((point_index * 37) % 29 - 14) / 14 * box_w * 0.36
            lines.append(f'<circle cx="{xx + jitter:.2f}" cy="{y(value):.2f}" r="2.2" fill="{color}" opacity="0.55"/>')
        lines.append(
            f'<text x="{xx:.2f}" y="{height - 54}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#222">{svg_escape(labels[index])}</text>'
        )
        lines.append(
            f'<text x="{xx:.2f}" y="{height - 36}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#666">n={len(values)}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def percentile(ordered_values: list[float], q: float) -> float:
    if not ordered_values:
        return float("nan")
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered_values[low]
    return ordered_values[low] * (high - position) + ordered_values[high] * (position - low)


def write_plots(rows: list[dict[str, Any]], metrics: list[str], output_dir: Path, max_plots: int) -> list[dict[str, str]]:
    selected = [metric for metric in COMPOSITE_REPORT_METRICS if metric in metrics]
    selected.extend(metric for metric in DEFAULT_REPORT_METRICS if metric in metrics and metric not in selected)
    selected.extend(metric for metric in metrics if metric not in selected and len(selected) < max_plots)
    selected = selected[:max_plots]
    plots: list[dict[str, str]] = []
    for eval_task in ("rewrite", "generate"):
        for metric in selected:
            path = plot_metric(rows, eval_task=eval_task, metric=metric, output_dir=output_dir)
            if path:
                plots.append({"eval_task": eval_task, "metric": metric, "path": path})
    return plots


def top_distribution_rows(rows: list[dict[str, Any]], eval_task: str, limit: int = 12) -> list[dict[str, Any]]:
    task_rows = [row for row in rows if row.get("eval_task") == eval_task]
    task_rows.sort(key=lambda row: abs(finite_float(row.get("cliffs_delta_left_vs_right"), 0.0)), reverse=True)
    return task_rows[:limit]


def write_report(
    path: Path,
    *,
    raw_csv: Path,
    summary_csv: Path,
    tests_csv: Path,
    plots: list[dict[str, str]],
    tests: list[dict[str, Any]],
    row_count: int,
) -> None:
    lines = [
        "# Phase-End GUI Metric Evaluation",
        "",
        f"Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"Rows scored: `{row_count}`",
        "",
        "## Outputs",
        "",
        f"- Raw metrics: `{raw_csv}`",
        f"- Group summaries: `{summary_csv}`",
        f"- Distribution comparisons: `{tests_csv}`",
        "",
        "## Composite AI-Likeness Axes",
        "",
        "Composite raw metrics are normalized per task with `human_control` mean as `0.0` and the AI control/source mean as `1.0`.",
        "The report plots use `*_ai_likeness_raw` so values below human mean or above AI mean remain visible.",
        "The clipped `*_ai_likeness` columns are still written to CSV for bounded summaries, but they are not the default plots.",
        "",
        "| composite | included metric family |",
        "|---|---|",
    ]
    for name, metrics in COMPOSITE_METRIC_GROUPS.items():
        lines.append(f"| `composite_{name}_ai_likeness` | {', '.join(f'`{metric}`' for metric in metrics)} |")
    lines.extend(
        [
            "",
        "## Strongest Distribution Differences",
        "",
        ]
    )
    for eval_task in ("rewrite", "generate"):
        lines.append(f"### {eval_task}")
        lines.append("")
        rows = top_distribution_rows(tests, eval_task)
        if not rows:
            lines.append("No comparison rows.")
            lines.append("")
            continue
        lines.append("| metric | left | right | mean delta | Cliff's delta |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                "| {metric} | {left_group} | {right_group} | {mean_delta:.6g} | {delta:.6g} |".format(
                    metric=row["metric"],
                    left_group=row["left_group"],
                    right_group=row["right_group"],
                    mean_delta=finite_float(row.get("mean_delta"), 0.0),
                    delta=finite_float(row.get("cliffs_delta_left_vs_right"), 0.0),
                )
            )
        lines.append("")

    lines.extend(["## Plots", ""])
    for plot in plots:
        rel = Path(plot["path"])
        try:
            rel = rel.relative_to(path.parent)
        except ValueError:
            pass
        lines.append(f"- `{plot['eval_task']}` `{plot['metric']}`: ![]({rel.as_posix()})")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[...truncated...]"


def write_sample_texts(path: Path, rows: list[dict[str, Any]], *, max_per_group: int, max_chars: int) -> None:
    lines = [
        "# Phase-End Evaluation Sample Texts",
        "",
        "This file stores readable samples for manual inspection. Full generated text is also preserved in the generation JSONL files.",
        "",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("eval_task") or ""), str(row.get("group") or ""))].append(row)

    order = [
        ("rewrite", "rewrite_output"),
        ("rewrite", "ai_source"),
        ("rewrite", "human_control"),
        ("generate", "generate_output"),
        ("generate", "ai_control"),
        ("generate", "human_control"),
    ]
    for eval_task, group in order:
        items = grouped.get((eval_task, group), [])
        if not items:
            continue
        lines.extend([f"## {eval_task} / {group}", ""])
        for index, row in enumerate(items[:max_per_group], start=1):
            text = truncate_text(str(row.get("analysis_text") or row.get("raw_text") or ""), max_chars)
            lines.extend(
                [
                    f"### {index}. `{row.get('id', '')}`",
                    "",
                    f"- source_file: `{row.get('source_file', '')}`",
                    f"- chunk_id: `{row.get('chunk_id', '')}`",
                    f"- chars: `{len(str(row.get('analysis_text') or ''))}`",
                    "",
                    "```text",
                    text,
                    "```",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score phase-end eval outputs with GUI-compatible raw metrics.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--generation-dir", default=str(DEFAULT_GENERATION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ai-source-controls", default="")
    parser.add_argument("--human-controls", default="")
    parser.add_argument("--rewrite-generations", default="")
    parser.add_argument("--generate-generations", default="")
    parser.add_argument(
        "--require-result-tags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Analyze generated text as a <result>...</result> contract output. "
            "Keep disabled for eval prompts that only ask for plain novel body text."
        ),
    )
    parser.add_argument(
        "--require-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if no plot images are produced. Use --no-require-plots for CSV-only smoke tests.",
    )
    parser.add_argument("--max-plots", type=int, default=24)
    parser.add_argument("--sample-text-max-per-group", type=int, default=5)
    parser.add_argument("--sample-text-chars", type=int, default=6000)
    parser.add_argument("--anti-slop-lexicon", default=str(DEFAULT_ANTI_SLOP_LEXICON))
    parser.add_argument("--translationese-model", default=str(DEFAULT_TRANSLATIONESE_MODEL))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    rows = score_rows(build_analysis_rows(args), args)
    if not rows:
        raise ValueError("No rows to score.")
    metrics = numeric_metric_names(rows)
    summaries = summarize_by_group(rows, metrics)
    tests = distribution_tests(rows, metrics)
    plots = write_plots(rows, metrics, plots_dir, args.max_plots)

    raw_csv = output_dir / "raw_metrics_by_sample.csv"
    summary_csv = output_dir / "metric_summary_by_group.csv"
    tests_csv = output_dir / "distribution_tests.csv"
    report_md = output_dir / "report.md"
    samples_md = output_dir / "sample_texts.md"
    summary_json = output_dir / "summary.json"
    write_csv(raw_csv, rows)
    write_csv(summary_csv, summaries)
    write_csv(tests_csv, tests)
    write_report(report_md, raw_csv=raw_csv, summary_csv=summary_csv, tests_csv=tests_csv, plots=plots, tests=tests, row_count=len(rows))
    write_sample_texts(samples_md, build_analysis_rows(args), max_per_group=args.sample_text_max_per_group, max_chars=args.sample_text_chars)
    write_json(
        summary_json,
        {
            "time": time.time(),
            "rows": len(rows),
            "metric_count": len(metrics),
            "plot_count": len(plots),
            "outputs": {
                "raw_metrics": str(raw_csv),
                "summary": str(summary_csv),
                "distribution_tests": str(tests_csv),
                "report": str(report_md),
                "sample_texts": str(samples_md),
                "plots": [plot["path"] for plot in plots],
            },
            "args": vars(args),
        },
    )
    print(json.dumps(json.loads(summary_json.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    if args.require_plots and not plots:
        raise RuntimeError(
            "No plot images were produced. Install matplotlib in the evaluation environment, "
            "or rerun with --no-require-plots for CSV/MD-only output."
        )


if __name__ == "__main__":
    main()
