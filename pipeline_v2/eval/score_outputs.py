#!/usr/bin/env python3
"""Score generated outputs from pipeline_v2 fixed-set evaluation.

Input rows may contain `generated_text`, `completion`, `text`, or
`raw_completion`. Output rows include flattened contract/style/rewrite metrics
and a compact summary JSON is written next to the scored JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parents[1]
TRAINING_ROOT = PIPELINE_ROOT.parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gui_style_scoring import GuiStyleScorer, finite_float, summarize_numeric
from pipeline_v2.lib.io import iter_jsonl, write_json, write_jsonl


DEFAULT_INPUT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "fixed_eval" / "generations.jsonl"
DEFAULT_OUTPUT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "fixed_eval" / "scored_outputs.jsonl"


def generated_text(row: dict[str, Any]) -> str:
    for key in ("generated_text", "raw_completion", "completion", "text", "output"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(str(item.get("content", "")))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            return str(value.get("content", ""))
        return str(value)
    return ""


def prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return "\n".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in prompt)
    return str(prompt or "")


def require_result_tags_for_row(row: dict[str, Any], policy: str, legacy_flag: bool) -> bool:
    if policy == "always":
        return True
    if policy == "never":
        return False
    if not legacy_flag:
        return False
    task = str(row.get("task") or "")
    return task in {"format_stop", "continuation"} or "<result>" in prompt_text(row) or "</result>" in prompt_text(row)


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), nested, out)
        return
    out[prefix] = value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_scored(rows: list[dict[str, Any]], *, input_path: Path, output_path: Path) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {}
    reason_counts: Counter[str] = Counter()
    collapse_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for row in rows:
        task_counts[str(row.get("task") or "")] += 1
        label_counts[str(row.get("model_label") or "")] += 1
        reason = str(row.get("metrics.result_contract_reason") or "")
        collapse = str(row.get("metrics.collapse_reason") or "")
        if reason:
            reason_counts[reason] += 1
        if collapse:
            collapse_counts[collapse] += 1
        for key, value in row.items():
            if key.startswith(("metrics.", "metric_scores.", "pos_usage_scores.")) or key == "score":
                number = finite_float(value)
                if math.isfinite(number):
                    metric_values.setdefault(key, []).append(number)
    return {
        "time": time.time(),
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "task_counts": dict(task_counts.most_common()),
        "model_label_counts": dict(label_counts.most_common()),
        "result_contract_reason_counts": dict(reason_counts.most_common()),
        "collapse_reason_counts": dict(collapse_counts.most_common()),
        "metric_summaries": {key: summarize_numeric(values) for key, values in sorted(metric_values.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score pipeline_v2 generated outputs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--anti-slop-lexicon", default="")
    parser.add_argument("--translationese-model", default="")
    parser.add_argument(
        "--result-tag-policy",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto requires tags only when the prompt/task asks for them.",
    )
    parser.add_argument("--require-result-tags", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    csv_path = Path(args.csv) if args.csv else output_path.with_suffix(".csv")

    scorer_kwargs: dict[str, Path] = {}
    if args.reference:
        scorer_kwargs["reference_path"] = Path(args.reference)
    if args.anti_slop_lexicon:
        scorer_kwargs["anti_slop_lexicon_path"] = Path(args.anti_slop_lexicon)
    if args.translationese_model:
        scorer_kwargs["translationese_model_path"] = Path(args.translationese_model)
    scorer = GuiStyleScorer(**scorer_kwargs)

    scored_rows: list[dict[str, Any]] = []
    for row in iter_jsonl(input_path):
        raw_text = generated_text(row)
        task = str(row.get("task") or "")
        require_tags = require_result_tags_for_row(row, args.result_tag_policy, args.require_result_tags)
        scored = scorer.score_text(
            raw_text,
            task=task,
            source_text=str(row.get("source_text") or ""),
            require_result_tags=require_tags,
        )
        out = {
            "id": row.get("id", ""),
            "task": task,
            "model_label": row.get("model_label", ""),
            "adapter_path": row.get("adapter_path", ""),
            "score": scored["score"],
            "generated_text": raw_text,
            "result_text": scored["result_text"],
            "source_text": row.get("source_text", ""),
            "reference_text": row.get("reference_text", ""),
            "require_result_tags": require_tags,
        }
        flatten("metrics", scored.get("metrics") or {}, out)
        flatten("metric_scores", scored.get("metric_scores") or {}, out)
        flatten("pos_usage_scores", scored.get("pos_usage_scores") or {}, out)
        flatten("scorer_status", scored.get("scorer_status") or {}, out)
        scored_rows.append(out)

    write_jsonl(output_path, scored_rows)
    write_csv(csv_path, scored_rows)
    summary = summarize_scored(scored_rows, input_path=input_path, output_path=output_path)
    summary["csv"] = str(csv_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
