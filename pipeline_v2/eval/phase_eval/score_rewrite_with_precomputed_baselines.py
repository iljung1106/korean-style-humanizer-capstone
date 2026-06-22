#!/usr/bin/env python3
"""Score rewrite generations and combine with precomputed baseline metrics.

Use this when human baseline text should not be uploaded to the training
instance. The baseline CSV is produced elsewhere and contains already-scored
`ai_source` and `human_control` rows. This script scores only model rewrite
outputs on the instance, appends them to the baseline rows, then writes the
same summary/report/plot artifacts as `score_and_report.py`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[3]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.eval.phase_eval.score_and_report import (  # noqa: E402
    DEFAULT_ANTI_SLOP_LEXICON,
    DEFAULT_GENERATION_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSLATIONESE_MODEL,
    group_text_rows,
    numeric_metric_names,
    read_jsonl,
    score_rows,
    summarize_by_group,
    distribution_tests,
    write_csv,
    write_plots,
    write_report,
    write_sample_texts,
    write_json,
)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score rewrite outputs with precomputed AI/human baseline metrics.")
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--generation-dir", default=str(DEFAULT_GENERATION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rewrite-generations", default="")
    parser.add_argument(
        "--require-result-tags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Analyze generated text as a <result>...</result> contract output.",
    )
    parser.add_argument("--max-plots", type=int, default=64)
    parser.add_argument("--sample-text-max-per-group", type=int, default=12)
    parser.add_argument("--sample-text-chars", type=int, default=9000)
    parser.add_argument("--anti-slop-lexicon", default=str(DEFAULT_ANTI_SLOP_LEXICON))
    parser.add_argument("--translationese-model", default=str(DEFAULT_TRANSLATIONESE_MODEL))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    generation_dir = Path(args.generation_dir)
    rewrite_generations = read_jsonl(
        Path(args.rewrite_generations) if args.rewrite_generations else generation_dir / "rewrite_generations.jsonl"
    )

    baseline_rows = read_csv_rows(Path(args.baseline_metrics))
    rewrite_text_rows = group_text_rows(
        eval_task="rewrite",
        group="rewrite_output",
        rows=rewrite_generations,
        text_key="generated_text",
        require_result_tags=args.require_result_tags,
    )
    rewrite_metric_rows = score_rows(rewrite_text_rows, args)
    rows = baseline_rows + rewrite_metric_rows
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
    write_sample_texts(samples_md, rows, max_per_group=args.sample_text_max_per_group, max_chars=args.sample_text_chars)
    write_json(
        summary_json,
        {
            "rows": len(rows),
            "baseline_rows": len(baseline_rows),
            "rewrite_rows": len(rewrite_metric_rows),
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
        },
    )


if __name__ == "__main__":
    main()
