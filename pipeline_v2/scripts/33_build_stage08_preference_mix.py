#!/usr/bin/env python3
"""Build a generate-heavy Stage 8 preference mix.

The current Stage 8 plan uses SimPO-CPO as a short preference re-anchor before
another mixed GRPO pass. This builder keeps all generate human-vs-AI rows,
excludes nochange rows, and takes a deterministic fraction of badstyle rows so
the preference step does not become mostly "avoid bad style" training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
DEFAULT_INPUT = TRAINING_ROOT / "data" / "pipeline_v2" / "simpo_curriculum" / "curated_mixed.jsonl"
DEFAULT_OUTPUT = TRAINING_ROOT / "data" / "pipeline_v2" / "simpo_curriculum" / "stage08_generate_heavy.jsonl"


def stable_score(row: dict[str, Any], seed: int) -> str:
    raw = f"{seed}|{row.get('id','')}|{row.get('bucket','')}|{row.get('prompt','')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 8 generate-heavy SimPO-CPO data.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--badstyle-fraction", type=float, default=0.25)
    parser.add_argument("--include-rewrite-bucket", action="store_true")
    parser.add_argument("--seed", type=int, default=4808)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    generate = [row for row in rows if str(row.get("bucket") or "") == "generate_human_vs_ai"]
    badstyle = [row for row in rows if str(row.get("bucket") or "") == "badstyle_rejected"]
    rewrite = [
        row
        for row in rows
        if args.include_rewrite_bucket and str(row.get("bucket") or "").startswith("rewrite")
    ]
    badstyle = sorted(badstyle, key=lambda row: stable_score(row, args.seed))
    keep_badstyle = int(round(len(badstyle) * max(0.0, min(1.0, args.badstyle_fraction))))
    selected = sorted(generate, key=lambda row: stable_score(row, args.seed)) + badstyle[:keep_badstyle] + rewrite
    selected = sorted(selected, key=lambda row: stable_score(row, args.seed + 17))
    write_jsonl(Path(args.output), selected)
    manifest = {
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "seed": args.seed,
        "badstyle_fraction": args.badstyle_fraction,
        "counts": {
            "input_total": len(rows),
            "generate_human_vs_ai": len(generate),
            "badstyle_rejected_input": len(badstyle),
            "badstyle_rejected_selected": keep_badstyle,
            "rewrite_selected": len(rewrite),
            "output_total": len(selected),
        },
    }
    Path(args.output).with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
