#!/usr/bin/env python3
"""Prepare curated SimPO curriculum splits.

Standalone by design: no imports from existing project scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]

DEFAULT_INPUT = TRAINING_ROOT / "data" / "processed" / "dpo_train.jsonl"
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "data" / "pipeline_v2" / "simpo_curriculum"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def message_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(item.get("content") or "") for item in value if isinstance(item, dict))
    return str(value or "")


def hangul_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    return sum(1 for char in chars if "\uac00" <= char <= "\ud7a3") / len(chars)


def bad_char_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 1.0
    bad = sum(1 for char in chars if char == "\ufffd" or ord(char) < 32)
    return bad / len(chars)


def compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def stable_hash(text: str, length: int = 16) -> str:
    compact = re.sub(r"\s+", "", text)
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()[:length]


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen_hashes: dict[str, int] = {}
    chunk_ids: dict[str, int] = {}
    for row in rows:
        chosen_hash = stable_hash(message_text(row.get("chosen")))
        chosen_hashes[chosen_hash] = chosen_hashes.get(chosen_hash, 0) + 1
        chunk_id = str(row.get("chunk_id") or row.get("metadata", {}).get("chunk_id") or "")
        if chunk_id:
            chunk_ids[chunk_id] = chunk_ids.get(chunk_id, 0) + 1
    duplicate_chosen_groups = sum(1 for count in chosen_hashes.values() if count > 1)
    duplicate_chosen_rows = sum(count for count in chosen_hashes.values() if count > 1)
    return {
        "rows": len(rows),
        "unique_chosen_texts": len(chosen_hashes),
        "duplicate_chosen_groups": duplicate_chosen_groups,
        "duplicate_chosen_rows": duplicate_chosen_rows,
        "unique_chunk_ids": len(chunk_ids),
        "top_chunk_repeats": sorted(chunk_ids.values(), reverse=True)[:5],
    }


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    chosen = message_text(row.get("chosen"))
    rejected = message_text(row.get("rejected"))
    if compact_len(chosen) < args.min_chosen_chars:
        return False, "chosen_too_short"
    if hangul_ratio(chosen) < args.min_chosen_hangul_ratio:
        return False, "chosen_low_hangul"
    if bad_char_ratio(chosen) > args.max_bad_char_ratio:
        return False, "chosen_bad_chars"
    if rejected and compact_len(rejected) > 0:
        ratio = len(chosen) / max(1, len(rejected))
        if ratio < args.min_chosen_rejected_len_ratio:
            return False, "chosen_much_shorter"
        if ratio > args.max_chosen_rejected_len_ratio:
            return False, "chosen_much_longer"
    return True, "kept"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SimPO curriculum splits from DPO JSONL.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-chosen-chars", type=int, default=800)
    parser.add_argument("--min-chosen-hangul-ratio", type=float, default=0.55)
    parser.add_argument("--max-bad-char-ratio", type=float, default=0.01)
    parser.add_argument("--min-chosen-rejected-len-ratio", type=float, default=0.45)
    parser.add_argument("--max-chosen-rejected-len-ratio", type=float, default=1.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict[str, Any]]] = {
        "01_generate_format": [],
        "02_nochange_anti_copy": [],
        "03_badstyle_rewrite": [],
        "curated_mixed": [],
    }
    dropped: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}

    for row in rows:
        bucket = str(row.get("bucket") or "")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        keep, reason = keep_row(row, args)
        if not keep:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        curated = dict(row)
        curated.setdefault("metadata", {})
        if isinstance(curated["metadata"], dict):
            curated["metadata"] = {**curated["metadata"], "pipeline_v2_curated": True}
        if bucket == "generate_human_vs_ai":
            splits["01_generate_format"].append(curated)
        elif bucket == "nochange_rejected":
            splits["02_nochange_anti_copy"].append(curated)
        else:
            splits["03_badstyle_rewrite"].append(curated)
        splits["curated_mixed"].append(curated)

    outputs = {}
    for name, split_rows in splits.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, split_rows)
        outputs[name] = {"path": str(path), **split_stats(split_rows)}

    manifest = {
        "time": time.time(),
        "input": str(args.input),
        "output_dir": str(output_dir),
        "input_rows": len(rows),
        "bucket_counts": bucket_counts,
        "dropped": dropped,
        "outputs": outputs,
        "args": vars(args),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
