#!/usr/bin/env python3
"""Probe pipeline_v2 Stage 1 loss masks.

Use `--mock-tokenizer` for fast local validation without ML dependencies. Use
the real tokenizer on the training machine before launching any training run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parents[1]
TRAINING_ROOT = PIPELINE_ROOT.parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gemma4_loader import DEFAULT_BASE_MODEL, maybe_apply_chat_template
from pipeline_v2.lib.io import read_jsonl, row_type_counts
from pipeline_v2.lib.masking import IGNORE_INDEX, MockGemma4Processor, decode_visible_labels, prepare_row


DEFAULT_DATASET = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_mixed_probe.jsonl"


def load_processor(args: argparse.Namespace) -> Any:
    if args.mock_tokenizer:
        return MockGemma4Processor()
    try:
        from pipeline_v2.lib.gemma4_loader import patch_gemma4_num_kv_shared_layers_for_config_validation

        patch_gemma4_num_kv_shared_layers_for_config_validation()
        from transformers import AutoProcessor, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Real tokenizer probe requires transformers/unsloth dependencies. "
            "Use --mock-tokenizer for local structural checks."
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    return maybe_apply_chat_template(processor, args.chat_template)


def select_rows(rows: list[dict[str, Any]], samples_per_type: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    known_types = ("raw_lm", "continuation_sft", "format_sft", "general_guard")
    required = [item for item in known_types if any(str(row.get("row_type") or "") == item for row in rows)]
    for row in rows:
        row_type = str(row.get("row_type") or "")
        if row_type not in known_types:
            continue
        count = seen.get(row_type, 0)
        if count >= samples_per_type:
            continue
        selected.append(row)
        seen[row_type] = count + 1
        if all(seen.get(item, 0) >= samples_per_type for item in required):
            break
    return selected


def evaluate_example(row: dict[str, Any], processor: Any, max_seq_length: int, preview_chars: int) -> dict[str, Any]:
    example = prepare_row(row, processor, max_seq_length)
    visible_text = decode_visible_labels(processor, example)
    prompt_labels_ok = all(label == IGNORE_INDEX for label in example.labels[: example.prompt_token_count])
    trailing_after_result = ""
    if "</result>" in visible_text:
        trailing_after_result = visible_text.split("</result>", 1)[1]
    normalized_trailing = (
        trailing_after_result.replace("<eos>", "")
        .replace("<end_of_turn>", "")
        .replace("<|end_of_turn|>", "")
        .strip()
    )
    checks = {
        "has_visible_labels": example.visible_label_count > 0,
        "raw_full_visible": True,
        "prompt_labels_masked": prompt_labels_ok,
        "has_result_open": True,
        "has_result_close": True,
        "post_result_text_empty": True,
    }
    if example.row_type == "raw_lm":
        checks["raw_full_visible"] = example.visible_label_count == example.total_token_count
    else:
        checks["raw_full_visible"] = True
        checks["has_result_open"] = "<result>" in visible_text
        checks["has_result_close"] = "</result>" in visible_text
        checks["post_result_text_empty"] = normalized_trailing == ""

    return {
        **example.summary(),
        "checks": checks,
        "visible_preview": visible_text[:preview_chars],
        "visible_tail": visible_text[-preview_chars:],
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.dataset)
    processor = load_processor(args)
    selected = select_rows(rows, args.samples_per_type)
    examples = [
        evaluate_example(row, processor, args.max_seq_length, args.preview_chars)
        for row in selected
    ]
    ok = bool(examples) and all(
        all(bool(value) for value in item["checks"].values())
        for item in examples
    )
    return {
        "ok": ok,
        "dataset": str(args.dataset),
        "dataset_rows": len(rows),
        "dataset_row_types": row_type_counts(rows),
        "selected": len(selected),
        "processor_class": processor.__class__.__name__,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe pipeline_v2 Stage 1 loss masks.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--samples-per-type", type=int, default=2)
    parser.add_argument("--preview-chars", type=int, default=240)
    parser.add_argument("--mock-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
