#!/usr/bin/env python3
"""Build a rewrite GRPO dataset with source text capped by tokenizer tokens.

The original GRPO rewrite rows can carry long source chunks. For GRPO this
inflates prompt+completion length and the lm_head logprob pass can OOM. This
script keeps rewrite rows only, trims source_text on paragraph/sentence
boundaries, and replaces the same source inside the user prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gemma4_loader import patch_gemma4_num_kv_shared_layers_for_config_validation


DEFAULT_INPUT = TRAINING_ROOT / "data" / "processed" / "grpo_mixed_prompts.jsonl"
DEFAULT_OUTPUT = TRAINING_ROOT / "data" / "processed" / "grpo_rewrite_source1536.jsonl"
DEFAULT_MODEL = "unsloth/gemma-4-31B-it"


SENTENCE_END_CHARS = set(".!?。！？")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def selected_task(row: dict[str, Any]) -> str:
    return str(row.get("task") or row.get("grpo_task") or "").strip()


def clean_text(text: str) -> str:
    text = str(text).replace("\ufeff", "")
    text = re.sub(r"[\u200b\u200c\u200d\u2060]", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraph_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    for paragraph in re.split(r"\n{2,}", clean_text(text)):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        start = 0
        for index, char in enumerate(paragraph):
            if char not in SENTENCE_END_CHARS:
                continue
            next_index = index + 1
            if next_index < len(paragraph) and not paragraph[next_index].isspace():
                continue
            piece = paragraph[start:next_index].strip()
            if piece:
                pieces.append(piece)
            start = next_index
        tail = paragraph[start:].strip()
        if tail:
            pieces.append(tail)
        if pieces:
            pieces[-1] = pieces[-1] + "\n\n"
    if pieces:
        pieces[-1] = pieces[-1].rstrip()
    return pieces


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def trim_to_token_limit(tokenizer: Any, text: str, max_tokens: int, min_tokens: int) -> str:
    text = clean_text(text)
    if token_count(tokenizer, text) <= max_tokens:
        return text

    pieces = split_paragraph_sentences(text)
    if not pieces:
        ids = tokenizer(text, add_special_tokens=False).input_ids[:max_tokens]
        return clean_text(tokenizer.decode(ids, skip_special_tokens=True))

    kept: list[str] = []
    best = ""
    for piece in pieces:
        candidate = clean_text("".join(kept + [piece]))
        count = token_count(tokenizer, candidate)
        if count <= max_tokens:
            kept.append(piece)
            best = candidate
            continue
        break

    if best and token_count(tokenizer, best) >= min_tokens:
        return best

    # If sentence boundaries are too coarse, fall back to a token slice.
    ids = tokenizer(text, add_special_tokens=False).input_ids[:max_tokens]
    return clean_text(tokenizer.decode(ids, skip_special_tokens=True))


def replace_prompt_source(prompt: Any, old_source: str, new_source: str) -> Any:
    if not isinstance(prompt, list):
        return prompt
    updated: list[dict[str, Any]] = []
    old_source = clean_text(old_source)
    for message in prompt:
        if not isinstance(message, dict):
            updated.append(message)
            continue
        new_message = dict(message)
        content = str(new_message.get("content") or "")
        if old_source and old_source in content:
            content = content.replace(old_source, new_source, 1)
        else:
            # Most rows use two blank lines between instruction and source.
            parts = content.rsplit("\n\n", 1)
            if len(parts) == 2 and len(parts[1]) > 300:
                content = parts[0].rstrip() + "\n\n" + new_source
        new_message["content"] = content
        updated.append(new_message)
    return updated


def load_tokenizer(model_name: str) -> Any:
    patch_gemma4_num_kv_shared_layers_for_config_validation()
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return getattr(processor, "tokenizer", processor)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-max-tokens", type=int, default=1536)
    parser.add_argument("--source-min-tokens", type=int, default=256)
    parser.add_argument("--limit-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    tokenizer = load_tokenizer(args.model)

    output_rows: list[dict[str, Any]] = []
    stats = {
        "input_rows": 0,
        "rewrite_rows": 0,
        "output_rows": 0,
        "trimmed_rows": 0,
        "source_max_tokens": args.source_max_tokens,
        "source_min_tokens": args.source_min_tokens,
        "source_tokens_before_max": 0,
        "source_tokens_after_max": 0,
    }
    before_counts: list[int] = []
    after_counts: list[int] = []

    for row in read_jsonl(input_path):
        stats["input_rows"] += 1
        if selected_task(row) != "rewrite":
            continue
        stats["rewrite_rows"] += 1
        source = clean_text(str(row.get("source_text") or ""))
        if not source:
            continue
        before = token_count(tokenizer, source)
        trimmed = trim_to_token_limit(tokenizer, source, args.source_max_tokens, args.source_min_tokens)
        after = token_count(tokenizer, trimmed)
        new_row = dict(row)
        new_row["source_text"] = trimmed
        new_row["prompt"] = replace_prompt_source(row.get("prompt"), source, trimmed)
        new_row["short_source_metadata"] = {
            "source_tokens_before": before,
            "source_tokens_after": after,
            "source_chars_before": len(source),
            "source_chars_after": len(trimmed),
            "source_max_tokens": args.source_max_tokens,
        }
        if after < before:
            stats["trimmed_rows"] += 1
        before_counts.append(before)
        after_counts.append(after)
        output_rows.append(new_row)
        if args.limit_rows > 0 and len(output_rows) >= args.limit_rows:
            break

    write_jsonl(output_path, output_rows)
    stats["output_rows"] = len(output_rows)
    stats["source_tokens_before_max"] = max(before_counts) if before_counts else 0
    stats["source_tokens_after_max"] = max(after_counts) if after_counts else 0
    stats["source_tokens_before_mean"] = round(sum(before_counts) / len(before_counts), 3) if before_counts else 0.0
    stats["source_tokens_after_mean"] = round(sum(after_counts) / len(after_counts), 3) if after_counts else 0.0
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
