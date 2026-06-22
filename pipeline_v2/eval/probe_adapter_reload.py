#!/usr/bin/env python3
"""Reload a saved adapter in a fresh process and generate a few probe outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parents[1]
TRAINING_ROOT = PIPELINE_ROOT.parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.eval.run_fixed_eval import (
    DEFAULT_EVAL_FILES,
    generate_one,
    generation_stop_token_ids,
    load_eval_rows,
    token_ids_for_text,
)
from pipeline_v2.lib.gemma4_loader import DEFAULT_BASE_MODEL, base_model_from_adapter, load_gemma4_model_and_processor
from pipeline_v2.lib.io import write_json, write_jsonl
from pipeline_v2.lib.lora import trainable_parameter_summary
from pipeline_v2.lib.result_contract import RESULT_CLOSE_TAG
from pipeline_v2.lib.trainer_utils import print_json, set_seed


DEFAULT_OUTPUT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "reload_probe" / "adapter_reload_probe.jsonl"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model != "auto":
        return args.model
    return base_model_from_adapter(args.adapter_path, DEFAULT_BASE_MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether a pipeline_v2 adapter reloads and generates.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--model-label", default="reload_probe")
    parser.add_argument("--eval-file", action="append", default=[])
    parser.add_argument("--limit-per-dataset", type=int, default=1)
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--max-prompt-length", type=positive_int, default=4096)
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--stop-at-result-close", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-at-turn-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_paths = [Path(path) for path in args.eval_file] if args.eval_file else list(DEFAULT_EVAL_FILES)
    rows = load_eval_rows(eval_paths, args.limit_per_dataset)
    model_name = resolve_model_name(args)

    print_json("[reload/load]", {"model_name": model_name, "adapter_path": str(adapter_path), "rows": len(rows)})
    model, processor, metadata = load_gemma4_model_and_processor(
        model_name=model_name,
        adapter_path=adapter_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        chat_template=args.chat_template,
        create_lora=False,
    )
    model.eval()
    result_close_ids = token_ids_for_text(processor, RESULT_CLOSE_TAG)
    stop_ids_by_text = generation_stop_token_ids(processor, args, result_close_ids)
    generated_rows: list[dict[str, Any]] = []
    for index, (path, row) in enumerate(rows, start=1):
        print_json("[reload/item]", {"index": index, "total": len(rows), "id": row.get("id")})
        generated = generate_one(model=model, processor=processor, row=row, args=args, stop_ids_by_text=stop_ids_by_text)
        generated["dataset_path"] = str(path)
        generated_rows.append(generated)

    write_jsonl(output_path, generated_rows)
    summary = {
        "time": time.time(),
        "ok": bool(generated_rows),
        "adapter_path": str(adapter_path),
        "output": str(output_path),
        "rows": len(generated_rows),
        "loader": metadata,
        "trainable": trainable_parameter_summary(model),
        "result_close_token_ids": result_close_ids,
        "generation_stop_token_ids": stop_ids_by_text,
        "args": vars(args),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
