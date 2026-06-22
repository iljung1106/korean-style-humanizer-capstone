#!/usr/bin/env python3
"""Generate outputs for pipeline_v2 fixed eval sets.

This script is intended for the paid-instance pilot run. It loads the selected
base model plus optional adapter once, iterates over fixed prompt JSONL files,
and writes a single generations JSONL that can be consumed by
`eval/score_outputs.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401,E402


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parents[1]
TRAINING_ROOT = PIPELINE_ROOT.parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gemma4_loader import DEFAULT_BASE_MODEL, base_model_from_adapter, load_gemma4_model_and_processor
from pipeline_v2.lib.io import read_jsonl, write_json, write_jsonl
from pipeline_v2.lib.masking import processor_tokenizer, render_chat
from pipeline_v2.lib.preference_data import (
    add_continuation_style_guidance_instruction,
    add_generate_length_instruction,
    add_generate_style_guidance_instruction,
    add_result_contract_instruction,
    add_rewrite_style_guidance_instruction,
)
from pipeline_v2.lib.result_contract import CHAT_STOP_TOKENS, RESULT_CLOSE_TAG, trim_after_first_stop_text
from pipeline_v2.lib.trainer_utils import print_json, set_seed


DEFAULT_EVAL_FILES = [
    TRAINING_ROOT / "data" / "eval_v2" / "fixed_generate_prompts.jsonl",
    TRAINING_ROOT / "data" / "eval_v2" / "fixed_rewrite_prompts.jsonl",
    TRAINING_ROOT / "data" / "eval_v2" / "fixed_continuation_prompts.jsonl",
    TRAINING_ROOT / "data" / "eval_v2" / "fixed_format_stop_prompts.jsonl",
]
DEFAULT_OUTPUT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "fixed_eval" / "generations.jsonl"


class StopOnAnyTokenSequence:
    def __init__(self, stop_sequences: list[list[int]], prompt_length: int) -> None:
        self.stop_sequences = [list(sequence) for sequence in stop_sequences if sequence]
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids: Any, _scores: Any, **_: Any) -> bool:
        if not self.stop_sequences:
            return False
        try:
            sequence = input_ids[0].tolist()
        except Exception:
            sequence = list(input_ids[0])
        generated = sequence[self.prompt_length :]
        if not generated:
            return False
        return any(len(generated) >= len(stop_ids) and generated[-len(stop_ids) :] == stop_ids for stop_ids in self.stop_sequences)


class ForceEosAfterAnyTokenSequence:
    def __init__(self, stop_sequences: list[list[int]], prompt_length: int, eos_token_id: int | None) -> None:
        self.stop_sequences = [list(sequence) for sequence in stop_sequences if sequence]
        self.prompt_length = int(prompt_length)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if not self.stop_sequences or self.eos_token_id is None:
            return scores
        try:
            sequence = input_ids[0].tolist()
        except Exception:
            sequence = list(input_ids[0])
        generated = sequence[self.prompt_length :]
        if any(len(generated) >= len(stop_ids) and generated[-len(stop_ids) :] == stop_ids for stop_ids in self.stop_sequences):
            scores[0, :] = -float("inf")
            scores[0, self.eos_token_id] = 0.0
        return scores


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model != "auto":
        return args.model
    if args.adapter_path:
        return base_model_from_adapter(args.adapter_path, DEFAULT_BASE_MODEL)
    return DEFAULT_BASE_MODEL


def load_eval_rows(paths: list[Path], limit_per_dataset: int) -> list[tuple[Path, dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        rows = read_jsonl(path)
        if limit_per_dataset > 0:
            rows = rows[:limit_per_dataset]
        loaded.extend((path, row) for row in rows)
    return loaded


def encode_prompt(processor: Any, prompt_text: str, max_prompt_length: int) -> dict[str, Any]:
    try:
        encoded = processor(
            text=prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
    except TypeError:
        tokenizer = processor_tokenizer(processor)
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
    return dict(encoded)


def move_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def grpo_compatible_messages(messages: Any, task: str, args: argparse.Namespace) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("eval row has invalid prompt messages")
    prepared = add_result_contract_instruction(messages)
    if task == "generate":
        prepared = add_generate_length_instruction(
            prepared,
            args.generate_max_output_chars,
            args.generate_min_output_chars,
        )
        prepared = add_generate_style_guidance_instruction(prepared)
    elif task == "rewrite":
        prepared = add_rewrite_style_guidance_instruction(prepared)
    elif task == "continuation":
        prepared = add_continuation_style_guidance_instruction(prepared)
    return prepared


def token_ids_for_text(processor: Any, text: str) -> list[int]:
    tokenizer = processor_tokenizer(processor)
    if hasattr(tokenizer, "encode"):
        try:
            return [int(item) for item in tokenizer.encode(text, add_special_tokens=False)]
        except TypeError:
            return [int(item) for item in tokenizer.encode(text)]
    encoded = tokenizer(text, return_tensors=None, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids")
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def generation_stop_token_ids(processor: Any, args: argparse.Namespace, result_close_ids: list[int]) -> dict[str, list[int]]:
    stop_ids_by_text: dict[str, list[int]] = {}
    if args.stop_at_result_close and result_close_ids:
        stop_ids_by_text[RESULT_CLOSE_TAG] = list(result_close_ids)
    if args.stop_at_turn_token:
        for stop_text in CHAT_STOP_TOKENS:
            ids = token_ids_for_text(processor, stop_text)
            if ids:
                stop_ids_by_text[stop_text] = ids

    deduped: dict[str, list[int]] = {}
    seen: set[tuple[int, ...]] = set()
    for stop_text, ids in stop_ids_by_text.items():
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        deduped[stop_text] = ids
    return deduped


def decode_ids(processor: Any, ids: Any, *, skip_special_tokens: bool = False) -> str:
    tokenizer = processor_tokenizer(processor)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    return str(ids)


def generate_one(
    *,
    model: Any,
    processor: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    stop_ids_by_text: dict[str, list[int]],
) -> dict[str, Any]:
    import torch
    from transformers import LogitsProcessorList
    from transformers import StoppingCriteriaList

    messages = row.get("prompt")
    if not isinstance(messages, list):
        raise ValueError(f"row {row.get('id')} has invalid prompt.")
    if args.grpo_compatible_prompts:
        messages = grpo_compatible_messages(messages, str(row.get("task") or ""), args)
    prompt_text = render_chat(processor, messages, add_generation_prompt=True)
    encoded = encode_prompt(processor, prompt_text, args.max_prompt_length)
    prompt_length = int(encoded["input_ids"].shape[-1])
    device = next(model.parameters()).device
    encoded = move_to_device(encoded, device)
    tokenizer = processor_tokenizer(processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    stopping_criteria = None
    stop_sequences = list(stop_ids_by_text.values())
    if stop_sequences:
        stopping_criteria = StoppingCriteriaList([StopOnAnyTokenSequence(stop_sequences, prompt_length)])

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature if args.temperature > 0.0 else None,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty if args.repetition_penalty > 0.0 else None,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "use_cache": True,
    }
    if stopping_criteria is not None:
        generate_kwargs["stopping_criteria"] = stopping_criteria
        if args.force_eos_after_stop:
            generate_kwargs["logits_processor"] = LogitsProcessorList(
                [ForceEosAfterAnyTokenSequence(stop_sequences, prompt_length, eos_token_id)]
            )
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**encoded, **generate_kwargs)
    elapsed = time.time() - start
    full_ids = output_ids[0]
    new_ids = full_ids[prompt_length:]
    generated_ids = new_ids.tolist() if hasattr(new_ids, "tolist") else list(new_ids)
    raw_generated_text = decode_ids(processor, new_ids, skip_special_tokens=False)
    generated_text, hit_stop_text, post_stop_text = trim_after_first_stop_text(raw_generated_text)
    return {
        "id": row.get("id", ""),
        "task": row.get("task", ""),
        "model_label": args.model_label,
        "adapter_path": args.adapter_path or "",
        "dataset_path": "",
        "prompt": messages,
        "source_text": row.get("source_text", ""),
        "reference_text": row.get("reference_text", ""),
        "generated_text": generated_text,
        "raw_generated_text": raw_generated_text,
        "prompt_tokens": prompt_length,
        "generated_tokens": len(generated_ids),
        "hit_eos": bool(eos_token_id is not None and int(eos_token_id) in [int(item) for item in generated_ids]),
        "hit_result_close": RESULT_CLOSE_TAG in generated_text,
        "hit_turn_stop": bool(hit_stop_text),
        "hit_stop_text": hit_stop_text,
        "post_stop_text": post_stop_text,
        "elapsed_sec": round(elapsed, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline_v2 fixed eval generation.")
    parser.add_argument("--eval-file", action="append", default=[], help="JSONL eval file. Defaults to all eval_v2 fixed sets.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--max-prompt-length", type=positive_int, default=4096)
    parser.add_argument("--max-new-tokens", type=positive_int, default=3072)
    parser.add_argument("--generate-min-output-chars", type=int, default=3000)
    parser.add_argument("--generate-max-output-chars", type=int, default=4500)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--stop-at-result-close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-at-turn-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-eos-after-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grpo-compatible-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    eval_paths = [Path(path) for path in args.eval_file] if args.eval_file else list(DEFAULT_EVAL_FILES)
    rows = load_eval_rows(eval_paths, args.limit_per_dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = resolve_model_name(args)
    print_json("[eval/load]", {"model_name": model_name, "adapter_path": args.adapter_path or None, "rows": len(rows)})
    model, processor, metadata = load_gemma4_model_and_processor(
        model_name=model_name,
        adapter_path=args.adapter_path or None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        chat_template=args.chat_template,
        create_lora=False,
    )
    model.eval()
    result_close_ids = token_ids_for_text(processor, RESULT_CLOSE_TAG)
    stop_ids_by_text = generation_stop_token_ids(processor, args, result_close_ids)
    generations: list[dict[str, Any]] = []
    for index, (path, row) in enumerate(rows, start=1):
        print_json("[eval/item]", {"index": index, "total": len(rows), "id": row.get("id"), "task": row.get("task")})
        generated = generate_one(model=model, processor=processor, row=row, args=args, stop_ids_by_text=stop_ids_by_text)
        generated["dataset_path"] = str(path)
        generations.append(generated)
        write_jsonl(output_path, generations)

    manifest = {
        "time": time.time(),
        "output": str(output_path),
        "rows": len(generations),
        "args": vars(args),
        "loader": metadata,
        "result_close_token_ids": result_close_ids,
        "generation_stop_token_ids": stop_ids_by_text,
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    print_json("[eval/done]", {"output": str(output_path), "rows": len(generations)})


if __name__ == "__main__":
    main()
