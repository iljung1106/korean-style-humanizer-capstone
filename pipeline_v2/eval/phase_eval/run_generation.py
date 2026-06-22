#!/usr/bin/env python3
"""Run phase-end rewrite/generate evaluation generations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parents[2]
TRAINING_ROOT = SCRIPT.parents[3]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gemma4_loader import DEFAULT_BASE_MODEL, base_model_from_adapter, load_gemma4_model_and_processor
from pipeline_v2.lib.io import write_json, write_jsonl
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


DEFAULT_EVAL_DIR = TRAINING_ROOT / "data" / "phase_eval_v2"
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "outputs" / "pipeline_v2" / "phase_eval"


def contains_subsequence(values: list[int], needle: list[int]) -> bool:
    if not needle or len(values) < len(needle):
        return False
    needed = len(needle)
    return any(values[index : index + needed] == needle for index in range(0, len(values) - needed + 1))


class StopWhenAllRowsContainAnyTokenSequence:
    def __init__(self, stop_sequences: list[list[int]], prompt_width: int) -> None:
        self.stop_sequences = [list(sequence) for sequence in stop_sequences if sequence]
        self.prompt_width = int(prompt_width)

    def __call__(self, input_ids: Any, _scores: Any, **_: Any) -> bool:
        if not self.stop_sequences:
            return False
        try:
            sequences = input_ids.tolist()
        except Exception:
            sequences = [list(row) for row in input_ids]
        stopped = 0
        for sequence in sequences:
            generated = sequence[self.prompt_width :]
            if any(contains_subsequence(generated, stop_ids) for stop_ids in self.stop_sequences):
                stopped += 1
        return stopped == len(sequences)


class ForceEosAfterAnyTokenSequence:
    def __init__(self, stop_sequences: list[list[int]], prompt_width: int, eos_token_id: int | None) -> None:
        self.stop_sequences = [list(sequence) for sequence in stop_sequences if sequence]
        self.prompt_width = int(prompt_width)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if not self.stop_sequences or self.eos_token_id is None:
            return scores
        for row_index in range(input_ids.shape[0]):
            sequence = input_ids[row_index].tolist()
            generated = sequence[self.prompt_width :]
            if any(len(generated) >= len(stop_ids) and generated[-len(stop_ids) :] == stop_ids for stop_ids in self.stop_sequences):
                scores[row_index, :] = -float("inf")
                scores[row_index, self.eos_token_id] = 0.0
        return scores


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


def encode_prompts(processor: Any, prompt_texts: list[str], max_prompt_length: int) -> dict[str, Any]:
    kwargs = {
        "text": prompt_texts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_prompt_length,
    }
    try:
        encoded = processor(**kwargs)
    except Exception:
        tokenizer = processor_tokenizer(processor)
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
    return dict(encoded)


def move_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


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


def result_text_flags(
    text: str,
    eos_token_id: int | None,
    generated_ids: list[int],
    *,
    hit_stop_text: str,
    post_stop_text: str,
) -> dict[str, Any]:
    return {
        "hit_result_close": RESULT_CLOSE_TAG in text,
        "hit_eos": bool(eos_token_id is not None and int(eos_token_id) in [int(item) for item in generated_ids]),
        "hit_turn_stop": bool(hit_stop_text),
        "hit_stop_text": hit_stop_text,
        "post_stop_text": post_stop_text,
        "post_result_chars": len(text.split(RESULT_CLOSE_TAG, 1)[1]) if RESULT_CLOSE_TAG in text else 0,
    }


def trim_terminal_special_ids(generated_ids: list[int], *, eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    terminal = {int(token_id) for token_id in (eos_token_id, pad_token_id) if token_id is not None}
    trimmed = list(generated_ids)
    while trimmed and int(trimmed[-1]) in terminal:
        trimmed.pop()
    return trimmed


def generate_batch(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    task: str,
    args: argparse.Namespace,
    stop_ids_by_text: dict[str, list[int]],
) -> list[dict[str, Any]]:
    import torch
    from transformers import LogitsProcessorList
    from transformers import StoppingCriteriaList

    prompt_texts = [
        render_chat(
            processor,
            grpo_compatible_messages(row["prompt"], task, args) if args.grpo_compatible_prompts else row["prompt"],
            add_generation_prompt=True,
        )
        for row in rows
    ]
    encoded = encode_prompts(processor, prompt_texts, args.max_prompt_length)
    prompt_width = int(encoded["input_ids"].shape[-1])
    device = next(model.parameters()).device
    encoded = move_to_device(encoded, device)
    tokenizer = processor_tokenizer(processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    max_new_tokens = args.max_new_tokens_rewrite if task == "rewrite" else args.max_new_tokens_generate
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature if args.temperature > 0.0 else None,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty if args.repetition_penalty > 0.0 else None,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "use_cache": True,
    }
    stop_sequences = list(stop_ids_by_text.values())
    if stop_sequences:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopWhenAllRowsContainAnyTokenSequence(stop_sequences, prompt_width)]
        )
        if args.force_eos_after_stop:
            generate_kwargs["logits_processor"] = LogitsProcessorList(
                [ForceEosAfterAnyTokenSequence(stop_sequences, prompt_width, eos_token_id)]
            )
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**encoded, **generate_kwargs)
    elapsed = time.time() - start

    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        full_ids = output_ids[index]
        new_ids = full_ids[prompt_width:]
        generated_ids = new_ids.tolist() if hasattr(new_ids, "tolist") else list(new_ids)
        decoded_ids = trim_terminal_special_ids(generated_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        raw_generated_text = decode_ids(processor, decoded_ids, skip_special_tokens=False)
        generated_text, hit_stop_text, post_stop_text = trim_after_first_stop_text(raw_generated_text)
        outputs.append(
            {
                "id": row.get("id", ""),
                "task": task,
                "model_label": args.model_label,
                "adapter_path": args.adapter_path or "",
                "source_file": row.get("source_file", ""),
                "source_chunk_id": row.get("source_chunk_id", ""),
                "prompt": row.get("prompt"),
                "source_text": row.get("source_text", ""),
                "reference_text": row.get("reference_text", ""),
                "generated_text": generated_text,
                "raw_generated_text": raw_generated_text,
                "prompt_tokens_padded": prompt_width,
                "generated_tokens": len(generated_ids),
                "decoded_tokens": len(decoded_ids),
                "elapsed_sec_batch": round(elapsed, 3),
                "batch_size": len(rows),
                "force_eos_after_stop": bool(args.force_eos_after_stop),
                "repetition_penalty": args.repetition_penalty,
                **result_text_flags(
                    generated_text,
                    eos_token_id,
                    generated_ids,
                    hit_stop_text=hit_stop_text,
                    post_stop_text=post_stop_text,
                ),
            }
        )
    return outputs


def run_task(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    task: str,
    args: argparse.Namespace,
    output_path: Path,
    stop_ids_by_text: dict[str, list[int]],
) -> list[dict[str, Any]]:
    generations: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        print_json("[phase-eval/generate]", {"task": task, "start": start, "count": len(batch), "total": len(rows)})
        generations.extend(
            generate_batch(
                model=model,
                processor=processor,
                rows=batch,
                task=task,
                args=args,
                stop_ids_by_text=stop_ids_by_text,
            )
        )
        write_jsonl(output_path, generations)
    return generations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-end generation eval.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--rewrite-prompts", default="")
    parser.add_argument("--generate-prompts", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--max-prompt-length", type=positive_int, default=4096)
    parser.add_argument("--max-new-tokens-rewrite", type=positive_int, default=4096)
    parser.add_argument("--max-new-tokens-generate", type=positive_int, default=3072)
    parser.add_argument("--generate-min-output-chars", type=int, default=3000)
    parser.add_argument("--generate-max-output-chars", type=int, default=4500)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--rewrite-limit", type=int, default=0)
    parser.add_argument("--generate-limit", type=int, default=0)
    parser.add_argument("--grpo-compatible-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-at-result-close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-at-turn-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-eos-after-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-eos-after-result-close", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-rewrite", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    eval_dir = Path(args.eval_dir)
    rewrite_path = Path(args.rewrite_prompts) if args.rewrite_prompts else eval_dir / "rewrite_prompts.jsonl"
    generate_path = Path(args.generate_prompts) if args.generate_prompts else eval_dir / "generate_prompts.jsonl"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rewrite_rows = [] if args.skip_rewrite else read_jsonl(rewrite_path)
    generate_rows = [] if args.skip_generate else read_jsonl(generate_path)
    if args.rewrite_limit > 0:
        rewrite_rows = rewrite_rows[: args.rewrite_limit]
    if args.generate_limit > 0:
        generate_rows = generate_rows[: args.generate_limit]
    if not rewrite_rows and not generate_rows:
        raise ValueError("No generation rows selected.")

    model_name = resolve_model_name(args)
    print_json(
        "[phase-eval/load]",
        {
            "model_name": model_name,
            "adapter_path": args.adapter_path or None,
            "rewrite_rows": len(rewrite_rows),
            "generate_rows": len(generate_rows),
        },
    )
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
    if args.force_eos_after_result_close is not None:
        args.force_eos_after_stop = bool(args.force_eos_after_result_close)
    stop_ids_by_text = generation_stop_token_ids(processor, args, result_close_ids)

    outputs: dict[str, Any] = {}
    if rewrite_rows:
        outputs["rewrite"] = {
            "path": str(output_dir / "rewrite_generations.jsonl"),
            "rows": len(
                run_task(
                    model=model,
                    processor=processor,
                    rows=rewrite_rows,
                    task="rewrite",
                    args=args,
                    output_path=output_dir / "rewrite_generations.jsonl",
                    stop_ids_by_text=stop_ids_by_text,
                )
            ),
        }
    if generate_rows:
        outputs["generate"] = {
            "path": str(output_dir / "generate_generations.jsonl"),
            "rows": len(
                run_task(
                    model=model,
                    processor=processor,
                    rows=generate_rows,
                    task="generate",
                    args=args,
                    output_path=output_dir / "generate_generations.jsonl",
                    stop_ids_by_text=stop_ids_by_text,
                )
            ),
        }

    manifest = {
        "time": time.time(),
        "args": vars(args),
        "model_name": model_name,
        "loader": metadata,
        "result_close_token_ids": result_close_ids,
        "generation_stop_token_ids": stop_ids_by_text,
        "outputs": outputs,
    }
    write_json(output_dir / "generation_manifest.json", manifest)
    print_json("[phase-eval/done]", {"output_dir": str(output_dir), "outputs": outputs})


if __name__ == "__main__":
    main()
