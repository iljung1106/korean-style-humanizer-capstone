#!/usr/bin/env python3
"""Compare multiple vLLM LoRA adapters in one process."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[3]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.eval.phase_eval.run_generation_vllm import (
    auto_lora_identity,
    grpo_compatible_messages,
    load_processor,
    make_sampling_params,
    output_token_ids,
    read_jsonl,
    resolve_model_name,
    stop_texts,
    vllm_init_kwargs,
)
from pipeline_v2.lib.io import write_json, write_jsonl
from pipeline_v2.lib.masking import render_chat
from pipeline_v2.lib.result_contract import trim_after_first_stop_text
from pipeline_v2.lib.trainer_utils import print_json, set_seed


DEFAULT_EVAL_DIR = TRAINING_ROOT / "data" / "phase_eval_v2"
DEFAULT_OUTPUT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "phase_eval" / "vllm_lora_compare_probe.jsonl"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_adapter_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.name or "adapter", value
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--adapter must be LABEL=PATH or PATH")
    return label, path


def is_base_adapter_path(path: str) -> bool:
    return path.strip().lower() in {"", "none", "base", "no_lora", "no-lora"}


def completion_prompt(prompt_text: str, completion: str) -> str:
    return prompt_text + completion


def prompt_logprob_summary(request_output: Any, prompt_token_count: int) -> dict[str, Any]:
    prompt_logprobs = getattr(request_output, "prompt_logprobs", None)
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None) or []
    if not prompt_logprobs:
        return {"available": False}
    values: list[float] = []
    selected_ids: list[int] = []
    for token_index in range(prompt_token_count, len(prompt_logprobs)):
        item = prompt_logprobs[token_index]
        token_id = int(prompt_token_ids[token_index]) if token_index < len(prompt_token_ids) else None
        if token_id is None or item is None:
            continue
        record = item.get(token_id) if isinstance(item, dict) else None
        if record is None:
            continue
        logprob = getattr(record, "logprob", record)
        try:
            values.append(float(logprob))
            selected_ids.append(token_id)
        except (TypeError, ValueError):
            continue
    return {
        "available": True,
        "tokens": len(values),
        "sum_logprob": sum(values),
        "mean_logprob": (sum(values) / len(values)) if values else None,
        "token_ids_prefix": selected_ids[:32],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe vLLM LoRA adapter differences in one process.")
    parser.add_argument("--adapter", action="append", required=True, help="LABEL=PATH or PATH. Repeat for each adapter.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--task", choices=["generate", "rewrite"], default="generate")
    parser.add_argument("--prompt-index", type=non_negative_int, default=0)
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--max-prompt-length", type=positive_int, default=4096)
    parser.add_argument("--max-new-tokens-generate", type=positive_int, default=256)
    parser.add_argument("--max-new-tokens-rewrite", type=positive_int, default=256)
    parser.add_argument("--generate-min-output-chars", type=int, default=3000)
    parser.add_argument("--generate-max-output-chars", type=int, default=4500)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--stop-at-result-close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-at-turn-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grpo-compatible-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--lora-id-mode", choices=["fingerprint", "sequential"], default="fingerprint")
    parser.add_argument("--fixed-completion", default="<result>\n비린내가 코끝을 찔렀다.\n</result>")
    parser.add_argument("--prompt-logprobs", type=non_negative_int, default=5)
    parser.add_argument("--vllm-tokenizer", default="")
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-quantization", default="auto_bnb")
    parser.add_argument("--vllm-load-format", default="auto_bnb")
    parser.add_argument("--vllm-kv-cache-dtype", default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--vllm-tensor-parallel-size", type=positive_int, default=1)
    parser.add_argument("--vllm-max-lora-rank", type=positive_int, default=64)
    parser.add_argument("--vllm-max-num-seqs", type=positive_int, default=8)
    parser.add_argument("--vllm-max-num-batched-tokens", type=non_negative_int, default=0)
    parser.add_argument("--vllm-attention-backend", default="")
    parser.add_argument("--vllm-attention-config-json", default="")
    parser.add_argument("--vllm-compilation-config", default="")
    parser.add_argument("--vllm-hf-overrides-json", default="")
    parser.add_argument("--vllm-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-disable-log-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    adapters = [parse_adapter_spec(value) for value in args.adapter]
    if args.reverse_order:
        adapters = list(reversed(adapters))
    first_adapter = next((path for _label, path in adapters if not is_base_adapter_path(path)), "")
    args.adapter_path = first_adapter
    model_name = resolve_model_name(args)
    processor = load_processor(model_name, args.chat_template)
    prompt_file = Path(args.prompt_file) if args.prompt_file else Path(args.eval_dir) / f"{args.task}_prompts.jsonl"
    rows = read_jsonl(prompt_file)
    if args.prompt_index >= len(rows):
        raise IndexError(f"--prompt-index {args.prompt_index} out of range for {prompt_file} with {len(rows)} rows")
    row = rows[args.prompt_index]
    messages = (
        grpo_compatible_messages(row["prompt"], str(row.get("task") or args.task), args)
        if args.grpo_compatible_prompts
        else row["prompt"]
    )
    prompt_text = render_chat(processor, messages, add_generation_prompt=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    init_kwargs = vllm_init_kwargs(args, model_name)
    print_json("[vllm-lora-probe/load]", {"model_name": model_name, "adapters": adapters, "init_kwargs": init_kwargs})
    llm = LLM(**init_kwargs)
    stops = stop_texts(args)
    sampling_params = make_sampling_params(args, args.task, stops)
    logprob_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=args.prompt_logprobs or None,
    )

    results: list[dict[str, Any]] = []
    prompt_token_count = len(llm.get_tokenizer().encode(prompt_text))
    fixed_full_prompt = completion_prompt(prompt_text, args.fixed_completion)
    next_sequential_lora_id = 1
    for label, adapter_path in adapters:
        if is_base_adapter_path(adapter_path):
            lora_name = "base"
            lora_int_id = 0
            fingerprint = ""
            lora_request = None
        else:
            lora_name, lora_int_id, fingerprint = auto_lora_identity(adapter_path)
            if args.lora_id_mode == "sequential":
                lora_int_id = next_sequential_lora_id
                next_sequential_lora_id += 1
            lora_request = LoRARequest(lora_name, lora_int_id, adapter_path)
        print_json(
            "[vllm-lora-probe/adapter]",
            {
                "label": label,
                "adapter_path": adapter_path,
                "lora_name": lora_name,
                "lora_int_id": lora_int_id,
                "fingerprint": fingerprint,
            },
        )
        started = time.time()
        request_kwargs = {"lora_request": lora_request} if lora_request is not None else {}
        generated = llm.generate([prompt_text], sampling_params, **request_kwargs)[0]
        elapsed = time.time() - started
        completion = generated.outputs[0]
        raw_text = str(getattr(completion, "text", ""))
        text, hit_stop_text, _post_stop_text = trim_after_first_stop_text(raw_text)
        logprob_output = llm.generate([fixed_full_prompt], logprob_params, **request_kwargs)[0]
        results.append(
            {
                "label": label,
                "adapter_path": adapter_path,
                "lora_name": lora_name,
                "lora_int_id": lora_int_id,
                "fingerprint": fingerprint,
                "elapsed_sec": round(elapsed, 3),
                "generated_text": text,
                "raw_generated_text": raw_text,
                "hit_stop_text": hit_stop_text,
                "token_ids_prefix": output_token_ids(completion)[:64],
                "stop_reason": str(getattr(completion, "stop_reason", None) or getattr(completion, "finish_reason", None)),
                "prompt_logprobs": prompt_logprob_summary(logprob_output, prompt_token_count),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, results)
    manifest = {
        "time": time.time(),
        "output": str(output_path),
        "args": vars(args),
        "model_name": model_name,
        "prompt_file": str(prompt_file),
        "prompt_index": args.prompt_index,
        "prompt_id": row.get("id", ""),
        "adapter_order": [label for label, _path in adapters],
        "results": [
            {
                "label": item["label"],
                "fingerprint": item["fingerprint"],
                "generated_chars": len(item["generated_text"]),
                "prompt_logprobs": item["prompt_logprobs"],
            }
            for item in results
        ],
    }
    write_json(output_path.with_suffix(".summary.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
