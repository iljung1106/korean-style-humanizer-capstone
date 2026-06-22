#!/usr/bin/env python3
"""Run phase-end rewrite/generate evaluation generations with vLLM."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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

from pipeline_v2.lib.gemma4_loader import DEFAULT_BASE_MODEL, base_model_from_adapter, maybe_apply_chat_template
from pipeline_v2.lib.io import write_json, write_jsonl
from pipeline_v2.lib.masking import processor_tokenizer, render_chat
from pipeline_v2.lib.preference_data import (
    add_continuation_style_guidance_instruction,
    add_generate_length_instruction,
    add_generate_style_guidance_instruction,
    add_result_contract_instruction,
    add_rewrite_style_guidance_instruction,
    rebuild_rewrite_prompt_with_current_guidance,
)
from pipeline_v2.lib.result_contract import CHAT_STOP_TOKENS, RESULT_CLOSE_TAG, trim_after_first_stop_text
from pipeline_v2.lib.trainer_utils import print_json, set_seed


DEFAULT_EVAL_DIR = TRAINING_ROOT / "data" / "phase_eval_v2"
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "outputs" / "pipeline_v2" / "phase_eval"


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


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_json_arg(value: str, *, name: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc


def parse_json_or_int_arg(value: str, *, name: str) -> Any:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return parse_json_arg(stripped, name=name)


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model != "auto":
        return args.model
    if args.adapter_path:
        return base_model_from_adapter(args.adapter_path, DEFAULT_BASE_MODEL)
    return DEFAULT_BASE_MODEL


def load_processor(model_name: str, chat_template: str) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return maybe_apply_chat_template(processor, chat_template)


def count_tokens(processor: Any, text: str) -> int:
    tokenizer = processor_tokenizer(processor)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def stop_texts(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    if args.stop_at_result_close:
        values.append(RESULT_CLOSE_TAG)
    if args.stop_at_turn_token:
        values.extend(CHAT_STOP_TOKENS)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def stop_token_id_labels(processor: Any) -> dict[int, str]:
    tokenizer = processor_tokenizer(processor)
    labels: dict[int, str] = {}
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        labels[int(eos_token_id)] = "eos"
    for token in CHAT_STOP_TOKENS:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(token)
        if len(token_ids) == 1:
            labels[int(token_ids[0])] = token
    return labels


def vllm_init_kwargs(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "max_model_len": args.max_seq_length,
        "seed": args.seed,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "enable_lora": bool(args.adapter_path),
        "enable_prefix_caching": args.vllm_enable_prefix_caching,
        "max_lora_rank": args.vllm_max_lora_rank,
        "max_num_seqs": args.vllm_max_num_seqs,
        "disable_log_stats": args.vllm_disable_log_stats,
    }
    if args.vllm_tokenizer:
        kwargs["tokenizer"] = args.vllm_tokenizer
    if args.vllm_dtype:
        kwargs["dtype"] = args.vllm_dtype
    quantization = args.vllm_quantization
    load_format = args.vllm_load_format
    model_name_lower = model_name.lower()
    if quantization in {"auto", "auto_bnb", "auto_quant"}:
        if args.load_in_4bit:
            quantization = "bitsandbytes"
        elif "nvfp4" in model_name_lower or "modelopt" in model_name_lower:
            quantization = "modelopt"
        elif "bnb" in model_name_lower:
            quantization = "bitsandbytes"
        else:
            quantization = ""
    if load_format in {"auto", "auto_bnb", "auto_quant"}:
        load_format = "bitsandbytes" if args.load_in_4bit or "bnb" in model_name.lower() else "auto"
    if quantization:
        kwargs["quantization"] = quantization
    if load_format:
        kwargs["load_format"] = load_format
    if args.vllm_enforce_eager:
        kwargs["enforce_eager"] = True
    if args.vllm_kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.vllm_kv_cache_dtype
    if args.vllm_max_num_batched_tokens > 0:
        kwargs["max_num_batched_tokens"] = args.vllm_max_num_batched_tokens
    attention_config: dict[str, Any] = {}
    if args.vllm_attention_config_json:
        parsed_attention_config = parse_json_arg(args.vllm_attention_config_json, name="--vllm-attention-config-json")
        if not isinstance(parsed_attention_config, dict):
            raise ValueError("--vllm-attention-config-json must decode to an object")
        attention_config.update(parsed_attention_config)
    if args.vllm_attention_backend:
        attention_config["backend"] = args.vllm_attention_backend
    if attention_config:
        kwargs["attention_config"] = attention_config
    if args.vllm_compilation_config:
        kwargs["compilation_config"] = parse_json_or_int_arg(
            args.vllm_compilation_config,
            name="--vllm-compilation-config",
        )
    if args.vllm_hf_overrides_json:
        parsed_hf_overrides = parse_json_arg(args.vllm_hf_overrides_json, name="--vllm-hf-overrides-json")
        if not isinstance(parsed_hf_overrides, dict):
            raise ValueError("--vllm-hf-overrides-json must decode to an object")
        kwargs["hf_overrides"] = parsed_hf_overrides
    if args.vllm_disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    return kwargs


def make_sampling_params(args: argparse.Namespace, task: str, stops: list[str]) -> Any:
    from vllm import SamplingParams

    max_tokens = args.max_new_tokens_rewrite if task == "rewrite" else args.max_new_tokens_generate
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "stop": stops or None,
        "include_stop_str_in_output": True,
    }
    try:
        signature = inspect.signature(SamplingParams)
        if "seed" in signature.parameters:
            kwargs["seed"] = args.seed
    except (TypeError, ValueError):
        pass
    return SamplingParams(**{key: value for key, value in kwargs.items() if value is not None})


def adapter_weight_path(adapter_path: str) -> Path:
    path = Path(adapter_path)
    if path.is_file():
        return path
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / filename
        if candidate.exists():
            return candidate
    return path


def file_fingerprint(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    remaining = max_bytes
    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    digest.update(str(path).encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def auto_lora_identity(adapter_path: str) -> tuple[str, int, str]:
    weight_path = adapter_weight_path(adapter_path)
    fingerprint = file_fingerprint(weight_path)
    lora_int_id = int(fingerprint[:8], 16) % 2_147_483_647
    if lora_int_id <= 0:
        lora_int_id = 1
    return f"adapter-{fingerprint[:12]}", lora_int_id, fingerprint


def make_lora_request(args: argparse.Namespace) -> Any | None:
    if not args.adapter_path:
        return None
    from vllm.lora.request import LoRARequest

    lora_name = args.vllm_lora_name
    lora_int_id = args.vllm_lora_int_id
    fingerprint = ""
    auto_identity = args.vllm_auto_lora_identity or (lora_name == "default" and lora_int_id == 1)
    if auto_identity:
        lora_name, lora_int_id, fingerprint = auto_lora_identity(args.adapter_path)
    print_json(
        "[phase-eval/vllm-lora-request]",
        {
            "adapter_path": args.adapter_path,
            "lora_name": lora_name,
            "lora_int_id": lora_int_id,
            "auto_identity": auto_identity,
            "fingerprint": fingerprint,
        },
    )
    return LoRARequest(lora_name, lora_int_id, args.adapter_path)


def grpo_compatible_messages(messages: Any, task: str, args: argparse.Namespace) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("eval row has invalid prompt messages")
    if task == "rewrite":
        return rebuild_rewrite_prompt_with_current_guidance(messages)
    prepared = add_result_contract_instruction(messages)
    if task == "generate":
        prepared = add_generate_length_instruction(
            prepared,
            args.generate_max_output_chars,
            args.generate_min_output_chars,
        )
        prepared = add_generate_style_guidance_instruction(prepared)
    elif task == "continuation":
        prepared = add_continuation_style_guidance_instruction(prepared)
    return prepared


def output_token_ids(output: Any) -> list[int]:
    token_ids = getattr(output, "token_ids", None)
    if token_ids is None:
        return []
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(item) for item in token_ids]


def result_text_flags(
    text: str,
    *,
    raw_text: str,
    hit_stop_text: str,
    token_ids: list[int],
    stop_reason: Any,
    stop_token_labels: dict[int, str],
) -> dict[str, Any]:
    stop_reason_label = ""
    try:
        stop_reason_label = stop_token_labels[int(stop_reason)]
    except (TypeError, ValueError, KeyError):
        stop_reason_label = "" if stop_reason is None else str(stop_reason)
    hit_turn_stop = bool(hit_stop_text) or stop_reason_label in CHAT_STOP_TOKENS
    return {
        "hit_result_close": RESULT_CLOSE_TAG in text,
        "hit_eos": str(stop_reason).lower() == "eos" or stop_reason_label == "eos",
        "hit_turn_stop": hit_turn_stop,
        "hit_stop_text": hit_stop_text or (stop_reason_label if hit_turn_stop else ""),
        "post_stop_text": raw_text.split(hit_stop_text, 1)[1] if hit_stop_text and hit_stop_text in raw_text else "",
        "post_result_chars": len(text.split(RESULT_CLOSE_TAG, 1)[1]) if RESULT_CLOSE_TAG in text else 0,
        "stop_reason": "" if stop_reason is None else str(stop_reason),
        "stop_reason_label": stop_reason_label,
        "generated_tokens": len(token_ids),
        "decoded_tokens": len(token_ids),
    }


def generate_task(
    *,
    llm: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    task: str,
    args: argparse.Namespace,
    output_path: Path,
    stops: list[str],
    lora_request: Any | None,
) -> list[dict[str, Any]]:
    generations: list[dict[str, Any]] = []
    sampling_params = make_sampling_params(args, task, stops)
    stop_token_labels = stop_token_id_labels(processor)
    for start_index in range(0, len(rows), args.batch_size):
        batch = rows[start_index : start_index + args.batch_size]
        prompts = [
            grpo_compatible_messages(row["prompt"], str(row.get("task") or task), args)
            if args.grpo_compatible_prompts
            else row["prompt"]
            for row in batch
        ]
        prompt_texts = [render_chat(processor, prompt, add_generation_prompt=True) for prompt in prompts]
        prompt_tokens = [count_tokens(processor, prompt) for prompt in prompt_texts]
        print_json("[phase-eval/vllm-generate]", {"task": task, "start": start_index, "count": len(batch), "total": len(rows)})
        start = time.time()
        request_kwargs = {"lora_request": lora_request} if lora_request is not None else {}
        outputs = llm.generate(prompt_texts, sampling_params, **request_kwargs)
        elapsed = time.time() - start
        for row, prompt_messages, prompt_len, request_output in zip(batch, prompts, prompt_tokens, outputs):
            completion = request_output.outputs[0]
            raw_generated_text = str(getattr(completion, "text", ""))
            generated_text, hit_stop_text, _post_stop_text = trim_after_first_stop_text(raw_generated_text)
            token_ids = output_token_ids(completion)
            stop_reason = getattr(completion, "stop_reason", None) or getattr(completion, "finish_reason", None)
            generations.append(
                {
                    "id": row.get("id", ""),
                    "task": task,
                    "model_label": args.model_label,
                    "adapter_path": args.adapter_path or "",
                    "source_file": row.get("source_file", ""),
                    "source_chunk_id": row.get("source_chunk_id", ""),
                    "prompt": prompt_messages,
                    "source_text": row.get("source_text", ""),
                    "reference_text": row.get("reference_text", ""),
                    "generated_text": generated_text,
                    "raw_generated_text": raw_generated_text,
                    "prompt_tokens_padded": prompt_len,
                    "elapsed_sec_batch": round(elapsed, 3),
                    "batch_size": len(batch),
                    "backend": "vllm",
                    **result_text_flags(
                        generated_text,
                        raw_text=raw_generated_text,
                        hit_stop_text=hit_stop_text,
                        token_ids=token_ids,
                        stop_reason=stop_reason,
                        stop_token_labels=stop_token_labels,
                    ),
                }
            )
        write_jsonl(output_path, generations)
    return generations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-end generation eval with vLLM.")
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
    parser.add_argument("--max-new-tokens-generate", type=positive_int, default=4096)
    parser.add_argument("--generate-min-output-chars", type=int, default=3000)
    parser.add_argument("--generate-max-output-chars", type=int, default=4500)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--stop-at-result-close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-at-turn-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grpo-compatible-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-rewrite", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--rewrite-limit", type=non_negative_int, default=0)
    parser.add_argument("--generate-limit", type=non_negative_int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
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
    parser.add_argument("--vllm-lora-name", default="default")
    parser.add_argument("--vllm-lora-int-id", type=positive_int, default=1)
    parser.add_argument("--vllm-auto-lora-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-disable-log-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=False)
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
    if args.rewrite_limit:
        rewrite_rows = rewrite_rows[: args.rewrite_limit]
    if args.generate_limit:
        generate_rows = generate_rows[: args.generate_limit]
    if not rewrite_rows and not generate_rows:
        raise ValueError("No generation rows selected.")

    model_name = resolve_model_name(args)
    stops = stop_texts(args)
    print_json(
        "[phase-eval/vllm-load]",
        {
            "model_name": model_name,
            "adapter_path": args.adapter_path or None,
            "rewrite_rows": len(rewrite_rows),
            "generate_rows": len(generate_rows),
            "stop": stops,
        },
    )
    processor = load_processor(model_name, args.chat_template)

    from vllm import LLM

    init_kwargs = vllm_init_kwargs(args, model_name)
    print_json("[phase-eval/vllm-kwargs]", init_kwargs)
    llm = LLM(**init_kwargs)
    lora_request = make_lora_request(args)

    outputs: dict[str, Any] = {}
    if rewrite_rows:
        outputs["rewrite"] = {
            "path": str(output_dir / "rewrite_generations.jsonl"),
            "rows": len(
                generate_task(
                    llm=llm,
                    processor=processor,
                    rows=rewrite_rows,
                    task="rewrite",
                    args=args,
                    output_path=output_dir / "rewrite_generations.jsonl",
                    stops=stops,
                    lora_request=lora_request,
                )
            ),
        }
    if generate_rows:
        outputs["generate"] = {
            "path": str(output_dir / "generate_generations.jsonl"),
            "rows": len(
                generate_task(
                    llm=llm,
                    processor=processor,
                    rows=generate_rows,
                    task="generate",
                    args=args,
                    output_path=output_dir / "generate_generations.jsonl",
                    stops=stops,
                    lora_request=lora_request,
                )
            ),
        }

    manifest = {
        "time": time.time(),
        "args": vars(args),
        "model_name": model_name,
        "backend": "vllm",
        "vllm_init_kwargs": init_kwargs,
        "stop_texts": stops,
        "outputs": outputs,
    }
    write_json(output_dir / "generation_manifest.json", manifest)
    print_json("[phase-eval/vllm-done]", {"output_dir": str(output_dir), "outputs": outputs})


if __name__ == "__main__":
    main()
