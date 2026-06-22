"""Shared GRPO helpers for standalone pipeline_v2 trainers."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

from .gemma4_loader import DEFAULT_BASE_MODEL, base_model_from_adapter, filter_kwargs, load_gemma4_model_and_processor, processor_tokenizer
from .gui_style_scoring import (
    CONTENT_TAG_PREFIXES,
    GuiStyleScorer,
    clip,
    finite_float,
    morphs,
    rewrite_edit_amount,
    rewrite_metrics,
    summarize_numeric,
)
from .io import TRAINING_ROOT, manifest_base, read_jsonl, write_json
from .lora import trainable_parameter_summary
from .preference_data import (
    GEMMA4_THINKING_TOKEN,
    GEMMA4_THOUGHT_CLOSE_TAG,
    GEMMA4_THOUGHT_OPEN_TAG,
    add_generate_length_instruction,
    add_generate_style_guidance_instruction,
    add_result_contract_instruction,
    add_rewrite_style_guidance_instruction,
    add_short_reasoning_instruction,
    apply_system_prompt_variant,
    as_chat_messages,
    dataset_from_rows,
    render_prompt,
    trim_messages,
)
from .result_contract import CHAT_STOP_TOKENS, RESULT_CLOSE_TAG, RESULT_OPEN_TAG, analyze_result_contract, collapse_reason, completion_text
from .trainer_utils import parse_report_to, print_json, set_seed


DEFAULT_GRPO_DATASET = TRAINING_ROOT / "data" / "processed" / "grpo_mixed_prompts.jsonl"
DEFAULT_OUTPUT_ROOT = TRAINING_ROOT / "outputs" / "pipeline_v2"
STOP_STRINGS = (RESULT_CLOSE_TAG, *CHAT_STOP_TOKENS)
DEFAULT_REASONING_OPEN_TAG = GEMMA4_THOUGHT_OPEN_TAG
DEFAULT_REASONING_CLOSE_TAG = GEMMA4_THOUGHT_CLOSE_TAG


def parse_gradient_checkpointing(value: Any) -> str | bool:
    text = str(value).strip().lower()
    if text in {"unsloth", "smart"}:
        return "unsloth"
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off", "none"}:
        return False
    raise ValueError("--gradient-checkpointing must be one of: unsloth, true, false")


def import_grpo_classes() -> tuple[Any, Any]:
    try:
        from trl import GRPOConfig, GRPOTrainer
        patch_grpo_tool_mask_alignment(GRPOTrainer)
        return GRPOConfig, GRPOTrainer
    except ImportError:
        from trl.experimental.grpo import GRPOConfig, GRPOTrainer

        patch_grpo_tool_mask_alignment(GRPOTrainer)
        return GRPOConfig, GRPOTrainer


def patch_paged_sdpa_attn_alias() -> None:
    """Map Unsloth/TRL's stale sdpa_paged name to Transformers 5.5's paged|sdpa name."""

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception as exc:
        print_json("[generation/paged_patch]", {"enabled": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    original = getattr(PreTrainedModel, "set_attn_implementation", None)
    if original is None or getattr(original, "_pipeline_v2_paged_sdpa_patch", False):
        return

    def patched_set_attn_implementation(self: Any, attn_implementation: Any, *args: Any, **kwargs: Any) -> Any:
        if attn_implementation == "paged|sdpa_paged":
            attn_implementation = "paged|sdpa"
        return original(self, attn_implementation, *args, **kwargs)

    setattr(patched_set_attn_implementation, "_pipeline_v2_paged_sdpa_patch", True)
    PreTrainedModel.set_attn_implementation = patched_set_attn_implementation  # type: ignore[method-assign]
    print_json("[generation/paged_patch]", {"enabled": True, "mapping": {"paged|sdpa_paged": "paged|sdpa"}})


def patch_gemma4_composite_config_for_paged_generation(model: Any) -> None:
    """Expose Gemma4 text_config dimensions on the composite config for paged generation."""

    attrs = (
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "num_hidden_layers",
        "sliding_window",
        "vocab_size",
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
    )
    patched: list[str] = []
    seen: set[int] = set()
    stack = [model]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        config = getattr(current, "config", None)
        text_config = getattr(config, "text_config", None)
        if config is not None and text_config is not None:
            config_name = type(config).__name__
            for attr in attrs:
                if hasattr(config, attr) or not hasattr(text_config, attr):
                    continue
                try:
                    setattr(config, attr, getattr(text_config, attr))
                    patched.append(f"{config_name}.{attr}")
                except Exception:
                    pass
        for child_name in ("base_model", "model", "module"):
            child = getattr(current, child_name, None)
            if child is not None:
                stack.append(child)
    if patched:
        print_json("[generation/paged_config_patch]", {"patched": sorted(set(patched))})


def patch_grpo_tool_mask_alignment(grpo_trainer_class: Any) -> None:
    """Patch Unsloth/TRL builds that call an unexported tool-mask helper."""

    import sys

    import torch

    module = sys.modules.get(getattr(grpo_trainer_class, "__module__", ""))
    if module is None or hasattr(module, "align_completion_tool_mask"):
        return

    def align_completion_tool_mask(tool_mask: Any, completion_mask: Any) -> Any:
        if tool_mask is None:
            return completion_mask

        if not torch.is_tensor(tool_mask):
            tool_mask_tensor = torch.tensor(tool_mask, device=completion_mask.device)
        else:
            tool_mask_tensor = tool_mask.to(device=completion_mask.device)

        if tool_mask_tensor.ndim == 1:
            tool_mask_tensor = tool_mask_tensor.unsqueeze(0)

        target_len = int(completion_mask.shape[-1])
        mask_len = int(tool_mask_tensor.shape[-1])
        if mask_len < target_len:
            pad_shape = (*tool_mask_tensor.shape[:-1], target_len - mask_len)
            left_pad = torch.ones(pad_shape, device=tool_mask_tensor.device, dtype=tool_mask_tensor.dtype)
            tool_mask_tensor = torch.cat([left_pad, tool_mask_tensor], dim=-1)
        elif mask_len > target_len:
            tool_mask_tensor = tool_mask_tensor[..., -target_len:]

        return completion_mask * tool_mask_tensor.to(dtype=completion_mask.dtype)

    module.align_completion_tool_mask = align_completion_tool_mask


def instantiate_config(config_class: Any, kwargs: dict[str, Any]) -> Any:
    """Create TRL configs while tolerating Unsloth wrapper signature drift."""

    current = dict(kwargs)
    while True:
        try:
            return config_class(**filter_kwargs(config_class, current))
        except TypeError as exc:
            message = str(exc)
            marker = "unexpected keyword argument "
            if marker not in message:
                raise
            bad_key = message.split(marker, 1)[1].strip().strip("'\"")
            if bad_key not in current:
                raise
            print_json("[config/drop_unsupported]", {"key": bad_key, "error": message})
            current.pop(bad_key, None)


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model != "auto":
        return args.model
    if args.adapter_path:
        return base_model_from_adapter(args.adapter_path, DEFAULT_BASE_MODEL)
    return DEFAULT_BASE_MODEL


def selected_task(row: dict[str, Any]) -> str:
    return str(row.get("task") or row.get("grpo_task") or "").strip()


def parse_csv_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def parse_metric_weight_overrides(value: str) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid metric weight override {item!r}; expected metric:multiplier")
        name, weight_text = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"invalid metric weight override {item!r}; metric name is empty")
        overrides[name] = float(weight_text.strip())
    return overrides


def load_grpo_rows(args: argparse.Namespace, processor: Any, *, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row in read_jsonl(args.dataset):
        row_task = selected_task(source_row)
        if task != "mixed" and row_task != task:
            continue
        effective_task = row_task or task
        variant_seed = ""
        if args.style_guidance_variant_mode == "row":
            variant_seed = f"{args.seed}:{effective_task}:{source_row.get('id') or source_row.get('task') or len(rows)}"
        messages = trim_messages(as_chat_messages(source_row.get("prompt")), args.max_prompt_chars)
        if variant_seed and args.system_prompt_variants:
            messages = apply_system_prompt_variant(messages, task=effective_task, variant_seed=variant_seed)
        messages = add_result_contract_instruction(messages)
        messages = add_short_reasoning_instruction(
            messages,
            budget_tokens=args.reasoning_budget_tokens if args.enable_short_reasoning else 0,
            open_tag=args.reasoning_open_tag,
            close_tag=args.reasoning_close_tag,
        )
        if row_task == "generate" or task == "generate":
            messages = add_generate_length_instruction(
                messages,
                args.generate_max_output_chars,
                args.generate_min_output_chars,
            )
            messages = add_generate_style_guidance_instruction(
                messages,
                variant_seed=variant_seed or None,
                shuffle_bullets=bool(args.shuffle_style_guidance_bullets),
            )
        elif row_task == "rewrite" or task == "rewrite":
            messages = add_rewrite_style_guidance_instruction(
                messages,
                variant_seed=variant_seed or None,
                shuffle_bullets=bool(args.shuffle_style_guidance_bullets),
            )
        rows.append(
            {
                "prompt": render_prompt(processor, messages, enable_thinking=bool(args.enable_short_reasoning)),
                "task": row_task or task,
                "source_text": str(source_row.get("source_text") or ""),
                "reference_text": str(source_row.get("reference_text") or ""),
                "id": str(source_row.get("id") or len(rows)),
            }
        )
        if args.limit_rows > 0 and len(rows) >= args.limit_rows:
            break
    if not rows:
        raise ValueError(f"No GRPO rows selected from {args.dataset} for task={task}")
    return rows


def order_mixed_task_rows(rows: list[dict[str, Any]], *, mode: str, seed: int) -> list[dict[str, Any]]:
    if mode == "source" or len(rows) <= 1:
        return list(rows)
    if mode == "shuffle":
        return list(rows)
    if mode != "alternating":
        raise ValueError(f"unsupported mixed task order: {mode}")

    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task") or "unknown"), []).append(row)
    for task_rows in by_task.values():
        rng.shuffle(task_rows)

    ordered_tasks = [task for task in ("generate", "rewrite") if task in by_task]
    ordered_tasks.extend(sorted(task for task in by_task if task not in set(ordered_tasks)))
    ordered: list[dict[str, Any]] = []
    while any(by_task.get(task) for task in ordered_tasks):
        for task in ordered_tasks:
            task_rows = by_task.get(task) or []
            if task_rows:
                ordered.append(task_rows.pop())
    return ordered


def finite_score(value: Any, default: float = -1.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def json_safe(value: Any) -> Any:
    """Convert non-standard floats to JSON-safe nulls for strict JSONL readers."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def hangul_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if "가" <= ch <= "힣") / len(chars)


def smoke_reward_score(text: str, *, require_result_tags: bool, max_output_chars: int) -> float:
    contract = analyze_result_contract(text, require_result_tags=require_result_tags, min_result_chars=20)
    result_text = contract.result_text.strip()
    reason = collapse_reason(text, require_result_tags=require_result_tags)
    score = 0.0
    score += 0.45 if contract.ok else -0.45
    score += 0.25 * min(1.0, len(result_text) / 400.0)
    score += 0.20 * hangul_ratio(result_text)
    if max_output_chars > 0 and len(result_text) > max_output_chars:
        score -= min(0.4, (len(result_text) - max_output_chars) / max(1.0, max_output_chars))
    if reason:
        score -= 0.35
    return float(max(-1.0, min(1.0, score)))


def generate_length_penalty(
    result_text: str,
    *,
    min_output_chars: int,
    max_output_chars: int,
    short_output_penalty: float,
    long_output_penalty: float,
) -> float:
    length = len(result_text.strip())
    penalty = 0.0
    if min_output_chars > 0 and length < min_output_chars:
        penalty -= max(0.0, short_output_penalty) * (1.0 - length / max(1.0, float(min_output_chars)))
    if max_output_chars > 0 and length > max_output_chars:
        penalty -= max(0.0, long_output_penalty) * min(
            1.0,
            (length - max_output_chars) / max(1.0, float(max_output_chars)),
        )
    return float(max(-1.0, penalty))


CONTRACT_ONLY_REASONS = {
    "missing_result_tags",
    "missing_result_open",
    "missing_result_close",
    "post_result_text",
    "multiple_result_open",
    "multiple_result_close",
}


def contract_penalty_for_reason(reason: str) -> float:
    if reason == "missing_result_close":
        return -0.18
    if reason in {"missing_result_tags", "missing_result_open"}:
        return -0.24
    if reason == "post_result_text":
        return -0.12
    if reason in {"multiple_result_open", "multiple_result_close"}:
        return -0.16
    if reason == "empty_or_short_result":
        return -0.35
    if reason == "thought_leak":
        return -0.30
    return -0.20 if reason else 0.0


def contract_or_collapse_fail_reward(scored: dict[str, Any], *, fail_reward: float) -> float | None:
    metrics = dict(scored.get("metrics") or {})
    collapse = str(metrics.get("collapse_reason") or "")
    result_text = str(scored.get("result_text") or "")
    content_collapse = collapse_reason(result_text, require_result_tags=False) or ""
    if content_collapse:
        metrics["content_collapse_reason"] = content_collapse
        scored["metrics"] = metrics
        return float(max(-1.0, min(0.0, fail_reward)))
    if collapse and collapse not in CONTRACT_ONLY_REASONS:
        return float(max(-1.0, min(0.0, fail_reward)))
    return None


def result_close_token_ids(processor: Any) -> list[int]:
    tokenizer = processor_tokenizer(processor)
    try:
        encoded = tokenizer(RESULT_CLOSE_TAG, add_special_tokens=False)
        token_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded["input_ids"]
    except Exception as exc:
        print_json("[generation/result_stop]", {"enabled": False, "reason": f"tokenize_failed:{type(exc).__name__}: {exc}"})
        return []
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def token_ids_for_text(processor: Any, text: str) -> list[int]:
    if not text:
        return []
    tokenizer = processor_tokenizer(processor)
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded["input_ids"]
    except Exception as exc:
        print_json("[generation/tokenize]", {"text": text, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return []
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def first_token_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return first_token_id(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_eos_token_id(processor: Any | None, model: Any | None) -> int | None:
    tokenizer = processor_tokenizer(processor) if processor is not None else None
    holders = [tokenizer, getattr(model, "generation_config", None), getattr(model, "config", None)]
    for holder in holders:
        token_id = first_token_id(getattr(holder, "eos_token_id", None))
        if token_id is not None:
            return token_id
    for holder in holders:
        token_id = first_token_id(getattr(holder, "pad_token_id", None))
        if token_id is not None:
            return token_id
    return None


def append_result_close_stopping_criteria(
    kwargs: dict[str, Any],
    gen_args: tuple[Any, ...],
    stop_ids: list[int],
    eos_token_id: int | None = None,
) -> None:
    if not stop_ids:
        return
    input_ids = kwargs.get("input_ids")
    if input_ids is None and gen_args:
        first = gen_args[0]
        if hasattr(first, "shape"):
            input_ids = first
    if input_ids is None or not hasattr(input_ids, "shape"):
        return
    try:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception as exc:
        print_json("[generation/result_stop]", {"enabled": False, "reason": f"import_failed:{type(exc).__name__}: {exc}"})
        return

    class ResultCloseState:
        def __init__(self, prompt_length: int, sequence: list[int]) -> None:
            self.prompt_length = int(prompt_length)
            self.sequence = [int(token_id) for token_id in sequence]
            self.done = None

        def update(self, input_ids: Any) -> Any:
            if self.done is None or self.done.shape[0] != input_ids.shape[0]:
                self.done = input_ids.new_zeros((input_ids.shape[0],), dtype=torch.bool)
            generated = input_ids[:, self.prompt_length :]
            needed = len(self.sequence)
            if generated.shape[-1] < needed:
                return self.done
            target = input_ids.new_tensor(self.sequence)
            suffix_matches = (generated[:, -needed:] == target).all(dim=-1)
            self.done = self.done | suffix_matches
            return self.done

    state = ResultCloseState(int(input_ids.shape[-1]), stop_ids)

    class StopWhenAllRowsSawTokenSequence(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **_kwargs: Any) -> bool:
            done = state.update(input_ids)
            return bool(done.all().item())

    class ForceEosAfterTokenSequence(LogitsProcessor):
        def __init__(self, eos_token_id: int) -> None:
            self.eos_token_id = int(eos_token_id)

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            done = state.update(input_ids)
            if done is None or not bool(done.any().item()):
                return scores
            floor = torch.finfo(scores.dtype).min if scores.is_floating_point() else -1_000_000_000
            forced_scores = torch.full_like(scores, floor)
            forced_scores[:, self.eos_token_id] = 0
            return torch.where(done[:, None], forced_scores, scores)

    existing = kwargs.get("stopping_criteria")
    criterion = StopWhenAllRowsSawTokenSequence()
    if existing is None:
        kwargs["stopping_criteria"] = StoppingCriteriaList([criterion])
    elif isinstance(existing, StoppingCriteriaList):
        existing.append(criterion)
        kwargs["stopping_criteria"] = existing
    else:
        kwargs["stopping_criteria"] = StoppingCriteriaList(list(existing) + [criterion])
    if eos_token_id is None:
        return
    existing_processors = kwargs.get("logits_processor")
    processor = ForceEosAfterTokenSequence(eos_token_id)
    if existing_processors is None:
        kwargs["logits_processor"] = LogitsProcessorList([processor])
    elif isinstance(existing_processors, LogitsProcessorList):
        existing_processors.append(processor)
        kwargs["logits_processor"] = existing_processors
    else:
        kwargs["logits_processor"] = LogitsProcessorList(list(existing_processors) + [processor])


def append_reasoning_budget_processor(
    kwargs: dict[str, Any],
    gen_args: tuple[Any, ...],
    open_ids: list[int],
    close_ids: list[int],
    result_open_ids: list[int] | None = None,
    *,
    budget_tokens: int,
    bias_start_tokens: int,
    max_bias: float,
) -> None:
    if not open_ids or not close_ids or budget_tokens <= 0:
        return
    input_ids = kwargs.get("input_ids")
    if input_ids is None and gen_args:
        first = gen_args[0]
        if hasattr(first, "shape"):
            input_ids = first
    if input_ids is None or not hasattr(input_ids, "shape"):
        return
    try:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
    except Exception as exc:
        print_json("[generation/reasoning_budget]", {"enabled": False, "reason": f"import_failed:{type(exc).__name__}: {exc}"})
        return

    def contains_sequence(values: list[int], sequence: list[int]) -> bool:
        needed = len(sequence)
        if needed <= 0 or len(values) < needed:
            return False
        return any(values[index : index + needed] == sequence for index in range(0, len(values) - needed + 1))

    def find_sequence_end(values: list[int], sequence: list[int]) -> int:
        needed = len(sequence)
        if needed <= 0 or len(values) < needed:
            return -1
        for index in range(0, len(values) - needed + 1):
            if values[index : index + needed] == sequence:
                return index + needed
        return -1

    def matched_prefix_len(values: list[int], sequence: list[int]) -> int:
        max_len = min(len(values), len(sequence) - 1)
        for size in range(max_len, 0, -1):
            if values[-size:] == sequence[:size]:
                return size
        return 0

    class ForceReasoningCloseAfterBudget(LogitsProcessor):
        def __init__(self, prompt_length: int, sequence: list[int]) -> None:
            self.prompt_length = int(prompt_length)
            self.sequence = [int(token_id) for token_id in sequence]
            self.result_open_sequence = [int(token_id) for token_id in (result_open_ids or [])]

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            generated = input_ids[:, self.prompt_length :]
            if generated.shape[-1] <= 0:
                forced = scores.clone()
                floor = torch.finfo(scores.dtype).min if scores.is_floating_point() else -1_000_000_000
                forced[:, :] = floor
                forced[:, open_ids[0]] = 0
                return forced
            forced = scores.clone()
            floor = torch.finfo(scores.dtype).min if scores.is_floating_point() else -1_000_000_000
            for row_index in range(generated.shape[0]):
                row_ids = [int(token_id) for token_id in generated[row_index].tolist()]
                if not contains_sequence(row_ids, open_ids):
                    if len(row_ids) < len(open_ids) and row_ids == open_ids[: len(row_ids)]:
                        forced[row_index, :] = floor
                        forced[row_index, open_ids[len(row_ids)]] = 0
                    continue
                open_end = find_sequence_end(row_ids, open_ids)
                if open_end < 0:
                    continue
                close_end = find_sequence_end(row_ids, self.sequence)
                if close_end >= 0:
                    if self.result_open_sequence and not contains_sequence(row_ids[close_end:], self.result_open_sequence):
                        after_close = row_ids[close_end:]
                        if len(after_close) < len(self.result_open_sequence) and after_close == self.result_open_sequence[: len(after_close)]:
                            forced[row_index, :] = floor
                            forced[row_index, self.result_open_sequence[len(after_close)]] = 0
                    continue
                reasoning_ids = row_ids[open_end:]
                prefix_len = matched_prefix_len(reasoning_ids, self.sequence)
                next_token = self.sequence[prefix_len]
                reasoning_len = len(reasoning_ids)
                if reasoning_len >= budget_tokens:
                    forced[row_index, :] = floor
                    forced[row_index, next_token] = 0
                    continue
                if max_bias > 0.0 and bias_start_tokens > 0:
                    ramp_start = max(0, budget_tokens - bias_start_tokens)
                    if reasoning_len >= ramp_start:
                        progress = (reasoning_len - ramp_start + 1) / max(1.0, float(bias_start_tokens))
                        forced[row_index, next_token] += float(max_bias) * min(1.0, max(0.0, progress))
            return forced

    existing_processors = kwargs.get("logits_processor")
    processor = ForceReasoningCloseAfterBudget(int(input_ids.shape[-1]), close_ids)
    if existing_processors is None:
        kwargs["logits_processor"] = LogitsProcessorList([processor])
    elif isinstance(existing_processors, LogitsProcessorList):
        existing_processors.append(processor)
        kwargs["logits_processor"] = existing_processors
    else:
        kwargs["logits_processor"] = LogitsProcessorList(list(existing_processors) + [processor])


def patch_generate_helpers(
    model: Any,
    *,
    processor: Any | None,
    logits_to_keep: int,
    stop_at_result_close: bool,
    force_eos_token_id: int | None = None,
    reasoning_open_ids: list[int] | None = None,
    reasoning_close_ids: list[int] | None = None,
    reasoning_result_open_ids: list[int] | None = None,
    reasoning_budget_tokens: int = 0,
    reasoning_bias_start_tokens: int = 96,
    reasoning_max_bias: float = 8.0,
) -> None:
    if logits_to_keep <= 0 and not stop_at_result_close and reasoning_budget_tokens <= 0:
        return
    stop_ids = result_close_token_ids(processor) if stop_at_result_close and processor is not None else []
    reasoning_open_ids = list(reasoning_open_ids or [])
    reasoning_close_ids = list(reasoning_close_ids or [])
    reasoning_result_open_ids = list(reasoning_result_open_ids or [])
    targets = [
        ("model", model),
        ("base_model", getattr(model, "base_model", None)),
        ("base_model.model", getattr(getattr(model, "base_model", None), "model", None)),
    ]
    patched = 0
    for label, target in targets:
        if target is None:
            continue
        original = getattr(target, "generate", None)
        if original is None or getattr(original, "_pipeline_v2_generation_patched", False):
            continue
        eos_token_id = force_eos_token_id if force_eos_token_id is not None and force_eos_token_id >= 0 else infer_eos_token_id(processor, target)

        def generate_with_pipeline_v2_helpers(
            *args: Any,
            _original: Any = original,
            _eos_token_id: int | None = eos_token_id,
            **kwargs: Any,
        ) -> Any:
            if logits_to_keep > 0:
                kwargs.setdefault("logits_to_keep", logits_to_keep)
            if reasoning_open_ids and reasoning_close_ids and reasoning_budget_tokens > 0:
                append_reasoning_budget_processor(
                    kwargs,
                    args,
                    reasoning_open_ids,
                    reasoning_close_ids,
                    reasoning_result_open_ids,
                    budget_tokens=reasoning_budget_tokens,
                    bias_start_tokens=reasoning_bias_start_tokens,
                    max_bias=reasoning_max_bias,
                )
            if stop_ids:
                append_result_close_stopping_criteria(kwargs, args, stop_ids, _eos_token_id)
            return _original(*args, **kwargs)

        generate_with_pipeline_v2_helpers._pipeline_v2_generation_patched = True
        setattr(target, "generate", generate_with_pipeline_v2_helpers)
        patched += 1
        print_json(
            "[generation/patch]",
            {
                "patched": label,
                "logits_to_keep": logits_to_keep if logits_to_keep > 0 else None,
                "result_close_stop": bool(stop_ids),
                "result_close_token_ids": stop_ids,
                "result_close_force_eos": bool(stop_ids and eos_token_id is not None),
                "eos_token_id": eos_token_id,
                "reasoning_budget_tokens": reasoning_budget_tokens if reasoning_close_ids else 0,
                "reasoning_open_token_ids": reasoning_open_ids,
                "reasoning_close_token_ids": reasoning_close_ids,
                "reasoning_result_open_token_ids": reasoning_result_open_ids,
            },
        )
    if patched == 0:
        print("[generation/patch] no generate target patched", flush=True)


def patch_grpo_generation_inference_mode(grpo_trainer_class: Any) -> None:
    """Force Unsloth inference mode around regular Transformers generation."""

    original = getattr(grpo_trainer_class, "_generate_single_turn", None)
    if original is None or getattr(original, "_pipeline_v2_inference_mode_patch", False):
        return

    def generate_single_turn_with_inference_mode(self: Any, prompts: list[str], images: Any) -> Any:
        use_vllm = bool(getattr(self, "use_vllm", False))
        was_training = getattr(getattr(self, "model", None), "training", None)
        switched = False
        start = time.time()
        if not use_vllm and hasattr(getattr(self, "model", None), "for_inference"):
            if not getattr(self, "_pipeline_v2_generation_inference_logged", False):
                print_json(
                    "[generation/inference_mode]",
                    {
                        "enabled": True,
                        "reason": "regular_transformers_generate",
                        "was_training": was_training,
                        "use_vllm": use_vllm,
                        "use_transformers_paged": bool(getattr(self, "use_transformers_paged", False)),
                        "batch_prompts": len(prompts),
                        "generation_batch_size": getattr(getattr(self, "args", None), "generation_batch_size", None),
                        "gradient_checkpointing": getattr(getattr(self, "args", None), "gradient_checkpointing", None),
                    },
                )
                self._pipeline_v2_generation_inference_logged = True
            self.model.for_inference()
            switched = True
        try:
            return original(self, prompts, images)
        finally:
            duration = time.time() - start
            print_json(
                "[generation/cycle_timing]",
                {
                    "seconds": round(duration, 3),
                    "batch_prompts": len(prompts),
                    "switched_to_inference": switched,
                    "was_training": was_training,
                    "use_vllm": use_vllm,
                    "use_transformers_paged": bool(getattr(self, "use_transformers_paged", False)),
                },
            )
            if switched and hasattr(getattr(self, "model", None), "for_training"):
                self.model.for_training(use_gradient_checkpointing=getattr(self.args, "gradient_checkpointing", True))

    generate_single_turn_with_inference_mode._pipeline_v2_inference_mode_patch = True
    grpo_trainer_class._generate_single_turn = generate_single_turn_with_inference_mode
    print_json("[generation/inference_mode_patch]", {"enabled": True})


def indexed_value(value: Any, index: int, default: Any = "") -> Any:
    if isinstance(value, list):
        return value[index] if index < len(value) else default
    return default if value is None else value


def reasoning_summary(text: str, *, open_tag: str, close_tag: str) -> dict[str, Any]:
    raw = str(text)
    start = raw.find(open_tag) if open_tag else -1
    end = raw.find(close_tag, start + len(open_tag)) if start >= 0 and close_tag else -1
    if start < 0:
        stripped = raw.lstrip()
        leading_offset = len(raw) - len(stripped)
        visible_prefixes = ("---\nthought\n", "thought\n")
        for prefix in visible_prefixes:
            if stripped.lower().startswith(prefix):
                prefix_end = leading_offset + len(prefix)
                result_start = raw.find("<result>", prefix_end)
                reasoning_text = raw[prefix_end : result_start if result_start >= 0 else len(raw)]
                return {
                    "reasoning_present": True,
                    "reasoning_closed": False,
                    "reasoning_chars": len(reasoning_text.strip()),
                    "reasoning_text": reasoning_text.strip(),
                }
        return {
            "reasoning_present": False,
            "reasoning_closed": False,
            "reasoning_chars": 0,
            "reasoning_text": "",
        }
    if end >= 0:
        reasoning_text = raw[start + len(open_tag) : end]
    else:
        result_start = raw.find("<result>", start + len(open_tag))
        reasoning_text = raw[start + len(open_tag) : result_start if result_start >= 0 else len(raw)]
    return {
        "reasoning_present": True,
        "reasoning_closed": end >= 0,
        "reasoning_chars": len(reasoning_text.strip()),
        "reasoning_text": reasoning_text.strip(),
    }


def token_words(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())


def jaccard_distance(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / max(1, len(left | right))


def ngram_set(items: list[str], n: int) -> set[tuple[str, ...]]:
    if len(items) < n:
        return set()
    return {tuple(items[index : index + n]) for index in range(len(items) - n + 1)}


def pos_ngram_set(text: str, n: int = 4) -> set[tuple[str, ...]]:
    try:
        from .gui_style_scoring import ngrams

        tags = [tag for _form, tag in morphs(text)]
        return set(ngrams(tags, n))
    except Exception:
        return set()


def content_ngram_set(text: str, n: int = 3) -> set[tuple[str, ...]]:
    try:
        from .gui_style_scoring import ngrams

        tokens = [
            form.lower()
            for form, tag in morphs(text)
            if any(str(tag).startswith(prefix) for prefix in CONTENT_TAG_PREFIXES)
        ]
        return set(ngrams(tokens, n))
    except Exception:
        return ngram_set(token_words(text), n)


def pairwise_jaccard_similarity(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def pairwise_text_similarity_components(left: str, right: str) -> dict[str, float]:
    left_words = token_words(left)
    right_words = token_words(right)
    char_sim = pairwise_jaccard_similarity(ngram_set(list(left), 5), ngram_set(list(right), 5))
    word_sim = pairwise_jaccard_similarity(ngram_set(left_words, 3), ngram_set(right_words, 3))
    content_sim = pairwise_jaccard_similarity(content_ngram_set(left, 3), content_ngram_set(right, 3))
    left_pos = pos_ngram_set(left, 4)
    right_pos = pos_ngram_set(right, 4)
    if left_pos or right_pos:
        pos_sim = pairwise_jaccard_similarity(left_pos, right_pos)
        combined = 0.30 * char_sim + 0.25 * word_sim + 0.25 * content_sim + 0.20 * pos_sim
        return {"combined": combined, "char": char_sim, "word": word_sim, "content": content_sim, "pos": pos_sim}
    combined = 0.35 * char_sim + 0.30 * word_sim + 0.35 * content_sim
    return {"combined": combined, "char": char_sim, "word": word_sim, "content": content_sim, "pos": float("nan")}


def pairwise_text_distance(left: str, right: str) -> float:
    return 1.0 - pairwise_text_similarity_components(left, right)["combined"]


def apply_distance_threshold_diversity_bonus(
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    groups: dict[str, list[int]],
    *,
    max_bonus: float,
) -> None:
    target = max(1e-6, float(args.group_diversity_target_distance))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for index in indices:
            distances: list[float] = []
            text = str(entries[index].get("result_text") or "")
            for other_index in indices:
                if other_index == index:
                    continue
                distances.append(pairwise_text_distance(text, str(entries[other_index].get("result_text") or "")))
            mean_distance = sum(distances) / max(1, len(distances))
            bonus = max_bonus * min(1.0, max(0.0, mean_distance) / target)
            entries[index]["diversity_distance"] = float(mean_distance)
            entries[index]["diversity_bonus"] = float(bonus)
            entries[index]["reward"] = float(max(-1.0, min(1.0, float(entries[index]["reward"]) + bonus)))


def apply_density_adjusted_diversity_bonus(
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    groups: dict[str, list[int]],
    *,
    max_bonus: float,
) -> None:
    smoothing = max(1e-6, float(args.group_diversity_density_smoothing))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        density_values: dict[int, float] = {}
        distance_values: dict[int, float] = {}
        rarity_values: dict[int, float] = {}
        component_values: dict[int, dict[str, float]] = {}
        for index in indices:
            similarities: list[float] = []
            distances: list[float] = []
            char_sims: list[float] = []
            word_sims: list[float] = []
            content_sims: list[float] = []
            pos_sims: list[float] = []
            text = str(entries[index].get("result_text") or "")
            for other_index in indices:
                if other_index == index:
                    continue
                components = pairwise_text_similarity_components(text, str(entries[other_index].get("result_text") or ""))
                similarity = finite_float(components.get("combined"), 0.0)
                similarities.append(similarity)
                distances.append(1.0 - similarity)
                char_sims.append(finite_float(components.get("char"), 0.0))
                word_sims.append(finite_float(components.get("word"), 0.0))
                content_sims.append(finite_float(components.get("content"), 0.0))
                pos_value = finite_float(components.get("pos"), float("nan"))
                if math.isfinite(pos_value):
                    pos_sims.append(pos_value)
            density = sum(similarities) / max(1, len(similarities))
            mean_distance = sum(distances) / max(1, len(distances))
            density_values[index] = float(density)
            distance_values[index] = float(mean_distance)
            rarity_values[index] = 1.0 / (smoothing + max(0.0, density))
            component_values[index] = {
                "char": sum(char_sims) / max(1, len(char_sims)),
                "word": sum(word_sims) / max(1, len(word_sims)),
                "content": sum(content_sims) / max(1, len(content_sims)),
                "pos": sum(pos_sims) / max(1, len(pos_sims)) if pos_sims else float("nan"),
            }
        mean_rarity = sum(rarity_values.values()) / max(1, len(rarity_values))
        for index in indices:
            relative_rarity = rarity_values[index] / max(1e-12, mean_rarity) - 1.0
            bonus = max_bonus * clip(relative_rarity, -1.0, 1.0)
            entries[index]["diversity_density"] = float(density_values[index])
            entries[index]["diversity_rarity"] = float(rarity_values[index])
            entries[index]["diversity_relative_rarity"] = float(relative_rarity)
            entries[index]["diversity_distance"] = float(distance_values[index])
            entries[index]["diversity_bonus"] = float(bonus)
            entries[index]["diversity_similarity_char"] = float(component_values[index]["char"])
            entries[index]["diversity_similarity_word"] = float(component_values[index]["word"])
            entries[index]["diversity_similarity_content"] = float(component_values[index]["content"])
            entries[index]["diversity_similarity_pos"] = float(component_values[index]["pos"])
            entries[index]["reward"] = float(max(-1.0, min(1.0, float(entries[index]["reward"]) + bonus)))


def apply_leave_one_out_diversity_bonus(
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    groups: dict[str, list[int]],
    *,
    max_bonus: float,
) -> None:
    """Redistribute set-level diversity through each rollout's marginal distance.

    This follows the SGRPO-style idea of using set diversity as a reward signal,
    but uses text/POS n-gram distances instead of domain-specific molecule metrics.
    The centered contribution keeps the group-average diversity bonus near zero,
    so it shapes the GRPO ranking without becoming a constant reward offset.
    """

    target = max(1e-6, float(args.group_diversity_target_distance))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        mean_distances: dict[int, float] = {}
        component_values: dict[int, dict[str, float]] = {}
        for index in indices:
            distances: list[float] = []
            char_sims: list[float] = []
            word_sims: list[float] = []
            content_sims: list[float] = []
            pos_sims: list[float] = []
            text = str(entries[index].get("result_text") or "")
            for other_index in indices:
                if other_index == index:
                    continue
                components = pairwise_text_similarity_components(text, str(entries[other_index].get("result_text") or ""))
                similarity = finite_float(components.get("combined"), 0.0)
                distances.append(1.0 - similarity)
                char_sims.append(finite_float(components.get("char"), 0.0))
                word_sims.append(finite_float(components.get("word"), 0.0))
                content_sims.append(finite_float(components.get("content"), 0.0))
                pos_value = finite_float(components.get("pos"), float("nan"))
                if math.isfinite(pos_value):
                    pos_sims.append(pos_value)
            mean_distances[index] = sum(distances) / max(1, len(distances))
            component_values[index] = {
                "char": sum(char_sims) / max(1, len(char_sims)),
                "word": sum(word_sims) / max(1, len(word_sims)),
                "content": sum(content_sims) / max(1, len(content_sims)),
                "pos": sum(pos_sims) / max(1, len(pos_sims)) if pos_sims else float("nan"),
            }
        group_mean_distance = sum(mean_distances.values()) / max(1, len(mean_distances))
        for index in indices:
            loo_contribution = mean_distances[index] - group_mean_distance
            scaled_contribution = loo_contribution / target
            bonus = max_bonus * clip(scaled_contribution, -1.0, 1.0)
            entries[index]["diversity_distance"] = float(mean_distances[index])
            entries[index]["diversity_loo_contribution"] = float(loo_contribution)
            entries[index]["diversity_bonus"] = float(bonus)
            entries[index]["diversity_similarity_char"] = float(component_values[index]["char"])
            entries[index]["diversity_similarity_word"] = float(component_values[index]["word"])
            entries[index]["diversity_similarity_content"] = float(component_values[index]["content"])
            entries[index]["diversity_similarity_pos"] = float(component_values[index]["pos"])
            entries[index]["reward"] = float(max(-1.0, min(1.0, float(entries[index]["reward"]) + bonus)))


def apply_mmr_reweighted_diversity_bonus(
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    groups: dict[str, list[int]],
    *,
    max_bonus: float,
) -> None:
    """MMR-style diversity reward reweighting for GRPO groups.

    MMR-GRPO prioritizes informative non-redundant completions. For open-ended
    prose, we approximate that by giving a small centered bonus to valid
    completions whose nearest equal-or-better peer is less similar. The quality
    gate prevents low-quality off-topic samples from receiving a large diversity
    bonus merely because they are different.
    """

    target = max(1e-6, float(args.group_diversity_target_distance))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        rewards = {index: finite_score(entries[index].get("reward"), default=-1.0) for index in indices}
        min_reward = min(rewards.values())
        max_reward = max(rewards.values())
        reward_span = max(1e-9, max_reward - min_reward)
        novelty_values: dict[int, float] = {}
        quality_values: dict[int, float] = {}
        component_values: dict[int, dict[str, float]] = {}
        for index in indices:
            peer_sims: list[float] = []
            equal_or_better_sims: list[float] = []
            char_sims: list[float] = []
            word_sims: list[float] = []
            content_sims: list[float] = []
            pos_sims: list[float] = []
            text = str(entries[index].get("result_text") or "")
            for other_index in indices:
                if other_index == index:
                    continue
                components = pairwise_text_similarity_components(text, str(entries[other_index].get("result_text") or ""))
                similarity = finite_float(components.get("combined"), 0.0)
                peer_sims.append(similarity)
                if rewards[other_index] >= rewards[index]:
                    equal_or_better_sims.append(similarity)
                char_sims.append(finite_float(components.get("char"), 0.0))
                word_sims.append(finite_float(components.get("word"), 0.0))
                content_sims.append(finite_float(components.get("content"), 0.0))
                pos_value = finite_float(components.get("pos"), float("nan"))
                if math.isfinite(pos_value):
                    pos_sims.append(pos_value)
            redundancy = max(equal_or_better_sims or peer_sims or [0.0])
            novelty_values[index] = 1.0 - redundancy
            normalized_quality = (rewards[index] - min_reward) / reward_span if max_reward > min_reward else 0.5
            quality_values[index] = 0.25 + 0.75 * clip(normalized_quality, 0.0, 1.0)
            component_values[index] = {
                "char": sum(char_sims) / max(1, len(char_sims)),
                "word": sum(word_sims) / max(1, len(word_sims)),
                "content": sum(content_sims) / max(1, len(content_sims)),
                "pos": sum(pos_sims) / max(1, len(pos_sims)) if pos_sims else float("nan"),
                "redundancy": redundancy,
            }
        group_mean_novelty = sum(novelty_values.values()) / max(1, len(novelty_values))
        for index in indices:
            mmr_contribution = quality_values[index] * (novelty_values[index] - group_mean_novelty)
            bonus = max_bonus * clip(mmr_contribution / target, -1.0, 1.0)
            entries[index]["diversity_distance"] = float(novelty_values[index])
            entries[index]["diversity_mmr_quality"] = float(quality_values[index])
            entries[index]["diversity_mmr_redundancy"] = float(component_values[index]["redundancy"])
            entries[index]["diversity_mmr_contribution"] = float(mmr_contribution)
            entries[index]["diversity_bonus"] = float(bonus)
            entries[index]["diversity_similarity_char"] = float(component_values[index]["char"])
            entries[index]["diversity_similarity_word"] = float(component_values[index]["word"])
            entries[index]["diversity_similarity_content"] = float(component_values[index]["content"])
            entries[index]["diversity_similarity_pos"] = float(component_values[index]["pos"])
            entries[index]["reward"] = float(max(-1.0, min(1.0, float(entries[index]["reward"]) + bonus)))


def apply_generate_diversity_bonus(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    max_bonus = max(0.0, float(args.group_diversity_bonus_max))
    if max_bonus <= 0.0:
        return
    all_group_counts: dict[str, int] = {}
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        if entry.get("task") != "generate":
            continue
        key = str(entry.get("row_id") or "")
        if not key:
            key = f"batch:{index // max(1, int(args.num_generations))}"
        all_group_counts[key] = all_group_counts.get(key, 0) + 1
        if entry.get("failed"):
            continue
        result_text = str(entry.get("result_text") or "").strip()
        if len(result_text) < max(120, int(args.group_diversity_min_chars)):
            continue
        groups.setdefault(key, []).append(index)
    for key, indices in groups.items():
        for index in indices:
            entries[index]["diversity_group_valid_count"] = len(indices)
            entries[index]["diversity_group_total_count"] = all_group_counts.get(key, len(indices))
    mode = str(getattr(args, "group_diversity_mode", "leave_one_out") or "leave_one_out")
    if mode == "none":
        return
    if mode == "distance_threshold":
        apply_distance_threshold_diversity_bonus(entries, args, groups, max_bonus=max_bonus)
        return
    if mode in {"leave_one_out", "sgrpo_leave_one_out"}:
        apply_leave_one_out_diversity_bonus(entries, args, groups, max_bonus=max_bonus)
        return
    if mode == "mmr_reweighted":
        apply_mmr_reweighted_diversity_bonus(entries, args, groups, max_bonus=max_bonus)
        return
    apply_density_adjusted_diversity_bonus(entries, args, groups, max_bonus=max_bonus)


def log_reward_aggregates(entries: list[dict[str, Any]], *, call_index: int, enabled: bool) -> None:
    if not enabled or not entries:
        return
    try:
        import wandb
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return

    def add_mean(payload: dict[str, float], key: str, values: list[float]) -> None:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if finite:
            payload[key] = statistics.fmean(finite)

    payload: dict[str, float] = {"reward_call_index": float(call_index)}
    tasks = sorted({str(entry.get("task") or "unknown") for entry in entries})
    for task in tasks:
        task_entries = [entry for entry in entries if str(entry.get("task") or "unknown") == task]
        prefix = f"reward_by_task/{task}"
        add_mean(payload, f"{prefix}/reward_mean", [finite_score(entry.get("reward"), default=float("nan")) for entry in task_entries])
        add_mean(
            payload,
            f"{prefix}/style_score_raw_mean",
            [finite_score((entry.get("components") or {}).get("style_score_raw"), default=float("nan")) for entry in task_entries],
        )
        payload[f"{prefix}/count"] = float(len(task_entries))
        payload[f"{prefix}/result_close_rate"] = statistics.fmean(
            1.0 if ((entry.get("scored") or {}).get("metrics") or {}).get("has_result_close") else 0.0
            for entry in task_entries
        )
        payload[f"{prefix}/contract_ok_rate"] = statistics.fmean(
            1.0 if ((entry.get("scored") or {}).get("metrics") or {}).get("result_contract_ok") else 0.0
            for entry in task_entries
        )
        for component in (
            "length_penalty",
            "diversity_bonus",
            "diversity_loo_contribution",
            "diversity_mmr_contribution",
            "rewrite_edit_amount",
            "rewrite_edit_score",
            "rewrite_style_improvement",
            "rewrite_edit_gate",
            "rewrite_improvement_score",
            "rewrite_family_anti_slop_score",
            "rewrite_family_comma_score",
            "rewrite_family_pos_4_6_score",
            "rewrite_family_pos_3_score",
            "rewrite_family_modifier_score",
            "rewrite_family_sentence_edge_score",
            "rewrite_family_sentence_length_score",
        ):
            add_mean(
                payload,
                f"{prefix}/{component}_mean",
                [finite_score((entry.get("components") or {}).get(component), default=float("nan")) for entry in task_entries],
            )
        family_names: set[str] = set()
        metric_names: set[str] = set()
        for entry in task_entries:
            scored = dict(entry.get("scored") or {})
            family_names.update(str(name) for name in (scored.get("family_scores") or {}).keys())
            metric_names.update(str(name) for name in (scored.get("metric_scores") or {}).keys())
        for family in sorted(family_names):
            add_mean(
                payload,
                f"{prefix}/family/{family}_mean",
                [finite_score(((entry.get("scored") or {}).get("family_scores") or {}).get(family), default=float("nan")) for entry in task_entries],
            )
        watched_metrics = {
            "anti_slop_density",
            "comma_per_1k_chars",
            "sentence_final_token_repeat_rate",
            "sentence_initial_token_repeat_rate",
            "sentence_length_cv",
            "sentence_length_iqr_ratio",
            "pos_3gram_repeat_rate",
            "pos_4gram_diversity",
            "pos_5gram_repeat_rate",
        }
        for metric in sorted(metric_names):
            if metric in watched_metrics or metric.startswith("simile_"):
                add_mean(
                    payload,
                    f"{prefix}/metric/{metric}_mean",
                    [finite_score(((entry.get("scored") or {}).get("metric_scores") or {}).get(metric), default=float("nan")) for entry in task_entries],
                )
    try:
        wandb.log(payload)
    except Exception:
        return


def make_reward_func(args: argparse.Namespace) -> Any:
    scorer: GuiStyleScorer | None = None
    rewrite_reference_stats = build_rewrite_reference_stats(Path(args.dataset))
    sample_log_calls = 0
    sample_log_path = Path(args.sample_log_path) if args.sample_log_path else Path(args.output) / "reward_samples.jsonl"
    if args.sample_log_every > 0:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reward_mode == "gui_style":
        scorer = GuiStyleScorer(
            reference_path=Path(args.gui_style_reference),
            anti_slop_lexicon_path=Path(args.anti_slop_lexicon),
            translationese_model_path=Path(args.translationese_model),
            disabled_metrics=parse_csv_set(args.disabled_style_metrics),
            metric_weight_overrides=parse_metric_weight_overrides(args.style_metric_weight_overrides),
        )
        scorer_status = scorer.score_text("<result>테스트 문장입니다.</result>", require_result_tags=True)["scorer_status"]
        scorer_status["disabled_metrics"] = sorted(parse_csv_set(args.disabled_style_metrics))
        scorer_status["metric_weight_overrides"] = parse_metric_weight_overrides(args.style_metric_weight_overrides)
        print_json("[reward/gui_style]", scorer_status)
    if rewrite_reference_stats.get("count", 0):
        print_json("[reward/rewrite_reference]", rewrite_reference_stats)

    def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
        nonlocal sample_log_calls
        sample_log_calls += 1
        entries: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        for index, completion in enumerate(completions):
            text = completion_text(completion)
            task = str(indexed_value(kwargs.get("task"), index, args.grpo_task) or args.grpo_task)
            source_text = str(indexed_value(kwargs.get("source_text"), index, ""))
            row_id = str(indexed_value(kwargs.get("id"), index, ""))
            scored: dict[str, Any] = {}
            reward_components: dict[str, Any] = {
                "style_score_raw": float("nan"),
                "length_penalty": 0.0,
                "diversity_bonus": 0.0,
                "diversity_distance": float("nan"),
                "diversity_density": float("nan"),
                "diversity_rarity": float("nan"),
                "contract_penalty": 0.0,
                "fail_reward_applied": False,
            }
            if scorer is not None:
                scored = scorer.score_text(
                    text,
                    task=task,
                    source_text=source_text,
                    require_result_tags=args.require_result_tags,
                )
                fail_reward = contract_or_collapse_fail_reward(scored, fail_reward=args.collapse_fail_reward)
                if fail_reward is not None:
                    reward_components["fail_reward_applied"] = True
                    reward = fail_reward
                    entries.append(
                        {
                            "index": index,
                            "row_id": row_id,
                            "task": task,
                            "text": text,
                            "scored": scored,
                            "reward": reward,
                            "components": reward_components,
                            "failed": True,
                            "result_text": str(scored.get("result_text") or ""),
                        }
                    )
                    continue
                style_reward = finite_score(scored.get("score"), default=-1.0)
                reward_components["style_score_raw"] = style_reward
                reward = style_reward
                metrics = dict(scored.get("metrics") or {})
                contract_reason = str(metrics.get("result_contract_reason") or "")
                if contract_reason:
                    contract_penalty = contract_penalty_for_reason(contract_reason)
                    reward_components["contract_penalty"] = contract_penalty
                    reward += contract_penalty
                if task == "rewrite":
                    reward = rewrite_weighted_style_reward(
                        style_reward=reward,
                        family_scores=dict(scored.get("family_scores") or {}),
                        args=args,
                        components=reward_components,
                    )
                    reward = combine_rewrite_reward(
                        style_reward=reward,
                        metrics=dict(scored.get("metrics") or {}),
                        reference_stats=rewrite_reference_stats,
                        edit_weight=args.rewrite_edit_weight,
                        improvement_weight=args.rewrite_improvement_weight,
                        improvement_scale=args.rewrite_improvement_scale,
                        low_edit_penalty_max=args.rewrite_low_edit_penalty_max,
                        edit_gate_min=args.rewrite_edit_gate_min,
                        edit_gate_q25=args.rewrite_edit_gate_q25,
                        edit_gate_q50=args.rewrite_edit_gate_q50,
                        components=reward_components,
                    )
                if task == "generate":
                    length_penalty = generate_length_penalty(
                        str(scored.get("result_text") or ""),
                        min_output_chars=args.generate_min_output_chars,
                        max_output_chars=args.generate_max_output_chars,
                        short_output_penalty=args.short_output_penalty,
                        long_output_penalty=args.long_output_penalty,
                    )
                    reward_components["length_penalty"] = length_penalty
                    reward += length_penalty
                reward = float(max(-1.0, min(1.0, reward)))
                entries.append(
                    {
                        "index": index,
                        "row_id": row_id,
                        "task": task,
                        "text": text,
                        "scored": scored,
                        "reward": reward,
                        "components": reward_components,
                        "failed": False,
                        "result_text": str(scored.get("result_text") or ""),
                    }
                )
            else:
                reward = smoke_reward_score(
                    text,
                    require_result_tags=args.require_result_tags,
                    max_output_chars=args.generate_max_output_chars,
                )
                entries.append(
                    {
                        "index": index,
                        "row_id": row_id,
                        "task": task,
                        "text": text,
                        "scored": {},
                        "reward": reward,
                        "components": reward_components,
                        "failed": False,
                        "result_text": analyze_result_contract(text, require_result_tags=False).result_text,
                    }
                )
        apply_generate_diversity_bonus(entries, args)
        apply_partial_std_reward_shaping(entries, args)
        apply_std_based_update_weight(entries, args)
        rewards = [float(entry["reward"]) for entry in entries]
        log_reward_aggregates(entries, call_index=sample_log_calls, enabled=bool(args.wandb_reward_component_log))
        if args.sample_log_every > 0 and sample_log_calls % args.sample_log_every == 0:
            for entry in entries[: max(1, args.sample_log_max_items)]:
                components = dict(entry.get("components") or {})
                components["diversity_bonus"] = finite_score(entry.get("diversity_bonus", components.get("diversity_bonus")), default=0.0)
                components["diversity_distance"] = finite_score(entry.get("diversity_distance", components.get("diversity_distance")), default=float("nan"))
                components["diversity_density"] = finite_score(entry.get("diversity_density", components.get("diversity_density")), default=float("nan"))
                components["diversity_rarity"] = finite_score(entry.get("diversity_rarity", components.get("diversity_rarity")), default=float("nan"))
                components["diversity_relative_rarity"] = finite_score(entry.get("diversity_relative_rarity"), default=float("nan"))
                components["diversity_loo_contribution"] = finite_score(entry.get("diversity_loo_contribution"), default=float("nan"))
                components["diversity_similarity_char"] = finite_score(entry.get("diversity_similarity_char"), default=float("nan"))
                components["diversity_similarity_word"] = finite_score(entry.get("diversity_similarity_word"), default=float("nan"))
                components["diversity_similarity_content"] = finite_score(entry.get("diversity_similarity_content"), default=float("nan"))
                components["diversity_similarity_pos"] = finite_score(entry.get("diversity_similarity_pos"), default=float("nan"))
                components["diversity_mmr_quality"] = finite_score(entry.get("diversity_mmr_quality"), default=float("nan"))
                components["diversity_mmr_redundancy"] = finite_score(entry.get("diversity_mmr_redundancy"), default=float("nan"))
                components["diversity_mmr_contribution"] = finite_score(entry.get("diversity_mmr_contribution"), default=float("nan"))
                for name in (
                    "rewrite_style_blended",
                    "rewrite_style_blend_weight_sum",
                    "rewrite_edit_amount",
                    "rewrite_edit_score",
                    "rewrite_style_improvement",
                    "rewrite_edit_gate",
                    "rewrite_improvement_score",
                    "rewrite_family_anti_slop_score",
                    "rewrite_family_translationese_score",
                    "rewrite_family_comma_score",
                    "rewrite_family_pos_4_6_score",
                    "rewrite_family_pos_3_score",
                    "rewrite_family_modifier_score",
                    "rewrite_family_lexical_repetition_score",
                    "rewrite_family_sentence_edge_score",
                    "rewrite_family_sentence_length_score",
                ):
                    if name in components:
                        components[name] = finite_score(components.get(name), default=float("nan"))
                components["diversity_group_valid_count"] = int(entry.get("diversity_group_valid_count", 0) or 0)
                components["diversity_group_total_count"] = int(entry.get("diversity_group_total_count", 0) or 0)
                sample_rows.append(
                    reward_sample_row(
                        args,
                        sample_log_calls,
                        int(entry["index"]),
                        str(entry.get("row_id") or ""),
                        str(entry.get("task") or ""),
                        str(entry.get("text") or ""),
                        dict(entry.get("scored") or {}),
                        float(entry["reward"]),
                        components=components,
                    )
                )
        if sample_rows:
            with sample_log_path.open("a", encoding="utf-8") as handle:
                for row in sample_rows[: max(1, args.sample_log_max_items)]:
                    handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
        return rewards

    return reward_func


def apply_partial_std_reward_shaping(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Pre-shape rewards so trainer scale_rewards can be disabled safely.

    Full GRPO group scaling divides by group std. This can amplify tiny reward
    differences when std is small. Here we keep a controlled std effect by
    dividing centered rewards by max(std, floor)**power. With power=0.5, std
    still matters, but much less than full normalization. The trainer must not
    apply another std normalization afterwards, or this transform cancels out.
    """

    power = float(getattr(args, "reward_std_shaping_power", 0.0) or 0.0)
    if power <= 0.0:
        return
    floor = max(1e-8, float(getattr(args, "reward_std_shaping_floor", 0.10) or 0.10))
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        row_id = str(entry.get("row_id") or "")
        if row_id:
            key = f"id:{row_id}"
        else:
            index = int(entry.get("index", 0) or 0)
            key = f"chunk:{index // max(1, int(getattr(args, 'num_generations', 1) or 1))}"
        groups.setdefault(key, []).append(entry)

    shaped_summaries: list[dict[str, Any]] = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        raw_rewards = [float(entry["reward"]) for entry in group]
        mean_reward = sum(raw_rewards) / len(raw_rewards)
        variance = sum((value - mean_reward) ** 2 for value in raw_rewards) / len(raw_rewards)
        std = math.sqrt(max(0.0, variance))
        divisor = max(std, floor) ** power
        shaped_rewards = [mean_reward + ((value - mean_reward) / divisor) for value in raw_rewards]
        shaped_rewards = [float(max(-1.0, min(1.0, value))) for value in shaped_rewards]
        shaped_mean = sum(shaped_rewards) / len(shaped_rewards)
        shaped_variance = sum((value - shaped_mean) ** 2 for value in shaped_rewards) / len(shaped_rewards)
        shaped_std = math.sqrt(max(0.0, shaped_variance))
        for entry, raw_reward, shaped_reward in zip(group, raw_rewards, shaped_rewards):
            components = dict(entry.get("components") or {})
            components["reward_raw_before_std_shaping"] = raw_reward
            components["reward_std_shaping_group_std"] = std
            components["reward_std_shaping_floor"] = floor
            components["reward_std_shaping_power"] = power
            components["reward_std_shaping_divisor"] = divisor
            entry["components"] = components
            entry["reward_raw_before_std_shaping"] = raw_reward
            entry["reward"] = shaped_reward
        shaped_summaries.append(
            {
                "key": key,
                "count": len(group),
                "std_raw": std,
                "std_shaped": shaped_std,
                "divisor": divisor,
            }
        )
    if shaped_summaries:
        min_std = min(item["std_raw"] for item in shaped_summaries)
        mean_std = sum(item["std_raw"] for item in shaped_summaries) / len(shaped_summaries)
        mean_divisor = sum(item["divisor"] for item in shaped_summaries) / len(shaped_summaries)
        try:
            import wandb

            wandb.log(
                {
                    "reward_std_shaping/groups": len(shaped_summaries),
                    "reward_std_shaping/raw_std_min": min_std,
                    "reward_std_shaping/raw_std_mean": mean_std,
                    "reward_std_shaping/divisor_mean": mean_divisor,
                    "reward_std_shaping/power": power,
                    "reward_std_shaping/floor": floor,
                }
            )
        except Exception:
            pass


def apply_std_based_update_weight(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Shrink low-std group reward dispersion without changing group mean.

    This approximates a group-level update weight while staying outside the
    trainer internals. With trainer reward scaling disabled, GRPO advantages
    are driven by within-group reward differences. Pulling rewards toward the
    group mean therefore reduces the effective update for low-signal groups,
    including the second policy update when num_iterations > 1.
    """

    min_weight = float(getattr(args, "reward_std_update_weight_min", 1.0) or 1.0)
    if min_weight >= 1.0:
        return
    min_weight = clip(min_weight, 0.0, 1.0)
    low = max(0.0, float(getattr(args, "reward_std_update_weight_std_low", 0.05) or 0.05))
    high = max(low + 1e-8, float(getattr(args, "reward_std_update_weight_std_high", 0.14) or 0.14))
    source = str(getattr(args, "reward_std_update_weight_source", "raw") or "raw")

    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        row_id = str(entry.get("row_id") or "")
        if row_id:
            key = f"id:{row_id}"
        else:
            index = int(entry.get("index", 0) or 0)
            key = f"chunk:{index // max(1, int(getattr(args, 'num_generations', 1) or 1))}"
        groups.setdefault(key, []).append(entry)

    summaries: list[dict[str, float]] = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        raw_values = []
        for entry in group:
            components = dict(entry.get("components") or {})
            raw = components.get("reward_raw_before_std_shaping")
            if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                raw = entry.get("reward")
            raw_values.append(float(raw))
        mean_raw = sum(raw_values) / len(raw_values)
        raw_std = math.sqrt(max(0.0, sum((value - mean_raw) ** 2 for value in raw_values) / len(raw_values)))
        if source == "shaped":
            basis_values = [float(entry["reward"]) for entry in group]
            basis_mean = sum(basis_values) / len(basis_values)
            basis_std = math.sqrt(max(0.0, sum((value - basis_mean) ** 2 for value in basis_values) / len(basis_values)))
            std_for_weight = basis_std
        else:
            std_for_weight = raw_std
        ramp = clip((std_for_weight - low) / max(1e-8, high - low), 0.0, 1.0)
        weight = min_weight + (1.0 - min_weight) * ramp
        shaped_values = [float(entry["reward"]) for entry in group]
        shaped_mean = sum(shaped_values) / len(shaped_values)
        for entry, current_reward in zip(group, shaped_values):
            weighted_reward = shaped_mean + (current_reward - shaped_mean) * weight
            weighted_reward = float(max(-1.0, min(1.0, weighted_reward)))
            components = dict(entry.get("components") or {})
            components["reward_std_update_weight"] = weight
            components["reward_std_update_weight_source_std"] = std_for_weight
            components["reward_std_update_weight_raw_std"] = raw_std
            components["reward_std_update_weight_min"] = min_weight
            components["reward_std_update_weight_std_low"] = low
            components["reward_std_update_weight_std_high"] = high
            components["reward_before_std_update_weight"] = float(current_reward)
            entry["components"] = components
            entry["reward_before_std_update_weight"] = float(current_reward)
            entry["reward"] = weighted_reward
        summaries.append({"raw_std": raw_std, "std_for_weight": std_for_weight, "weight": weight})
    if summaries:
        try:
            import wandb

            wandb.log(
                {
                    "reward_std_update_weight/groups": len(summaries),
                    "reward_std_update_weight/weight_min": min(item["weight"] for item in summaries),
                    "reward_std_update_weight/weight_mean": sum(item["weight"] for item in summaries) / len(summaries),
                    "reward_std_update_weight/source_std_mean": sum(item["std_for_weight"] for item in summaries) / len(summaries),
                    "reward_std_update_weight/raw_std_mean": sum(item["raw_std"] for item in summaries) / len(summaries),
                    "reward_std_update_weight/std_low": low,
                    "reward_std_update_weight/std_high": high,
                    "reward_std_update_weight/min": min_weight,
                }
            )
        except Exception:
            pass


def reward_sample_row(
    args: argparse.Namespace,
    call_index: int,
    completion_index: int,
    row_id: str,
    task: str,
    text: str,
    scored: dict[str, Any],
    reward: float,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = dict(scored.get("metrics") or {})
    components = dict(components or {})
    result_text = str(scored.get("result_text") or analyze_result_contract(text, require_result_tags=False).result_text)
    reasoning = reasoning_summary(text, open_tag=args.reasoning_open_tag, close_tag=args.reasoning_close_tag)
    length_penalty = finite_score(components.get("length_penalty"), default=0.0)
    diversity_bonus = finite_score(components.get("diversity_bonus"), default=0.0)
    diversity_distance = finite_score(components.get("diversity_distance"), default=float("nan"))
    diversity_density = finite_score(components.get("diversity_density"), default=float("nan"))
    diversity_rarity = finite_score(components.get("diversity_rarity"), default=float("nan"))
    rewrite_component_names = (
        "rewrite_style_blended",
        "rewrite_style_blend_weight_sum",
        "rewrite_edit_amount",
        "rewrite_edit_score",
        "rewrite_style_improvement",
        "rewrite_edit_gate",
        "rewrite_improvement_score",
        "rewrite_family_anti_slop_score",
        "rewrite_family_translationese_score",
        "rewrite_family_comma_score",
        "rewrite_family_pos_4_6_score",
        "rewrite_family_pos_3_score",
        "rewrite_family_modifier_score",
        "rewrite_family_lexical_repetition_score",
        "rewrite_family_sentence_edge_score",
        "rewrite_family_sentence_length_score",
    )
    rewrite_component_values = {
        name: finite_score(components.get(name), default=float("nan"))
        for name in rewrite_component_names
        if name in components
    }
    reward_std_shaping_values = {
        name: finite_score(components.get(name), default=float("nan"))
        for name in (
            "reward_raw_before_std_shaping",
            "reward_std_shaping_group_std",
            "reward_std_shaping_floor",
            "reward_std_shaping_power",
            "reward_std_shaping_divisor",
            "reward_before_std_update_weight",
            "reward_std_update_weight",
            "reward_std_update_weight_source_std",
            "reward_std_update_weight_raw_std",
            "reward_std_update_weight_min",
            "reward_std_update_weight_std_low",
            "reward_std_update_weight_std_high",
        )
        if name in components
    }
    return {
        "time": time.time(),
        "call_index": call_index,
        "completion_index": completion_index,
        "id": row_id,
        "task": task,
        "reward": float(reward),
        "score": finite_score(scored.get("score"), default=float("nan")),
        "style_score_raw": finite_score(components.get("style_score_raw"), default=float("nan")),
        "length_penalty": length_penalty,
        "diversity_bonus": diversity_bonus,
        "diversity_distance": diversity_distance,
        "diversity_density": diversity_density,
        "diversity_rarity": diversity_rarity,
        "diversity_relative_rarity": finite_score(components.get("diversity_relative_rarity"), default=float("nan")),
        "diversity_loo_contribution": finite_score(components.get("diversity_loo_contribution"), default=float("nan")),
        "diversity_similarity_char": finite_score(components.get("diversity_similarity_char"), default=float("nan")),
        "diversity_similarity_word": finite_score(components.get("diversity_similarity_word"), default=float("nan")),
        "diversity_similarity_content": finite_score(components.get("diversity_similarity_content"), default=float("nan")),
        "diversity_similarity_pos": finite_score(components.get("diversity_similarity_pos"), default=float("nan")),
        "diversity_mmr_quality": finite_score(components.get("diversity_mmr_quality"), default=float("nan")),
        "diversity_mmr_redundancy": finite_score(components.get("diversity_mmr_redundancy"), default=float("nan")),
        "diversity_mmr_contribution": finite_score(components.get("diversity_mmr_contribution"), default=float("nan")),
        **rewrite_component_values,
        **reward_std_shaping_values,
        "diversity_group_valid_count": int(components.get("diversity_group_valid_count", 0) or 0),
        "diversity_group_total_count": int(components.get("diversity_group_total_count", 0) or 0),
        "fail_reward_applied": bool(components.get("fail_reward_applied", False)),
        "collapse_reason": str(metrics.get("collapse_reason") or ""),
        "result_contract_reason": str(metrics.get("result_contract_reason") or ""),
        "result_contract_ok": bool(metrics.get("result_contract_ok")) if "result_contract_ok" in metrics else None,
        "has_result_open": bool(metrics.get("has_result_open")) if "has_result_open" in metrics else None,
        "has_result_close": bool(metrics.get("has_result_close")) if "has_result_close" in metrics else None,
        "result_open_count": int(metrics.get("result_open_count", 0) or 0),
        "result_close_count": int(metrics.get("result_close_count", 0) or 0),
        "post_result_chars": int(metrics.get("post_result_chars", 0) or 0),
        "result_text_chars": len(result_text),
        "raw_text_chars": len(text),
        "reasoning_present": bool(reasoning["reasoning_present"]),
        "reasoning_closed": bool(reasoning["reasoning_closed"]),
        "reasoning_chars": int(reasoning["reasoning_chars"]),
        "reasoning_text": str(reasoning["reasoning_text"])[:1000],
        "family_scores": dict(scored.get("family_scores") or {}),
        "metric_scores": dict(scored.get("metric_scores") or {}),
        "pos_usage_scores": dict(scored.get("pos_usage_scores") or {}),
        "components": {
            "style_score_raw": finite_score(components.get("style_score_raw"), default=float("nan")),
            "length_penalty": length_penalty,
            "diversity_bonus": diversity_bonus,
            "diversity_distance": diversity_distance,
            "diversity_density": diversity_density,
            "diversity_rarity": diversity_rarity,
            "diversity_relative_rarity": finite_score(components.get("diversity_relative_rarity"), default=float("nan")),
            "diversity_loo_contribution": finite_score(components.get("diversity_loo_contribution"), default=float("nan")),
            "diversity_similarity_char": finite_score(components.get("diversity_similarity_char"), default=float("nan")),
            "diversity_similarity_word": finite_score(components.get("diversity_similarity_word"), default=float("nan")),
            "diversity_similarity_content": finite_score(components.get("diversity_similarity_content"), default=float("nan")),
            "diversity_similarity_pos": finite_score(components.get("diversity_similarity_pos"), default=float("nan")),
            "diversity_mmr_quality": finite_score(components.get("diversity_mmr_quality"), default=float("nan")),
            "diversity_mmr_redundancy": finite_score(components.get("diversity_mmr_redundancy"), default=float("nan")),
            "diversity_mmr_contribution": finite_score(components.get("diversity_mmr_contribution"), default=float("nan")),
            **rewrite_component_values,
            **reward_std_shaping_values,
            "fail_reward_applied": bool(components.get("fail_reward_applied", False)),
        },
        "text": text[: max(0, int(args.sample_log_text_chars))],
    }


def build_rewrite_reference_stats(dataset_path: Path) -> dict[str, Any]:
    edit_amounts: list[float] = []
    for row in read_jsonl(dataset_path):
        task = selected_task(row)
        if task != "rewrite":
            continue
        source_text = str(row.get("source_text") or "")
        reference_text = str(row.get("reference_text") or "")
        if not source_text.strip() or not reference_text.strip():
            continue
        metrics = rewrite_metrics(source_text, reference_text)
        edit_amounts.append(rewrite_edit_amount(metrics))
    summary = summarize_numeric(edit_amounts)
    return {
        "count": int(summary.get("count", 0)),
        "edit_q25": finite_float(summary.get("p25")),
        "edit_q50": finite_float(summary.get("median")),
        "edit_q75": finite_float(summary.get("p75")),
        "edit_mean": finite_float(summary.get("mean")),
    }


def rewrite_edit_score(edit_amount: float, reference_stats: dict[str, Any]) -> float:
    if not math.isfinite(edit_amount):
        return -0.6
    q25 = finite_float(reference_stats.get("edit_q25"), 0.22)
    q50 = finite_float(reference_stats.get("edit_q50"), 0.38)
    q75 = finite_float(reference_stats.get("edit_q75"), max(q50, 0.52))
    if not math.isfinite(q25) or q25 <= 0:
        q25 = 0.22
    if not math.isfinite(q50) or q50 <= q25:
        q50 = q25 + 0.12
    if not math.isfinite(q75) or q75 <= q50:
        q75 = q50 + 0.14
    if edit_amount < q25:
        return clip(-1.0 + edit_amount / max(1e-9, q25), -1.0, 0.0)
    if edit_amount <= q75:
        return clip(0.15 + 0.10 * (1.0 - abs(edit_amount - q50) / max(1e-9, q75 - q25)), 0.05, 0.25)
    return clip(0.10 - 0.65 * min(1.0, (edit_amount - q75) / max(1e-9, q75 - q50)), -0.55, 0.10)


def combine_rewrite_reward(
    *,
    style_reward: float,
    metrics: dict[str, Any],
    reference_stats: dict[str, Any],
    edit_weight: float,
    improvement_weight: float,
    improvement_scale: float = 0.35,
    low_edit_penalty_max: float = 0.35,
    edit_gate_min: float = 0.20,
    edit_gate_q25: float = 0.40,
    edit_gate_q50: float = 1.00,
    components: dict[str, Any] | None = None,
) -> float:
    edit_amount = finite_float(metrics.get("rewrite_edit_amount"))
    edit_score = rewrite_edit_score(edit_amount, reference_stats)
    improvement = finite_float(metrics.get("rewrite_style_improvement"), 0.0)
    q25 = finite_float(reference_stats.get("edit_q25"), 0.22)
    q50 = finite_float(reference_stats.get("edit_q50"), max(q25 + 0.12, 0.38))
    edit_gate_min = clip(float(edit_gate_min), 0.0, 1.0)
    edit_gate_q25 = clip(float(edit_gate_q25), edit_gate_min, 1.0)
    edit_gate_q50 = clip(float(edit_gate_q50), edit_gate_q25, 1.0)
    if math.isfinite(edit_amount):
        if edit_amount < q25:
            edit_gate = edit_gate_min + (edit_gate_q25 - edit_gate_min) * clip(edit_amount / max(1e-9, q25), 0.0, 1.0)
        else:
            edit_gate = edit_gate_q25 + (edit_gate_q50 - edit_gate_q25) * clip((edit_amount - q25) / max(1e-9, q50 - q25), 0.0, 1.0)
    else:
        edit_gate = edit_gate_min
    improvement_scale = max(1e-6, float(improvement_scale))
    improvement_score = clip(improvement / improvement_scale, -1.0, 1.0) * edit_gate
    if math.isfinite(edit_amount) and edit_amount < q25:
        low_edit_ratio = 1.0 - clip(edit_amount / max(1e-9, q25), 0.0, 1.0)
        improvement_score -= max(0.0, float(low_edit_penalty_max)) * low_edit_ratio
    if components is not None:
        components["rewrite_edit_amount"] = edit_amount
        components["rewrite_edit_score"] = edit_score
        components["rewrite_style_improvement"] = improvement
        components["rewrite_edit_gate"] = edit_gate
        components["rewrite_improvement_score"] = improvement_score
    total_weight = 1.0 + max(0.0, edit_weight) + max(0.0, improvement_weight)
    return (
        style_reward
        + max(0.0, edit_weight) * edit_score
        + max(0.0, improvement_weight) * improvement_score
    ) / max(1e-9, total_weight)


def rewrite_weighted_style_reward(
    *,
    style_reward: float,
    family_scores: dict[str, Any],
    args: argparse.Namespace,
    components: dict[str, Any],
) -> float:
    """Reweight rewrite style reward toward the axes that currently separate AI/human most."""

    parts: list[tuple[float, float, str]] = []
    base_weight = max(0.0, float(getattr(args, "rewrite_style_base_weight", 1.0)))
    if math.isfinite(style_reward) and base_weight > 0:
        parts.append((style_reward, base_weight, "style_base"))
    family_weights = {
        "anti_slop": float(getattr(args, "rewrite_anti_slop_family_weight", 0.0)),
        "translationese": float(getattr(args, "rewrite_translationese_family_weight", 0.0)),
        "comma": float(getattr(args, "rewrite_comma_family_weight", 0.0)),
        "pos_4_6": float(getattr(args, "rewrite_pos_family_weight", 0.0)),
        "pos_3": 0.35 * float(getattr(args, "rewrite_pos_family_weight", 0.0)),
        "modifier": float(getattr(args, "rewrite_modifier_family_weight", 0.0)),
        "lexical_repetition": float(getattr(args, "rewrite_lexical_family_weight", 0.0)),
        "sentence_edge": float(getattr(args, "rewrite_sentence_edge_family_weight", 0.0)),
        "sentence_length": float(getattr(args, "rewrite_sentence_length_family_weight", 0.0)),
    }
    for family, weight in family_weights.items():
        score = finite_float(family_scores.get(family))
        if weight > 0 and math.isfinite(score):
            parts.append((score, weight, family))
            components[f"rewrite_family_{family}_score"] = score
            components[f"rewrite_family_{family}_weight"] = weight
    if not parts:
        return style_reward
    total_weight = sum(weight for _score, weight, _name in parts)
    blended = sum(score * weight for score, weight, _name in parts) / max(1e-9, total_weight)
    components["rewrite_style_blended"] = blended
    components["rewrite_style_blend_weight_sum"] = total_weight
    components["rewrite_style_blend_parts"] = {name: {"score": score, "weight": weight} for score, weight, name in parts}
    return clip(blended)


def add_common_grpo_args(parser: argparse.ArgumentParser, *, task: str, default_output_name: str) -> None:
    parser.add_argument("--dataset", default=str(DEFAULT_GRPO_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT / default_output_name))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--grpo-task", default=task, choices=["generate", "rewrite", "mixed"])
    parser.add_argument("--reward-mode", default="gui_style", choices=["gui_style", "smoke"])
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--max-completion-length", type=int, default=3072)
    parser.add_argument("--generate-min-output-chars", type=int, default=3000)
    parser.add_argument("--generate-max-output-chars", type=int, default=4500)
    parser.add_argument("--short-output-penalty", type=float, default=0.20)
    parser.add_argument("--long-output-penalty", type=float, default=0.10)
    parser.add_argument("--enable-short-reasoning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reasoning-budget-tokens", type=int, default=384)
    parser.add_argument("--reasoning-bias-start-tokens", type=int, default=96)
    parser.add_argument("--reasoning-max-bias", type=float, default=8.0)
    parser.add_argument("--reasoning-open-tag", default=DEFAULT_REASONING_OPEN_TAG)
    parser.add_argument("--reasoning-close-tag", default=DEFAULT_REASONING_CLOSE_TAG)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init-lora", action="store_true", help="Attach a fresh LoRA adapter when training from a merged/base checkpoint.")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-last-layer-fraction", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--loss-type", default="dr_grpo")
    parser.add_argument("--importance-sampling-level", choices=["token", "sequence"], default="token")
    parser.add_argument("--mask-truncated-completions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=0.2)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument("--scale-rewards", default="group")
    parser.add_argument(
        "--reward-std-shaping-power",
        type=float,
        default=0.0,
        help=(
            "If >0, pre-shape rewards inside each prompt group as "
            "mean + (reward - mean) / max(std, floor)**power, then disable trainer reward scaling. "
            "Use 0.5 for partial std influence instead of full group-std normalization."
        ),
    )
    parser.add_argument(
        "--reward-std-shaping-floor",
        type=float,
        default=0.10,
        help="Std floor used by --reward-std-shaping-power to prevent low-variance amplification.",
    )
    parser.add_argument(
        "--reward-std-update-weight-min",
        type=float,
        default=1.0,
        help=(
            "If <1, shrink within-group reward dispersion for low-std groups after std shaping. "
            "This approximates a lower effective update weight while preserving the group mean."
        ),
    )
    parser.add_argument(
        "--reward-std-update-weight-std-low",
        type=float,
        default=0.05,
        help="Raw/std source value at or below which --reward-std-update-weight-min is applied.",
    )
    parser.add_argument(
        "--reward-std-update-weight-std-high",
        type=float,
        default=0.14,
        help="Raw/std source value at or above which the std update weight becomes 1.0.",
    )
    parser.add_argument(
        "--reward-std-update-weight-source",
        choices=["raw", "shaped"],
        default="raw",
        help="Use raw pre-shaping group std or shaped reward std when computing update weight.",
    )
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--typical-p", type=float, default=0.0)
    parser.add_argument("--epsilon-cutoff", type=float, default=0.0)
    parser.add_argument("--eta-cutoff", type=float, default=0.0)
    parser.add_argument("--use-transformers-paged", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-implementation", default="")
    parser.add_argument("--generation-use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-batch-size", type=int, default=0)
    parser.add_argument("--steps-per-generation", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=default_output_name)
    parser.add_argument("--require-result-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-strings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-at-result-close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-logits-to-keep", type=int, default=1)
    parser.add_argument("--force-eos-token-id", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing",
        default="unsloth",
        help="Unsloth/Trainer gradient checkpointing mode: unsloth, true, or false.",
    )
    parser.add_argument("--unsloth-grpo-mini-batch", type=int, default=1)
    parser.add_argument("--unsloth-logit-chunk-multiplier", type=int, default=8)
    parser.add_argument("--unsloth-num-chunks", type=int, default=0)
    parser.add_argument("--collapse-fail-reward", type=float, default=-1.0)
    parser.add_argument("--group-diversity-bonus-max", type=float, default=0.03)
    parser.add_argument(
        "--group-diversity-mode",
        choices=["none", "distance_threshold", "density_adjusted", "dra_density", "leave_one_out", "sgrpo_leave_one_out", "mmr_reweighted"],
        default="leave_one_out",
    )
    parser.add_argument("--group-diversity-target-distance", type=float, default=0.55)
    parser.add_argument("--group-diversity-density-smoothing", type=float, default=0.10)
    parser.add_argument("--group-diversity-min-chars", type=int, default=120)
    parser.add_argument("--sample-log-every", type=int, default=1)
    parser.add_argument("--sample-log-max-items", type=int, default=4)
    parser.add_argument("--sample-log-text-chars", type=int, default=3000)
    parser.add_argument("--sample-log-path", default="")
    parser.add_argument("--style-guidance-variant-mode", choices=["none", "row"], default="none")
    parser.add_argument("--system-prompt-variants", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle-style-guidance-bullets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rewrite-style-base-weight", type=float, default=1.0)
    parser.add_argument("--rewrite-anti-slop-family-weight", type=float, default=0.0)
    parser.add_argument("--rewrite-translationese-family-weight", type=float, default=0.0)
    parser.add_argument("--rewrite-comma-family-weight", type=float, default=0.0)
    parser.add_argument("--rewrite-pos-family-weight", type=float, default=0.0)
    parser.add_argument("--rewrite-modifier-family-weight", type=float, default=0.0)
    parser.add_argument("--rewrite-lexical-family-weight", type=float, default=0.0)
    parser.add_argument("--rewrite-sentence-edge-family-weight", type=float, default=0.0)
    parser.add_argument("--rewrite-sentence-length-family-weight", type=float, default=0.0)
    parser.add_argument("--rewrite-edit-weight", type=float, default=0.35)
    parser.add_argument("--rewrite-improvement-weight", type=float, default=0.30)
    parser.add_argument("--rewrite-improvement-scale", type=float, default=0.35)
    parser.add_argument("--rewrite-low-edit-penalty-max", type=float, default=0.35)
    parser.add_argument("--rewrite-edit-gate-min", type=float, default=0.20)
    parser.add_argument("--rewrite-edit-gate-q25", type=float, default=0.40)
    parser.add_argument("--rewrite-edit-gate-q50", type=float, default=1.00)
    parser.add_argument(
        "--disabled-style-metrics",
        default="",
        help="Comma-separated scalar style metrics to remove from GUI-style reward scoring.",
    )
    parser.add_argument(
        "--style-metric-weight-overrides",
        default="",
        help="Comma-separated metric:multiplier overrides applied after default scalar metric weights.",
    )
    parser.add_argument("--wandb-reward-component-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-task-order", choices=["shuffle", "source", "alternating"], default="shuffle")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--gui-style-reference", default=str(TRAINING_ROOT / "data" / "processed" / "gui_style_reward_reference.json"))
    parser.add_argument("--anti-slop-lexicon", default=str(TRAINING_ROOT / "data" / "processed" / "anti_slop_lexicon.json"))
    parser.add_argument("--translationese-model", default=str(TRAINING_ROOT / "models" / "translationese_svm" / "svm_detector.joblib"))
    parser.add_argument("--probe-only", action="store_true")


def run_grpo_stage(args: argparse.Namespace, *, stage: str, task: str) -> None:
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model_name = resolve_model_name(args)
    adapter_path = args.adapter_path or None
    gradient_checkpointing = parse_gradient_checkpointing(args.gradient_checkpointing)
    if adapter_path and args.init_lora:
        raise ValueError("--adapter-path and --init-lora are mutually exclusive for GRPO loading.")
    if not adapter_path and not args.init_lora:
        raise ValueError("Pass either --adapter-path to train an existing LoRA or --init-lora to create a fresh LoRA.")

    print_json(
        "[model]",
        {
            "model_name": model_name,
            "adapter_path": adapter_path,
            "init_lora": bool(args.init_lora),
            "stage": stage,
            "task": task,
        },
    )
    model, processor, loader_metadata = load_gemma4_model_and_processor(
        model_name=model_name,
        adapter_path=adapter_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        gradient_checkpointing=gradient_checkpointing,
        chat_template=args.chat_template,
        create_lora=bool(args.init_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_last_layer_fraction=args.lora_last_layer_fraction,
    )
    print_json("[loader]", loader_metadata)
    print_json("[trainable]", trainable_parameter_summary(model))
    if args.use_transformers_paged:
        patch_gemma4_composite_config_for_paged_generation(model)
    reasoning_open_ids: list[int] = []
    reasoning_close_ids: list[int] = []
    reasoning_result_open_ids: list[int] = []
    if args.enable_short_reasoning:
        reasoning_open_text = args.reasoning_open_tag if args.reasoning_open_tag.endswith("\n") else args.reasoning_open_tag + "\n"
        reasoning_open_ids = token_ids_for_text(processor, reasoning_open_text)
        reasoning_close_ids = token_ids_for_text(processor, args.reasoning_close_tag)
        reasoning_result_open_ids = token_ids_for_text(processor, RESULT_OPEN_TAG)
        print_json(
            "[reasoning]",
            {
                "enabled": True,
                "mode": "gemma4_processor_enable_thinking_with_thought_channel_budget",
                "control_token": GEMMA4_THINKING_TOKEN,
                "open_tag": args.reasoning_open_tag,
                "open_text_for_tokens": reasoning_open_text,
                "close_tag": args.reasoning_close_tag,
                "result_open_tag": RESULT_OPEN_TAG,
                "budget_tokens": args.reasoning_budget_tokens,
                "bias_start_tokens": args.reasoning_bias_start_tokens,
                "max_bias": args.reasoning_max_bias,
                "open_token_ids": reasoning_open_ids,
                "close_token_ids": reasoning_close_ids,
                "result_open_token_ids": reasoning_result_open_ids,
            },
        )
    patch_generate_helpers(
        model,
        processor=processor,
        logits_to_keep=args.generation_logits_to_keep,
        stop_at_result_close=args.require_result_tags and args.stop_at_result_close,
        force_eos_token_id=args.force_eos_token_id,
        reasoning_open_ids=reasoning_open_ids,
        reasoning_close_ids=reasoning_close_ids,
        reasoning_result_open_ids=reasoning_result_open_ids,
        reasoning_budget_tokens=args.reasoning_budget_tokens if args.enable_short_reasoning else 0,
        reasoning_bias_start_tokens=args.reasoning_bias_start_tokens,
        reasoning_max_bias=args.reasoning_max_bias,
    )

    rows = load_grpo_rows(args, processor, task=args.grpo_task)
    task_counts: dict[str, int] = {}
    for row in rows:
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
    print_json("[data]", {"rows": len(rows), "task_counts": task_counts, "path": args.dataset})
    rows = order_mixed_task_rows(rows, mode=args.mixed_task_order if args.grpo_task == "mixed" else "shuffle", seed=args.seed)
    print_json(
        "[data/order]",
        {
            "mixed_task_order": args.mixed_task_order,
            "preview_tasks": [str(row.get("task") or "") for row in rows[: min(12, len(rows))]],
        },
    )
    dataset = dataset_from_rows(rows)
    if args.grpo_task != "mixed" or args.mixed_task_order == "shuffle":
        dataset = dataset.shuffle(seed=args.seed)

    GRPOConfig, GRPOTrainer = import_grpo_classes()
    patch_grpo_generation_inference_mode(GRPOTrainer)
    if args.use_transformers_paged:
        patch_paged_sdpa_attn_alias()
    try:
        from unsloth import is_bfloat16_supported

        use_bf16 = bool(is_bfloat16_supported())
    except Exception:
        use_bf16 = True

    tokenizer = processor_tokenizer(processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    generation_kwargs = {
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "use_cache": bool(args.generation_use_cache),
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty if args.repetition_penalty > 0.0 else None,
        "no_repeat_ngram_size": args.no_repeat_ngram_size if args.no_repeat_ngram_size > 0 else None,
        "min_p": args.min_p if args.min_p > 0.0 else None,
        "typical_p": args.typical_p if args.typical_p > 0.0 else None,
        "epsilon_cutoff": args.epsilon_cutoff if args.epsilon_cutoff > 0.0 else None,
        "eta_cutoff": args.eta_cutoff if args.eta_cutoff > 0.0 else None,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    trainer_scale_rewards: Any = args.scale_rewards
    if float(getattr(args, "reward_std_shaping_power", 0.0) or 0.0) > 0.0:
        if str(args.scale_rewards).lower() not in {"false", "none", "0"}:
            print_json(
                "[reward/std_shaping]",
                {
                    "message": "Disabling trainer scale_rewards because reward_std_shaping_power is enabled.",
                    "requested_scale_rewards": args.scale_rewards,
                    "effective_scale_rewards": False,
                    "power": args.reward_std_shaping_power,
                    "floor": args.reward_std_shaping_floor,
                },
            )
        trainer_scale_rewards = False

    config_kwargs = {
        "output_dir": str(output),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": "linear",
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "optim": args.optim,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "seed": args.seed,
        "report_to": parse_report_to(args.report_to),
        "run_name": args.run_name,
        "remove_unused_columns": False,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "importance_sampling_level": args.importance_sampling_level,
        "mask_truncated_completions": args.mask_truncated_completions,
        "epsilon": args.epsilon,
        "epsilon_high": args.epsilon_high,
        "num_iterations": args.num_iterations,
        "scale_rewards": trainer_scale_rewards,
        "generation_kwargs": generation_kwargs,
        "use_transformers_paged": args.use_transformers_paged,
        "cache_implementation": str(args.cache_implementation or "").strip() or None,
        "stop_strings": list(STOP_STRINGS) if args.stop_strings else None,
        "shuffle_dataset": False if args.grpo_task == "mixed" and args.mixed_task_order == "alternating" else True,
    }
    if args.generation_batch_size > 0:
        config_kwargs["generation_batch_size"] = args.generation_batch_size
    if args.steps_per_generation > 0:
        config_kwargs["steps_per_generation"] = args.steps_per_generation
    if isinstance(gradient_checkpointing, bool):
        config_kwargs["gradient_checkpointing"] = gradient_checkpointing
    if args.unsloth_grpo_mini_batch > 0:
        config_kwargs["unsloth_grpo_mini_batch"] = args.unsloth_grpo_mini_batch
    if args.unsloth_logit_chunk_multiplier > 0:
        config_kwargs["unsloth_logit_chunk_multiplier"] = args.unsloth_logit_chunk_multiplier
    if args.unsloth_num_chunks > 0:
        config_kwargs["unsloth_num_chunks"] = args.unsloth_num_chunks
    config_kwargs = {key: value for key, value in config_kwargs.items() if value is not None}
    training_args = instantiate_config(GRPOConfig, config_kwargs)
    print_json(
        "[grpo/effective_config]",
        {
            "per_device_train_batch_size": getattr(training_args, "per_device_train_batch_size", None),
            "gradient_accumulation_steps": getattr(training_args, "gradient_accumulation_steps", None),
            "num_generations": getattr(training_args, "num_generations", None),
            "num_iterations": getattr(training_args, "num_iterations", None),
            "generation_batch_size": getattr(training_args, "generation_batch_size", None),
            "steps_per_generation": getattr(training_args, "steps_per_generation", None),
            "use_transformers_paged": getattr(training_args, "use_transformers_paged", None),
            "cache_implementation": getattr(training_args, "cache_implementation", None),
            "generation_use_cache": generation_kwargs.get("use_cache"),
            "use_vllm": getattr(training_args, "use_vllm", None),
            "torch_compile": getattr(training_args, "torch_compile", None),
            "gradient_checkpointing": getattr(training_args, "gradient_checkpointing", None),
            "max_prompt_length": getattr(training_args, "max_prompt_length", None),
            "max_completion_length": getattr(training_args, "max_completion_length", None),
            "generation_kwargs": generation_kwargs,
        },
    )
    processing_class = tokenizer
    reward_func = make_reward_func(args)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "reward_funcs": [reward_func],
        "processing_class": processing_class,
        "tokenizer": processing_class,
    }
    trainer = GRPOTrainer(**filter_kwargs(GRPOTrainer, trainer_kwargs))
    if args.probe_only:
        batch = next(iter(trainer.get_train_dataloader()))
        if isinstance(batch, dict):
            payload = {"type": "dict", "keys": sorted(batch.keys())}
        elif isinstance(batch, (list, tuple)):
            first = batch[0] if batch else None
            payload = {
                "type": type(batch).__name__,
                "len": len(batch),
                "first_type": type(first).__name__ if first is not None else None,
                "first_keys": sorted(first.keys()) if isinstance(first, dict) else None,
            }
        else:
            payload = {"type": type(batch).__name__}
        print_json("[probe/batch]", payload)
        return

    resume_from_checkpoint = str(args.resume_from_checkpoint or "").strip() or None
    print_json(
        "[train/start]",
        {
            "stage": stage,
            "task": task,
            "time": time.time(),
            "resume_from_checkpoint": resume_from_checkpoint,
        },
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    final_dir = output / "policy"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    metrics = dict(result.metrics)
    trainer.save_metrics("train", metrics)
    write_json(
        output / f"{stage}_manifest.json",
        {
            **manifest_base(stage=stage, args=vars(args)),
            "dataset_rows": len(rows),
            "task_counts": task_counts,
            "loader": loader_metadata,
            "final_adapter": str(final_dir),
            "train_metrics": metrics,
        },
    )
    print_json("[done]", {"stage": stage, "final_adapter": str(final_dir), "train_metrics": metrics})
