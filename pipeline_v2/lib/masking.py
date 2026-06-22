"""Row rendering and loss masking for pipeline_v2.

The core contract is explicit:

- `raw_lm` rows train on every non-padding token.
- `continuation_sft` and `format_sft` rows train only on assistant content.

This replaces generic response-only masking because Stage 1 mixes raw LM and
chat-SFT rows in one dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


IGNORE_INDEX = -100


@dataclass
class PreparedExample:
    row_id: str
    row_type: str
    text: str
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    prompt_token_count: int
    visible_label_count: int
    total_token_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.row_id,
            "row_type": self.row_type,
            "total_tokens": self.total_token_count,
            "prompt_tokens": self.prompt_token_count,
            "visible_labels": self.visible_label_count,
            "visible_ratio": round(self.visible_label_count / max(1, self.total_token_count), 6),
        }


def processor_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _unwrap_input_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Could not unwrap input_ids from {type(value).__name__}")
    return [int(item) for item in value]


def tokenize_text(processor: Any, text: str, max_seq_length: int) -> list[int]:
    kwargs = {
        "return_tensors": None,
        "truncation": True,
        "max_length": max_seq_length,
    }
    try:
        encoded = processor(text=text, **kwargs)
    except TypeError:
        tokenizer = processor_tokenizer(processor)
        encoded = tokenizer(text, **kwargs)
    if isinstance(encoded, dict):
        return _unwrap_input_ids(encoded["input_ids"])
    return _unwrap_input_ids(getattr(encoded, "input_ids"))


def _unwrap_offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Could not unwrap offset_mapping from {type(value).__name__}")
    offsets: list[tuple[int, int]] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(f"Invalid offset item: {item!r}")
        offsets.append((int(item[0]), int(item[1])))
    return offsets


def tokenize_text_with_offsets(
    processor: Any,
    text: str,
    max_seq_length: int,
) -> tuple[list[int], list[tuple[int, int]]] | None:
    kwargs = {
        "return_tensors": None,
        "truncation": True,
        "max_length": max_seq_length,
        "return_offsets_mapping": True,
    }
    try:
        encoded = processor(text=text, **kwargs)
    except Exception:
        try:
            tokenizer = processor_tokenizer(processor)
            encoded = tokenizer(text, **kwargs)
        except Exception:
            return None
    if not isinstance(encoded, dict):
        encoded = {
            "input_ids": getattr(encoded, "input_ids"),
            "offset_mapping": getattr(encoded, "offset_mapping"),
        }
    try:
        input_ids = _unwrap_input_ids(encoded["input_ids"])
        offsets = _unwrap_offsets(encoded["offset_mapping"])
    except Exception:
        return None
    if len(input_ids) != len(offsets):
        return None
    return input_ids, offsets


def render_chat(processor: Any, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    renderer = processor
    if not hasattr(renderer, "apply_chat_template"):
        renderer = processor_tokenizer(processor)
    text = renderer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}, expected str")
    return text.removeprefix("<bos>")


def first_assistant_index(messages: Sequence[dict[str, Any]]) -> int:
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            return index
    raise ValueError("SFT row has no assistant message.")


def message_content_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def raw_lm_example(row: dict[str, Any], processor: Any, max_seq_length: int) -> PreparedExample:
    text = str(row.get("text") or "")
    if not text.strip():
        raise ValueError(f"raw_lm row {row.get('id')} has empty text.")
    input_ids = tokenize_text(processor, text, max_seq_length)
    if not input_ids:
        raise ValueError(f"raw_lm row {row.get('id')} tokenized to zero tokens.")
    labels = list(input_ids)
    return PreparedExample(
        row_id=str(row.get("id") or ""),
        row_type="raw_lm",
        text=text,
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
        prompt_token_count=0,
        visible_label_count=len([item for item in labels if item != IGNORE_INDEX]),
        total_token_count=len(input_ids),
    )


def sft_example(row: dict[str, Any], processor: Any, max_seq_length: int) -> PreparedExample:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"SFT row {row.get('id')} has invalid messages.")
    assistant_index = first_assistant_index(messages)
    prompt_messages = messages[:assistant_index]
    full_text = render_chat(processor, messages, add_generation_prompt=False)
    prompt_text = render_chat(processor, prompt_messages, add_generation_prompt=True)
    input_with_offsets = tokenize_text_with_offsets(processor, full_text, max_seq_length)
    input_ids = input_with_offsets[0] if input_with_offsets is not None else tokenize_text(processor, full_text, max_seq_length)
    assistant_content = message_content_text(messages[assistant_index])
    assistant_char_start = full_text.find(assistant_content) if assistant_content else -1
    if input_with_offsets is not None and assistant_char_start >= 0:
        _, offsets = input_with_offsets
        labels = [
            token if start >= assistant_char_start else IGNORE_INDEX
            for token, (start, _end) in zip(input_ids, offsets)
        ]
        visible = sum(1 for item in labels if item != IGNORE_INDEX)
        if visible > 0:
            prompt_len = len(input_ids) - visible
            return PreparedExample(
                row_id=str(row.get("id") or ""),
                row_type=str(row.get("row_type") or "sft"),
                text=full_text,
                input_ids=input_ids,
                labels=labels,
                attention_mask=[1] * len(input_ids),
                prompt_token_count=prompt_len,
                visible_label_count=visible,
                total_token_count=len(input_ids),
            )
    prompt_ids = tokenize_text(processor, prompt_text, max_seq_length)
    prompt_len = min(len(prompt_ids), len(input_ids))
    if input_ids[:prompt_len] != prompt_ids[:prompt_len] and assistant_char_start >= 0:
        prompt_text = full_text[:assistant_char_start]
        prompt_ids = tokenize_text(processor, prompt_text, max_seq_length)
        prompt_len = min(len(prompt_ids), len(input_ids))
    if input_ids[:prompt_len] != prompt_ids[:prompt_len] and "<result>" in full_text:
        # Some processors handle generation-prompt rendering differently from
        # full-message rendering. Falling back to the literal result start keeps
        # the prompt/header masked while making `<result>` itself trainable.
        prompt_text = full_text[: full_text.index("<result>")]
        prompt_ids = tokenize_text(processor, prompt_text, max_seq_length)
        prompt_len = min(len(prompt_ids), len(input_ids))
    if input_ids[:prompt_len] != prompt_ids[:prompt_len]:
        raise ValueError(
            f"SFT row {row.get('id')} prompt tokens are not a prefix of full tokens; "
            "loss mask would be unsafe."
        )
    labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
    visible = sum(1 for item in labels if item != IGNORE_INDEX)
    if visible <= 0:
        raise ValueError(
            f"SFT row {row.get('id')} has no visible assistant labels after truncation "
            f"(prompt_tokens={len(prompt_ids)}, total_tokens={len(input_ids)})."
        )
    return PreparedExample(
        row_id=str(row.get("id") or ""),
        row_type=str(row.get("row_type") or "sft"),
        text=full_text,
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
        prompt_token_count=prompt_len,
        visible_label_count=visible,
        total_token_count=len(input_ids),
    )


def prepare_row(row: dict[str, Any], processor: Any, max_seq_length: int) -> PreparedExample:
    row_type = str(row.get("row_type") or "")
    if row_type == "raw_lm":
        return raw_lm_example(row, processor, max_seq_length)
    if row_type in {"continuation_sft", "format_sft", "general_guard"}:
        return sft_example(row, processor, max_seq_length)
    raise ValueError(f"Unsupported row_type for pipeline_v2 Stage 1: {row_type!r}")


class PipelineV2TokenizedDataset:
    def __init__(self, rows: Sequence[dict[str, Any]], processor: Any, max_seq_length: int):
        self.rows = list(rows)
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = prepare_row(self.rows[index], self.processor, self.max_seq_length)
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }


class PipelineV2DataCollator:
    def __init__(self, processor: Any, pad_to_multiple_of: int | None = 8):
        tokenizer = processor_tokenizer(processor)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", 0)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        batch_input_ids: list[list[int]] = []
        batch_attention: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature["attention_mask"])
            labels = list(feature["labels"])
            pad_count = max_length - len(input_ids)
            batch_input_ids.append(input_ids + [int(self.pad_token_id)] * pad_count)
            batch_attention.append(attention_mask + [0] * pad_count)
            batch_labels.append(labels + [IGNORE_INDEX] * pad_count)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def decode_visible_labels(processor: Any, example: PreparedExample) -> str:
    tokenizer = processor_tokenizer(processor)
    visible_ids = [token for token, label in zip(example.input_ids, example.labels) if label != IGNORE_INDEX]
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(visible_ids, skip_special_tokens=False)
    return "".join(chr(item) for item in visible_ids)


class MockGemma4Processor:
    """A tiny character tokenizer for local mask tests without ML packages."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def __init__(self) -> None:
        self.tokenizer = self

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        rendered = ["<bos>"]
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            rendered.append(f"<|turn>{role}\n{content}\n")
        if add_generation_prompt:
            rendered.append("<|turn>assistant\n")
        text = "".join(rendered)
        if tokenize:
            return self.encode(text)
        return text

    def __call__(
        self,
        text: str | None = None,
        *,
        return_tensors: Any = None,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, list[list[int]]]:
        if text is None:
            raise TypeError("MockGemma4Processor requires text=...")
        ids = self.encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    def encode(self, text: str) -> list[int]:
        return [ord(char) + 3 for char in text]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        chars: list[str] = []
        for item in ids:
            if skip_special_tokens and item in {self.pad_token_id, self.eos_token_id, self.bos_token_id}:
                continue
            if item >= 3:
                chars.append(chr(item - 3))
        return "".join(chars)
