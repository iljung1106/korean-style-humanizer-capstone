"""Preference and RL dataset helpers for standalone pipeline_v2 trainers."""

from __future__ import annotations

import re
import random
from typing import Any, Iterable

from .gemma4_loader import processor_tokenizer


RESULT_OPEN_TAG = "<result>"
RESULT_CLOSE_TAG = "</result>"
GEMMA4_THINKING_TOKEN = "<|think|>"
GEMMA4_THOUGHT_OPEN_TAG = "<|channel>thought"
GEMMA4_THOUGHT_CLOSE_TAG = "<channel|>"


def dataset_from_rows(rows: Iterable[dict[str, Any]]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(list(rows))


def strip_bos(text: str) -> str:
    return text.removeprefix("<bos>").removeprefix("<s>")


def message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def as_chat_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise ValueError(f"expected chat messages list, got {type(value).__name__}")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"invalid message item: {item!r}")
        role = str(item.get("role") or "user")
        messages.append({"role": role, "content": message_content(item)})
    return messages


def as_assistant_messages(value: Any) -> list[dict[str, str]]:
    messages = as_chat_messages(value)
    if messages and all(message.get("role") != "assistant" for message in messages):
        return [{"role": "assistant", "content": "\n".join(message["content"] for message in messages)}]
    return [message for message in messages if message.get("role") == "assistant"]


def assistant_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages if message.get("content")).strip()


def trim_messages(messages: list[dict[str, str]], max_prompt_chars: int) -> list[dict[str, str]]:
    if max_prompt_chars <= 0:
        return messages
    used = 0
    trimmed: list[dict[str, str]] = []
    for message in reversed(messages):
        content = message["content"]
        remaining = max_prompt_chars - used
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[-remaining:]
        trimmed.append({"role": message["role"], "content": content})
        used += len(content)
    return list(reversed(trimmed))


def render_prompt(processor: Any, messages: list[dict[str, str]], *, enable_thinking: bool = False) -> str:
    tokenizer = processor_tokenizer(processor)
    renderer = processor if hasattr(processor, "apply_chat_template") else tokenizer
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    try:
        rendered = renderer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = renderer.apply_chat_template(messages, **kwargs)
    return strip_bos(str(rendered))


def add_result_contract_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    instruction = (
        "\n\n출력 형식:\n"
        f"- 최종 결과물만 {RESULT_OPEN_TAG}와 {RESULT_CLOSE_TAG} 사이에 작성하세요.\n"
        f"- {RESULT_CLOSE_TAG} 이후에는 아무것도 출력하지 마세요.\n"
        f"- {RESULT_OPEN_TAG} 안에는 최종 본문만 넣고, 분석, 메모, 시스템 문구는 넣지 마세요."
    )
    updated = [dict(message) for message in messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "user":
            if RESULT_OPEN_TAG not in updated[index]["content"]:
                updated[index]["content"] = updated[index]["content"].rstrip() + instruction
            return updated
    updated.append({"role": "user", "content": instruction.strip()})
    return updated


def add_short_reasoning_instruction(
    messages: list[dict[str, str]],
    *,
    budget_tokens: int,
    open_tag: str,
    close_tag: str,
) -> list[dict[str, str]]:
    if budget_tokens <= 0:
        return messages
    updated = [dict(message) for message in messages]
    for index, message in enumerate(updated):
        if message.get("role") == "system":
            content = message["content"].lstrip()
            if content.startswith(GEMMA4_THINKING_TOKEN):
                content = content[len(GEMMA4_THINKING_TOKEN) :].lstrip("\n")
            updated[index]["content"] = content.rstrip()
            break
    else:
        updated.insert(0, {"role": "system", "content": ""})
    return updated


def normalize_generate_length_instruction(content: str, *, min_output_chars: int, max_output_chars: int) -> str:
    text = re.sub(r"\n\n분량 조건(?:\([^)]*\))?:[^\n]+", "", str(content).rstrip())
    parts: list[str] = []
    if min_output_chars > 0 and max_output_chars > 0:
        parts.append(f"{min_output_chars:,}자 이상, {max_output_chars:,}자 이내")
    elif min_output_chars > 0:
        parts.append(f"{min_output_chars:,}자 이상")
    elif max_output_chars > 0:
        parts.append(f"{max_output_chars:,}자 이내")
    if not parts:
        return text
    length_line = f"- 최종 본문은 {parts[0]}로 작성하세요."
    length_pattern = re.compile(
        r"(?m)^- .*?(?:\d[\d,]*\s*자|최소\s*\d[\d,]*|최종 본문).*?(?:작성하세요|마무리하세요|쓰되).*$"
    )
    if length_pattern.search(text):
        return length_pattern.sub(length_line, text, count=1)
    if "요구사항:" in text:
        return text.replace("요구사항:\n", f"요구사항:\n{length_line}\n", 1)
    return text + "\n\n요구사항:\n" + length_line


def add_generate_length_instruction(
    messages: list[dict[str, str]],
    max_output_chars: int,
    min_output_chars: int = 3000,
) -> list[dict[str, str]]:
    if max_output_chars <= 0 and min_output_chars <= 0:
        return messages
    updated = [dict(message) for message in messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "user":
            updated[index]["content"] = normalize_generate_length_instruction(
                updated[index]["content"],
                min_output_chars=min_output_chars,
                max_output_chars=max_output_chars,
            )
            return updated
    updated.append(
        {
            "role": "user",
            "content": normalize_generate_length_instruction(
                "",
                min_output_chars=min_output_chars,
                max_output_chars=max_output_chars,
            ),
        }
    )
    return updated


GENERATE_STYLE_GUIDANCE_MARKER = "문체 지침:"
REWRITE_STYLE_GUIDANCE_MARKER = "재작성 목표:"


COMMON_STYLE_GUIDANCE_BULLETS = (
    "같은 구조의 문장이 세 개 이상 연속되거나, 감정 어휘가 한 문단에 밀집되거나, 형용사·부사가 중첩 수식되는 패턴을 피하세요.",
    "동일한 서술어, 감정어 및 내용어가 너무 잦게 나오지 않도록 다양한 표현을 섞어 사용하세요. 꼭 필요한 경우가 아니라면 의미가 달라지지 않는 선에서 표현을 조정하세요.",
    "유사한 길이의 문장을 연속하지 마세요. 긴 호흡과 짧은 호흡의 문장을 섞어 사용하세요. 글 내에 충분히 다양한 길이의 문장이 있어야 합니다.",
    "같은 종결 어미가 반복되지 않도록 '-ㅆ다' 외에도 '-까?', '-지', '-나', 명사형 등 다양한 종결 방식을 문맥에 맞게 사용하세요.",
    "'그 순간', '마치', '압도적인', '미묘한', '운명처럼', '그것은 단순한' 같은 상투적 강조 표현을 반복하지 마세요.",
    "'그것', '그', '그녀', '그들' 같은 대명사가 반복되지 않도록, 구체적인 대상 이름이나 자연스러운 생략을 사용하세요.",
    "'마치 ~같았다', '~한 것처럼', '~에 가까웠다' 같은 비유 표현을 줄이세요.",
    "의미가 불명확한 신체 반응 묘사나 과장된 감정 강조를 자제하세요.",
    "요약, 작가 메모 등을 포함하지 말고, 실제 연재분 발췌와 같은 형태로 본문만 작성하세요.",
)

SYSTEM_PROMPT_VARIANTS = {
    "generate": (
        "당신은 한국 웹소설 작가입니다. 사용자가 제시한 조건으로 바로 읽히는 장르소설 본문을 씁니다.",
        "당신은 한국어 장르소설을 쓰는 웹소설 작가입니다. 주어진 조건을 바탕으로 소설 본문을 작성합니다.",
        "당신은 한국 웹소설 본문을 작성하는 작가입니다. 조건을 설명하지 않고 장면으로 바로 이어지는 글을 씁니다.",
    ),
    "continuation": (
        "당신은 한국 웹소설 작가입니다. 자연스럽고 흡입력 있는 한국어 장르소설 본문을 씁니다.",
        "당신은 한국어 장르소설을 이어 쓰는 웹소설 작가입니다. 앞 문맥에 맞는 다음 본문을 작성합니다.",
        "당신은 한국 웹소설의 다음 장면을 쓰는 작가입니다. 기존 문맥을 이어 자연스러운 본문을 작성합니다.",
    ),
    "rewrite": (
        "당신은 한국 웹소설 문체 교정 작가입니다. AI가 쓴 듯한 문장을 자연스러운 한국 웹소설 본문으로 고칩니다.",
        "당신은 한국어 웹소설 원고를 다듬는 작가입니다. 원문의 의미를 유지하면서 자연스러운 소설 본문으로 다시 씁니다.",
        "당신은 한국 장르소설 문장을 재작성하는 작가입니다. 주어진 본문을 자연스러운 연재 본문으로 고칩니다.",
    ),
}


def common_style_guidance_text(*, variant_seed: str | None = None, shuffle_bullets: bool = False) -> str:
    bullets = list(COMMON_STYLE_GUIDANCE_BULLETS)
    if shuffle_bullets and variant_seed:
        random.Random(f"style-guidance:{variant_seed}").shuffle(bullets)
    return "\n\n문체 지침:\n" + "\n".join(f"- {bullet}" for bullet in bullets)


def apply_system_prompt_variant(
    messages: list[dict[str, str]],
    *,
    task: str,
    variant_seed: str | None = None,
) -> list[dict[str, str]]:
    if not variant_seed:
        return [dict(message) for message in messages]
    updated = [dict(message) for message in messages]
    variants = SYSTEM_PROMPT_VARIANTS.get(task) or SYSTEM_PROMPT_VARIANTS["generate"]
    variant = variants[random.Random(f"system-prompt:{task}:{variant_seed}").randrange(len(variants))]
    for index, message in enumerate(updated):
        if message.get("role") == "system":
            updated[index]["content"] = variant
            return updated
    updated.insert(0, {"role": "system", "content": variant})
    return updated


def add_generate_style_guidance_instruction(
    messages: list[dict[str, str]],
    *,
    variant_seed: str | None = None,
    shuffle_bullets: bool = False,
) -> list[dict[str, str]]:
    guidance = (
        "\n\n작성 목표:\n"
        "- 실제 한국 웹소설의 한 장면처럼 바로 이어지는 본문을 작성하세요.\n"
        "- 요약, 설정 설명, 작가 메모, 분석 문구를 쓰지 말고 장면 안에서 사건과 감정이 진행되게 하세요.\n"
        "- 제시된 조건을 설명하지 말고, 바로 소설 본문으로 시작하세요."
        + common_style_guidance_text(variant_seed=variant_seed, shuffle_bullets=shuffle_bullets)
    )
    updated = [dict(message) for message in messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "user":
            if GENERATE_STYLE_GUIDANCE_MARKER not in updated[index]["content"]:
                updated[index]["content"] = updated[index]["content"].rstrip() + guidance
            return updated
    updated.append({"role": "user", "content": guidance.strip()})
    return updated


def add_continuation_style_guidance_instruction(
    messages: list[dict[str, str]],
    *,
    variant_seed: str | None = None,
    shuffle_bullets: bool = False,
) -> list[dict[str, str]]:
    guidance = (
        "\n\n이어쓰기 목표:\n"
        "- 앞 문맥의 시점, 호칭, 말투, 사건 흐름을 유지하세요.\n"
        "- 앞 내용을 요약하지 말고, 바로 다음 장면을 이어 쓰세요."
        + common_style_guidance_text(variant_seed=variant_seed, shuffle_bullets=shuffle_bullets)
    )
    updated = [dict(message) for message in messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "user":
            if GENERATE_STYLE_GUIDANCE_MARKER not in updated[index]["content"]:
                updated[index]["content"] = updated[index]["content"].rstrip() + guidance
            return updated
    updated.append({"role": "user", "content": guidance.strip()})
    return updated


def add_rewrite_style_guidance_instruction(
    messages: list[dict[str, str]],
    *,
    variant_seed: str | None = None,
    shuffle_bullets: bool = False,
) -> list[dict[str, str]]:
    guidance = (
        "\n\n재작성 목표:\n"
        "- 사건 순서, 인물 관계, 설정 정보, 장면의 핵심 의미는 유지하세요.\n"
        "- 단순 교정이나 일부 단어 치환에 그치지 말고, 문장 구조와 서술 호흡을 충분히 고치세요. 목표는 주어진 본문과 비교했을 때 더 몰입이 되는 글을 만드는 것입니다.\n"
        "- 원문을 짧게 요약하거나 과도하게 늘리지 말고, 원문 분량의 85~115% 수준을 유지하세요. 원문이 부적절한 경우에는 범위 내에서 조정해도 좋습니다."
        + common_style_guidance_text(variant_seed=variant_seed, shuffle_bullets=shuffle_bullets)
    )
    updated = [dict(message) for message in messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "user":
            if REWRITE_STYLE_GUIDANCE_MARKER not in updated[index]["content"]:
                updated[index]["content"] = updated[index]["content"].rstrip() + guidance
            return updated
    updated.append({"role": "user", "content": guidance.strip()})
    return updated


def current_rewrite_guidance_text() -> str:
    """Return the current rewrite guidance without relying on an existing prompt."""

    seed = [{"role": "user", "content": ""}]
    return add_rewrite_style_guidance_instruction(seed)[-1]["content"].strip()


def rewrite_source_from_prompt(content: str) -> str:
    text = str(content)
    marker = "\n원문:\n"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    marker = "원문:\n"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    return text.strip()


def rebuild_rewrite_prompt_with_current_guidance(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Rebuild rewrite prompts so stale style guidance is not mixed into eval/training.

    Older eval rows already contain a full prompt, including previous style guidance.
    Appending new guidance to those rows puts instructions after the source text and
    leaves contradictory old bullets in place. This function keeps the system message
    and source text, then reconstructs the user message from the current contract and
    rewrite guidance.
    """

    updated = [dict(message) for message in messages]
    user_index = next(
        (index for index in range(len(updated) - 1, -1, -1) if updated[index].get("role") == "user"),
        None,
    )
    if user_index is None:
        source = ""
        updated.append({"role": "user", "content": ""})
        user_index = len(updated) - 1
    else:
        source = rewrite_source_from_prompt(updated[user_index].get("content", ""))

    contract_seed = [{"role": "user", "content": ""}]
    contract = add_result_contract_instruction(contract_seed)[-1]["content"].strip()
    guidance = current_rewrite_guidance_text()
    updated[user_index]["content"] = (
        "아래 원문을 자연스러운 한국 웹소설 본문으로 다시 쓰세요.\n\n"
        f"{contract}\n\n"
        f"{guidance}\n\n"
        f"원문:\n{source}"
    ).strip()
    return updated


class TextFirstProcessor:
    """Adapt Gemma4 processors for TRL trainers that call tokenizer(text)."""

    def __init__(self, processor: Any):
        self.processor = processor

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "processor"), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        scalar_text = False
        if args:
            if len(args) > 1:
                raise TypeError("TextFirstProcessor accepts at most one positional text argument.")
            if "text" in kwargs:
                raise TypeError("TextFirstProcessor got both positional text and text=.")
            kwargs["text"] = args[0]
            scalar_text = isinstance(args[0], str)
        else:
            scalar_text = isinstance(kwargs.get("text"), str)
        output = self.processor(**kwargs)
        if scalar_text and kwargs.get("return_tensors") is None:
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                value = output.get(key) if hasattr(output, "get") else None
                if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
                    output[key] = value[0]
        return output

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        return self.processor.save_pretrained(*args, **kwargs)
