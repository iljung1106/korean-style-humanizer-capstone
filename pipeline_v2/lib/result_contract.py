"""Result-tag and output-integrity helpers for pipeline_v2.

These checks are intentionally dependency-light. They are used both during
fixed-set evaluation and during post-training adapter probes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


RESULT_OPEN_TAG = "<result>"
RESULT_CLOSE_TAG = "</result>"
CHAT_TURN_STOP_TOKEN = "<turn|>"
CHAT_STOP_TOKENS = (CHAT_TURN_STOP_TOKEN, "<end_of_turn>", "<|end_of_turn|>")
GENERATION_STOP_TEXTS = (RESULT_CLOSE_TAG, *CHAT_STOP_TOKENS)

THOUGHT_LEAK_RE = re.compile(
    r"(<\/?think\b|<\|/?thought|(?:^|\n)\s*(?:thought|analysis|reasoning)\s*:|"
    r"\[System instruction\]|<start_of_turn>|<end_of_turn>|<\|end_of_turn\|>)",
    re.IGNORECASE,
)
SPECIAL_TOKEN_LEAK_RE = re.compile(
    r"(<\|[^>]{1,80}\|>|<start_of_turn>|<end_of_turn>|<bos>|<eos>|<pad>|"
    r"<unused\d+>|<image>|<video>)",
    re.IGNORECASE,
)
FOREIGN_SCRIPT_PATTERNS = {
    "arabic": r"[\u0600-\u06ff]",
    "cyrillic": r"[\u0400-\u04ff]",
    "devanagari": r"[\u0900-\u097f]",
    "kana": r"[\u3040-\u30ff]",
    "thai": r"[\u0e00-\u0e7f]",
    "hangul_jamo": r"[\u3130-\u318f]",
    "replacement": r"\ufffd",
    "private_use": r"[\ue000-\uf8ff]",
}


@dataclass(frozen=True)
class ResultContract:
    raw_text: str
    result_text: str
    has_open: bool
    has_close: bool
    open_count: int
    close_count: int
    post_result_text: str
    reason: str | None

    @property
    def ok(self) -> bool:
        return self.reason is None

    def summary(self) -> dict[str, Any]:
        return {
            "result_contract_ok": self.ok,
            "result_contract_reason": self.reason or "",
            "has_result_open": self.has_open,
            "has_result_close": self.has_close,
            "result_open_count": self.open_count,
            "result_close_count": self.close_count,
            "raw_chars": len(self.raw_text),
            "result_text_chars": len(self.result_text),
            "post_result_chars": len(self.post_result_text.strip()),
        }


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def extract_result_text(text: str) -> str:
    start = text.find(RESULT_OPEN_TAG)
    if start < 0:
        return text
    start += len(RESULT_OPEN_TAG)
    end = text.find(RESULT_CLOSE_TAG, start)
    if end < 0:
        return text[start:]
    return text[start:end]


def trim_after_first_stop_text(
    text: str,
    stop_texts: tuple[str, ...] = CHAT_STOP_TOKENS,
) -> tuple[str, str, str]:
    """Remove generated chat-control text after the first stop marker.

    The returned tuple is `(trimmed_text, hit_stop_text, post_stop_text)`.
    `</result>` is intentionally not stripped here because downstream result
    contract checks need to see it.
    """

    raw_text = str(text)
    best_index: int | None = None
    best_stop = ""
    for stop_text in stop_texts:
        if not stop_text:
            continue
        index = raw_text.find(stop_text)
        if index >= 0 and (best_index is None or index < best_index):
            best_index = index
            best_stop = stop_text
    if best_index is None:
        return raw_text, "", ""
    post_start = best_index + len(best_stop)
    return raw_text[:best_index], best_stop, raw_text[post_start:]


def analyze_result_contract(text: str, *, require_result_tags: bool = True, min_result_chars: int = 40) -> ResultContract:
    raw_text = str(text)
    stripped = raw_text.strip()
    open_count = stripped.count(RESULT_OPEN_TAG)
    close_count = stripped.count(RESULT_CLOSE_TAG)
    start = stripped.find(RESULT_OPEN_TAG)
    end = stripped.find(RESULT_CLOSE_TAG, start + len(RESULT_OPEN_TAG)) if start >= 0 else stripped.find(RESULT_CLOSE_TAG)
    has_open = start >= 0
    has_close = end >= 0 and (not has_open or end > start)

    if has_open:
        result_start = start + len(RESULT_OPEN_TAG)
        result_end = end if has_close else len(stripped)
        result_text = stripped[result_start:result_end]
        post_result_text = stripped[end + len(RESULT_CLOSE_TAG) :] if has_close else ""
    else:
        result_text = stripped
        post_result_text = ""

    reason: str | None = None
    if require_result_tags and not has_open and not has_close:
        reason = "missing_result_tags"
    elif not has_open and has_close:
        reason = "missing_result_open"
    elif has_open and not has_close:
        reason = "missing_result_close"
    elif open_count > 1:
        reason = "multiple_result_open"
    elif close_count > 1:
        reason = "multiple_result_close"
    elif post_result_text.strip():
        reason = "post_result_text"
    elif require_result_tags and len(result_text.strip()) < min_result_chars:
        reason = "empty_or_short_result"
    elif THOUGHT_LEAK_RE.search(result_text):
        reason = "thought_leak"

    return ResultContract(
        raw_text=raw_text,
        result_text=result_text,
        has_open=has_open,
        has_close=has_close,
        open_count=open_count,
        close_count=close_count,
        post_result_text=post_result_text,
        reason=reason,
    )


def foreign_script_counts(text: str) -> dict[str, int]:
    return {name: len(re.findall(pattern, text)) for name, pattern in FOREIGN_SCRIPT_PATTERNS.items()}


def symbol_noise_ratio(text: str) -> float:
    allowed = ".,!?;:()[]{}'\"“”‘’…-—·（），<>/"
    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace() and ch not in allowed)
    return symbols / max(1, len(text))


def repeated_line_rate(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 12]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def repeated_ngram_rate(text: str, n: int = 8) -> float:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+|[^\s]", text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def max_repeated_char_run(text: str) -> int:
    best = 0
    current = 0
    previous = ""
    for char in text:
        if char == previous:
            current += 1
        else:
            previous = char
            current = 1
        best = max(best, current)
    return best


def max_repeated_hangul_or_alpha_run(text: str) -> int:
    best = 0
    current = 0
    previous = ""
    for char in text:
        if not (("가" <= char <= "힣") or ("A" <= char <= "Z") or ("a" <= char <= "z")):
            previous = ""
            current = 0
            continue
        if char == previous:
            current += 1
        else:
            previous = char
            current = 1
        best = max(best, current)
    return best


def max_repeated_digit_run(text: str) -> int:
    best = 0
    current = 0
    previous = ""
    for char in text:
        if not char.isdigit():
            previous = ""
            current = 0
            continue
        if char == previous:
            current += 1
        else:
            previous = char
            current = 1
        best = max(best, current)
    return best


def short_token_burst_reason(text: str) -> str | None:
    tokens = re.findall(r"[가-힣]+|[A-Za-z]+|[0-9]+", text)
    if len(tokens) < 60:
        return None
    short_tokens = [token for token in tokens if len(token) <= 2]
    if len(short_tokens) >= 40:
        counts: dict[str, int] = {}
        for token in short_tokens:
            counts[token] = counts.get(token, 0) + 1
        if counts and max(counts.values()) / max(1, len(short_tokens)) >= 0.28:
            return "short_token_burst"
    window = 90
    for start in range(0, max(1, len(tokens) - window + 1), 15):
        subset = [token for token in tokens[start : start + window] if len(token) <= 2]
        if len(subset) < 25:
            continue
        counts = {}
        for token in subset:
            counts[token] = counts.get(token, 0) + 1
        if counts and max(counts.values()) >= 14:
            return "short_token_burst"
    return None


def mixed_latin_noise(text: str) -> bool:
    latin_tokens = re.findall(r"\b[A-Za-z]{1,12}\b", text)
    if len(latin_tokens) < 12:
        return False
    hangul_chars = sum(1 for char in text if "가" <= char <= "힣")
    if hangul_chars < 200:
        return False
    repeated_hangul_fragments = len(re.findall(r"([가-힣]{1,3})\1{2,}", text))
    short_latin_ratio = sum(1 for token in latin_tokens if len(token) <= 4) / max(1, len(latin_tokens))
    return repeated_hangul_fragments >= 2 and short_latin_ratio >= 0.55


def collapse_reason(text: str, *, require_result_tags: bool = False) -> str | None:
    contract = analyze_result_contract(text, require_result_tags=require_result_tags)
    if contract.reason in {
        "missing_result_tags",
        "missing_result_open",
        "missing_result_close",
        "post_result_text",
        "thought_leak",
        "multiple_result_open",
        "multiple_result_close",
    }:
        return contract.reason
    result = contract.result_text.strip()
    if not result:
        return "empty"
    if len(result) < 120:
        return "too_short"
    if SPECIAL_TOKEN_LEAK_RE.search(result):
        return "special_token_leak"
    foreign_counts = foreign_script_counts(result)
    if any(value > 0 for name, value in foreign_counts.items() if name in {"replacement", "private_use", "hangul_jamo"}):
        return "corrupt_character"
    if any(value > 0 for name, value in foreign_counts.items() if name in {"arabic", "cyrillic", "devanagari", "kana", "thai"}):
        return "foreign_script_leak"
    if symbol_noise_ratio(result) > 0.08:
        return "symbol_noise"
    if max_repeated_hangul_or_alpha_run(result) >= 10:
        return "char_run_repeat"
    if max_repeated_digit_run(result) >= 12:
        return "char_run_repeat"
    if max_repeated_char_run(result) >= 20:
        return "char_run_repeat"
    if re.search(r"(.{2,30})\1{5,}", result, flags=re.DOTALL):
        return "char_run_repeat"
    short_token_reason = short_token_burst_reason(result)
    if short_token_reason:
        return short_token_reason
    if mixed_latin_noise(result):
        return "mixed_latin_noise"
    if repeated_line_rate(result) > 0.25:
        return "line_repeat"
    if repeated_ngram_rate(result, n=8) > 0.20:
        return "ngram_repeat"
    return None
