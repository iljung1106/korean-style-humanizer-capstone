"""GUI-compatible raw style metrics for phase-end evaluation.

This module intentionally does not import the legacy GUI/eval modules.  The
metric formulas are copied into pipeline_v2 so phase-end evaluation can run as
a self-contained artifact while preserving the GUI metric semantics.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONTENT_POS_PREFIXES = ("N", "V", "M", "XR")
PUNCT_TAG_PREFIXES = ("S",)
MODIFIER_TAG_PREFIXES = ("MAG", "MAJ", "MM")
PREDICATE_TAGS = {"VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"}
BOUND_NOUNS = {"것", "수", "점", "바", "데", "줄"}
POS_WINDOW_SIZE = 200
POS_WINDOW_STRIDE = 100
POS_WINDOW_MIN_SIZE = 80
SENTENCE_WINDOW_SIZE = 20
SENTENCE_WINDOW_STRIDE = 10
SENTENCE_WINDOW_MIN_SIZE = 10
MODIFIER_BURST_TAU = 20.0
CONTENT_POS_GROUPS = {
    "content_noun": ("N",),
    "content_predicate": ("V",),
    "content_modifier": ("M",),
    "content_root": ("XR",),
}

DEFAULT_REPORT_METRICS = [
    "char_len",
    "sentence_count",
    "morph_count",
    "comma_per_1k_chars",
    "comma_sentence_rate",
    "commas_per_sentence",
    "parenthesis_pair_per_1k_chars",
    "parenthesis_sentence_rate",
    "sentence_length_mean_morphs",
    "sentence_length_cv",
    "sentence_length_median_morphs",
    "sentence_length_iqr_ratio",
    "sentence_initial_token_top3_window_coverage",
    "sentence_initial_pos2_top3_window_coverage",
    "sentence_final_token_top3_window_coverage",
    "sentence_final_pos2_top3_window_coverage",
    "sentence_initial_token_repeat_rate",
    "sentence_initial_pos_bigram_repeat_rate",
    "sentence_final_token_repeat_rate",
    "sentence_final_pos_bigram_repeat_rate",
    "pos_3gram_diversity",
    "pos_4gram_diversity",
    "pos_5gram_diversity",
    "pos_6gram_diversity",
    "pos_3gram_repeat_rate",
    "pos_4gram_repeat_rate",
    "pos_5gram_repeat_rate",
    "pos_6gram_repeat_rate",
    "content_modifier_repeat_occurrence_rate",
    "content_modifier_simpson_concentration",
    "content_modifier_gini_frequency",
    "content_modifier_yule_k",
    "content_modifier_2gram_repeat_rate",
    "modifier_hill_d2_norm",
    "modifier_repetition_mass",
    "modifier_repeat_burst_mass",
]


@dataclass
class Chunk:
    group: str
    source_file: str
    chunk_id: str
    text: str
    sampled_offset: int | None = None


def hangul_ratio(text: str) -> float:
    return sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3") / max(1, len(text))


def detect_encoding(data: bytes) -> str:
    if data.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if data.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if len(data) >= 4:
        even_nulls = data[0::2].count(0)
        odd_nulls = data[1::2].count(0)
        half = max(1, len(data) // 2)
        if odd_nulls / half > 0.25 and even_nulls / half < 0.05:
            return "utf-16-le"
        if even_nulls / half > 0.25 and odd_nulls / half < 0.05:
            return "utf-16-be"
    best_enc = "utf-8"
    best_score = -1.0
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        decoded = data.decode(enc, errors="ignore")
        score = hangul_ratio(decoded)
        if score > best_score:
            best_score = score
            best_enc = enc
    return best_enc


def decode_bytes(data: bytes, encoding: str | None = None) -> str:
    enc = encoding or detect_encoding(data)
    return data.decode(enc, errors="ignore")


def read_text(path: Path) -> str:
    return decode_bytes(path.read_bytes())


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _boundary_index(text: str, start: int, target_end: int, max_end: int) -> int:
    target_end = min(target_end, len(text))
    max_end = min(max_end, len(text))
    for marker in ("\n\n", "\n", "다.", "요.", "까.", "죠.", ". ", "! ", "? "):
        idx = text.rfind(marker, start + 1, max_end)
        if idx >= start + 200:
            return idx + len(marker)
    return target_end


def paragraph_chunks(
    text: str,
    *,
    target_chars: int = 5000,
    min_chars: int = 2500,
    max_chars: int = 7000,
) -> list[str]:
    """Build evaluation chunks on paragraph boundaries where possible."""

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", clean_text(text)) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        piece = "\n\n".join(current).strip()
        if len(piece) >= min_chars:
            chunks.append(piece)
        current = []
        current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            start = 0
            while start < len(paragraph):
                end = _boundary_index(paragraph, start, start + target_chars, start + max_chars)
                piece = paragraph[start:end].strip()
                if len(piece) >= min_chars:
                    chunks.append(piece)
                start = max(end, start + 1)
            continue
        next_len = current_len + len(paragraph) + (2 if current else 0)
        if current and next_len > max_chars:
            flush()
        current.append(paragraph)
        current_len += len(paragraph) + (2 if len(current) > 1 else 0)
        if current_len >= target_chars:
            flush()
    flush()
    return chunks


def ngram_stats(items: list[str], n: int) -> tuple[float, float]:
    if len(items) < n:
        return float("nan"), float("nan")
    grams = [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]
    diversity = len(set(grams)) / len(grams)
    entropy = shannon_entropy(["/".join(g) for g in grams])
    return diversity, entropy


def repeated_ngram_rate(items: list[str], n: int) -> float:
    if len(items) < n:
        return float("nan")
    grams = [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def windows(items: list[str], window_size: int, stride: int, min_size: int) -> list[list[str]]:
    if not items:
        return []
    if len(items) < window_size:
        return [items] if len(items) >= min_size else []
    return [items[i : i + window_size] for i in range(0, len(items) - window_size + 1, stride)]


def window_recurrence(items: list[str], n: int, window_size: int, stride: int, min_size: int) -> float:
    values: list[float] = []
    for window in windows(items, window_size, stride, min_size):
        values.append(repeated_ngram_rate(window, n))
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else 0.0


def topk_window_coverage(items: list[str], k: int, window_size: int, stride: int, min_size: int) -> float:
    values: list[float] = []
    for window in windows(items, window_size, stride, min_size):
        if not window:
            continue
        counts = Counter(window)
        values.append(sum(count for _item, count in counts.most_common(k)) / len(window))
    return statistics.fmean(values) if values else 0.0


def hill_d2_norm(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    vocab = len(counts)
    if total <= 0 or vocab <= 0:
        return 0.0
    simpson = sum((count / total) ** 2 for count in counts.values())
    if simpson <= 0:
        return 0.0
    return (1.0 / simpson) / vocab


def repetition_mass(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    return sum(count - 1 for count in counts.values() if count >= 2) / len(items)


def repeat_burst_mass(items: list[str], tau: float = MODIFIER_BURST_TAU) -> float:
    if not items:
        return 0.0
    last_seen: dict[str, int] = {}
    burst = 0.0
    for index, item in enumerate(items):
        if item in last_seen:
            burst += math.exp(-(index - last_seen[item]) / tau)
        last_seen[item] = index
    return burst / len(items)


def shannon_entropy(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def mtld(lemmas: list[str], threshold: float = 0.72) -> float:
    def one_direction(seq: list[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        token_count = 0
        for tok in seq:
            token_count += 1
            types.add(tok)
            ttr = len(types) / token_count
            if ttr <= threshold:
                factors += 1
                types.clear()
                token_count = 0
        if token_count:
            ttr = len(types) / token_count
            factors += (1 - ttr) / (1 - threshold)
        return len(seq) / factors if factors else float(len(seq))

    if len(lemmas) < 20:
        return float("nan")
    return (one_direction(lemmas) + one_direction(list(reversed(lemmas)))) / 2


def gini(values: list[int]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    total = sum(ordered)
    if total == 0:
        return 0.0
    n = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return float((2 * weighted) / (n * total) - (n + 1) / n)


def yule_k(items: list[str]) -> float:
    if not items:
        return float("nan")
    counts = Counter(items)
    freq_of_freq = Counter(counts.values())
    m1 = sum(freq * n_types for freq, n_types in freq_of_freq.items())
    m2 = sum((freq**2) * n_types for freq, n_types in freq_of_freq.items())
    if m1 == 0:
        return float("nan")
    return 10000 * (m2 - m1) / (m1 * m1)


def iqr_ratio(values: list[int]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    if med == 0:
        return 0.0
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return (q3 - q1) / med


def coverage(counts: Counter[str], top_n: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return sum(count for _item, count in counts.most_common(top_n)) / total


def lexical_distribution_metrics(items: list[str], prefix: str) -> dict[str, float]:
    counts = Counter(items)
    total = sum(counts.values())
    types = len(counts)
    if total == 0 or types == 0:
        return {
            f"{prefix}_repeat_occurrence_rate": float("nan"),
            f"{prefix}_hapax_rate": float("nan"),
            f"{prefix}_top_10_coverage": float("nan"),
            f"{prefix}_top_20_coverage": float("nan"),
            f"{prefix}_max_frequency_rate": float("nan"),
            f"{prefix}_simpson_concentration": float("nan"),
            f"{prefix}_gini_frequency": float("nan"),
            f"{prefix}_yule_k": float("nan"),
        }
    repeated_occurrences = sum(count for count in counts.values() if count > 1)
    freqs = list(counts.values())
    probs = [count / total for count in freqs]
    return {
        f"{prefix}_repeat_occurrence_rate": repeated_occurrences / total,
        f"{prefix}_hapax_rate": sum(1 for count in freqs if count == 1) / types,
        f"{prefix}_top_10_coverage": coverage(counts, 10),
        f"{prefix}_top_20_coverage": coverage(counts, 20),
        f"{prefix}_max_frequency_rate": max(freqs) / total,
        f"{prefix}_simpson_concentration": sum(prob * prob for prob in probs),
        f"{prefix}_gini_frequency": gini(freqs),
        f"{prefix}_yule_k": yule_k(items),
    }


def content_pos_group_metrics(tokens: list[Any], max_ngram: int = 3) -> dict[str, float]:
    row: dict[str, float] = {}
    for prefix, tag_prefixes in CONTENT_POS_GROUPS.items():
        items = [f"{t.form}/{t.tag}" for t in tokens if t.tag.startswith(tag_prefixes) and t.tag != "NNP"]
        row[f"{prefix}_token_count"] = float(len(items))
        row.update(lexical_distribution_metrics(items, prefix))
        for n in range(1, max_ngram + 1):
            row[f"{prefix}_{n}gram_repeat_rate"] = repeated_ngram_rate(items, n)
            diversity, entropy = ngram_stats(items, n)
            row[f"{prefix}_{n}gram_distinct_rate"] = diversity
            row[f"{prefix}_{n}gram_entropy"] = entropy
    return row


def sentence_edge_sequences(
    sent_texts: list[str],
    kiwi: Any,
    token_n: int,
    pos_n: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    initial_token_sequences: list[str] = []
    initial_pos_sequences: list[str] = []
    final_token_sequences: list[str] = []
    final_pos_sequences: list[str] = []
    for sent in sent_texts:
        tokens = [t for t in kiwi.tokenize(sent) if not t.tag.startswith(PUNCT_TAG_PREFIXES)]
        if len(tokens) >= token_n:
            initial_token_sequences.append("/".join(f"{t.form}/{t.tag}" for t in tokens[:token_n]))
            final_token_sequences.append("/".join(f"{t.form}/{t.tag}" for t in tokens[-token_n:]))
        if len(tokens) >= pos_n:
            initial_pos_sequences.append("/".join(t.tag for t in tokens[:pos_n]))
            final_pos_sequences.append("/".join(t.tag for t in tokens[-pos_n:]))
    return initial_token_sequences, initial_pos_sequences, final_token_sequences, final_pos_sequences


def analyze_chunk(chunk: Chunk, kiwi: Any) -> tuple[dict[str, Any], Counter[str]]:
    text = chunk.text
    tokens = kiwi.tokenize(text)
    sents = kiwi.split_into_sents(text)
    sent_texts = [sent.text for sent in sents if sent.text.strip()]
    if not sent_texts:
        sent_texts = [part for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if part.strip()]

    comma_count = text.count(",") + text.count("，")
    parenthesis_open_count = text.count("(") + text.count("（")
    parenthesis_close_count = text.count(")") + text.count("）")
    parenthesis_count = parenthesis_open_count + parenthesis_close_count
    parenthesis_pair_count = min(parenthesis_open_count, parenthesis_close_count)
    sent_count = len(sent_texts)
    comma_sent_count = sum(1 for sent in sent_texts if "," in sent or "，" in sent)
    parenthesis_sent_count = sum(1 for sent in sent_texts if re.search(r"[()（）]", sent))
    rel_positions: list[float] = []
    segment_lens: list[int] = []
    context_pos: list[str] = []

    for sent in sent_texts:
        comma_positions = [match.start() for match in re.finditer(r"[,，]", sent)]
        if comma_positions:
            rel_positions.extend(pos / max(1, len(sent)) for pos in comma_positions)
            last = 0
            for pos in comma_positions:
                segment_lens.append(pos - last)
                last = pos + 1
            segment_lens.append(len(sent) - last)

    for index, tok in enumerate(tokens):
        if tok.form in {",", "，"}:
            for token in tokens[max(0, index - 3) : index] + tokens[index + 1 : index + 4]:
                if not token.tag.startswith(PUNCT_TAG_PREFIXES):
                    context_pos.append(token.tag)

    pos_tags = [token.tag for token in tokens if not token.tag.startswith(PUNCT_TAG_PREFIXES)]
    surface_tokens = [token.form for token in tokens if not token.tag.startswith(PUNCT_TAG_PREFIXES)]
    modifier_tokens = [
        f"{token.form}/{token.tag}"
        for token in tokens
        if token.tag.startswith(MODIFIER_TAG_PREFIXES)
    ]
    content_lemmas = [
        f"{token.form}/{token.tag}"
        for token in tokens
        if token.tag.startswith(CONTENT_POS_PREFIXES) and token.tag != "NNP"
    ]
    content_with_nnp = [f"{token.form}/{token.tag}" for token in tokens if token.tag.startswith(CONTENT_POS_PREFIXES)]

    etn_count = sum(1 for token in tokens if token.tag == "ETN")
    etm_bound_count = 0
    for index in range(len(tokens) - 1):
        if tokens[index].tag == "ETM" and tokens[index + 1].tag == "NNB" and tokens[index + 1].form in BOUND_NOUNS:
            etm_bound_count += 1
    predicate_count = sum(1 for token in tokens if token.tag in PREDICATE_TAGS)
    sentence_lengths = [
        len([token for token in kiwi.tokenize(sent) if not token.tag.startswith(PUNCT_TAG_PREFIXES)])
        for sent in sent_texts
    ]
    initial_tokens, initial_pos_bigrams, final_tokens, final_pos_bigrams = sentence_edge_sequences(
        sent_texts, kiwi, 1, 2
    )
    sent_len_mean = statistics.fmean(sentence_lengths) if sentence_lengths else float("nan")
    sent_len_std = statistics.pstdev(sentence_lengths) if sentence_lengths else float("nan")
    sent_len_median = statistics.median(sentence_lengths) if sentence_lengths else 0.0

    row: dict[str, Any] = {
        "group": chunk.group,
        "source_file": chunk.source_file,
        "chunk_id": chunk.chunk_id,
        "sampled_offset": chunk.sampled_offset if chunk.sampled_offset is not None else -1,
        "char_len": len(text),
        "sentence_count": sent_count,
        "morph_count": len(tokens),
        "pos_token_count": len(pos_tags),
        "modifier_token_count": len(modifier_tokens),
        "modifier_per_1k_morphs": len(modifier_tokens) / max(1, len(tokens)) * 1000,
        "content_lemma_count": len(content_lemmas),
        "comma_count": comma_count,
        "comma_per_1k_chars": comma_count / max(1, len(text)) * 1000,
        "comma_sentence_rate": comma_sent_count / max(1, sent_count),
        "commas_per_sentence": comma_count / max(1, sent_count),
        "parenthesis_count": parenthesis_count,
        "parenthesis_pair_count": parenthesis_pair_count,
        "parenthesis_per_1k_chars": parenthesis_count / max(1, len(text)) * 1000,
        "parenthesis_pair_per_1k_chars": parenthesis_pair_count / max(1, len(text)) * 1000,
        "parenthesis_sentence_rate": parenthesis_sent_count / max(1, sent_count),
        "parentheses_per_sentence": parenthesis_count / max(1, sent_count),
        "comma_mean_relative_position": statistics.fmean(rel_positions) if rel_positions else float("nan"),
        "comma_segment_mean_chars": statistics.fmean(segment_lens) if segment_lens else float("nan"),
        "comma_context_pos_entropy": shannon_entropy(context_pos),
        "clausal_nominal_etn_per_1k_morphs": etn_count / max(1, len(tokens)) * 1000,
        "clausal_nominal_etn_per_100_predicates": etn_count / max(1, predicate_count) * 100,
        "etm_bound_noun_per_1k_morphs": etm_bound_count / max(1, len(tokens)) * 1000,
        "lexical_mtld_no_nnp": mtld(content_lemmas),
        "lexical_entropy_no_nnp": shannon_entropy(content_lemmas),
        "lexical_ttr_no_nnp": len(set(content_lemmas)) / max(1, len(content_lemmas)),
        "lexical_ttr_with_nnp": len(set(content_with_nnp)) / max(1, len(content_with_nnp)),
        "sentence_length_mean_morphs": sent_len_mean,
        "sentence_length_cv": sent_len_std / sent_len_mean if sent_len_mean and not math.isnan(sent_len_mean) else float("nan"),
        "sentence_length_median_morphs": float(sent_len_median),
        "sentence_length_iqr_ratio": iqr_ratio(sentence_lengths),
        "sentence_initial_token_top3_window_coverage": topk_window_coverage(
            initial_tokens, 3, SENTENCE_WINDOW_SIZE, SENTENCE_WINDOW_STRIDE, SENTENCE_WINDOW_MIN_SIZE
        ),
        "sentence_initial_pos2_top3_window_coverage": topk_window_coverage(
            initial_pos_bigrams, 3, SENTENCE_WINDOW_SIZE, SENTENCE_WINDOW_STRIDE, SENTENCE_WINDOW_MIN_SIZE
        ),
        "sentence_final_token_top3_window_coverage": topk_window_coverage(
            final_tokens, 3, SENTENCE_WINDOW_SIZE, SENTENCE_WINDOW_STRIDE, SENTENCE_WINDOW_MIN_SIZE
        ),
        "sentence_final_pos2_top3_window_coverage": topk_window_coverage(
            final_pos_bigrams, 3, SENTENCE_WINDOW_SIZE, SENTENCE_WINDOW_STRIDE, SENTENCE_WINDOW_MIN_SIZE
        ),
        "sentence_initial_token_repeat_rate": repeated_ngram_rate(initial_tokens, 1),
        "sentence_initial_pos_bigram_repeat_rate": repeated_ngram_rate(initial_pos_bigrams, 1),
        "sentence_final_token_repeat_rate": repeated_ngram_rate(final_tokens, 1),
        "sentence_final_pos_bigram_repeat_rate": repeated_ngram_rate(final_pos_bigrams, 1),
        "modifier_hill_d2_norm": hill_d2_norm(modifier_tokens),
        "modifier_repetition_mass": repetition_mass(modifier_tokens),
        "modifier_repeat_burst_mass": repeat_burst_mass(modifier_tokens),
    }
    row.update(lexical_distribution_metrics(content_lemmas, "content"))
    row.update(content_pos_group_metrics(tokens))
    for n in range(1, 6):
        row[f"content_{n}gram_repeat_rate"] = repeated_ngram_rate(content_lemmas, n)
        diversity, entropy = ngram_stats(content_lemmas, n)
        row[f"content_{n}gram_distinct_rate"] = diversity
        row[f"content_{n}gram_entropy"] = entropy
    row["content_bigram_repeat_rate"] = row["content_2gram_repeat_rate"]
    row["content_trigram_repeat_rate"] = row["content_3gram_repeat_rate"]
    for n in range(2, 7):
        diversity, entropy = ngram_stats(pos_tags, n)
        row[f"pos_{n}gram_diversity"] = diversity
        row[f"pos_{n}gram_entropy"] = entropy
        row[f"pos_{n}gram_repeat_rate"] = repeated_ngram_rate(pos_tags, n)
        row[f"pos_{n}gram_window_recurrence"] = window_recurrence(
            pos_tags,
            n,
            POS_WINDOW_SIZE,
            POS_WINDOW_STRIDE,
            POS_WINDOW_MIN_SIZE,
        )
        row[f"pos_{n}gram_entropy_norm"] = entropy / math.log(max(2, len(set(tuple(pos_tags[i : i + n]) for i in range(max(0, len(pos_tags) - n + 1))))))
        if not math.isfinite(row[f"pos_{n}gram_entropy_norm"]):
            row[f"pos_{n}gram_entropy_norm"] = 0.0

    return row, Counter(surface_tokens)


def analyze_text(text: str, *, group: str, source_file: str = "", chunk_id: str = "") -> dict[str, Any]:
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    row, _surface = analyze_chunk(Chunk(group=group, source_file=source_file, chunk_id=chunk_id, text=text), kiwi)
    return row


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize_numeric(values: Iterable[Any]) -> dict[str, Any]:
    finite = [finite_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite) if len(finite) >= 2 else 0.0,
        "median": quantile(finite, 0.50),
        "p10": quantile(finite, 0.10),
        "p25": quantile(finite, 0.25),
        "p75": quantile(finite, 0.75),
        "p90": quantile(finite, 0.90),
        "min": min(finite),
        "max": max(finite),
    }


def cliffs_delta(left: Iterable[Any], right: Iterable[Any]) -> float:
    a = [finite_float(value) for value in left]
    b = [finite_float(value) for value in right]
    a = [value for value in a if math.isfinite(value)]
    b = [value for value in b if math.isfinite(value)]
    if not a or not b:
        return float("nan")
    greater = 0
    lower = 0
    for value in a:
        greater += sum(1 for other in b if value > other)
        lower += sum(1 for other in b if value < other)
    return (greater - lower) / (len(a) * len(b))
