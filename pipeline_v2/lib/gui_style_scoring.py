"""Standalone GUI-style scoring utilities for pipeline_v2.

This module deliberately does not import legacy `train/`, `scripts/`, or
`eval/` modules. It implements the subset of metrics needed for pilot-stage
comparison: result contract, anti-slop density, translationese SVM when
available, comma overuse, POS n-gram shape, simple repetition, and rewrite
copy/length diagnostics.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from difflib import SequenceMatcher
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .io import TRAINING_ROOT
from .result_contract import analyze_result_contract, collapse_reason


DEFAULT_REFERENCE = TRAINING_ROOT / "data" / "processed" / "gui_style_reward_reference.json"
DEFAULT_ANTI_SLOP_LEXICON = TRAINING_ROOT / "data" / "processed" / "anti_slop_lexicon.json"
DEFAULT_TRANSLATIONESE_MODEL = TRAINING_ROOT / "models" / "translationese_svm" / "svm_detector.joblib"

PUNCT_TAG_PREFIXES = ("S",)
MODIFIER_TAG_PREFIXES = ("MAG", "MAJ", "MM")
CONTENT_TAG_PREFIXES = ("N", "V", "M")
MODIFIER_BURST_TAU = 20.0
SIMILE_MARKER_PATTERNS = (
    re.compile(r"마치"),
    re.compile(r"처럼"),
    re.compile(r"같이"),
    re.compile(r"듯이"),
    re.compile(r"마냥"),
    re.compile(r"[가-힣A-Za-z0-9]+같은"),
    re.compile(r"[가-힣A-Za-z0-9]+듯한"),
)
CONTENT_POS_GROUPS = {
    "content_noun": ("N",),
    "content_predicate": ("V",),
    "content_modifier": ("M",),
    "content_root": ("XR",),
}
PRIMARY_SCALAR_METRICS = {
    "comma_per_1k_chars",
    "comma_sentence_rate",
    "pos_3gram_diversity",
    "pos_3gram_repeat_rate",
    "pos_4gram_diversity",
    "pos_4gram_repeat_rate",
    "pos_5gram_diversity",
    "pos_5gram_repeat_rate",
    "pos_6gram_diversity",
    "pos_6gram_repeat_rate",
    "sentence_initial_pos_bigram_repeat_rate",
    "sentence_final_token_repeat_rate",
    "sentence_length_cv",
    "sentence_length_iqr_ratio",
    "simile_marker_per_1k_chars",
    "simile_sentence_rate",
    "modifier_repetition_mass",
    "modifier_repeat_burst_mass",
}
METRIC_FAMILY_WEIGHTS = {
    "anti_slop": 0.42,
    "translationese": 0.32,
    "comma": 0.22,
    "pos_4_6": 0.36,
    "pos_3": 0.12,
    "pos_1_2": 0.02,
    "pos_usage": 0.16,
    "modifier": 0.08,
    "sentence_edge": 0.04,
    "sentence_length": 0.025,
    "other": 0.05,
}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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


def summarize_numeric(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": float("nan"),
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
        "median": quantile(finite, 0.50),
        "p10": quantile(finite, 0.10),
        "p25": quantile(finite, 0.25),
        "p75": quantile(finite, 0.75),
        "p90": quantile(finite, 0.90),
        "min": min(finite),
        "max": max(finite),
    }


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    try:
        kiwi = get_kiwi()
        return [sent.text.strip() for sent in kiwi.split_into_sents(text) if sent.text.strip()]
    except Exception:
        return [piece.strip() for piece in re.split(r"(?<=[.!?。！？…])\s+|\n+", text) if piece.strip()]


@lru_cache(maxsize=1)
def get_kiwi() -> Any:
    from kiwipiepy import Kiwi

    return Kiwi()


def kiwi_available() -> bool:
    try:
        get_kiwi()
        return True
    except Exception:
        return False


def morphs(text: str) -> list[tuple[str, str]]:
    kiwi = get_kiwi()
    return [(token.form, token.tag) for token in kiwi.tokenize(text) if token.form and token.form.strip()]


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    if len(items) < n:
        return []
    return [tuple(items[index : index + n]) for index in range(len(items) - n + 1)]


def repeated_occurrence_rate(items: list[Any]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    return sum(count for count in counts.values() if count > 1) / len(items)


def repetition_mass(items: list[Any]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    return sum(count - 1 for count in counts.values() if count >= 2) / len(items)


def repeat_burst_mass(items: list[Any], tau: float = MODIFIER_BURST_TAU) -> float:
    if not items:
        return 0.0
    last_seen: dict[Any, int] = {}
    burst = 0.0
    for index, item in enumerate(items):
        if item in last_seen:
            burst += math.exp(-(index - last_seen[item]) / tau)
        last_seen[item] = index
    return burst / len(items)


def diversity(items: list[Any]) -> float:
    if not items:
        return 0.0
    return len(set(items)) / len(items)


def distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items() if value > 0}


def js_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return float("nan")
    midpoint = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in keys}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for key, value in a.items():
            if value <= 0.0:
                continue
            total += value * math.log(value / max(b.get(key, 0.0), 1e-12), 2)
        return total

    return math.sqrt(max(0.0, 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)))


def human_band_score(value: float, stats: dict[str, Any]) -> float:
    if not math.isfinite(value):
        return float("nan")
    q10 = finite_float(stats.get("human_q10"))
    q25 = finite_float(stats.get("human_q25"))
    q50 = finite_float(stats.get("human_q50"), finite_float(stats.get("human_mean")))
    q75 = finite_float(stats.get("human_q75"))
    q90 = finite_float(stats.get("human_q90"))
    iqr = finite_float(stats.get("human_iqr"))
    if all(math.isfinite(item) for item in (q25, q75)) and q25 <= value <= q75:
        return 1.0
    if all(math.isfinite(item) for item in (q10, q25)) and value < q25:
        if value >= q10:
            return clip((value - q10) / max(1e-9, q25 - q10))
        return clip(-min(1.0, (q10 - value) / max(1e-9, q25 - q10)))
    if all(math.isfinite(item) for item in (q75, q90)) and value > q75:
        if value <= q90:
            return clip((q90 - value) / max(1e-9, q90 - q75))
        return clip(-min(1.0, (value - q90) / max(1e-9, q90 - q75)))
    if not math.isfinite(iqr) or iqr <= 1e-9:
        iqr = max(abs(q50) * 0.10, 1e-6) if math.isfinite(q50) else 1.0
    if math.isfinite(q50):
        return clip(1.0 - abs(value - q50) / max(1e-9, iqr))
    return float("nan")


def human_banded_reward_score(value: float, stats: dict[str, Any]) -> float:
    """Reward shape: weak inside human IQR, strong outside the broad band."""

    if not math.isfinite(value):
        return float("nan")
    q10 = finite_float(stats.get("human_q10"))
    q25 = finite_float(stats.get("human_q25"))
    q50 = finite_float(stats.get("human_q50"), finite_float(stats.get("human_mean")))
    q75 = finite_float(stats.get("human_q75"))
    q90 = finite_float(stats.get("human_q90"))
    if not all(math.isfinite(item) for item in (q10, q25, q50, q75, q90)):
        return human_band_score(value, stats)

    left_iqr = max(1e-9, q25 - q10)
    right_iqr = max(1e-9, q90 - q75)
    median_span = max(1e-9, max(q75 - q25, abs(q50) * 0.05, 1e-6))

    if q25 <= value <= q75:
        center_distance = abs(value - q50) / median_span
        return clip(0.22 - 0.10 * center_distance, 0.08, 0.22)
    if q10 <= value < q25:
        return clip(-0.42 * (q25 - value) / left_iqr, -0.42, 0.0)
    if q75 < value <= q90:
        return clip(-0.42 * (value - q75) / right_iqr, -0.42, 0.0)
    if value < q10:
        return clip(-0.42 - 0.58 * min(1.0, (q10 - value) / left_iqr), -1.0, -0.42)
    return clip(-0.42 - 0.58 * min(1.0, (value - q90) / right_iqr), -1.0, -0.42)


def anti_ai_score(value: float, stats: dict[str, Any]) -> float:
    human_mean = finite_float(stats.get("human_mean"))
    ai_mean = finite_float(stats.get("ai_mean"))
    if not all(math.isfinite(item) for item in (value, human_mean, ai_mean)):
        return float("nan")
    gap = human_mean - ai_mean
    if abs(gap) < 1e-9:
        return 0.0
    progress = (value - ai_mean) / gap
    return clip(2.0 * progress - 1.0)


def scalar_metric_score(value: float, stats: dict[str, Any]) -> float:
    band = human_banded_reward_score(value, stats)
    anti = anti_ai_score(value, stats)
    parts = []
    if math.isfinite(band):
        parts.append((band, 0.90))
    if math.isfinite(anti):
        parts.append((0.25 * anti, 0.10))
    if not parts:
        return float("nan")
    return clip(sum(value * weight for value, weight in parts) / sum(weight for _value, weight in parts))


def separation_weight(stats: dict[str, Any]) -> float:
    base = abs(finite_float(stats.get("weight"), 1.0))
    ai_in_iqr = finite_float(stats.get("ai_in_human_iqr_rate"))
    if math.isfinite(ai_in_iqr):
        base *= 0.35 + 0.65 * (1.0 - clip(ai_in_iqr, 0.0, 1.0))
    return max(0.0, base)


def metric_family(name: str) -> str:
    if name == "anti_slop_density":
        return "anti_slop"
    if name == "translationese_raw":
        return "translationese"
    if name.startswith("comma_"):
        return "comma"
    if name.startswith("pos_"):
        if any(name.startswith(f"pos_{n}gram") for n in (4, 5, 6)):
            return "pos_4_6"
        if name.startswith("pos_3gram"):
            return "pos_3"
        return "pos_1_2"
    if "modifier" in name:
        return "modifier"
    if name.startswith("sentence_initial") or name.startswith("sentence_final"):
        return "sentence_edge"
    if name.startswith("sentence_length"):
        return "sentence_length"
    return "other"


def text_basic_metrics(text: str) -> dict[str, float | int]:
    stripped = text.strip()
    sentences = split_sentences(stripped)
    sentence_count = max(1, len(sentences))
    char_len = max(1, len(stripped))
    comma_count = stripped.count(",") + stripped.count("，")
    hangul_count = len(re.findall(r"[가-힣]", stripped))
    letter_count = len(re.findall(r"[가-힣A-Za-z]", stripped))
    sentence_lengths = [len(sentence.strip()) for sentence in sentences if sentence.strip()]
    sentence_initial_tokens = []
    sentence_final_tokens = []
    for sentence in sentences:
        tokens = lexical_tokens(sentence)
        if tokens:
            sentence_initial_tokens.append(tokens[0])
            sentence_final_tokens.append(tokens[-1])
    mean_sentence_len = statistics.fmean(sentence_lengths) if sentence_lengths else 0.0
    sentence_len_std = statistics.pstdev(sentence_lengths) if len(sentence_lengths) >= 2 else 0.0
    sentence_length_median = quantile([float(value) for value in sentence_lengths], 0.5) if sentence_lengths else 0.0
    sentence_length_iqr_ratio = (
        (quantile([float(value) for value in sentence_lengths], 0.75) - quantile([float(value) for value in sentence_lengths], 0.25))
        / max(1.0, sentence_length_median)
        if sentence_lengths
        else 0.0
    )
    simile_marker_count = sum(len(pattern.findall(stripped)) for pattern in SIMILE_MARKER_PATTERNS)
    simile_sentence_count = sum(
        1 for sentence in sentences if any(pattern.search(sentence) for pattern in SIMILE_MARKER_PATTERNS)
    )
    return {
        "char_len": len(stripped),
        "sentence_count": len(sentences),
        "hangul_ratio": hangul_count / max(1, letter_count),
        "comma_per_1k_chars": 1000.0 * comma_count / char_len,
        "comma_sentence_rate": sum(1 for sentence in sentences if "," in sentence or "，" in sentence) / sentence_count,
        "sentence_length_mean_chars": mean_sentence_len,
        "sentence_length_cv": sentence_len_std / max(1.0, mean_sentence_len),
        "sentence_length_iqr_ratio": sentence_length_iqr_ratio,
        "sentence_initial_token_repeat_rate": repeated_occurrence_rate(sentence_initial_tokens),
        "sentence_final_token_repeat_rate": repeated_occurrence_rate(sentence_final_tokens),
        "simile_marker_count": simile_marker_count,
        "simile_marker_per_1k_chars": 1000.0 * simile_marker_count / char_len,
        "simile_sentence_rate": simile_sentence_count / sentence_count,
    }


def text_pos_metrics(text: str, *, max_ngram_n: int = 6) -> tuple[dict[str, float | int], dict[str, dict[str, float]]]:
    try:
        analyzed = morphs(text)
    except Exception:
        return {"kiwi_available": 0}, {}
    non_punct = [(form, tag) for form, tag in analyzed if not tag.startswith(PUNCT_TAG_PREFIXES)]
    tags = [tag for _form, tag in non_punct]
    modifier_items = [f"{form}/{tag}" for form, tag in non_punct if tag.startswith(MODIFIER_TAG_PREFIXES)]
    metrics: dict[str, float | int] = {
        "kiwi_available": 1,
        "morph_count": len(analyzed),
        "pos_token_count": len(tags),
        "modifier_repetition_mass": repetition_mass(modifier_items),
        "modifier_repeat_burst_mass": repeat_burst_mass(modifier_items),
    }
    metrics.update(lexical_distribution_metrics(modifier_items, "content_modifier"))
    distributions: dict[str, dict[str, float]] = {}
    for n in range(1, max_ngram_n + 1):
        grams = ngrams(tags, n)
        metrics[f"pos_{n}gram_diversity"] = diversity(grams)
        metrics[f"pos_{n}gram_repeat_rate"] = repeated_occurrence_rate(grams)
        distributions[str(n)] = distribution(Counter("/".join(gram) for gram in grams))

    sentence_tokens: list[list[tuple[str, str]]] = []
    for sentence in split_sentences(text):
        try:
            sentence_tokens.append([(form, tag) for form, tag in morphs(sentence) if not tag.startswith(PUNCT_TAG_PREFIXES)])
        except Exception:
            continue
    initial_pos2: list[tuple[str, str]] = []
    final_pos2: list[tuple[str, str]] = []
    for tokens in sentence_tokens:
        if len(tokens) >= 2:
            initial_pos2.append((tokens[0][1], tokens[1][1]))
            final_pos2.append((tokens[-2][1], tokens[-1][1]))
    metrics["sentence_initial_pos_bigram_repeat_rate"] = repeated_occurrence_rate(initial_pos2)
    metrics["sentence_final_pos_bigram_repeat_rate"] = repeated_occurrence_rate(final_pos2)
    return metrics, distributions


def shannon_entropy(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values())


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


def lexical_distribution_metrics(items: list[str], prefix: str) -> dict[str, float]:
    counts = Counter(items)
    total = sum(counts.values())
    if total == 0:
        return {
            f"{prefix}_repeat_occurrence_rate": float("nan"),
            f"{prefix}_simpson_concentration": float("nan"),
            f"{prefix}_gini_frequency": float("nan"),
            f"{prefix}_yule_k": float("nan"),
        }
    freqs = list(counts.values())
    probs = [count / total for count in freqs]
    repeated_occurrences = sum(count for count in counts.values() if count > 1)
    return {
        f"{prefix}_repeat_occurrence_rate": repeated_occurrences / total,
        f"{prefix}_simpson_concentration": sum(prob * prob for prob in probs),
        f"{prefix}_gini_frequency": gini(freqs),
        f"{prefix}_yule_k": yule_k(items),
    }


def anti_slop_tokens(text: str) -> list[str]:
    return re.findall(r"[가-힣]{2,12}", text)


def anti_slop_feature_key(kind: str, parts: tuple[str, ...], skip: int = 0) -> str:
    return "\t".join((kind, str(skip), *parts))


def anti_slop_feature_counts(tokens: list[str], *, max_skip: int = 3) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts.update(anti_slop_feature_key("unigram", (token,)) for token in tokens)
    counts.update(anti_slop_feature_key("bigram", (tokens[index], tokens[index + 1])) for index in range(max(0, len(tokens) - 1)))
    for skip in range(1, max_skip + 1):
        step = skip + 1
        counts.update(
            anti_slop_feature_key("skip_bigram", (tokens[index], tokens[index + step]), skip)
            for index in range(max(0, len(tokens) - step))
        )
    for first_gap in (1, 2):
        for second_gap in (1, 2):
            if first_gap == 1 and second_gap == 1:
                continue
            j_offset = first_gap
            k_offset = first_gap + second_gap
            counts.update(
                anti_slop_feature_key(
                    "skip_trigram",
                    (tokens[index], tokens[index + j_offset], tokens[index + k_offset]),
                    max(first_gap, second_gap) - 1,
                )
                for index in range(max(0, len(tokens) - k_offset))
            )
    return counts


class AntiSlopDensityScorer:
    def __init__(self, lexicon_path: Path = DEFAULT_ANTI_SLOP_LEXICON) -> None:
        self.features: list[dict[str, Any]] = []
        if not lexicon_path.exists():
            return
        payload = json.loads(lexicon_path.read_text(encoding="utf-8"))
        for item in payload.get("terms", []):
            parts = item.get("parts")
            if not isinstance(parts, list) or not parts:
                term = str(item.get("term") or "").strip()
                parts = term.split() if term else []
            if not parts:
                continue
            kind = str(item.get("kind") or ("unigram" if len(parts) == 1 else "bigram"))
            skip = int(item.get("skip") or 0)
            self.features.append(
                {
                    "key": anti_slop_feature_key(kind, tuple(str(part) for part in parts), skip),
                    "weight": max(0.05, finite_float(item.get("weight"), 1.0)),
                }
            )

    def density(self, text: str) -> float:
        if not self.features:
            return 0.0
        counts = anti_slop_feature_counts(anti_slop_tokens(text))
        feature_total = max(1, sum(counts.values()))
        weighted_hits = 0.0
        for feature in self.features:
            count = counts.get(str(feature["key"]), 0)
            if count:
                weighted_hits += 1000.0 * count * float(feature["weight"]) / feature_total
        return weighted_hits


class TranslationeseSVMScorer:
    def __init__(self, model_path: Path = DEFAULT_TRANSLATIONESE_MODEL) -> None:
        self.available = False
        self.error = ""
        self.model: Any = None
        self.char_vectorizer: Any = None
        self.pos_vectorizer: Any = None
        self.classes: list[Any] = []
        if not model_path.exists():
            self.error = f"missing_model:{model_path}"
            return
        try:
            import joblib

            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.char_vectorizer = bundle["feature_parts"]["char_vectorizer"]
            self.pos_vectorizer = bundle["feature_parts"]["pos_vectorizer"]
            self.classes = list(getattr(self.model, "classes_", []))
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def pos_doc(self, text: str) -> str:
        tokens: list[str] = []
        for form, tag in morphs(text):
            if form and tag:
                tokens.append(f"{form}/{tag}")
                tokens.append(f"POS_{tag}")
        return " ".join(tokens)

    def score(self, text: str) -> float:
        if not self.available:
            return float("nan")
        try:
            from scipy.sparse import hstack

            x_char = self.char_vectorizer.transform([text])
            x_pos = self.pos_vectorizer.transform([self.pos_doc(text)])
            x = hstack([x_char, x_pos], format="csr")
            raw = self.model.decision_function(x)
            value = float(raw[0] if getattr(raw, "ndim", 1) != 0 else raw)
            if len(self.classes) == 2 and self.classes[0] == "HUMAN":
                value = -value
            return value
        except Exception:
            return float("nan")


def counter_jaccard(left: Counter[str], right: Counter[str]) -> float:
    union = sum(max(left[key], right[key]) for key in set(left) | set(right))
    if union <= 0:
        return 0.0
    overlap = sum(min(left[key], right[key]) for key in set(left) | set(right))
    return overlap / union


def counter_containment(left: Counter[str], right: Counter[str]) -> float:
    denom = min(sum(left.values()), sum(right.values()))
    if denom <= 0:
        return 0.0
    overlap = sum(min(left[key], right[key]) for key in set(left) | set(right))
    return overlap / denom


def lexical_tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[가-힣A-Za-z0-9]+", text) if len(token.strip()) >= 2]


def rewrite_metrics(source_text: str, result_text: str) -> dict[str, float]:
    if not source_text.strip() or not result_text.strip():
        return {}
    source_tokens = lexical_tokens(source_text)
    result_tokens = lexical_tokens(result_text)
    source_counter = Counter(source_tokens)
    result_counter = Counter(result_tokens)
    metrics = {
        "rewrite_length_ratio": len(result_text.strip()) / max(1, len(source_text.strip())),
        "rewrite_surface_jaccard": counter_jaccard(source_counter, result_counter),
        "rewrite_surface_containment": counter_containment(source_counter, result_counter),
        "rewrite_char_sequence_ratio": SequenceMatcher(None, source_text, result_text).ratio(),
    }
    try:
        source_tags = [tag for _form, tag in morphs(source_text) if not tag.startswith(PUNCT_TAG_PREFIXES)]
        result_tags = [tag for _form, tag in morphs(result_text) if not tag.startswith(PUNCT_TAG_PREFIXES)]
        metrics["rewrite_pos_3gram_jaccard"] = counter_jaccard(
            Counter("/".join(item) for item in ngrams(source_tags, 3)),
            Counter("/".join(item) for item in ngrams(result_tags, 3)),
        )
    except Exception:
        metrics["rewrite_pos_3gram_jaccard"] = float("nan")
    metrics["rewrite_edit_amount"] = rewrite_edit_amount(metrics)
    return metrics


def rewrite_edit_amount(metrics: dict[str, Any]) -> float:
    surface_change = 1.0 - finite_float(metrics.get("rewrite_surface_jaccard"), 1.0)
    containment_change = 1.0 - finite_float(metrics.get("rewrite_surface_containment"), 1.0)
    char_change = 1.0 - finite_float(metrics.get("rewrite_char_sequence_ratio"), 1.0)
    pos_change = 1.0 - finite_float(metrics.get("rewrite_pos_3gram_jaccard"), 1.0)
    length_ratio = finite_float(metrics.get("rewrite_length_ratio"), 1.0)
    length_change = min(1.0, abs(math.log(max(length_ratio, 1e-6))) / math.log(2.0)) if math.isfinite(length_ratio) else 0.0
    parts = [
        (clip(surface_change, 0.0, 1.0), 0.25),
        (clip(containment_change, 0.0, 1.0), 0.20),
        (clip(char_change, 0.0, 1.0), 0.25),
        (clip(pos_change, 0.0, 1.0), 0.20),
        (clip(length_change, 0.0, 1.0), 0.10),
    ]
    return sum(value * weight for value, weight in parts) / sum(weight for _value, weight in parts)


class GuiStyleScorer:
    def __init__(
        self,
        *,
        reference_path: Path = DEFAULT_REFERENCE,
        anti_slop_lexicon_path: Path = DEFAULT_ANTI_SLOP_LEXICON,
        translationese_model_path: Path = DEFAULT_TRANSLATIONESE_MODEL,
        disabled_metrics: Iterable[str] | None = None,
        metric_weight_overrides: dict[str, float] | None = None,
    ) -> None:
        self.reference_path = reference_path
        self.reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else {}
        self.metrics_reference = dict(self.reference.get("metrics") or {})
        if isinstance(self.reference.get("anti_slop"), dict):
            self.metrics_reference["anti_slop_density"] = self.reference["anti_slop"]
        if isinstance(self.reference.get("translationese"), dict):
            self.metrics_reference["translationese_raw"] = self.reference["translationese"]
        self.disabled_metrics = {str(name).strip() for name in (disabled_metrics or []) if str(name).strip()}
        self.metric_weight_overrides = {
            str(name).strip(): float(weight)
            for name, weight in (metric_weight_overrides or {}).items()
            if str(name).strip()
        }
        self.pos_usage_reference = dict(self.reference.get("pos_ngram_usage") or {})
        self.anti_slop = AntiSlopDensityScorer(anti_slop_lexicon_path)
        self.translationese = TranslationeseSVMScorer(translationese_model_path)

    def score_metric_bundle(
        self,
        metrics: dict[str, Any],
        pos_distributions: dict[str, dict[str, float]],
        *,
        contract_ok: bool,
        collapse: str,
    ) -> tuple[float, dict[str, float], dict[str, float], dict[str, float]]:
        metric_scores: dict[str, float] = {}
        family_parts: dict[str, list[tuple[float, float]]] = {}
        for name, stats in self.metrics_reference.items():
            if name in self.disabled_metrics:
                continue
            value = finite_float(metrics.get(name))
            if not math.isfinite(value):
                continue
            score = scalar_metric_score(value, stats)
            if not math.isfinite(score):
                continue
            family = metric_family(name)
            base_weight = separation_weight(stats)
            if name in PRIMARY_SCALAR_METRICS:
                base_weight *= 1.15
            base_weight *= METRIC_FAMILY_WEIGHTS.get(family, METRIC_FAMILY_WEIGHTS["other"])
            base_weight *= self.metric_weight_overrides.get(name, 1.0)
            if base_weight <= 0:
                continue
            metric_scores[name] = score
            family_parts.setdefault(family, []).append((score, base_weight))

        pos_usage_scores: dict[str, float] = {}
        for n_text, stats in self.pos_usage_reference.items():
            generated = pos_distributions.get(str(n_text), {})
            human = {str(key): finite_float(value, 0.0) for key, value in (stats.get("human_distribution") or {}).items()}
            ai = {str(key): finite_float(value, 0.0) for key, value in (stats.get("ai_distribution") or {}).items()}
            if not generated or not human:
                continue
            dist_human = js_distance(generated, human)
            score = clip(0.20 - 1.70 * dist_human)
            if ai:
                dist_ai = js_distance(generated, ai)
                anti = clip((dist_ai - dist_human) / max(1e-6, dist_ai + dist_human))
                score = clip(0.85 * score + 0.15 * anti)
            pos_usage_scores[f"pos_{n_text}gram_usage_score"] = score
            base_weight = abs(finite_float(stats.get("weight"), 0.5)) * METRIC_FAMILY_WEIGHTS["pos_usage"]
            family_parts.setdefault("pos_usage", []).append((score, base_weight))

        family_scores: dict[str, float] = {}
        weighted_parts: list[tuple[float, float]] = []
        for family, parts in family_parts.items():
            weight_sum = sum(weight for _score, weight in parts)
            if weight_sum <= 0:
                continue
            score = sum(score * weight for score, weight in parts) / weight_sum
            family_scores[family] = score
            weighted_parts.append((score, METRIC_FAMILY_WEIGHTS.get(family, METRIC_FAMILY_WEIGHTS["other"])))

        if not contract_ok:
            weighted_parts.append((-1.0, 0.70))
        if collapse:
            weighted_parts.append((-0.90, 0.50))
        gui_score = sum(score * weight for score, weight in weighted_parts) / max(1e-9, sum(abs(weight) for _score, weight in weighted_parts)) if weighted_parts else float("nan")
        return gui_score, metric_scores, pos_usage_scores, family_scores

    def score_text(self, raw_text: str, *, task: str = "", source_text: str = "", require_result_tags: bool = True) -> dict[str, Any]:
        contract = analyze_result_contract(raw_text, require_result_tags=require_result_tags)
        result_text = contract.result_text.strip()
        metrics: dict[str, Any] = {}
        metrics.update(contract.summary())
        metrics["collapse_reason"] = collapse_reason(raw_text, require_result_tags=require_result_tags) or ""
        metrics.update(text_basic_metrics(result_text))
        pos_metrics, pos_distributions = text_pos_metrics(result_text)
        metrics.update(pos_metrics)

        anti_density = self.anti_slop.density(result_text)
        metrics["anti_slop_density"] = anti_density
        translationese_raw = self.translationese.score(result_text)
        metrics["translationese_raw"] = translationese_raw

        gui_score, metric_scores, pos_usage_scores, family_scores = self.score_metric_bundle(
            metrics,
            pos_distributions,
            contract_ok=contract.ok,
            collapse=str(metrics["collapse_reason"]),
        )

        if task == "rewrite" or source_text.strip():
            metrics.update(rewrite_metrics(source_text, result_text))
            source_scored = self.score_text(source_text, task="", source_text="", require_result_tags=False) if source_text.strip() else {}
            source_score = finite_float(source_scored.get("score"))
            metrics["rewrite_source_style_score"] = source_score
            metrics["rewrite_style_improvement"] = gui_score - source_score if math.isfinite(source_score) and math.isfinite(gui_score) else float("nan")

        return {
            "score": gui_score,
            "result_text": result_text,
            "metrics": metrics,
            "metric_scores": metric_scores,
            "pos_usage_scores": pos_usage_scores,
            "family_scores": family_scores,
            "scorer_status": {
                "reference": str(self.reference_path),
                "anti_slop_terms": len(self.anti_slop.features),
                "translationese_available": self.translationese.available,
                "translationese_error": self.translationese.error,
            },
        }
