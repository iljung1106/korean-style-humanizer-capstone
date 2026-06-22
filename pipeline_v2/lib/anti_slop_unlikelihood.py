from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


TRAINING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANTI_SLOP_LEXICON = TRAINING_ROOT / "data" / "processed" / "anti_slop_lexicon.json"


@dataclass(frozen=True)
class AntiSlopLexicalTerm:
    kind: str
    term: str
    parts: tuple[str, ...]
    weight: float
    lift: float
    human_count: int
    ai_count: int


@dataclass(frozen=True)
class AntiSlopContinuation:
    term: AntiSlopLexicalTerm
    token_ids: tuple[int, ...]
    continuation_weight: float
    start_weight: float


@dataclass
class AntiSlopULConfig:
    continuations: list[AntiSlopContinuation]
    start_token_weights: dict[int, float]
    pad_token_id: int | None = None


def _parts_from_item(item: dict[str, Any]) -> tuple[str, ...]:
    parts = tuple(str(part).strip() for part in item.get("parts", []) if str(part).strip())
    if parts:
        return parts
    term = str(item.get("term", "")).strip()
    return tuple(part for part in term.split() if part)


def load_anti_slop_lexical_terms(
    path: str | Path = DEFAULT_ANTI_SLOP_LEXICON,
    *,
    unigram_top_k: int = 300,
    unigram_min_lift: float = 7.5,
    bigram_top_k: int = 300,
    bigram_min_lift: float = 4.0,
    bigram_min_weight: float = 0.05,
    trigram_top_k: int = 0,
    trigram_min_lift: float = 0.0,
    trigram_min_weight: float = 0.0,
    include_skip: bool = False,
) -> list[AntiSlopLexicalTerm]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    terms_by_kind: dict[str, list[AntiSlopLexicalTerm]] = {"unigram": [], "bigram": [], "trigram": []}
    for item in payload.get("terms", []):
        kind = str(item.get("kind", "")).strip()
        if not include_skip and kind.startswith("skip_"):
            continue
        parts = _parts_from_item(item)
        if not parts:
            continue
        if kind not in {"unigram", "bigram", "trigram"}:
            continue
        term = str(item.get("term", "")).strip() or " ".join(parts)
        lexical = AntiSlopLexicalTerm(
            kind=kind,
            term=term,
            parts=parts,
            weight=max(0.05, float(item.get("weight", 1.0))),
            lift=float(item.get("lift", 0.0)),
            human_count=int(item.get("human_count", 0)),
            ai_count=int(item.get("ai_count", 0)),
        )
        if kind == "unigram":
            if lexical.lift >= unigram_min_lift:
                terms_by_kind["unigram"].append(lexical)
        elif kind == "bigram":
            if lexical.lift >= bigram_min_lift and lexical.weight >= bigram_min_weight:
                terms_by_kind["bigram"].append(lexical)
        elif kind == "trigram":
            if lexical.lift >= trigram_min_lift and lexical.weight >= trigram_min_weight:
                terms_by_kind["trigram"].append(lexical)

    selected: list[AntiSlopLexicalTerm] = []
    for kind, top_k in (("bigram", bigram_top_k), ("trigram", trigram_top_k), ("unigram", unigram_top_k)):
        kind_terms = terms_by_kind[kind]
        kind_terms.sort(key=lambda item: (item.weight, item.lift, item.ai_count), reverse=True)
        if top_k > 0:
            kind_terms = kind_terms[:top_k]
        selected.extend(kind_terms)
    return selected


def _resolve_encode_tokenizer(tokenizer_or_processor: Any) -> Any:
    if hasattr(tokenizer_or_processor, "encode"):
        return tokenizer_or_processor
    for attr in ("tokenizer", "text_tokenizer"):
        candidate = getattr(tokenizer_or_processor, attr, None)
        if candidate is not None and hasattr(candidate, "encode"):
            return candidate
    raise AttributeError(
        f"{tokenizer_or_processor.__class__.__name__} does not expose encode() or an inner tokenizer with encode()."
    )


def _tokenize_variants(tokenizer: Any, text: str) -> list[tuple[int, ...]]:
    encode_tokenizer = _resolve_encode_tokenizer(tokenizer)
    variants: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for variant in (text, " " + text):
        ids = tuple(int(token_id) for token_id in encode_tokenizer.encode(variant, add_special_tokens=False))
        if len(ids) == 0 or ids in seen:
            continue
        seen.add(ids)
        variants.append(ids)
    return variants


def build_anti_slop_ul_config(
    tokenizer: Any,
    *,
    lexicon_path: str | Path = DEFAULT_ANTI_SLOP_LEXICON,
    unigram_top_k: int = 300,
    unigram_min_lift: float = 7.5,
    bigram_top_k: int = 300,
    bigram_min_lift: float = 4.0,
    bigram_min_weight: float = 0.05,
    trigram_top_k: int = 0,
    trigram_min_lift: float = 0.0,
    trigram_min_weight: float = 0.0,
    start_weight_multiplier: float = 0.08,
    continuation_weight_multiplier: float = 1.0,
    max_terms: int = 0,
) -> AntiSlopULConfig:
    encode_tokenizer = _resolve_encode_tokenizer(tokenizer)
    lexical_terms = load_anti_slop_lexical_terms(
        lexicon_path,
        unigram_top_k=unigram_top_k,
        unigram_min_lift=unigram_min_lift,
        bigram_top_k=bigram_top_k,
        bigram_min_lift=bigram_min_lift,
        bigram_min_weight=bigram_min_weight,
        trigram_top_k=trigram_top_k,
        trigram_min_lift=trigram_min_lift,
        trigram_min_weight=trigram_min_weight,
        include_skip=False,
    )
    if max_terms > 0:
        lexical_terms = lexical_terms[:max_terms]

    continuations: list[AntiSlopContinuation] = []
    start_token_weights: dict[int, float] = {}
    for term in lexical_terms:
        base_weight = max(0.01, min(3.0, term.weight))
        for token_ids in _tokenize_variants(encode_tokenizer, term.term):
            if len(token_ids) >= 2:
                continuations.append(
                    AntiSlopContinuation(
                        term=term,
                        token_ids=token_ids,
                        continuation_weight=base_weight * continuation_weight_multiplier,
                        start_weight=base_weight * start_weight_multiplier,
                    )
                )
        if len(term.parts) < 2:
            continue
        start_variants = _tokenize_variants(encode_tokenizer, term.parts[0])
        if start_variants:
            start_id = start_variants[0][0]
            if start_id != getattr(encode_tokenizer, "eos_token_id", None) and start_id != getattr(
                encode_tokenizer, "pad_token_id", None
            ):
                start_token_weights[start_id] = max(
                    start_token_weights.get(start_id, 0.0),
                    base_weight * start_weight_multiplier,
                )

    return AntiSlopULConfig(
        continuations=continuations,
        start_token_weights=start_token_weights,
        pad_token_id=getattr(encode_tokenizer, "pad_token_id", None),
    )


def summarize_anti_slop_ul_terms(
    *,
    lexicon_path: str | Path = DEFAULT_ANTI_SLOP_LEXICON,
    unigram_top_k: int = 300,
    unigram_min_lift: float = 7.5,
    bigram_top_k: int = 300,
    bigram_min_lift: float = 4.0,
    bigram_min_weight: float = 0.05,
    trigram_top_k: int = 0,
    trigram_min_lift: float = 0.0,
    trigram_min_weight: float = 0.0,
) -> dict[str, Any]:
    terms = load_anti_slop_lexical_terms(
        lexicon_path,
        unigram_top_k=unigram_top_k,
        unigram_min_lift=unigram_min_lift,
        bigram_top_k=bigram_top_k,
        bigram_min_lift=bigram_min_lift,
        bigram_min_weight=bigram_min_weight,
        trigram_top_k=trigram_top_k,
        trigram_min_lift=trigram_min_lift,
        trigram_min_weight=trigram_min_weight,
        include_skip=False,
    )
    by_kind: dict[str, int] = {}
    for term in terms:
        by_kind[term.kind] = by_kind.get(term.kind, 0) + 1
    examples = {
        kind: [
            {
                "term": term.term,
                "parts": list(term.parts),
                "weight": term.weight,
                "lift": term.lift,
                "human_count": term.human_count,
                "ai_count": term.ai_count,
            }
            for term in terms
            if term.kind == kind
        ][:20]
        for kind in ("bigram", "trigram", "unigram")
    }
    return {
        "lexicon_path": str(lexicon_path),
        "total_terms": len(terms),
        "by_kind": by_kind,
        "selection": {
            "unigram": {"top_k": unigram_top_k, "min_lift": unigram_min_lift},
            "bigram": {"top_k": bigram_top_k, "min_lift": bigram_min_lift, "min_weight": bigram_min_weight},
            "trigram": {"top_k": trigram_top_k, "min_lift": trigram_min_lift, "min_weight": trigram_min_weight},
        },
        "skip_terms_included": False,
        "examples": examples,
    }


def anti_slop_unlikelihood_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    config: AntiSlopULConfig,
) -> torch.Tensor:
    if not config.continuations and not config.start_token_weights:
        return logits.new_zeros(())
    if logits.ndim != 3 or input_ids.ndim != 2 or labels.ndim != 2:
        return logits.new_zeros(())
    if logits.shape[1] < 2:
        return logits.new_zeros(())

    # logits[:, t - 1] predicts token labels[:, t].
    pred_logits = logits[:, :-1, :]
    target_labels = labels[:, 1:]
    valid_targets = target_labels.ne(-100)
    if not bool(valid_targets.any()):
        return logits.new_zeros(())
    log_den = torch.logsumexp(pred_logits, dim=-1)
    losses: list[torch.Tensor] = []

    if config.start_token_weights:
        token_ids = torch.tensor(list(config.start_token_weights), device=logits.device, dtype=torch.long)
        weights = torch.tensor(
            [config.start_token_weights[int(token_id)] for token_id in token_ids.tolist()],
            device=logits.device,
            dtype=torch.float32,
        )
        selected = pred_logits.index_select(dim=-1, index=token_ids)
        probs = torch.exp(selected - log_den.unsqueeze(-1))
        weighted_mass = (probs * weights.view(1, 1, -1)).sum(dim=-1) / max(float(weights.sum().item()), 1e-6)
        start_loss = -torch.log1p(-weighted_mass.clamp(max=1.0 - 1e-6))
        losses.append(start_loss[valid_targets].mean())

    seq_len = input_ids.shape[1]
    for continuation in config.continuations:
        token_ids = continuation.token_ids
        if len(token_ids) < 2:
            continue
        prefix = token_ids[:-1]
        target = token_ids[-1]
        prefix_len = len(prefix)
        if prefix_len >= seq_len:
            continue
        prefix_tensor = torch.tensor(prefix, device=input_ids.device, dtype=input_ids.dtype)
        windows = input_ids.unfold(dimension=1, size=prefix_len, step=1)
        windows = windows[:, : seq_len - prefix_len, :]
        matches = windows.eq(prefix_tensor.view(1, 1, -1)).all(dim=-1)
        valid = labels[:, prefix_len:].ne(-100)
        mask = matches & valid
        if not bool(mask.any()):
            continue
        log_p = pred_logits[:, prefix_len - 1 : seq_len - 1, int(target)] - log_den[:, prefix_len - 1 : seq_len - 1]
        prob = torch.exp(log_p).clamp(max=1.0 - 1e-6)
        ul = -torch.log1p(-prob)
        losses.append(ul[mask].mean() * float(continuation.continuation_weight))

    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def install_anti_slop_unlikelihood(trainer: Any, config: AntiSlopULConfig, *, weight: float) -> None:
    if weight <= 0.0:
        return
    original_compute_loss = trainer.compute_loss

    def compute_loss_with_anti_slop_ul(model, inputs, return_outputs=False, num_items_in_batch=None):
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        try:
            result = original_compute_loss(model, inputs, return_outputs=True, **kwargs)
        except TypeError:
            result = original_compute_loss(model, inputs, return_outputs=True)
        if not isinstance(result, tuple) or len(result) != 2:
            return result
        loss, outputs = result
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        logits = getattr(outputs, "logits", None)
        if labels is not None and input_ids is not None and logits is not None:
            ul_loss = anti_slop_unlikelihood_loss(logits, input_ids, labels, config)
            if torch.isfinite(ul_loss):
                loss = loss + float(weight) * ul_loss.to(loss.device)
        if return_outputs:
            return loss, outputs
        return loss

    trainer.compute_loss = compute_loss_with_anti_slop_ul
