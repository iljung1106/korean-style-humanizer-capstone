"""LoRA target helpers for standalone pipeline_v2 trainers."""

from __future__ import annotations

import re
from typing import Any


LANGUAGE_LORA_TARGET_REGEX = (
    r".*language_model\.layers\.(?:{layers})\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)


def get_language_num_hidden_layers(model: Any) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        config,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "num_hidden_layers", None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not infer language model layer count from model config.")


def target_regex_for_last_fraction(num_layers: int, fraction: float) -> str:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1].")
    start = max(0, int(num_layers * (1.0 - fraction)))
    layers = "|".join(str(index) for index in range(start, num_layers))
    return LANGUAGE_LORA_TARGET_REGEX.format(layers=layers)


def trainable_parameter_summary(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    lora_total = 0
    lora_trainable = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        if "lora_" in name:
            lora_total += count
            if parameter.requires_grad:
                lora_trainable += count
    return {
        "total": total,
        "trainable": trainable,
        "lora_total": lora_total,
        "lora_trainable": lora_trainable,
    }


def freeze_non_lora_parameters(model: Any) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = bool(re.search(r"\.lora_[AB]\.", name) or "lora_embedding_" in name)

