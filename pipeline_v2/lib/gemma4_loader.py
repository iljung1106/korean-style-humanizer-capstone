"""Gemma 4 loading utilities for standalone pipeline_v2 trainers."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from .io import TRAINING_ROOT
from .lora import get_language_num_hidden_layers, target_regex_for_last_fraction


DEFAULT_BASE_MODEL = "unsloth/gemma-4-31B-it"
LEGACY_BNB_BASE_MODELS = {
    "unsloth/gemma-4-31B-it-unsloth-bnb-4bit": DEFAULT_BASE_MODEL,
    "unsloth/gemma-4-31b-it-unsloth-bnb-4bit": DEFAULT_BASE_MODEL,
}


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


@contextlib.contextmanager
def clean_project_sys_path_for_unsloth_loader() -> Iterator[None]:
    """Avoid local project modules shadowing packages during Unsloth imports."""

    original = list(sys.path)
    training_root = str(TRAINING_ROOT)
    cwd = os.getcwd()
    try:
        sys.path = [
            item
            for item in sys.path
            if item not in ("", cwd, training_root)
            and not item.endswith("/gemma4_style_rl_training")
            and not item.endswith("\\gemma4_style_rl_training")
        ]
        yield
    finally:
        sys.path = original


def patch_gemma4_num_kv_shared_layers_for_config_validation() -> bool:
    """Patch a Transformers 5.5 / Unsloth Gemma4 config-validation edge case.

    Unsloth hides `num_kv_shared_layers == 0` via `AttributeError` to avoid a
    cache-constructor bug. Some Transformers config validation paths still read
    the attribute. Returning 0 in that validation path lets AutoConfig load.
    """

    patched = False
    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    except Exception:
        return False

    original_getattr = getattr(Gemma4TextConfig, "__getattr__", None)
    if original_getattr is not None and getattr(original_getattr, "_pipeline_v2_patched", False):
        return False

    def patched_getattr(self: Any, name: str) -> Any:
        if name == "num_kv_shared_layers":
            raw_value = self.__dict__.get("num_kv_shared_layers", None)
            if raw_value == 0:
                return 0
        if original_getattr is not None:
            return original_getattr(self, name)
        raise AttributeError(name)

    setattr(patched_getattr, "_pipeline_v2_patched", True)
    Gemma4TextConfig.__getattr__ = patched_getattr  # type: ignore[method-assign]
    patched = True
    return patched


def base_model_from_adapter(adapter_path: str | Path, fallback: str = DEFAULT_BASE_MODEL) -> str:
    adapter_config = Path(adapter_path) / "adapter_config.json"
    if not adapter_config.exists():
        return fallback
    try:
        data = json.loads(adapter_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    value = data.get("base_model_name_or_path") or data.get("base_model_name")
    if not value:
        return fallback
    model_name = str(value)
    return LEGACY_BNB_BASE_MODELS.get(model_name, model_name)


def processor_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def maybe_apply_chat_template(processor: Any, template: str | None) -> Any:
    if not template:
        return processor
    try:
        from unsloth.chat_templates import get_chat_template
    except Exception:
        return processor
    tokenizer = processor_tokenizer(processor)
    patched = get_chat_template(tokenizer, chat_template=template)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = patched
        return processor
    return patched


def load_gemma4_model_and_processor(
    *,
    model_name: str,
    adapter_path: str | Path | None = None,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    load_in_16bit: bool = False,
    chat_template: str = "gemma-4",
    gradient_checkpointing: str | bool = "unsloth",
    create_lora: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_last_layer_fraction: float = 0.6,
    random_state: int = 3407,
) -> tuple[Any, Any, dict[str, Any]]:
    patch_gemma4_num_kv_shared_layers_for_config_validation()
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    with clean_project_sys_path_for_unsloth_loader():
        import unsloth  # noqa: F401
        from unsloth import FastModel

    loader_kwargs = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "load_in_16bit": load_in_16bit,
        "use_gradient_checkpointing": gradient_checkpointing,
    }
    loader_kwargs = filter_kwargs(FastModel.from_pretrained, loader_kwargs)
    model, processor = FastModel.from_pretrained(**loader_kwargs)
    processor = maybe_apply_chat_template(processor, chat_template)

    metadata: dict[str, Any] = {
        "loader": "FastModel",
        "loader_kwargs": {
            key: (str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value)
            for key, value in loader_kwargs.items()
        },
        "model_class": model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
    }

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name="default", is_trainable=True)
        metadata["adapter_path"] = str(adapter_path)
        metadata["model_class_after_adapter"] = model.__class__.__name__
    elif create_lora:
        target_modules = target_regex_for_last_fraction(
            get_language_num_hidden_layers(model),
            lora_last_layer_fraction,
        )
        lora_kwargs = {
            "r": lora_r,
            "target_modules": target_modules,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": "none",
            "use_gradient_checkpointing": gradient_checkpointing,
            "random_state": random_state,
        }
        lora_kwargs = filter_kwargs(FastModel.get_peft_model, lora_kwargs)
        model = FastModel.get_peft_model(model, **lora_kwargs)
        metadata["created_lora"] = {
            key: value for key, value in lora_kwargs.items() if key != "model"
        }
        metadata["model_class_after_lora"] = model.__class__.__name__

    return model, processor, metadata
