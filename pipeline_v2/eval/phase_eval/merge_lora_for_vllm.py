#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a standalone model directory for vLLM eval.

Native vLLM runtime LoRA is not reliable for Gemma4 in our current environment,
so phase evaluation can use this script to materialize a merged model once and
then run vLLM without `LoRARequest`.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[3]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gemma4_loader import base_model_from_adapter, patch_gemma4_num_kv_shared_layers_for_config_validation
from pipeline_v2.lib.io import write_json
from pipeline_v2.lib.trainer_utils import print_json


DEFAULT_BASE_MODEL = "/workspace/modelscope_cache/unsloth/gemma-4-31B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA into a standalone vLLM-loadable model.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-if-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-processor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_base_model(adapter_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    adapter_base = base_model_from_adapter(adapter_path, DEFAULT_BASE_MODEL)
    if adapter_base.startswith("/workspace/") and Path(adapter_base).exists():
        return adapter_base
    default_path = Path(DEFAULT_BASE_MODEL)
    if default_path.exists():
        return str(default_path)
    return adapter_base


def torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def already_complete(output_dir: Path) -> bool:
    if not (output_dir / "merge_manifest.json").exists():
        return False
    if (output_dir / "model.safetensors.index.json").exists():
        return True
    return any(output_dir.glob("model-*.safetensors")) or (output_dir / "pytorch_model.bin.index.json").exists()


def load_base_model(base_model: str, args: argparse.Namespace) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype(args.dtype),
        "device_map": args.device_map,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
    }
    print_json("[merge-lora/load-base]", {"base_model": base_model, **{k: str(v) for k, v in kwargs.items()}})
    return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def save_processor(base_model: str, output_dir: Path, *, trust_remote_code: bool) -> dict[str, str]:
    saved: dict[str, str] = {}
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        processor.save_pretrained(output_dir)
        saved["processor"] = processor.__class__.__name__
    except Exception as exc:
        saved["processor_error"] = repr(exc)
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
            tokenizer.save_pretrained(output_dir)
            saved["tokenizer"] = tokenizer.__class__.__name__
        except Exception as tokenizer_exc:
            saved["tokenizer_error"] = repr(tokenizer_exc)
    return saved


def copy_adapter_metadata(adapter_path: Path, output_dir: Path) -> None:
    metadata_dir = output_dir / "merged_from_adapter"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "README.md"):
        source = adapter_path / filename
        if source.exists():
            shutil.copy2(source, metadata_dir / filename)


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    if args.skip_if_complete and already_complete(output_dir):
        print_json("[merge-lora/skip-complete]", {"output_dir": str(output_dir)})
        return

    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_gemma4_num_kv_shared_layers_for_config_validation()

    base_model = resolve_base_model(adapter_path, args.base_model)
    model = load_base_model(base_model, args)

    from peft import PeftModel

    print_json("[merge-lora/load-adapter]", {"adapter_path": str(adapter_path)})
    peft_model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name="default", is_trainable=False)
    print_json("[merge-lora/merge]", {"model_class": peft_model.__class__.__name__})
    merged = peft_model.merge_and_unload()
    merged.eval()

    print_json(
        "[merge-lora/save-model]",
        {
            "output_dir": str(output_dir),
            "max_shard_size": args.max_shard_size,
            "safe_serialization": args.safe_serialization,
        },
    )
    merged.save_pretrained(
        output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    processor_info = save_processor(base_model, output_dir, trust_remote_code=args.trust_remote_code) if args.save_processor else {}
    copy_adapter_metadata(adapter_path, output_dir)

    manifest = {
        "time": time.time(),
        "elapsed_sec": round(time.time() - start, 3),
        "adapter_path": str(adapter_path),
        "base_model": base_model,
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_shard_size": args.max_shard_size,
        "safe_serialization": args.safe_serialization,
        "processor_info": processor_info,
    }
    write_json(output_dir / "merge_manifest.json", manifest)
    print_json("[merge-lora/done]", manifest)

    del merged
    del peft_model
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
