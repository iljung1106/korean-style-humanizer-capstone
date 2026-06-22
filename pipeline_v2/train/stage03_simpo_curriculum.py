#!/usr/bin/env python3
"""Stage 3 SimPO curriculum trainer for pipeline_v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parents[1]
TRAINING_ROOT = PIPELINE_ROOT.parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gemma4_loader import (
    DEFAULT_BASE_MODEL,
    base_model_from_adapter,
    filter_kwargs,
    load_gemma4_model_and_processor,
    processor_tokenizer,
)
from pipeline_v2.lib.io import manifest_base, read_jsonl, write_json
from pipeline_v2.lib.lora import trainable_parameter_summary
from pipeline_v2.lib.preference_data import (
    TextFirstProcessor,
    add_generate_style_guidance_instruction,
    as_assistant_messages,
    as_chat_messages,
    assistant_text,
    dataset_from_rows,
    render_prompt,
    trim_messages,
)
from pipeline_v2.lib.trainer_utils import parse_report_to, print_json, set_seed


DEFAULT_DATASET = TRAINING_ROOT / "data" / "pipeline_v2" / "simpo_curriculum" / "curated_mixed.jsonl"
DEFAULT_OUTPUT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage03_simpo_curriculum"


def import_cpo_classes() -> tuple[Any, Any]:
    try:
        from trl.experimental.cpo import CPOConfig, CPOTrainer

        return CPOConfig, CPOTrainer
    except ImportError:
        from trl import CPOConfig, CPOTrainer

        return CPOConfig, CPOTrainer


def import_dpo_classes(*, patch_unsloth: bool = True) -> tuple[Any, Any]:
    try:
        if not patch_unsloth:
            raise AttributeError("disabled by --no-unsloth-dpo-patch")
        from unsloth import PatchDPOTrainer

        PatchDPOTrainer()
        print("[unsloth] PatchDPOTrainer applied", flush=True)
    except (ImportError, AttributeError) as exc:
        print(f"[unsloth] PatchDPOTrainer unavailable: {type(exc).__name__}: {exc}", flush=True)
    from trl import DPOConfig, DPOTrainer

    return DPOConfig, DPOTrainer


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model != "auto":
        return args.model
    if args.adapter_path:
        return base_model_from_adapter(args.adapter_path, DEFAULT_BASE_MODEL)
    return DEFAULT_BASE_MODEL


def parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_loss_type(raw: str) -> str | list[str]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else raw
    return parts


def patch_text_only_mm_token_type_ids(model: Any) -> None:
    """Gemma4 text-only DPO batches may omit the VLM token type tensor."""

    patched = 0
    seen: set[int] = set()
    candidates: list[Any] = [model]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        inner_model = getattr(base_model, "model", None)
        if inner_model is not None:
            candidates.append(inner_model)
            inner_text_model = getattr(inner_model, "model", None)
            if inner_text_model is not None:
                candidates.append(inner_text_model)

    for candidate in candidates:
        if candidate is None or id(candidate) in seen or not hasattr(candidate, "forward"):
            continue
        seen.add(id(candidate))
        original_forward = candidate.forward

        def forward_with_mm_token_type_ids(*forward_args: Any, _original_forward=original_forward, **forward_kwargs: Any) -> Any:
            if forward_kwargs.get("mm_token_type_ids") is None:
                input_ids = forward_kwargs.get("input_ids")
                if input_ids is None and forward_args:
                    input_ids = forward_args[0]
                if input_ids is not None and hasattr(input_ids, "new_zeros"):
                    forward_kwargs["mm_token_type_ids"] = input_ids.new_zeros(input_ids.shape)
            return _original_forward(*forward_args, **forward_kwargs)

        candidate.forward = forward_with_mm_token_type_ids  # type: ignore[method-assign]
        patched += 1
    print_json("[compat]", {"patch": "text_only_mm_token_type_ids", "patched_forwards": patched})


def add_reference_adapter_for_dpo(model: Any, adapter_path: str | Path, *, policy_name: str = "default", ref_name: str = "ref") -> None:
    """Load a frozen copy of the starting LoRA so DPO compares policy vs start policy.

    PEFT DPO without explicit adapter names can compare the trainable adapter
    against the base/no-adapter model. For this curriculum stage we want the
    Stage 2 policy to be the reference, not the raw base model.
    """

    if not hasattr(model, "load_adapter"):
        print_json("[dpo/ref_adapter]", {"enabled": False, "reason": "model_has_no_load_adapter"})
        return
    existing = list(getattr(model, "peft_config", {}) or {})
    if ref_name not in existing:
        model.load_adapter(str(adapter_path), adapter_name=ref_name, is_trainable=False)
    if hasattr(model, "set_adapter"):
        model.set_adapter(policy_name)
    print_json(
        "[dpo/ref_adapter]",
        {
            "enabled": True,
            "policy_adapter": policy_name,
            "ref_adapter": ref_name,
            "peft_config": list(getattr(model, "peft_config", {}) or {}),
            "active_adapter": getattr(model, "active_adapter", None),
        },
    )


def load_simpo_rows(args: argparse.Namespace, processor: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted_buckets = {item for item in args.bucket if item}
    for row in read_jsonl(args.dataset):
        bucket = str(row.get("bucket") or "")
        if wanted_buckets and bucket not in wanted_buckets:
            continue
        prompt_messages = trim_messages(as_chat_messages(row.get("prompt")), args.max_prompt_chars)
        prompt_messages = add_generate_style_guidance_instruction(prompt_messages)
        chosen_messages = as_assistant_messages(row.get("chosen"))
        rejected_messages = as_assistant_messages(row.get("rejected"))
        rows.append(
            {
                "prompt": render_prompt(processor, prompt_messages),
                "chosen": assistant_text(chosen_messages),
                "rejected": assistant_text(rejected_messages),
                "id": str(row.get("id") or len(rows)),
                "bucket": bucket,
            }
        )
        if args.limit_rows > 0 and len(rows) >= args.limit_rows:
            break
    if not rows:
        raise ValueError(f"No SimPO rows selected from {args.dataset}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pipeline_v2 Stage 3 SimPO curriculum.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--init-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-last-layer-fraction", type=float, default=1.0)
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--bucket", action="append", default=[])
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--preference-loss", choices=["simpo", "cpo", "dpo"], default="simpo")
    parser.add_argument("--dpo-loss-type", default="sigmoid")
    parser.add_argument("--rpo-alpha", type=float, default=0.0)
    parser.add_argument("--unsloth-dpo-patch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simpo-gamma", type=float, default=0.01)
    parser.add_argument("--cpo-alpha", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="pipeline_v2_stage03_simpo")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    adapter_path = args.adapter_path or None
    if adapter_path and args.init_lora:
        raise ValueError("--adapter-path and --init-lora are mutually exclusive")
    if not adapter_path and not args.init_lora:
        raise ValueError("Provide --adapter-path to continue an adapter, or --init-lora to create a new LoRA")

    model_name = resolve_model_name(args)
    print_json("[model]", {"model_name": model_name, "adapter_path": adapter_path, "init_lora": args.init_lora})
    model, processor, loader_metadata = load_gemma4_model_and_processor(
        model_name=model_name,
        adapter_path=adapter_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        chat_template=args.chat_template,
        create_lora=args.init_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_last_layer_fraction=args.lora_last_layer_fraction,
        random_state=args.seed,
    )
    print_json("[loader]", loader_metadata)
    print_json("[trainable]", trainable_parameter_summary(model))

    rows = load_simpo_rows(args, processor)
    bucket_counts: dict[str, int] = {}
    for row in rows:
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
    print_json("[data]", {"rows": len(rows), "bucket_counts": bucket_counts, "path": args.dataset})
    dataset = dataset_from_rows(rows).shuffle(seed=args.seed)

    try:
        from unsloth import is_bfloat16_supported

        use_bf16 = bool(is_bfloat16_supported())
    except Exception:
        use_bf16 = True

    config_kwargs = {
        "output_dir": str(output),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
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
        "dataset_num_proc": args.dataset_num_proc,
        "max_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
        "beta": args.beta,
        "loss_type": "simpo" if args.preference_loss == "simpo" else parse_loss_type(args.dpo_loss_type),
    }
    if args.preference_loss in {"simpo", "cpo"}:
        processing_class = TextFirstProcessor(processor)
        CPOConfig, CPOTrainer = import_cpo_classes()
        config_kwargs.update(
            {
                "cpo_alpha": args.cpo_alpha,
                "simpo_gamma": args.simpo_gamma,
                "alpha": args.alpha,
            }
        )
        training_args = CPOConfig(**filter_kwargs(CPOConfig, config_kwargs))
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": dataset,
            "processing_class": processing_class,
            "tokenizer": processing_class,
        }
        trainer = CPOTrainer(**filter_kwargs(CPOTrainer, trainer_kwargs))
    else:
        processing_class = processor
        DPOConfig, DPOTrainer = import_dpo_classes(patch_unsloth=args.unsloth_dpo_patch)
        patch_text_only_mm_token_type_ids(model)
        if adapter_path:
            add_reference_adapter_for_dpo(model, adapter_path)
            config_kwargs.update(
                {
                    "model_adapter_name": "default",
                    "ref_adapter_name": "ref",
                    "force_use_ref_model": False,
                }
            )
        else:
            print_json(
                "[dpo/ref_adapter]",
                {
                    "enabled": False,
                    "reason": "init_lora_uses_disabled_adapter_reference",
                    "ref_model": None,
                },
            )
        dpo_config_kwargs = dict(config_kwargs)
        dpo_config_kwargs.pop("max_length", None)
        dpo_config_kwargs.pop("max_prompt_length", None)
        if args.rpo_alpha > 0:
            base_loss = parse_loss_type(args.dpo_loss_type)
            if isinstance(base_loss, str):
                dpo_config_kwargs["loss_type"] = [base_loss, "sft"]
                dpo_config_kwargs["loss_weights"] = [1.0, args.rpo_alpha]
            else:
                dpo_config_kwargs["loss_type"] = base_loss
                dpo_config_kwargs["loss_weights"] = [1.0] * len(base_loss)
            print_json(
                "[dpo/nll_mix]",
                {
                    "mode": "loss_type_sft",
                    "loss_type": dpo_config_kwargs["loss_type"],
                    "loss_weights": dpo_config_kwargs.get("loss_weights"),
                    "requested_rpo_alpha": args.rpo_alpha,
                },
            )
        training_args = DPOConfig(**filter_kwargs(DPOConfig, dpo_config_kwargs))
        trainer_kwargs = {
            "model": model,
            "ref_model": None,
            "args": training_args,
            "train_dataset": dataset,
            "processing_class": processing_class,
            "tokenizer": processing_class,
            "beta": args.beta,
            "max_length": args.max_seq_length,
            "max_prompt_length": args.max_prompt_length,
        }
        trainer = DPOTrainer(**filter_kwargs(DPOTrainer, trainer_kwargs))
    if args.probe_only:
        batch = next(iter(trainer.get_train_dataloader()))
        print_json("[probe/batch_keys]", {"keys": sorted(batch.keys())})
        return

    print(f"[train] starting Stage 3 {args.preference_loss.upper()}", flush=True)
    result = trainer.train()
    trainer.save_state()
    final_dir = output / "policy"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    metrics = dict(result.metrics)
    trainer.save_metrics("train", metrics)
    write_json(
        output / "stage03_manifest.json",
        {
            **manifest_base(stage=f"stage03_{args.preference_loss}_curriculum", args=vars(args)),
            "dataset_rows": len(rows),
            "bucket_counts": bucket_counts,
            "loader": loader_metadata,
            "final_adapter": str(final_dir),
            "train_metrics": metrics,
        },
    )
    print_json("[done]", {"final_adapter": str(final_dir), "train_metrics": metrics})


if __name__ == "__main__":
    main()
