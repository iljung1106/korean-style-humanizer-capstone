#!/usr/bin/env python3
"""Stage 1 CPT-lite trainer for pipeline_v2.

This trainer is standalone. It deliberately uses the custom Stage 1 row masks
from `pipeline_v2.lib.masking` instead of generic chat-response masking.
"""

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
    load_gemma4_model_and_processor,
)
from pipeline_v2.lib.anti_slop_unlikelihood import (
    DEFAULT_ANTI_SLOP_LEXICON,
    build_anti_slop_ul_config,
    install_anti_slop_unlikelihood,
    summarize_anti_slop_ul_terms,
)
from pipeline_v2.lib.io import manifest_base, read_jsonl, row_type_counts, write_json
from pipeline_v2.lib.lora import trainable_parameter_summary
from pipeline_v2.lib.masking import (
    MockGemma4Processor,
    PipelineV2DataCollator,
    PipelineV2TokenizedDataset,
    decode_visible_labels,
    prepare_row,
)
from pipeline_v2.lib.preference_data import add_generate_style_guidance_instruction
from pipeline_v2.lib.sampling import BalancedRowTypeSampler, parse_row_type_weights
from pipeline_v2.lib.trainer_utils import parse_report_to, print_json, set_seed


DEFAULT_DATASET = TRAINING_ROOT / "data" / "pipeline_v2" / "cpt_mixed_probe.jsonl"
DEFAULT_OUTPUT = TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage01_cpt_lite"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.dataset)
    include_types = {str(item) for item in args.include_row_type if str(item)}
    exclude_types = {str(item) for item in args.exclude_row_type if str(item)}
    if include_types:
        rows = [row for row in rows if str(row.get("row_type") or "") in include_types]
    if exclude_types:
        rows = [row for row in rows if str(row.get("row_type") or "") not in exclude_types]
    rows = [
        {
            **row,
            "messages": add_generate_style_guidance_instruction(row["messages"]),
        }
        if str(row.get("row_type") or "") == "continuation_sft" and isinstance(row.get("messages"), list)
        else row
        for row in rows
    ]
    if args.limit_rows > 0:
        rows = rows[: args.limit_rows]
    if not rows:
        raise ValueError(f"No rows loaded from {args.dataset}")
    return rows


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model != "auto":
        return args.model
    if args.adapter_path:
        return base_model_from_adapter(args.adapter_path, DEFAULT_BASE_MODEL)
    return DEFAULT_BASE_MODEL


def run_mask_probe(rows: list[dict[str, Any]], processor: Any, args: argparse.Namespace) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        row_type = str(row.get("row_type") or "")
        if row_type in seen:
            continue
        example = prepare_row(row, processor, args.max_seq_length)
        visible_text = decode_visible_labels(processor, example)
        samples.append(
            {
                **example.summary(),
                "visible_preview": visible_text[: args.mask_probe_chars],
                "visible_tail": visible_text[-args.mask_probe_chars :],
            }
        )
        seen.add(row_type)
        if seen >= {"raw_lm", "continuation_sft", "format_sft"}:
            break
    ok = bool(samples) and all(item["visible_labels"] > 0 for item in samples)
    return {"ok": ok, "samples": samples}


def build_train_sampler(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[Any | None, dict[str, Any]]:
    if args.row_type_sampling == "shuffle":
        return None, {"type": "trainer_default_shuffle"}
    weights = parse_row_type_weights(args.row_type_balance)
    dataset_types = set(row_type_counts(rows))
    missing_weights = sorted(dataset_types - set(weights))
    if missing_weights:
        raise ValueError(
            "Balanced sampling would silently drop row types not present in "
            f"--row-type-balance: {missing_weights}"
        )
    sampler = BalancedRowTypeSampler(
        rows,
        weights,
        seed=args.seed,
        num_samples=args.sampler_epoch_size or len(rows),
    )
    return sampler, sampler.metadata(preview_samples=max(1, args.batch_size * args.grad_accum * 4))


def save_manifest(args: argparse.Namespace, rows: list[dict[str, Any]], output: Path, extra: dict[str, Any]) -> None:
    manifest = manifest_base(stage=args.manifest_stage, args=vars(args))
    manifest.update(
        {
            "dataset": str(args.dataset),
            "dataset_rows": len(rows),
            "row_type_counts": row_type_counts(rows),
            "output": str(output),
            **extra,
        }
    )
    write_json(output / "stage01_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pipeline_v2 Stage 1 CPT-lite adapter.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="auto", help="Base model name/path, or auto.")
    parser.add_argument("--adapter-path", default="", help="Existing adapter to continue from.")
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", default="unsloth")
    parser.add_argument("--lora-r", type=positive_int, default=32)
    parser.add_argument("--lora-alpha", type=positive_int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-last-layer-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--grad-accum", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=positive_int, default=1)
    parser.add_argument("--save-steps", type=positive_int, default=10)
    parser.add_argument("--save-total-limit", type=positive_int, default=2)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anti-slop-ul-weight", type=float, default=0.0)
    parser.add_argument("--anti-slop-ul-lexicon", default=str(DEFAULT_ANTI_SLOP_LEXICON))
    parser.add_argument("--anti-slop-ul-unigram-top-k", type=int, default=300)
    parser.add_argument("--anti-slop-ul-unigram-min-lift", type=float, default=7.5)
    parser.add_argument("--anti-slop-ul-bigram-top-k", type=int, default=300)
    parser.add_argument("--anti-slop-ul-bigram-min-lift", type=float, default=4.0)
    parser.add_argument("--anti-slop-ul-bigram-min-weight", type=float, default=0.05)
    parser.add_argument("--anti-slop-ul-trigram-top-k", type=int, default=0)
    parser.add_argument("--anti-slop-ul-trigram-min-lift", type=float, default=0.0)
    parser.add_argument("--anti-slop-ul-trigram-min-weight", type=float, default=0.0)
    parser.add_argument("--anti-slop-ul-start-weight-multiplier", type=float, default=0.08)
    parser.add_argument("--anti-slop-ul-preview", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="pipeline_v2_stage01_cpt_lite")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--include-row-type", action="append", default=[])
    parser.add_argument("--exclude-row-type", action="append", default=[])
    parser.add_argument("--manifest-stage", default="stage01_cpt_lite")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument(
        "--row-type-sampling",
        choices=["balanced", "shuffle"],
        default="balanced",
        help="Use balanced row-type replacement sampling or Trainer's default shuffle.",
    )
    parser.add_argument(
        "--row-type-balance",
        default="raw_lm=3,continuation_sft=3,format_sft=2,general_guard=1",
        help="Comma-separated row_type=count quotas. Missing row types are ignored.",
    )
    parser.add_argument(
        "--sampler-epoch-size",
        type=int,
        default=0,
        help="If >0, replacement samples per epoch for balanced sampling.",
    )
    parser.add_argument("--mask-probe-only", action="store_true")
    parser.add_argument("--mock-tokenizer", action="store_true", help="Only valid with --mask-probe-only.")
    parser.add_argument("--mask-probe-chars", type=positive_int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.anti_slop_ul_preview:
        print_json(
            "[anti-slop-ul/preview]",
            summarize_anti_slop_ul_terms(
                lexicon_path=args.anti_slop_ul_lexicon,
                unigram_top_k=args.anti_slop_ul_unigram_top_k,
                unigram_min_lift=args.anti_slop_ul_unigram_min_lift,
                bigram_top_k=args.anti_slop_ul_bigram_top_k,
                bigram_min_lift=args.anti_slop_ul_bigram_min_lift,
                bigram_min_weight=args.anti_slop_ul_bigram_min_weight,
                trigram_top_k=args.anti_slop_ul_trigram_top_k,
                trigram_min_lift=args.anti_slop_ul_trigram_min_lift,
                trigram_min_weight=args.anti_slop_ul_trigram_min_weight,
            ),
        )
        return
    if args.sampler_epoch_size < 0:
        raise ValueError("--sampler-epoch-size must be >= 0.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.anti_slop_ul_weight > 0.0:
        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    rows = load_rows(args)
    print_json("[data]", {"rows": len(rows), "row_types": row_type_counts(rows), "path": str(args.dataset)})
    train_sampler, sampler_metadata = build_train_sampler(args, rows)
    print_json("[sampler]", sampler_metadata)

    if args.mask_probe_only and args.mock_tokenizer:
        processor = MockGemma4Processor()
        probe = run_mask_probe(rows, processor, args)
        print_json("[mask_probe]", probe)
        save_manifest(args, rows, output, {"mask_probe": probe, "mock_tokenizer": True, "sampler": sampler_metadata})
        raise SystemExit(0 if probe["ok"] else 1)

    model_name = resolve_model_name(args)
    print_json("[model]", {"model_name": model_name, "adapter_path": args.adapter_path or None})
    model, processor, loader_metadata = load_gemma4_model_and_processor(
        model_name=model_name,
        adapter_path=args.adapter_path or None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        chat_template=args.chat_template,
        gradient_checkpointing=args.gradient_checkpointing,
        create_lora=not bool(args.adapter_path),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_last_layer_fraction=args.lora_last_layer_fraction,
        random_state=args.seed,
    )
    print_json("[loader]", loader_metadata)
    print_json("[trainable]", trainable_parameter_summary(model))

    probe = run_mask_probe(rows, processor, args)
    print_json("[mask_probe]", probe)
    if not probe["ok"]:
        save_manifest(args, rows, output, {"loader": loader_metadata, "mask_probe": probe})
        raise SystemExit("Mask probe failed; refusing to train.")
    if args.mask_probe_only:
        save_manifest(args, rows, output, {"loader": loader_metadata, "mask_probe": probe})
        return

    try:
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError("Stage 1 training requires transformers.") from exc

    dataset = PipelineV2TokenizedDataset(rows, processor, args.max_seq_length)
    collator = PipelineV2DataCollator(processor)

    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        report_to=parse_report_to(args.report_to),
        run_name=args.run_name,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    class PipelineV2Trainer(Trainer):
        def __init__(self, *trainer_args: Any, train_sampler: Any | None = None, **trainer_kwargs: Any) -> None:
            self._pipeline_v2_train_sampler = train_sampler
            super().__init__(*trainer_args, **trainer_kwargs)

        def _get_train_sampler(self, *sampler_args: Any, **sampler_kwargs: Any) -> Any:
            if self._pipeline_v2_train_sampler is not None:
                return self._pipeline_v2_train_sampler
            return super()._get_train_sampler(*sampler_args, **sampler_kwargs)

    trainer = PipelineV2Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        train_sampler=train_sampler,
    )
    anti_slop_ul_summary: dict[str, Any] = {"enabled": False, "weight": float(args.anti_slop_ul_weight)}
    if args.anti_slop_ul_weight > 0.0:
        ul_config = build_anti_slop_ul_config(
            processor,
            lexicon_path=args.anti_slop_ul_lexicon,
            unigram_top_k=args.anti_slop_ul_unigram_top_k,
            unigram_min_lift=args.anti_slop_ul_unigram_min_lift,
            bigram_top_k=args.anti_slop_ul_bigram_top_k,
            bigram_min_lift=args.anti_slop_ul_bigram_min_lift,
            bigram_min_weight=args.anti_slop_ul_bigram_min_weight,
            trigram_top_k=args.anti_slop_ul_trigram_top_k,
            trigram_min_lift=args.anti_slop_ul_trigram_min_lift,
            trigram_min_weight=args.anti_slop_ul_trigram_min_weight,
            start_weight_multiplier=args.anti_slop_ul_start_weight_multiplier,
        )
        anti_slop_ul_summary = {
            "enabled": True,
            "weight": float(args.anti_slop_ul_weight),
            "lexicon": str(args.anti_slop_ul_lexicon),
            "continuations": len(ul_config.continuations),
            "start_tokens": len(ul_config.start_token_weights),
            "unigram_top_k": int(args.anti_slop_ul_unigram_top_k),
            "unigram_min_lift": float(args.anti_slop_ul_unigram_min_lift),
            "bigram_top_k": int(args.anti_slop_ul_bigram_top_k),
            "bigram_min_lift": float(args.anti_slop_ul_bigram_min_lift),
            "bigram_min_weight": float(args.anti_slop_ul_bigram_min_weight),
            "trigram_top_k": int(args.anti_slop_ul_trigram_top_k),
            "trigram_min_lift": float(args.anti_slop_ul_trigram_min_lift),
            "trigram_min_weight": float(args.anti_slop_ul_trigram_min_weight),
            "start_weight_multiplier": float(args.anti_slop_ul_start_weight_multiplier),
        }
        print_json("[anti-slop-ul]", anti_slop_ul_summary)
        install_anti_slop_unlikelihood(trainer, ul_config, weight=args.anti_slop_ul_weight)
    print("[train] starting Stage 1 CPT-lite", flush=True)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    final_dir = output / "policy"
    trainer.save_model(str(final_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(final_dir))
    elif hasattr(getattr(processor, "tokenizer", None), "save_pretrained"):
        processor.tokenizer.save_pretrained(str(final_dir))

    train_metrics = dict(train_result.metrics)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    save_manifest(
        args,
        rows,
        output,
        {
            "loader": loader_metadata,
            "mask_probe": probe,
            "sampler": sampler_metadata,
            "anti_slop_ul": anti_slop_ul_summary,
            "final_adapter": str(final_dir),
            "train_metrics": train_metrics,
        },
    )
    print_json("[done]", {"final_adapter": str(final_dir), "train_metrics": train_metrics})


if __name__ == "__main__":
    main()
