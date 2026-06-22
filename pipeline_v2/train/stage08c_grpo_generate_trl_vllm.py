#!/usr/bin/env python3
"""Stage 8C generate GRPO via vanilla TRL + vLLM server mode.

This path is intentionally separate from the Unsloth GRPO runner.  It is for a
two-GPU setup where one GPU runs the trainer and another GPU runs
`trl vllm-serve`, avoiding the slow regular Transformers `generate()` path seen
on the B200 smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.gemma4_loader import filter_kwargs
from pipeline_v2.lib.grpo_training import (
    add_common_grpo_args,
    instantiate_config,
    load_grpo_rows,
    make_reward_func,
    order_mixed_task_rows,
    resolve_model_name,
)
from pipeline_v2.lib.io import manifest_base, write_json
from pipeline_v2.lib.lora import target_regex_for_last_fraction
from pipeline_v2.lib.preference_data import dataset_from_rows
from pipeline_v2.lib.trainer_utils import parse_report_to, print_json, set_seed


DEFAULT_DYNAMIC_WEIGHTS = TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08_dynamic_style_weights.json"
FALLBACK_STYLE_OVERRIDES = (
    "sentence_final_token_repeat_rate:3.2,"
    "sentence_length_cv:3.4,"
    "sentence_length_iqr_ratio:3.4,"
    "pos_3gram_repeat_rate:2.2,"
    "anti_slop_density:2.4,"
    "pos_5gram_repeat_rate:1.6,"
    "pos_4gram_diversity:1.4,"
    "simile_marker_per_1k_chars:1.4,"
    "simile_sentence_rate:1.4,"
    "content_modifier_repeat_occurrence_rate:1.2,"
    "modifier_repetition_mass:1.1,"
    "modifier_repeat_burst_mass:1.0,"
    "sentence_initial_token_repeat_rate:0.9,"
    "comma_per_1k_chars:0.30"
)


def load_style_overrides(path: Path) -> str:
    if not path.exists():
        return FALLBACK_STYLE_OVERRIDES
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return FALLBACK_STYLE_OVERRIDES
    value = str(payload.get("style_metric_weight_overrides") or "")
    return value or FALLBACK_STYLE_OVERRIDES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 8C generate GRPO with vanilla TRL + vLLM server mode.")
    add_common_grpo_args(parser, task="generate", default_output_name="stage08c_grpo_generate_trl_vllm_from_stage08b")
    parser.add_argument("--dynamic-style-weights", default=str(DEFAULT_DYNAMIC_WEIGHTS))
    parser.add_argument("--vllm-server-base-url", default="")
    parser.add_argument("--vllm-server-host", default="127.0.0.1")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--vllm-server-timeout", type=float, default=240.0)
    parser.add_argument("--vllm-group-port", type=int, default=51216)
    parser.add_argument("--vllm-model-impl", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-length", type=int, default=8192)
    parser.add_argument("--vllm-enable-sleep-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-importance-sampling-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    parser.add_argument("--lora-num-layers", type=int, default=60)
    parser.add_argument("--save-final-policy", action=argparse.BooleanOptionalAction, default=True)
    probe_args, _unknown = parser.parse_known_args()
    style_overrides = probe_args.style_metric_weight_overrides or load_style_overrides(Path(probe_args.dynamic_style_weights))
    parser.set_defaults(
        dataset=str(TRAINING_ROOT / "data" / "processed" / "grpo_mixed_source1536.jsonl"),
        output=str(TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08c_grpo_generate_trl_vllm_from_stage08b"),
        adapter_path="",
        gui_style_reference=str(TRAINING_ROOT / "data" / "processed" / "gui_style_reward_reference_stage06.json"),
        disabled_style_metrics=(
            "translationese_raw,"
            "content_top_10_coverage,"
            "content_gini_frequency,"
            "content_repeat_occurrence_rate,"
            "parenthesis_pair_per_1k_chars"
        ),
        style_metric_weight_overrides=style_overrides,
        rewrite_style_base_weight=0.8,
        rewrite_anti_slop_family_weight=0.25,
        rewrite_translationese_family_weight=0.0,
        rewrite_comma_family_weight=0.03,
        rewrite_pos_family_weight=0.22,
        rewrite_modifier_family_weight=0.16,
        rewrite_lexical_family_weight=0.0,
        rewrite_sentence_edge_family_weight=0.32,
        rewrite_sentence_length_family_weight=0.45,
        rewrite_edit_weight=0.20,
        rewrite_improvement_weight=0.45,
        rewrite_improvement_scale=0.25,
        rewrite_low_edit_penalty_max=0.12,
        rewrite_edit_gate_min=0.40,
        rewrite_edit_gate_q25=0.70,
        rewrite_edit_gate_q50=1.0,
        generate_min_output_chars=3000,
        generate_max_output_chars=4500,
        short_output_penalty=0.10,
        long_output_penalty=0.06,
        collapse_fail_reward=-0.65,
        max_steps=24,
        max_completion_length=3072,
        batch_size=3,
        num_generations=5,
        grad_accum=1,
        num_iterations=1,
        learning_rate=7.0e-7,
        max_grad_norm=0.2,
        beta=0.005,
        loss_type="dr_grpo",
        scale_rewards="group",
        reward_std_shaping_power=0.5,
        reward_std_shaping_floor=0.10,
        reward_std_update_weight_min=0.45,
        reward_std_update_weight_std_low=0.05,
        reward_std_update_weight_std_high=0.16,
        reward_std_update_weight_source="raw",
        repetition_penalty=1.05,
        no_repeat_ngram_size=0,
        group_diversity_mode="leave_one_out",
        group_diversity_bonus_max=0.025,
        mixed_task_order="shuffle",
        save_steps=12,
        save_total_limit=2,
        sample_log_every=1,
        sample_log_max_items=12,
        sample_log_text_chars=3500,
        style_guidance_variant_mode="row",
        system_prompt_variants=True,
        shuffle_style_guidance_bullets=True,
        wandb_reward_component_log=True,
        enable_short_reasoning=False,
        init_lora=True,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    return parser.parse_args()


def torch_dtype_value(name: str) -> Any:
    if name == "auto":
        return "auto"
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_tokenizer(model_name: str, *, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_peft_config(args: argparse.Namespace) -> Any:
    if args.adapter_path:
        raise ValueError("This TRL+vLLM runner starts from a merged model and creates a fresh LoRA; do not pass --adapter-path.")
    if not args.init_lora:
        raise ValueError("This TRL+vLLM runner requires --init-lora so the Stage08B merged model is trained via fresh LoRA.")
    from peft import LoraConfig

    target_modules = target_regex_for_last_fraction(args.lora_num_layers, float(args.lora_last_layer_fraction))
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def build_grpo_config(args: argparse.Namespace, output: Path) -> Any:
    from trl import GRPOConfig

    trainer_scale_rewards: Any = args.scale_rewards
    if float(getattr(args, "reward_std_shaping_power", 0.0) or 0.0) > 0.0:
        if str(args.scale_rewards).lower() not in {"false", "none", "0"}:
            print_json(
                "[reward/std_shaping]",
                {
                    "message": "Disabling trainer scale_rewards because reward_std_shaping_power is enabled.",
                    "requested_scale_rewards": args.scale_rewards,
                    "effective_scale_rewards": False,
                    "power": args.reward_std_shaping_power,
                    "floor": args.reward_std_shaping_floor,
                },
            )
        trainer_scale_rewards = False

    model_init_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_value(args.torch_dtype),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.model_attn_implementation:
        model_init_kwargs["attn_implementation"] = args.model_attn_implementation

    generation_kwargs = {
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "use_cache": bool(args.generation_use_cache),
    }

    config_kwargs = {
        "output_dir": str(output),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": args.warmup_steps,
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
        "bf16": args.torch_dtype == "bfloat16",
        "fp16": args.torch_dtype == "float16",
        "seed": args.seed,
        "report_to": parse_report_to(args.report_to),
        "run_name": args.run_name,
        "remove_unused_columns": False,
        "model_init_kwargs": model_init_kwargs,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "generation_batch_size": args.generation_batch_size if args.generation_batch_size > 0 else args.batch_size,
        "steps_per_generation": args.steps_per_generation if args.steps_per_generation > 0 else 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "generation_kwargs": generation_kwargs,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "importance_sampling_level": args.importance_sampling_level,
        "mask_truncated_completions": args.mask_truncated_completions,
        "epsilon": args.epsilon,
        "epsilon_high": args.epsilon_high,
        "num_iterations": args.num_iterations,
        "scale_rewards": trainer_scale_rewards,
        "shuffle_dataset": True,
        "gradient_checkpointing": True,
        "use_vllm": True,
        "vllm_mode": "server",
        "vllm_model_impl": args.vllm_model_impl,
        "vllm_server_base_url": str(args.vllm_server_base_url or "").strip() or None,
        "vllm_server_host": args.vllm_server_host,
        "vllm_server_port": args.vllm_server_port,
        "vllm_server_timeout": args.vllm_server_timeout,
        "vllm_group_port": args.vllm_group_port,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_model_length": args.vllm_max_model_length if args.vllm_max_model_length > 0 else None,
        "vllm_enable_sleep_mode": bool(args.vllm_enable_sleep_mode),
        "vllm_importance_sampling_correction": bool(args.vllm_importance_sampling_correction),
        "torch_compile": False,
    }
    config_kwargs = {key: value for key, value in config_kwargs.items() if value is not None}
    return instantiate_config(GRPOConfig, config_kwargs)


def jsonable_model_init_kwargs(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: str(item) if key == "torch_dtype" else item for key, item in value.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.no_repeat_ngram_size > 0:
        raise ValueError("no_repeat_ngram_size remains disabled for Korean long-form GRPO.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_name = resolve_model_name(args)
    tokenizer = load_tokenizer(model_name, trust_remote_code=bool(args.trust_remote_code))
    rows = load_grpo_rows(args, tokenizer, task=args.grpo_task)
    task_counts: dict[str, int] = {}
    for row in rows:
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
    rows = order_mixed_task_rows(rows, mode=args.mixed_task_order if args.grpo_task == "mixed" else "shuffle", seed=args.seed)
    dataset = dataset_from_rows(rows).shuffle(seed=args.seed)
    training_args = build_grpo_config(args, output)
    peft_config = build_peft_config(args)
    reward_func = make_reward_func(args)

    from trl import GRPOTrainer

    print_json(
        "[trl_vllm/model]",
        {
            "model_name": model_name,
            "stage": "stage08c_grpo_generate_trl_vllm_from_stage08b",
            "task": args.grpo_task,
            "load_in_4bit": bool(args.load_in_4bit),
            "load_in_16bit": bool(args.load_in_16bit),
            "peft": {
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_num_layers": args.lora_num_layers,
                "target_modules": target_regex_for_last_fraction(args.lora_num_layers, float(args.lora_last_layer_fraction)),
            },
        },
    )
    print_json(
        "[trl_vllm/data]",
        {"rows": len(rows), "task_counts": task_counts, "path": args.dataset},
    )
    print_json(
        "[trl_vllm/effective_config]",
        {
            "per_device_train_batch_size": getattr(training_args, "per_device_train_batch_size", None),
            "gradient_accumulation_steps": getattr(training_args, "gradient_accumulation_steps", None),
            "num_generations": getattr(training_args, "num_generations", None),
            "num_iterations": getattr(training_args, "num_iterations", None),
            "generation_batch_size": getattr(training_args, "generation_batch_size", None),
            "steps_per_generation": getattr(training_args, "steps_per_generation", None),
            "max_completion_length": getattr(training_args, "max_completion_length", None),
            "use_vllm": getattr(training_args, "use_vllm", None),
            "vllm_mode": getattr(training_args, "vllm_mode", None),
            "vllm_model_impl": getattr(training_args, "vllm_model_impl", None),
            "vllm_server_base_url": getattr(training_args, "vllm_server_base_url", None),
            "vllm_server_host": getattr(training_args, "vllm_server_host", None),
            "vllm_server_port": getattr(training_args, "vllm_server_port", None),
            "vllm_tensor_parallel_size": getattr(training_args, "vllm_tensor_parallel_size", None),
            "vllm_max_model_length": getattr(training_args, "vllm_max_model_length", None),
            "vllm_importance_sampling_correction": getattr(training_args, "vllm_importance_sampling_correction", None),
            "model_init_kwargs": jsonable_model_init_kwargs(getattr(training_args, "model_init_kwargs", None)),
        },
    )

    trainer_kwargs = {
        "model": model_name,
        "args": training_args,
        "train_dataset": dataset,
        "reward_funcs": [reward_func],
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
    }
    trainer = GRPOTrainer(**filter_kwargs(GRPOTrainer, trainer_kwargs))
    if args.probe_only:
        batch = next(iter(trainer.get_train_dataloader()))
        print_json("[probe/batch]", {"type": type(batch).__name__, "keys": sorted(batch.keys()) if isinstance(batch, dict) else None})
        return

    print_json(
        "[train/start]",
        {
            "stage": "stage08c_grpo_generate_trl_vllm_from_stage08b",
            "task": args.grpo_task,
            "resume_from_checkpoint": str(args.resume_from_checkpoint or "").strip() or None,
        },
    )
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint or "").strip() or None)
    trainer.save_state()
    final_dir = output / "policy"
    if args.save_final_policy:
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
    metrics = dict(result.metrics)
    trainer.save_metrics("train", metrics)
    write_json(
        output / "stage08c_grpo_generate_trl_vllm_from_stage08b_manifest.json",
        {
            **manifest_base(stage="stage08c_grpo_generate_trl_vllm_from_stage08b", args=vars(args)),
            "dataset_rows": len(rows),
            "task_counts": task_counts,
            "final_adapter": str(final_dir) if args.save_final_policy else None,
            "train_metrics": metrics,
        },
    )
    print_json("[done]", {"stage": "stage08c_grpo_generate_trl_vllm_from_stage08b", "final_adapter": str(final_dir), "train_metrics": metrics})


if __name__ == "__main__":
    main()
