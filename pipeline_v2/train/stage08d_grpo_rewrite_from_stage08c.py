#!/usr/bin/env python3
"""Stage 8D rewrite-only GRPO from Stage 8C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import unsloth  # noqa: F401,E402

from pipeline_v2.lib.grpo_training import add_common_grpo_args, run_grpo_stage


DEFAULT_DYNAMIC_WEIGHTS = TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08_dynamic_style_weights.json"
FALLBACK_STYLE_OVERRIDES = (
    "sentence_final_token_repeat_rate:3.2,"
    "sentence_length_cv:3.2,"
    "sentence_length_iqr_ratio:3.2,"
    "pos_3gram_repeat_rate:2.2,"
    "anti_slop_density:2.2,"
    "pos_5gram_repeat_rate:1.5,"
    "pos_4gram_diversity:1.3,"
    "simile_marker_per_1k_chars:1.3,"
    "simile_sentence_rate:1.3,"
    "content_modifier_repeat_occurrence_rate:1.2,"
    "modifier_repetition_mass:1.1,"
    "modifier_repeat_burst_mass:1.0,"
    "sentence_initial_token_repeat_rate:0.8,"
    "comma_per_1k_chars:0.25"
)


def load_style_overrides(path: Path) -> str:
    if not path.exists():
        return FALLBACK_STYLE_OVERRIDES
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return FALLBACK_STYLE_OVERRIDES
    return str(payload.get("style_metric_weight_overrides") or FALLBACK_STYLE_OVERRIDES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pipeline_v2 Stage 8D rewrite-only GRPO.")
    add_common_grpo_args(parser, task="rewrite", default_output_name="stage08d_grpo_rewrite_from_stage08c")
    parser.add_argument("--dynamic-style-weights", default=str(DEFAULT_DYNAMIC_WEIGHTS))
    probe_args, _unknown = parser.parse_known_args()
    style_overrides = probe_args.style_metric_weight_overrides or load_style_overrides(Path(probe_args.dynamic_style_weights))
    parser.set_defaults(
        dataset=str(TRAINING_ROOT / "data" / "processed" / "grpo_mixed_source1536.jsonl"),
        output=str(TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08d_grpo_rewrite_from_stage08c"),
        adapter_path=str(TRAINING_ROOT / "outputs" / "pipeline_v2" / "stage08c_grpo_generate_from_stage08b" / "policy"),
        gui_style_reference=str(TRAINING_ROOT / "data" / "processed" / "gui_style_reward_reference_stage06.json"),
        disabled_style_metrics=(
            "translationese_raw,"
            "content_top_10_coverage,"
            "content_gini_frequency,"
            "content_repeat_occurrence_rate,"
            "parenthesis_pair_per_1k_chars"
        ),
        style_metric_weight_overrides=style_overrides,
        rewrite_style_base_weight=0.80,
        rewrite_anti_slop_family_weight=0.25,
        rewrite_translationese_family_weight=0.0,
        rewrite_comma_family_weight=0.03,
        rewrite_pos_family_weight=0.22,
        rewrite_modifier_family_weight=0.16,
        rewrite_lexical_family_weight=0.0,
        rewrite_sentence_edge_family_weight=0.32,
        rewrite_sentence_length_family_weight=0.45,
        rewrite_edit_weight=0.22,
        rewrite_improvement_weight=0.48,
        rewrite_improvement_scale=0.25,
        rewrite_low_edit_penalty_max=0.12,
        rewrite_edit_gate_min=0.40,
        rewrite_edit_gate_q25=0.70,
        rewrite_edit_gate_q50=1.0,
        short_output_penalty=0.08,
        long_output_penalty=0.06,
        collapse_fail_reward=-0.65,
        max_steps=24,
        max_completion_length=2048,
        batch_size=2,
        num_generations=3,
        grad_accum=1,
        num_iterations=1,
        learning_rate=6.0e-7,
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
        group_diversity_bonus_max=0.015,
        mixed_task_order="shuffle",
        save_steps=12,
        save_total_limit=2,
        sample_log_every=1,
        sample_log_max_items=8,
        sample_log_text_chars=3500,
        style_guidance_variant_mode="row",
        system_prompt_variants=True,
        shuffle_style_guidance_bullets=True,
        wandb_reward_component_log=True,
        enable_short_reasoning=False,
    )
    return parser.parse_args()


def main() -> None:
    run_grpo_stage(parse_args(), stage="stage08d_grpo_rewrite_from_stage08c", task="rewrite")


if __name__ == "__main__":
    main()
