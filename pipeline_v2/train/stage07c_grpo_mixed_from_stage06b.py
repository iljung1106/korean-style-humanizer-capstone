#!/usr/bin/env python3
"""Stage 7C mixed generate/rewrite GRPO from the Stage 6B merged model.

Stage 7C is intentionally separate from Stage 7A/7B probes:
- start from the Stage 6B merged model,
- open a fresh LoRA,
- alternate generate and rewrite rows,
- use G5 when memory allows,
- keep thinking and no-repeat-ngram constraints disabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT.parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from pipeline_v2.lib.grpo_training import add_common_grpo_args, run_grpo_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pipeline_v2 Stage 7C mixed GRPO from Stage 6B.")
    add_common_grpo_args(parser, task="mixed", default_output_name="stage07c_grpo_mixed_from_stage06b")
    parser.set_defaults(
        dataset=str(TRAINING_ROOT / "data" / "processed" / "grpo_mixed_source1536.jsonl"),
        gui_style_reference=str(TRAINING_ROOT / "data" / "processed" / "gui_style_reward_reference_stage06.json"),
        disabled_style_metrics=(
            "translationese_raw,"
            "content_top_10_coverage,"
            "content_gini_frequency,"
            "content_repeat_occurrence_rate,"
            "parenthesis_pair_per_1k_chars"
        ),
        style_metric_weight_overrides=(
            "anti_slop_density:2.4,"
            "sentence_length_cv:3.0,"
            "sentence_length_iqr_ratio:2.8,"
            "sentence_final_token_repeat_rate:2.4,"
            "simile_marker_per_1k_chars:1.7,"
            "simile_sentence_rate:1.7,"
            "pos_5gram_repeat_rate:1.4,"
            "pos_4gram_diversity:1.3,"
            "pos_3gram_repeat_rate:1.2,"
            "content_modifier_repeat_occurrence_rate:1.2,"
            "modifier_repetition_mass:1.1,"
            "modifier_repeat_burst_mass:1.0,"
            "sentence_initial_token_repeat_rate:0.9,"
            "comma_per_1k_chars:0.40"
        ),
        rewrite_style_base_weight=0.9,
        rewrite_anti_slop_family_weight=0.34,
        rewrite_translationese_family_weight=0.0,
        rewrite_comma_family_weight=0.05,
        rewrite_pos_family_weight=0.25,
        rewrite_modifier_family_weight=0.22,
        rewrite_lexical_family_weight=0.0,
        rewrite_sentence_edge_family_weight=0.38,
        rewrite_sentence_length_family_weight=0.58,
        rewrite_edit_weight=0.25,
        rewrite_improvement_weight=0.55,
        rewrite_improvement_scale=0.25,
        rewrite_low_edit_penalty_max=0.16,
        rewrite_edit_gate_min=0.40,
        rewrite_edit_gate_q25=0.70,
        rewrite_edit_gate_q50=1.0,
        generate_min_output_chars=3000,
        generate_max_output_chars=4500,
        short_output_penalty=0.16,
        long_output_penalty=0.08,
        collapse_fail_reward=-1.0,
        max_steps=150,
        max_completion_length=3072,
        batch_size=3,
        num_generations=5,
        grad_accum=1,
        num_iterations=2,
        learning_rate=1.0e-6,
        max_grad_norm=0.2,
        beta=0.005,
        loss_type="dr_grpo",
        scale_rewards="group",
        repetition_penalty=1.05,
        no_repeat_ngram_size=0,
        group_diversity_mode="leave_one_out",
        group_diversity_bonus_max=0.03,
        mixed_task_order="alternating",
        save_steps=50,
        save_total_limit=2,
        sample_log_every=1,
        sample_log_max_items=10,
        sample_log_text_chars=3500,
        style_guidance_variant_mode="row",
        system_prompt_variants=True,
        shuffle_style_guidance_bullets=True,
        wandb_reward_component_log=True,
        enable_short_reasoning=False,
    )
    return parser.parse_args()


def main() -> None:
    run_grpo_stage(parse_args(), stage="stage07c_grpo_mixed_from_stage06b", task="mixed")


if __name__ == "__main__":
    main()
