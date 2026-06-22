#!/usr/bin/env python3
"""Stage 6 mixed generate/rewrite GRPO trainer for pipeline_v2."""

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
    parser = argparse.ArgumentParser(description="Train pipeline_v2 Stage 6 mixed generate/rewrite GRPO.")
    add_common_grpo_args(parser, task="mixed", default_output_name="stage06_grpo_mixed")
    parser.set_defaults(
        dataset=str(TRAINING_ROOT / "data" / "processed" / "grpo_mixed_source1536.jsonl"),
        gui_style_reference=str(TRAINING_ROOT / "data" / "processed" / "gui_style_reward_reference_stage06.json"),
        disabled_style_metrics="translationese_raw,content_top_10_coverage,content_gini_frequency,content_repeat_occurrence_rate",
        style_metric_weight_overrides=(
            "sentence_final_token_repeat_rate:3.0,"
            "sentence_length_cv:2.6,"
            "sentence_length_iqr_ratio:2.4,"
            "anti_slop_density:2.2,"
            "simile_marker_per_1k_chars:1.8,"
            "simile_sentence_rate:1.8,"
            "pos_5gram_repeat_rate:1.5,"
            "pos_4gram_diversity:1.4,"
            "pos_3gram_repeat_rate:1.2,"
            "comma_per_1k_chars:0.45,"
            "sentence_initial_token_repeat_rate:0.55,"
            "modifier_repetition_mass:0.55,"
            "modifier_repeat_burst_mass:0.55,"
            "content_modifier_repeat_occurrence_rate:0.55"
        ),
        rewrite_style_base_weight=1.0,
        rewrite_anti_slop_family_weight=0.35,
        rewrite_translationese_family_weight=0.0,
        rewrite_comma_family_weight=0.05,
        rewrite_pos_family_weight=0.30,
        rewrite_modifier_family_weight=0.06,
        rewrite_lexical_family_weight=0.0,
        rewrite_sentence_edge_family_weight=0.35,
        rewrite_sentence_length_family_weight=0.45,
        rewrite_edit_weight=0.35,
        rewrite_improvement_weight=0.35,
        rewrite_low_edit_penalty_max=0.35,
        max_completion_length=3072,
        batch_size=2,
        num_generations=2,
        grad_accum=1,
        num_iterations=3,
        learning_rate=8e-7,
        max_grad_norm=0.2,
        beta=0.005,
        loss_type="dr_grpo",
        scale_rewards="group",
        repetition_penalty=1.05,
        no_repeat_ngram_size=0,
        group_diversity_mode="leave_one_out",
        group_diversity_bonus_max=0.025,
        save_steps=20,
        save_total_limit=2,
        sample_log_every=1,
        sample_log_max_items=8,
        sample_log_text_chars=3500,
        wandb_reward_component_log=True,
    )
    return parser.parse_args()


def main() -> None:
    run_grpo_stage(parse_args(), stage="stage06_grpo_mixed", task="mixed")


if __name__ == "__main__":
    main()
