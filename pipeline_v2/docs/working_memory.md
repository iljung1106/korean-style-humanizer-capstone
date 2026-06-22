# Pipeline V2 Working Memory

Date: 2026-06-01

This memo records the current implementation state, the intended next work, and
the constraints for the standalone `pipeline_v2` training stack.

## Main Objective

Build a new Gemma 4 31B Korean webnovel training pipeline that can add a human
webnovel prose prior without breaking instruction following, multi-turn chat
format, or `<result>...</result>` termination discipline.

## Hard Constraints

- `pipeline_v2` runtime must not import legacy project modules from `train/`,
  `scripts/`, or `eval/`.
- Existing legacy code may be inspected and selectively copied into standalone
  `pipeline_v2` files when useful.
- Do not proceed to long training until smoke gates pass.
- Do not train raw CPT alone as the first stage.
- Keep all stages measurable against fixed eval sets.

## Data Prepared So Far

- Raw CPT chunks:
  - `gemma4_style_rl_training/data/pipeline_v2/cpt_raw_chunks.jsonl`
  - 148,527 rows after audit
  - no replacement-character rows
  - no metadata/front-matter head rows after audit
- CPT/SFT probe mix:
  - `gemma4_style_rl_training/data/pipeline_v2/cpt_mixed_probe.jsonl`
  - 2,000 rows
  - `raw_lm=1000`, `continuation_sft=700`, `format_sft=300`
- SimPO curriculum:
  - `data/pipeline_v2/simpo_curriculum/01_generate_format.jsonl`
  - `data/pipeline_v2/simpo_curriculum/02_nochange_anti_copy.jsonl`
  - `data/pipeline_v2/simpo_curriculum/03_badstyle_rewrite.jsonl`
  - `data/pipeline_v2/simpo_curriculum/curated_mixed.jsonl`
- Fixed eval sets:
  - `data/eval_v2/fixed_generate_prompts.jsonl`
  - `data/eval_v2/fixed_rewrite_prompts.jsonl`
  - `data/eval_v2/fixed_continuation_prompts.jsonl`
  - `data/eval_v2/fixed_format_stop_prompts.jsonl`

## Implemented Code

Data prep:

- `pipeline_v2/scripts/01_clean_human_novels.py`
- `pipeline_v2/scripts/02_build_cpt_mix.py`
- `pipeline_v2/scripts/03_prepare_simpo_curriculum.py`
- `pipeline_v2/scripts/04_prepare_eval_sets.py`

Stage 1 runtime:

- `pipeline_v2/lib/io.py`
- `pipeline_v2/lib/gemma4_loader.py`
- `pipeline_v2/lib/lora.py`
- `pipeline_v2/lib/masking.py`
- `pipeline_v2/lib/sampling.py`
- `pipeline_v2/lib/trainer_utils.py`
- `pipeline_v2/lib/anti_slop_unlikelihood.py`

Stage 1 tools:

- `pipeline_v2/eval/probe_loss_masks.py`
- `pipeline_v2/train/stage01_cpt_lite.py`
- `pipeline_v2/configs/stage01_cpt_lite.example.json`

Stage 1 pilot/eval bundle:

- `pipeline_v2/lib/result_contract.py`
- `pipeline_v2/lib/gui_style_scoring.py`
- `pipeline_v2/eval/run_fixed_eval.py`
- `pipeline_v2/eval/score_outputs.py`
- `pipeline_v2/eval/probe_adapter_reload.py`
- `pipeline_v2/scripts/10_run_stage01_pilot.py`

## Stage 1 Behavior

Input row types:

- `raw_lm`
  - render raw text directly
  - full-token loss
- `continuation_sft`
  - render with Gemma 4 chat template
  - system/user/prompt masked to `-100`
  - assistant `<result>...</result>` visible
- `format_sft`
  - same assistant-only masking as `continuation_sft`
- `general_guard`
  - supported by masking/sampler, but no builder yet
  - intended for small format/multiturn preservation subset if drift appears

Slop lexicon unlikelihood:

- Stage 1 now has optional `--anti-slop-ul-*` flags.
- It is implemented inside `pipeline_v2/lib/anti_slop_unlikelihood.py`.
- Default weight is `0.0` for clean ablations.
- Recommended next paid run weight: `0.003`, using the processed anti-slop
  lexicon and the default top-k/lift filters.

Sampler:

- Stage 1 defaults to row-type balanced replacement sampling.
- Default quota:
  - `raw_lm=3`
  - `continuation_sft=3`
  - `format_sft=2`
  - `general_guard=1`
- Missing row types are ignored.
- Present row types not listed in the quota raise an error to avoid silent
  dropping.

LoRA target:

- Default is language-only LoRA on:
  - `q_proj/k_proj/v_proj/o_proj`
  - `gate_proj/up_proj/down_proj`
- Default layer fraction is last 60% of language layers.
- Recommended comparison later:
  - 0.6 smoke
  - 0.8 smoke
  - 1.0 only if needed
- Do not open vision tower, multimodal projector, embeddings, lm_head, or norms
  by default.

## Validation Completed Locally

- Python compile checks passed for Stage 1 files.
- Mock tokenizer `probe_loss_masks.py` passed.
- Mock tokenizer `stage01_cpt_lite.py --mask-probe-only` passed and printed
  balanced sampler metadata.

## Not Yet Completed

The current training instance is unavailable, so the following remain pending:

- Real Gemma 4 tokenizer loss-mask probe.
- 10-step or 200-step Stage 1 real-model pilot.
- Actual adapter reload probe execution on a saved real adapter.
- Actual fixed eval generation on base/start adapter and Stage 1 adapter.
- Stage 2 format/task SFT trainer.
- Stage 3 SimPO curriculum trainer.
- Stage 4 generate GRPO trainer.
- Stage 5 rewrite GRPO trainer.

## Immediate Next Work

Because full model training cannot currently run, continue implementing
standalone analysis/eval code:

1. Run `scripts/10_run_stage01_pilot.py --dry-run` before instance launch.
2. On the paid instance, run the Stage 1 pilot bundle, not isolated micro-tests.
3. Inspect `pilot_summary.json`, scored summaries, and sample generations.
4. Only then decide whether to expand Stage 1 steps, compare
   `lora_last_layer_fraction=0.8`, or move to Stage 2.

Metric policy:

- Added `pipeline_v2/docs/gui_metric_policy.md`.
- Primary style signals:
  - anti-slop density
  - translationese SVM
  - comma overuse
  - POS 4-6gram diversity/repeat
- Low-weight signals:
  - POS 3gram
  - POS n-gram usage distribution
  - modifier shape
  - sentence-initial POS repeat
- Do not use directly as reward:
  - POS 1-2gram scalar metrics
  - sentence length CV
  - sentence-final POS repeat
  - sentence-initial token repeat
  - parenthesis metrics
  - individual high-order POS n-gram target matching
