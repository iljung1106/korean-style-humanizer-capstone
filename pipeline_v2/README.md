# Gemma 4 Webnovel Pipeline V2

This folder is intentionally standalone. Scripts here must not import project-local
modules from `train/`, `scripts/`, or `eval/`. Existing code may be used as a
reference, but the v2 pipeline should remain portable and auditable.

## Goals

- Add a light human-webnovel style prior without breaking instruction following.
- Re-anchor `<result>...</result>` formatting after any raw text training.
- Split preference and RL stages by task so generate and rewrite signals do not
  cancel each other out.
- Keep every stage measurable against a fixed evaluation set.

## Planned Stages

1. `clean_human_novels`: clean raw `.txt` novels into body-only chunks.
2. `build_cpt_mix`: build a mixed `raw_lm` / `continuation_sft` / `format_sft`
   dataset with explicit row types.
3. `stage01_cpt_lite`: train a small style prior adapter.
4. `prepare_eval_sets`: create fixed generate/rewrite/continuation/format-stop
   eval prompts before training.
5. `stage02_format_task_sft`: restore instruction and result-tag discipline.
6. `stage03_simpo_curriculum`: run generate -> nochange -> badstyle preference
   curriculum.
7. `stage04_grpo_generate`: optimize generation only.
8. `stage05_grpo_rewrite`: optimize rewrite only.
9. `stage06_final_polish`: short curated SimPO or mixed GRPO, not both by default.

## Files

- `docs/training_pipeline_v2_plan.md`: current plan and rationale.
- `docs/implementation_plan.md`: concrete build order, stage contracts, and
  smoke gates.
- `docs/runbook.md`: concrete preparation commands.
- `docs/working_memory.md`: current state and next-work memo.
- `docs/gui_metric_policy.md`: metric-priority policy for GUI-style reward/eval.
- `scripts/01_clean_human_novels.py`: standalone raw text cleaner/chunker.
- `scripts/02_build_cpt_mix.py`: standalone mixed CPT/SFT dataset builder.
- `scripts/03_prepare_simpo_curriculum.py`: standalone DPO filtering/splitting
  script for SimPO curriculum stages.
- `scripts/04_prepare_eval_sets.py`: standalone fixed eval-set builder.
- `scripts/10_run_stage01_pilot.py`: paid-instance Stage 1 pilot bundle:
  preflight, training, reload probe, fixed eval, and scoring.
- `configs/pipeline_v2.example.json`: example paths and defaults.
- `configs/stage01_cpt_lite.example.json`: conservative Stage 1 smoke defaults.
- `lib/`: standalone shared runtime helpers for IO, Gemma 4 loading, LoRA
  targets, row-type balanced sampling, result-contract checks, GUI-style
  scoring, and Stage 1 loss masking.
- `eval/probe_loss_masks.py`: checks raw-LM and assistant-only SFT masks before
  training.
- `eval/run_fixed_eval.py`: generates fixed eval outputs for a base model or
  adapter.
- `eval/score_outputs.py`: scores generated outputs for result-contract,
  anti-slop, translationese, comma, POS, repetition, and rewrite diagnostics.
- `eval/probe_adapter_reload.py`: reloads a saved adapter in a fresh process and
  generates short probes.
- `train/stage01_cpt_lite.py`: Stage 1 mixed raw-LM / continuation-SFT /
  format-SFT trainer.
