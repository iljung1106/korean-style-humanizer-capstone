# Pipeline V2 Implementation Plan

Date: 2026-06-01

This document is the build plan for the standalone `pipeline_v2` training
pipeline. It is intentionally stricter than `training_pipeline_v2_plan.md`: the
goal here is to define exactly what to build, in what order, and what must pass
before moving to the next stage.

## Current State

Data preparation is implemented and audited.

- Raw CPT chunks:
  - path: `gemma4_style_rl_training/data/pipeline_v2/cpt_raw_chunks.jsonl`
  - rows: `148527`
  - replacement-character rows after audit: `0`
  - metadata/front-matter head rows after audit: `0`
- CPT/SFT probe mix:
  - path: `gemma4_style_rl_training/data/pipeline_v2/cpt_mixed_probe.jsonl`
  - rows: `2000`
  - composition: `raw_lm=1000`, `continuation_sft=700`, `format_sft=300`
  - result-tag failures after audit: `0`
  - post-result text after audit: `0`
  - metadata/front-matter head rows after audit: `0`
- SimPO curriculum:
  - `01_generate_format.jsonl`: `74` rows
  - `02_nochange_anti_copy.jsonl`: `177` rows
  - `03_badstyle_rewrite.jsonl`: `424` rows
  - `curated_mixed.jsonl`: `675` rows
- Fixed eval:
  - `fixed_generate_prompts.jsonl`: `48` unique prompts
  - `fixed_rewrite_prompts.jsonl`: `48` unique prompts
  - `fixed_continuation_prompts.jsonl`: `48` unique prompts
  - `fixed_format_stop_prompts.jsonl`: `24` unique prompts

Generated data is intentionally ignored by git. The code and docs under
`pipeline_v2/` are the tracked artifacts.

## Implementation Status

Stage 1 foundation is implemented.

- Shared runtime:
  - `pipeline_v2/lib/io.py`
  - `pipeline_v2/lib/gemma4_loader.py`
  - `pipeline_v2/lib/lora.py`
  - `pipeline_v2/lib/masking.py`
  - `pipeline_v2/lib/trainer_utils.py`
- Mask probe:
  - `pipeline_v2/eval/probe_loss_masks.py`
- Stage 1 trainer:
  - `pipeline_v2/train/stage01_cpt_lite.py`
- Stage 1 pilot/eval bundle:
  - `pipeline_v2/lib/result_contract.py`
  - `pipeline_v2/lib/gui_style_scoring.py`
  - `pipeline_v2/eval/run_fixed_eval.py`
  - `pipeline_v2/eval/score_outputs.py`
  - `pipeline_v2/eval/probe_adapter_reload.py`
  - `pipeline_v2/scripts/10_run_stage01_pilot.py`
- Config:
  - `pipeline_v2/configs/stage01_cpt_lite.example.json`
- Sampling:
  - Stage 1 defaults to row-type balanced replacement sampling.
  - Current default quota is
    `raw_lm=3,continuation_sft=3,format_sft=2,general_guard=1`.
  - Missing row types are ignored, but row types present in the dataset must be
    listed in the quota to avoid silent dropping.

Local validation completed:

- Python syntax compile passed for all new Stage 1 files.
- Mock-tokenizer loss-mask probe passed on
  `data/pipeline_v2/cpt_mixed_probe.jsonl`.
- Mock-tokenizer `stage01_cpt_lite.py --mask-probe-only` passed on a 20-row
  subset.
- `score_outputs.py` ran on an existing 54-row GUI reward sample log and wrote
  scored JSONL/CSV/summary.
- `10_run_stage01_pilot.py --dry-run` produced the expected preflight,
  training, reload, baseline eval, Stage 1 eval, and scoring commands.

Still required on the training machine:

- Real Gemma 4 tokenizer mask probe.
- Stage 1 real-model pilot training.
- Saved adapter reload check on the real Stage 1 output.
- Fixed eval comparison before and after Stage 1 on the real model.

## Non-Negotiable Constraints

- Do not import from existing project modules under `train/`, `scripts/`, or
  `eval/`.
- Existing code may be read and copied from selectively, but the v2 runtime must
  remain standalone.
- Do not train Stage 1 with a generic chat SFT trainer.
- Every stage must emit:
  - exact command manifest
  - package/model metadata
  - input dataset paths and row counts
  - adapter path
  - fixed eval summary
- Do not run long training before the relevant smoke gates pass.

## Repository Layout To Build

```text
pipeline_v2/
  configs/
    pipeline_v2.example.json
    stage01_cpt_lite.example.json
    stage02_format_sft.example.json
    stage03_simpo_curriculum.example.json
    stage04_grpo_generate.example.json
    stage05_grpo_rewrite.example.json
  docs/
    training_pipeline_v2_plan.md
    implementation_plan.md
    runbook.md
  scripts/
    01_clean_human_novels.py
    02_build_cpt_mix.py
    03_prepare_simpo_curriculum.py
    04_prepare_eval_sets.py
    10_run_stage01_pilot.py
  lib/
    __init__.py
    io.py
    gemma4_loader.py
    masking.py
    lora.py
    result_contract.py
    gui_style_scoring.py
    trainer_utils.py
  train/
    stage01_cpt_lite.py
    stage02_format_task_sft.py
    stage03_simpo_curriculum.py
    stage04_grpo_generate.py
    stage05_grpo_rewrite.py
  eval/
    run_fixed_eval.py
    score_outputs.py
    probe_adapter_reload.py
    probe_loss_masks.py
```

The first implementation pass should build only the shared `lib/`, Stage 1, and
the mask probe. Later stages are documented now but should not be implemented
until Stage 1 output is evaluated.

## Data Contracts

### `raw_lm`

Input row:

```json
{
  "row_type": "raw_lm",
  "id": "...",
  "text": "...",
  "source_file": "...",
  "source_row_id": "..."
}
```

Training contract:

- Render exactly `text`.
- Labels are full-token labels.
- Only padding is masked to `-100`.
- No chat template is applied.
- No `<result>` tags are added.

### `continuation_sft`

Input row:

```json
{
  "row_type": "continuation_sft",
  "id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<result>...</result>"}
  ]
}
```

Training contract:

- Render with the Gemma 4 chat template.
- Labels for system/user/prompt tokens are `-100`.
- Labels for assistant content, including `<result>` and `</result>`, are
  visible.
- Padding is `-100`.

### `format_sft`

Same masking contract as `continuation_sft`.

`format_sft` is oversampled in small probe mixes because the clean source pool
is small. This is acceptable for Stage 1 probing, but the manifest must record
source-row repetition.

## Stage 0: Eval Baseline

Purpose: establish fixed baseline before any new training.

Build:

- `pipeline_v2/eval/evaluate_fixed_sets.py`

Minimum behavior:

- Load base model or adapter.
- Generate on all four fixed eval files.
- Record JSONL samples.
- Record aggregate metrics:
  - result open/close success
  - post-result text count
  - thought/meta leakage count
  - output length stats
  - rewrite source-copy stats when `source_text` exists

Smoke gate:

- Must run on a tiny local selection before full eval.
- Must not require W&B.

## Stage 1: CPT-Lite Trainer

Purpose: add a light human webnovel prose prior while preserving result-format
discipline.

Build:

- `pipeline_v2/lib/io.py`
- `pipeline_v2/lib/gemma4_loader.py`
- `pipeline_v2/lib/masking.py`
- `pipeline_v2/lib/lora.py`
- `pipeline_v2/train/stage01_cpt_lite.py`
- `pipeline_v2/eval/probe_loss_masks.py`

Key implementation requirements:

- Load Gemma 4 31B through Unsloth `FastModel`.
- Apply the Transformers 5.5 Gemma4 compatibility patch if required.
- Support `--load-in-4bit`.
- Support a start adapter and a fresh LoRA adapter.
- Default LoRA target should remain language-only:
  `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`.
- Support `--lora-last-layer-fraction`.
- Implement a custom collator that respects `row_type`.
- Do not rely on `train_on_responses_only`.
- Support optional slop lexicon unlikelihood in Stage 1. It should be disabled
  by default for ablations, but enabled in the next paid pilot with a small
  weight such as `0.003`.

Recommended first settings:

```text
dataset: data/pipeline_v2/cpt_mixed_probe.jsonl
max_seq_length: 8192
max_steps: 10 for smoke, then 100-300 for probe
learning_rate: 5e-7 to 1e-6
batch_size: 1
grad_accum: 8
lora_r: 32
lora_alpha: 64
lora_last_layer_fraction: 1.0 for the full-layer LoRA recipe
anti_slop_ul_weight: 0.003 for the next paid run
anti_slop_ul_bigram_top_k: 300 by default
anti_slop_ul_bigram_min_lift: 4.0 by default
anti_slop_ul_bigram_min_weight: 0.05 by default
```

Stage 1 smoke gates:

- `probe_loss_masks.py` must show:
  - `raw_lm`: visible label ratio near non-padding token ratio
  - `continuation_sft`: prompt label count is `0`
  - `format_sft`: prompt label count is `0`
  - assistant visible labels include `<result>` and `</result>`
- 10-step smoke must complete without NaN.
- Saved adapter must reload.
- Fixed `format_stop` eval must not regress catastrophically.

Stop conditions:

- Any prompt token is label-visible for SFT rows.
- Any all-`-100` label batch appears.
- Loss becomes NaN/Inf.
- Eval shows high post-`</result>` text rate.

## Stage 2: Format / Task SFT

Purpose: re-anchor instruction following after Stage 1.

Build after Stage 1 passes:

- `pipeline_v2/train/stage02_format_task_sft.py`

Inputs:

- existing clean `sft_train.jsonl`
- continuation SFT rows
- fixed format/stop contract examples
- generated task SFT rows if created later

Training contract:

- No `raw_lm` rows.
- Assistant-only loss only.
- Explicit `<result>...</result>` discipline.

Initial settings:

```text
max_steps: 50-150
learning_rate: 1e-6 to 2e-6
```

Gate:

- Result contract must improve or remain stable relative to Stage 1.
- Rewrite/generate instruction following must not degrade.

## Stage 3: SimPO Curriculum

Purpose: preference training without flattening all preference buckets.

Build after Stage 2 passes:

- `pipeline_v2/train/stage03_simpo_curriculum.py`

Order:

1. `01_generate_format.jsonl`
2. `02_nochange_anti_copy.jsonl`
3. `03_badstyle_rewrite.jsonl`
4. optional `curated_mixed.jsonl` for short polish only

Important interpretation:

- `nochange_rejected` means anti-copy, not preservation.
- `03_badstyle_rewrite` has many repeated chosen texts. Treat row count as
  preference weight, not unique text count.

Initial settings:

```text
generate stage: 10-30 steps
nochange stage: 10-20 steps
badstyle stage: 30-80 steps
beta: start near current SimPO beta, then tune
```

Gate:

- No large increase in rewrite semantic drift.
- No large increase in output copying.
- Fixed eval result contract remains stable.

## Stage 4: Generate-Only GRPO

Purpose: optimize free generation without rewrite constraints.

Build after SimPO curriculum passes:

- `pipeline_v2/train/stage04_grpo_generate.py`

Initial conservative smoke:

```text
num_generations: 2
max_completion_length: 768-1536
reward: cheap/configurable first
report_to: none for smoke
```

Full probe can raise completion length and generations only after the smoke is
stable.

Gate:

- Non-zero reward spread.
- No result-contract collapse.
- No large increase in repetition or meta leakage.

## Stage 5: Rewrite-Only GRPO

Purpose: improve rewrite quality separately from generation.

Build after generate GRPO is stable:

- `pipeline_v2/train/stage05_grpo_rewrite.py`

Requirements:

- Do not disable rewrite fidelity for the main run.
- Keep source-copy penalty and length-ratio checks.
- Semantic preservation should be present, but initially weaker than anti-copy
  pressure.

Initial settings:

```text
temperature: 1.0 to 1.05
top_p: 0.98 to 1.0
num_generations: 2 for smoke, 3-4 only if stable
```

Gate:

- Rewrite source 3-gram similarity decreases.
- Semantic drift does not spike.
- Reward spread remains usable.

## Stage 6: Final Polish

Use one short final stage only:

- short curated SimPO, or
- short mixed GRPO.

Do not run both for long.

## Discriminator Work

Do not block the main pipeline on a token-level AI discriminator.

Recommended timing:

1. Build Stage 1-3 baseline first.
2. Save fixed eval and generation samples.
3. Use those samples to train an E2B prefix/token discriminator as an offline
   scorer.
4. Use discriminator attribution first for analysis.
5. Only then consider token-weighted SFT/UL or token-weighted SimPO.

Do not add discriminator reward directly to GRPO until it has source-file,
generator, and prompt-family holdout validation.

## Immediate Next Run Step

Run the Stage 1 gates on the training machine:

1. Real-tokenizer `probe_loss_masks.py`.
2. 10-step `stage01_cpt_lite.py` smoke.
3. Saved adapter reload check.
4. Fixed eval before/after comparison.

No later trainer stage should be coded until these gates pass.
