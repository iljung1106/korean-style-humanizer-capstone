# Training Pipeline V2 Plan

Date: 2026-05-31

## Background

The current SFT set contains only Korean webnovel-related tasks from the larger
SFT seed. That filtering is intentional, but it leaves the model with a small
amount of supervised style data. The human novel corpus under
`data/raw/human_novels` is much larger and should be used to teach sentence
rhythm, scene flow, dialogue cadence, and genre prose patterns.

Raw continued pretraining on an instruction model is risky if done too heavily:
it can weaken instruction following, chat boundaries, and result-tag discipline.
The v2 pipeline therefore uses a light raw LM stage mixed with continuation and
format SFT, then re-anchors instruction behavior before preference and RL stages.

## Similarity Criteria

Two similarity families are used for different purposes.

### Fast Dataset Diagnostics

Fast diagnostics can use character-level and token-level similarity:

- character `SequenceMatcher` ratio for source/chosen/rejected sanity checks
- whitespace/punctuation-normalized surface token overlap
- line/chunk hash and n-gram hash for deduplication

These checks do not require `kiwipiepy`.

### Rewrite Reward / Detailed Analysis

Rewrite quality should use Korean-aware tokenization when available:

- surface token Jaccard
- surface 3-gram Jaccard
- POS 3-gram Jaccard
- content containment
- length ratio

The current reward code uses `kiwipiepy` for detailed rewrite profiling. The v2
data-prep scripts stay dependency-light, but training/reward code may keep using
`kiwipiepy` for Korean morphology-sensitive metrics.

## Stage 0: Fixed Evaluation Set

Create fixed eval files before training:

- `data/eval_v2/fixed_generate_prompts.jsonl`
- `data/eval_v2/fixed_continuation_prompts.jsonl`
- `data/eval_v2/fixed_rewrite_prompts.jsonl`
- `data/eval_v2/fixed_format_stop_prompts.jsonl`

Track:

- `<result>` open/close success
- text after `</result>`
- thought/meta leakage
- length compliance
- GUI style score
- anti-slop and translationese scores
- rewrite source similarity and semantic drift
- small human review sample

## Stage 1: CPT-lite + Continuation SFT

Purpose: inject a human webnovel prose prior without letting raw LM training
overwrite instruction behavior.

Dataset row types:

- `raw_lm`: cleaned body-only novel text, full-token loss
- `continuation_sft`: previous context -> next passage, assistant-only loss
- `format_sft`: instruction examples with `<result>...</result>`, assistant-only loss

Initial ratio:

- `raw_lm`: 50%
- `continuation_sft`: 35%
- `format_sft`: 15%

Initial training:

- short probe first: 100-300 steps
- low LR: `5e-7` to `1e-6`
- checkpoint and eval every 50 steps if practical

## Stage 2: Format / Task SFT

Purpose: restore and strengthen instruction-following after CPT-lite.

Use no raw LM rows here. Train only assistant-side losses on:

- existing Korean webnovel SFT rows
- continuation SFT
- generate SFT
- rewrite SFT
- format/stop contract examples

Expected output: a format-stable task adapter ready for preference training.

## Stage 3: SimPO Curriculum

SimPO should be curriculumized instead of using the entire DPO file as one flat
mixture.

Recommended order:

1. `generate_human_vs_ai` plus format preference, short
2. `nochange_rejected`, short anti-copy stage
3. `badstyle_rejected`, moderate rewrite/style cleanup stage
4. optional short curated mixed SimPO after GRPO

Notes:

- `nochange_rejected` rejects outputs that copied the input too closely. It is
  not a preservation bucket.
- Because nochange chosen outputs can be much farther from source than normal
  rewrite should be, this stage should be short.
- Remove broken chosen rows and extreme length outliers before curriculum use.

## Stage 4: Generate-only GRPO

Purpose: optimize free generation without rewrite constraints.

Requirements:

- expand unique generate prompts beyond the current small set
- include 2500-4500 character targets if long generation is desired
- keep strict result contract and meta-leak penalties

Initial settings:

- temperature around `0.95`
- `num_generations`: 3-4
- `max_completion_length`: 3072
- character limit as prompt contract, not token truncation

## Stage 5: Rewrite-only GRPO

Purpose: improve rewrite behavior separately from generation.

Prompt direction:

- preserve events, information, and character intent
- do not add new events
- do not copy sentence-by-sentence
- freely change expression, sentence order, and prose rhythm

Reward direction:

- increase direct penalty for excessive source 3-gram similarity
- keep semantic drift penalty weaker than anti-copy pressure at first
- raise rewrite profile influence

Initial settings:

- temperature `1.0` to `1.05`
- `top_p` `0.98` to `1.0`
- `num_generations`: preferably 4 if memory allows

## Stage 6: Final Polish

Use only one short final stage by default:

- curated SimPO for 20-40 steps, or
- mixed GRPO for 20-30 steps

Do not run both for long. The final stage is for balance and cleanup, not for
large behavioral movement.

## Things To Avoid

- long raw CPT before any eval
- raw CPT after SimPO/GRPO
- mixing raw LM and SFT rows without explicit loss masks
- long mixed GRPO before task-specific GRPO
- reusing the full unfiltered DPO set as final polish
- repeatedly upsampling a small generate prompt set

