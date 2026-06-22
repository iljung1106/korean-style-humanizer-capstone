# Pipeline V2 Preparation Runbook

Run these from the workspace root that contains `gemma4_style_rl_training` and
`data/raw/human_novels`.

## 1. Smoke-clean a few raw files

```bash
python3 gemma4_style_rl_training/pipeline_v2/scripts/01_clean_human_novels.py \
  --limit-files 2 \
  --limit-chunks 5 \
  --output gemma4_style_rl_training/data/pipeline_v2/smoke_cpt_raw_chunks.jsonl \
  --manifest gemma4_style_rl_training/data/pipeline_v2/smoke_cpt_raw_chunks.manifest.json
```

Inspect the output manually before a full run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("gemma4_style_rl_training/data/pipeline_v2/smoke_cpt_raw_chunks.jsonl")
for line in p.read_text(encoding="utf-8").splitlines()[:3]:
    row = json.loads(line)
    print(row["id"], row["chars"], row["hangul_ratio"])
    print(row["text"][:500].replace("\n", "\\n"))
    print("---")
PY
```

## 2. Full raw clean

```bash
python3 gemma4_style_rl_training/pipeline_v2/scripts/01_clean_human_novels.py \
  --input-dir data/raw/human_novels \
  --output gemma4_style_rl_training/data/pipeline_v2/cpt_raw_chunks.jsonl \
  --manifest gemma4_style_rl_training/data/pipeline_v2/cpt_raw_chunks.manifest.json \
  --min-chars 1200 \
  --target-chars 4200 \
  --max-chars 6000 \
  --min-hangul-ratio 0.45
```

## 3. Build mixed CPT/SFT rows

For a small probe set:

```bash
python3 gemma4_style_rl_training/pipeline_v2/scripts/02_build_cpt_mix.py \
  --raw-chunks gemma4_style_rl_training/data/pipeline_v2/cpt_raw_chunks.jsonl \
  --format-sft gemma4_style_rl_training/data/processed/sft_train.jsonl \
  --output gemma4_style_rl_training/data/pipeline_v2/cpt_mixed_probe.jsonl \
  --manifest gemma4_style_rl_training/data/pipeline_v2/cpt_mixed_probe.manifest.json \
  --total-rows 2000 \
  --raw-lm-ratio 0.50 \
  --continuation-sft-ratio 0.35 \
  --format-sft-ratio 0.15
```

For the full available mixture, omit `--total-rows`.

Do not omit `--total-rows` for the first training runs. The available raw and
continuation rows are much larger than the format rows, so an unbounded full
mixture can still make epoch accounting noisy. Stage 1 now supports row-type
balanced replacement sampling, but keep the first runs on the explicit probe
mix until the training/eval gates pass.

## 4. Prepare SimPO curriculum splits

```bash
python3 gemma4_style_rl_training/pipeline_v2/scripts/03_prepare_simpo_curriculum.py \
  --input gemma4_style_rl_training/data/processed/dpo_train.jsonl \
  --output-dir gemma4_style_rl_training/data/pipeline_v2/simpo_curriculum
```

Expected split files:

- `01_generate_format.jsonl`
- `02_nochange_anti_copy.jsonl`
- `03_badstyle_rewrite.jsonl`
- `curated_mixed.jsonl`

## 5. Prepare fixed eval sets

```bash
python3 gemma4_style_rl_training/pipeline_v2/scripts/04_prepare_eval_sets.py \
  --grpo-prompts gemma4_style_rl_training/data/processed/grpo_mixed_prompts.jsonl \
  --raw-chunks gemma4_style_rl_training/data/pipeline_v2/cpt_raw_chunks.jsonl \
  --sft gemma4_style_rl_training/data/processed/sft_train.jsonl \
  --output-dir gemma4_style_rl_training/data/eval_v2
```

Expected files:

- `fixed_generate_prompts.jsonl`
- `fixed_continuation_prompts.jsonl`
- `fixed_rewrite_prompts.jsonl`
- `fixed_format_stop_prompts.jsonl`

## 6. Probe Stage 1 Loss Masks

Fast structural check without loading ML packages:

```bash
python3 gemma4_style_rl_training/pipeline_v2/eval/probe_loss_masks.py \
  --dataset gemma4_style_rl_training/data/pipeline_v2/cpt_mixed_probe.jsonl \
  --mock-tokenizer \
  --json
```

Real tokenizer check on the training machine:

```bash
/workspace/venvs/torch-cu128-clean/bin/python3 /workspace/gemma4_style_rl_training/pipeline_v2/eval/probe_loss_masks.py \
  --dataset /workspace/gemma4_style_rl_training/data/pipeline_v2/cpt_mixed_probe.jsonl \
  --tokenizer-model unsloth/gemma-4-31B-it \
  --max-seq-length 8192 \
  --json
```

Do not start Stage 1 if this fails. For SFT rows, prompt labels must be masked
and the visible assistant span must include `<result>` and `</result>`.

## 7. Audit GUI-Style Metric Reference

When the training instance is unavailable, inspect the existing GUI-style
human-vs-AI reference and reward samples locally:

```bash
python3 gemma4_style_rl_training/pipeline_v2/eval/audit_gui_style_reference.py \
  --reference gemma4_style_rl_training/data/processed/gui_style_reward_reference.json \
  --diagnostics gemma4_style_rl_training/outputs/diagnostics/latest_vs_ai_vs_human_reward_comparison.json \
  --samples "gemma4_style_rl_training/outputs/202605301522/gui_style_reward_samples (1).jsonl" \
  --output-dir gemma4_style_rl_training/outputs/pipeline_v2/gui_metric_audit
```

Primary outputs:

- `scalar_metric_separation.csv`
- `pos_ngram_usage_separation.csv`
- `existing_diagnostics_comparison.csv`
- `reward_sample_component_summary.csv`
- `summary.json`

## 8. Stage 1 Paid-Instance Pilot Bundle

Use this when the training instance is available and the goal is to spend one
setup cost on a meaningful end-to-end pilot rather than on isolated tiny tests.

Default bundle:

- real tokenizer mask preflight
- Stage 1 CPT-lite training
- saved adapter reload probe
- fixed eval generation for baseline/start adapter
- fixed eval generation for Stage 1 adapter
- standalone scoring for both eval outputs

Recommended first paid run:

```bash
/workspace/venvs/torch-cu128-clean/bin/python3 /workspace/gemma4_style_rl_training/pipeline_v2/scripts/10_run_stage01_pilot.py \
  --dataset /workspace/gemma4_style_rl_training/data/pipeline_v2/cpt_mixed_probe.jsonl \
  --output-root /workspace/gemma4_style_rl_training/outputs/pipeline_v2/stage01_pilot_v1 \
  --model unsloth/gemma-4-31B-it \
  --max-seq-length 8192 \
  --max-steps 200 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 8e-7 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-last-layer-fraction 1.0 \
  --row-type-sampling balanced \
  --row-type-balance raw_lm=3,continuation_sft=3,format_sft=2,general_guard=1 \
  --anti-slop-ul-weight 0.003 \
  --anti-slop-ul-unigram-top-k 300 \
  --anti-slop-ul-unigram-min-lift 7.5 \
  --anti-slop-ul-bigram-top-k 300 \
  --anti-slop-ul-bigram-min-lift 4.0 \
  --anti-slop-ul-bigram-min-weight 0.05 \
  --eval-limit-per-dataset 4 \
  --eval-max-new-tokens 2048 \
  --reload-probe-limit-per-dataset 1 \
  --report-to none
```

If continuing from an existing adapter, add:

```bash
  --adapter-path /workspace/gemma4_style_rl_training/outputs/.../policy
```

Stage 1 supports optional slop lexicon unlikelihood during the SFT/CPT-lite
loss. Use it for the next paid run, not for interpreting an already completed
adapter. The implementation lives inside `pipeline_v2/lib` and does not depend
on legacy training scripts.

Before paying for the run, verify the exact subcommands:

```bash
python3 gemma4_style_rl_training/pipeline_v2/scripts/10_run_stage01_pilot.py \
  --dry-run \
  --max-steps 2 \
  --eval-limit-per-dataset 1 \
  --reload-probe-limit-per-dataset 1 \
  --output-root gemma4_style_rl_training/outputs/pipeline_v2/stage01_pilot_dryrun
```

Primary outputs:

- `stage01_cpt_lite/policy`: final Stage 1 adapter
- `stage01_cpt_lite/stage01_manifest.json`: training manifest and metrics
- `reload_probe/generations.jsonl`: fresh-process adapter reload generations
- `eval_baseline/generations.jsonl`: baseline/start-adapter fixed eval
- `eval_baseline/scored_outputs.summary.json`: baseline score summary
- `eval_stage01/generations.jsonl`: Stage 1 fixed eval
- `eval_stage01/scored_outputs.summary.json`: Stage 1 score summary
- `pilot_summary.json`: command log and copied summaries

Pass/fail gates:

- Stage 1 training completes and writes `policy`.
- Reload probe loads `policy` in a fresh process and emits generations.
- `eval_stage01/scored_outputs.summary.json` exists.
- `result_contract_reason_counts.post_result_text` does not increase versus
  baseline.
- `metric_summaries.metrics.anti_slop_density.mean`,
  `metrics.translationese_raw.mean`, and comma metrics do not worsen materially.
- Format/continuation rows keep acceptable `<result>...</result>` behavior.

Do not move to Stage 2/SimPO/GRPO until this pilot produces a readable
`pilot_summary.json` and sample outputs are manually inspected.

## 9. Stage 1 CPT-Lite Smoke

Use this for the first 10-step run:

```bash
/workspace/venvs/torch-cu128-clean/bin/python3 /workspace/gemma4_style_rl_training/pipeline_v2/train/stage01_cpt_lite.py \
  --dataset /workspace/gemma4_style_rl_training/data/pipeline_v2/cpt_mixed_probe.jsonl \
  --output /workspace/gemma4_style_rl_training/outputs/pipeline_v2/stage01_cpt_lite_smoke \
  --model unsloth/gemma-4-31B-it \
  --max-seq-length 8192 \
  --load-in-4bit \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 8e-7 \
  --max-steps 10 \
  --logging-steps 1 \
  --save-steps 10 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-last-layer-fraction 1.0 \
  --row-type-sampling balanced \
  --row-type-balance raw_lm=3,continuation_sft=3,format_sft=2,general_guard=1 \
  --report-to none
```

Expected final adapter path:

```text
/workspace/gemma4_style_rl_training/outputs/pipeline_v2/stage01_cpt_lite_smoke/policy
```

## 10. Stage 3-5 Reward And Training Signals

Stage 3 SimPO is preference training. Stage 4/5 GRPO use scalar rewards.

Stage 4 generate GRPO uses the GUI-style reward bundle:

- result contract and collapse penalties
- anti-slop density
- translationese SVM score
- comma overuse metrics
- POS 3-6gram diversity/repetition, with 4-6gram weighted higher
- POS 1-5gram usage-distribution alignment
- modifier distribution metrics
- low-weight sentence-edge and sentence-length metrics
- optional group diversity shaping for generate rollouts

Scalar metrics are scored against the human reference distribution with banded
rewarding:

- inside human q25-q75: weak positive distribution alignment
- q10-q25 or q75-q90: moderate negative pressure
- outside q10-q90: strong negative pressure

This avoids optimizing every metric monotonically toward a single point while
still penalizing clearly AI-like outliers.

Generate diversity shaping is intentionally small and only applied to valid
generate outputs in the same prompt group. The default is `leave_one_out` with
`--group-diversity-bonus-max 0.03`. Set the max bonus to `0` for ablations.

- `density_adjusted` / `dra_density`: DRA-GRPO-inspired semantic density / inverse-propensity
  shaping. Redundant samples get lower or negative relative rarity; rarer
  samples get a small positive bonus.
- `leave_one_out` / `sgrpo_leave_one_out`: SGRPO-inspired set-diversity shaping. Each rollout receives
  a centered marginal contribution based on its mean distance to the other
  rollouts.
- `mmr_reweighted`: MMR-GRPO-inspired reweighting. A completion receives a
  small centered novelty contribution when its nearest equal-or-better peer is
  less similar. A quality gate limits the bonus for low-quality off-topic
  variation.
- `distance_threshold`: older simple mean-distance bonus. Keep this for ablation
  only; it gives every sufficiently different sample a positive offset and is
  less well aligned with group-relative ranking.
- Similarity uses Korean text features: character 5grams, word 3grams, and
  `kiwipiepy` content-token 3grams / POS 4grams when available.
- Keep `--group-diversity-bonus-max` low, around `0.03-0.06`, so diversity
  does not override result contract, collapse, or human-distribution style
  rewards.
- Probe before a paid run:

```bash
python3 gemma4_style_rl_training/pipeline_v2/eval/probe_diversity_reward.py \
  --mode all \
  --json
```

The research basis is:

- DRA-GRPO: semantic-density / inverse-propensity reward adjustment for GRPO
  when scalar rewards do not distinguish diverse completions.
- MMR-GRPO: maximal-marginal-relevance reward reweighting, prioritizing
  semantically non-redundant completions because redundant rollouts carry less
  marginal learning signal.
- GAPO: group-aware reward vectors for diversity and coverage; frequency-aware
  rewards are most direct when the valid answer set is known.
- SGRPO: set-level diversity rewards redistributed to individual rollouts via
  leave-one-out contribution. This is the best fit for open-ended webnovel
  generation because the output space has no fixed discrete answer set.
- Distinct-n and Self-BLEU are useful evaluation diagnostics, but not used as
  the primary training reward here. Distinct-n is length-biased, and Self-BLEU
  alone can reward low-quality off-topic variation.

Sources checked:

- DRA-GRPO: https://arxiv.org/abs/2505.09655
- MMR-GRPO: https://arxiv.org/abs/2601.09085
- GAPO: https://arxiv.org/abs/2511.12596
- GCPO: https://arxiv.org/abs/2605.11461
- SGRPO summary: https://huggingface.co/papers/2605.08659
- Joint diversity/quality RL for language generation:
  https://arxiv.org/abs/2509.02534
- Diversity-incentivized exploration:
  https://huggingface.co/papers/2509.26209

Stage 5 rewrite GRPO adds rewrite-specific rewards on top of the GUI-style
bundle:

- source/reference rewrite edit profile estimated from the rewrite dataset
- penalty when edit amount is below the learned lower band
- penalty when edit amount is excessive
- style-improvement reward compared with the source text
- gating so style improvement does not dominate when the output barely edits
  the source

Detailed rewrite similarity uses Korean-aware tokenization when `kiwipiepy` is
available, including surface token overlap, POS 3gram overlap, content
containment, character sequence ratio, and length ratio.

## 11. Phase-End GUI Metric Evaluation

Use this after each substantial training phase when the goal is not a reward
score, but a raw GUI-compatible metric report.

The evaluator writes:

- `eval_sets/*.jsonl`: fixed rewrite/generate/control samples
- `generations/rewrite_generations.jsonl`
- `generations/generate_generations.jsonl`
- `metrics/raw_metrics_by_sample.csv`
- `metrics/metric_summary_by_group.csv`
- `metrics/distribution_tests.csv`
- `metrics/plots/*.png`
- `metrics/report.md`

The scoring step requires `matplotlib` for PNG plots. If plot generation is
not available, CSV/MD-only smoke scoring can be run with `--no-require-plots`
on `score_and_report.py`, but phase-end bundle runs should keep plots enabled.

Default full bundle:

```bash
/workspace/venvs/torch-cu128-clean/bin/python3 /workspace/gemma4_style_rl_training/pipeline_v2/eval/phase_eval/run_phase_eval_bundle.py \
  --eval-dir /workspace/gemma4_style_rl_training/data/phase_eval_v2 \
  --output-dir /workspace/gemma4_style_rl_training/outputs/pipeline_v2/phase_eval_stage01 \
  --model unsloth/gemma-4-31B-it \
  --adapter-path /workspace/gemma4_style_rl_training/outputs/.../policy \
  --model-label stage01 \
  --max-seq-length 8192 \
  --max-prompt-length 4096 \
  --max-new-tokens-rewrite 4096 \
  --max-new-tokens-generate 4096 \
  --generation-batch-size 2 \
  --load-in-4bit
```

vLLM generation backend:

```bash
/workspace/venvs/torch-cu128-clean/bin/python3 /workspace/gemma4_style_rl_training/pipeline_v2/eval/phase_eval/patch_vllm_gemma4_keqv.py

/workspace/venvs/torch-cu128-clean/bin/python3 /workspace/gemma4_style_rl_training/pipeline_v2/eval/phase_eval/run_phase_eval_bundle.py \
  --eval-dir /workspace/gemma4_style_rl_training/data/phase_eval_v2 \
  --output-dir /workspace/gemma4_style_rl_training/outputs/pipeline_v2/phase_eval_stage01_vllm \
  --model unsloth/gemma-4-31B-it \
  --adapter-path /workspace/gemma4_style_rl_training/outputs/.../policy \
  --model-label stage01_vllm \
  --max-seq-length 8192 \
  --max-prompt-length 4096 \
  --max-new-tokens-rewrite 4096 \
  --max-new-tokens-generate 4096 \
  --generation-backend vllm \
  --generation-batch-size 4 \
  --vllm-max-num-seqs 4 \
  --vllm-kv-cache-dtype fp8 \
  --load-in-4bit
```

Small vLLM throughput matrix after a phase eval set has been prepared:

```bash
/workspace/venvs/torch-cu128-clean/bin/python3 /workspace/gemma4_style_rl_training/pipeline_v2/eval/phase_eval/benchmark_vllm_matrix.py \
  --eval-dir /workspace/gemma4_style_rl_training/data/phase_eval_v2 \
  --output-root /workspace/gemma4_style_rl_training/outputs/pipeline_v2/phase_eval_bench_stage01 \
  --adapter-path /workspace/gemma4_style_rl_training/outputs/.../policy \
  --variants bnb_b4,bnb_b8,nvfp4_b4,nvfp4_b8 \
  --rewrite-limit 4 \
  --generate-limit 0 \
  --max-new-tokens 1024 \
  --vllm-kv-cache-dtype fp8
```

For NVFP4 inference, use a model such as
`nvidia/Gemma-4-31B-IT-NVFP4` with `--vllm-quantization modelopt`.
This is an inference/evaluation optimization path; QLoRA training can still
use BNB 4-bit weights.

Generation stops at `</result>` by default and forces EOS for the row that
already emitted `</result>`, so batched evaluation does not add artificial
post-result text while waiting for slower rows. To inspect raw post-result
behavior, run `run_generation.py` directly with
`--no-force-eos-after-result-close`.

Optional corpus overrides:

```bash
  --ai-novel-dir /workspace/data/raw/ai_novels \
  --ai-control-dir /workspace/data/raw/heldout_ai_novels \
  --human-novel-dir /workspace/data/raw/human_novels \
  --generate-prompts /workspace/gemma4_style_rl_training/data/processed/grpo_generate_prompts.jsonl
```

For a command-only check without loading the model:

```bash
python3 gemma4_style_rl_training/pipeline_v2/eval/phase_eval/run_phase_eval_bundle.py \
  --dry-run \
  --output-dir gemma4_style_rl_training/outputs/pipeline_v2/phase_eval_dryrun \
  --adapter-path outputs/.../policy
```

The evaluator is independent of the legacy GUI modules. It duplicates the GUI
metric formulas in `pipeline_v2/eval/phase_eval/metrics_gui_compatible.py` and
records the raw metric values such as comma rates, POS n-gram repetition,
sentence edge repetition, sentence length CV, parenthesis rates, and modifier
concentration metrics.

Rewrite comparison groups:

- `human_control`
- `ai_source`
- `rewrite_output`

Generate comparison groups:

- `human_control`
- `ai_control`
- `generate_output`

The default generate set uses only `prompt_kind=new_writing` rows from
`data/processed/grpo_generate_prompts.jsonl`.

## 12. Standalone Fixed Eval And Scoring

To run only fixed eval for a given adapter:

```bash
/workspace/venvs/torch-cu128-clean/bin/python3 /workspace/gemma4_style_rl_training/pipeline_v2/eval/run_fixed_eval.py \
  --adapter-path /workspace/gemma4_style_rl_training/outputs/pipeline_v2/stage01_pilot_v1/stage01_cpt_lite/policy \
  --output /workspace/gemma4_style_rl_training/outputs/pipeline_v2/stage01_pilot_v1/eval_stage01/generations.jsonl \
  --model auto \
  --limit-per-dataset 4 \
  --max-new-tokens 2048 \
  --load-in-4bit
```

Then score:

```bash
python3 gemma4_style_rl_training/pipeline_v2/eval/score_outputs.py \
  --input gemma4_style_rl_training/outputs/pipeline_v2/stage01_pilot_v1/eval_stage01/generations.jsonl \
  --output gemma4_style_rl_training/outputs/pipeline_v2/stage01_pilot_v1/eval_stage01/scored_outputs.jsonl \
  --result-tag-policy auto
```

`--result-tag-policy auto` requires `<result>` tags only for prompts/tasks that
ask for them, such as `format_stop` and `continuation`.

## 13. Stage Order

After the data prep outputs look sane:

```text
stage00_fixed_eval on data/eval_v2
stage01_cpt_lite on cpt_mixed_probe/full
stage02_format_task_sft
stage03_simpo_curriculum:
  01_generate_format -> 02_nochange_anti_copy -> 03_badstyle_rewrite
stage04_grpo_generate
stage05_grpo_rewrite
stage06_final_polish
```

Keep the first CPT-lite run short. The first question is not whether the model
can memorize the corpus; it is whether the style prior improves while result
contract and instruction following remain intact.

## 14. Current Data Audit Notes

- `raw_lm` rows are cleaned body chunks; metadata, URL, ISBN/copyright lines,
  and replacement characters are filtered.
- `continuation_sft` rows are generated from raw chunks and must use
  assistant-only loss.
- `format_sft` rows are intentionally oversampled in small probe mixes because
  the clean source pool is small. Track this in manifests.
- SimPO splits have repeated chosen texts, especially `03_badstyle_rewrite`.
  Treat row count as preference weight, not as unique text count.
- Stage 1 cannot be trained safely with a generic chat SFT trainer unless it is
  row-type aware: `raw_lm` needs full-token loss, while SFT rows need
  assistant-only loss.
