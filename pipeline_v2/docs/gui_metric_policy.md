# GUI-Style Metric Policy

Date: 2026-06-01

This document records which GUI-style metrics should be used as training/eval
signals, which should be kept at low weight, and which should not be used as
direct optimization targets.

The classification is based on the current audit output:

- `outputs/pipeline_v2/gui_metric_audit/summary.json`
- `outputs/pipeline_v2/gui_metric_audit/scalar_metric_separation.csv`
- `outputs/pipeline_v2/gui_metric_audit/pos_ngram_usage_separation.csv`
- `outputs/pipeline_v2/gui_metric_audit/reward_sample_component_summary.csv`

## Decision Rules

Use a metric strongly only when most of these are true:

- It separates human and AI text clearly in the reference data.
- It is not mostly a proxy for output length or result-format failure.
- It is hard to game with a trivial local hack.
- It remains meaningful across generate and rewrite tasks.
- It has a clear failure direction.

Keep a metric at low weight when:

- It has some separation but overlaps heavily with other stronger metrics.
- It is noisy for short samples.
- It is useful mainly as a guardrail or diagnostics dimension.

Do not use a metric as a direct reward when:

- It is low-signal in the current human-vs-AI reference.
- It is too sparse or easy to game.
- It encourages degenerate behavior.
- It duplicates a stronger metric but adds noise.

Important nuance:

- High AI-in-human-q10-q90 rate does not mean there is no distribution shift.
- It only means most AI samples are not extreme outliers under the human
  reference.
- A metric can still be useful as a soft distributional regularizer when the
  mean/median shift, Mann-Whitney test, KS test, and effect size are consistent.
- Such metrics should use low family-level weight and band/center scoring, not
  hard monotonic optimization.

## Use As Primary Signals

These are strong enough to use as core reward/eval axes.

### Anti-Slop Density

Use: yes, high priority.

Current audit:

- Cliff's delta: `0.924`
- human mean: `1.48`
- AI mean: `8.06`
- AI-in-human-IQR rate: `0.058`

Interpretation:

- This is the strongest current human-vs-AI separator.
- AI text contains the anti-slop lexicon far more densely than human text.

Training use:

- Keep as a primary reward component.
- Also useful for SFT/UL candidate mining.
- Do not raise weight so high that the model simply avoids varied descriptive
  phrases altogether.

### Translationese SVM

Use: yes, high priority.

Current audit:

- Cliff's delta: `0.875`
- human mean: `0.606`
- AI mean: `-0.441`
- AI-in-human-IQR rate: `0.035`

Interpretation:

- Strong global detector of AI/translation-like prose.
- It catches distributional awkwardness that lexical penalties miss.

Training use:

- Keep as a primary reward/eval axis.
- Use as a scalar text-level signal, not token-local supervision yet.
- Validate on held-out source families before increasing weight.

### Comma Overuse Metrics

Use: yes, medium-high priority.

Metrics:

- `comma_per_1k_chars`
- `comma_sentence_rate`

Current audit:

- `comma_per_1k_chars`
  - Cliff's delta: `0.747`
  - human mean: `5.47`
  - AI mean: `10.69`
- `comma_sentence_rate`
  - Cliff's delta: `0.659`
  - human mean: `0.137`
  - AI mean: `0.261`

Interpretation:

- Current AI text overuses comma-like segmentation.
- This is easy to understand and useful.

Training use:

- Use as a medium-high reward axis.
- Avoid making it a hard ban. Some styles legitimately use commas.
- Prefer band scoring against human ranges, not monotonic minimization.

### POS 4-6gram Diversity And Repeat

Use: yes, medium-high priority.

Metrics:

- `pos_4gram_diversity`
- `pos_5gram_diversity`
- `pos_6gram_diversity`
- `pos_4gram_repeat_rate`
- `pos_5gram_repeat_rate`
- `pos_6gram_repeat_rate`

Current audit:

- Human text has higher high-order POS diversity.
- AI text has higher high-order POS repetition.
- Useful separation range: Cliff's delta roughly `0.60-0.66`.

Interpretation:

- These capture repetitive syntactic templates.
- They are more meaningful than POS 1-2gram scalar metrics.

Training use:

- Use as core structure metrics.
- Prefer aggregate family scoring instead of many independent large weights.
- Keep 5-6gram reward weight moderate because sparse high-order patterns can be
  noisy on short samples.

## Use At Low Weight

These can stay in evaluation and low-weight reward, but should not dominate.

### POS 3gram Diversity And Repeat

Use: low-to-medium.

Current audit:

- `pos_3gram_diversity`: weak but usable.
- `pos_3gram_repeat_rate`: weak but usable.

Interpretation:

- Meaningful, but less discriminative than 4-6gram.
- Useful as a smoother companion to high-order POS metrics.

Training use:

- Keep as low/medium weight.
- Do not duplicate too much weight with 4-6gram metrics.

### POS N-Gram Usage Distribution

Use: low-to-medium.

Current audit JS distances:

- 1gram: `0.081`
- 2gram: `0.145`
- 3gram: `0.226`
- 4gram: `0.350`
- 5gram: `0.511`

Interpretation:

- Higher-order usage distributions distinguish human and AI better.
- 5gram is informative but sparse.

Training use:

- Keep as a low/medium distributional alignment term.
- Use 3-5gram more than 1-2gram.
- Do not let 5gram dominate; it can become brittle.

### Modifier Shape Metrics

Use: low weight, mostly diagnostic.

Metrics:

- `content_modifier_simpson_concentration`
- `content_modifier_yule_k`
- `content_modifier_gini_frequency`
- `content_modifier_repeat_occurrence_rate`

Current audit:

- Separation exists but is weaker/noisier than anti-slop, translationese, comma,
  and POS 4-6gram metrics.

Interpretation:

- These may reflect repetitive modifier usage, but overlap with anti-slop and
  POS recurrence.
- They can be unstable by genre, speaker density, and scene type.

Training use:

- Keep in reports.
- If used in reward, use small family-level weight.
- Do not optimize each modifier metric independently with high weights.

### Sentence Initial POS Bigram Repeat

Use: low-to-medium weight.

Current audit:

- Weak but usable.
- human mean: `0.707`
- AI mean: `0.773`
- Cliff's delta: `0.561`
- AI-in-human-IQR rate: `0.407`

Interpretation:

- Can detect repeated sentence-start shapes.
- Scene style and dialogue density can shift it.
- This is not meaningless. It has a moderate distribution difference, but the
  overlap is still large enough that it should not dominate training.

Training use:

- Use as a light distributional regularizer.
- Keep below the main POS 4-6gram diversity/repeat family.
- Prefer family-level scoring with sentence-edge metrics rather than a large
  standalone weight.

### Result Contract And Collapse Checks

Use: yes, but as hard task/format gates, not style metrics.

Metrics/checks:

- missing `<result>`
- missing `</result>`
- post-`</result>` text
- thought/meta leakage
- empty/short result
- repetition collapse

Interpretation:

- These are not human-vs-AI style metrics.
- They are task validity checks.

Training use:

- Keep as hard penalties or filtering gates.
- Do not mix them into conclusions about prose style.
- Track separately from GUI-style score.

## Do Not Use As Direct Reward Targets

These should not be direct optimization targets in the current pipeline.

### POS 1-2gram Scalar Metrics

Use: no direct reward.

Examples:

- `pos_2gram_diversity`
- `pos_2gram_repeat_rate`

Current audit:

- Low or weak/redundant signal.
- Large overlap between AI and human ranges.

Reason:

- Too shallow.
- Easy to satisfy while prose remains AI-like.
- Duplicates stronger POS 4-6gram signals.

Allowed use:

- Diagnostics only.

### Sentence Length CV

Use: low-weight distributional regularizer only if needed.

Current audit:

- Weak/redundant as a reward target, not meaningless as a diagnostic.
- human mean: `0.709`
- AI mean: `0.649`
- Cliff's delta: `0.284`
- AI-in-human-IQR rate: `0.616`
- AI-in-human-q10-q90 rate: `0.884`

Additional one-chunk-per-file probe:

- AI vs human mean: `0.650` vs `0.705`
- AI vs human median: `0.642` vs `0.688`
- Mann-Whitney p: `0.011`
- KS p: `0.004`
- Cohen's d: `-0.374`
- AI-in-human-IQR rate: `0.535`

Reason:

- Too genre- and scene-dependent.
- Can be gamed by alternating short and long sentences.
- Does not reliably imply human-like prose.
- The current reference suggests human chunks have slightly higher sentence
  length variation, but most AI chunks still fall inside the broad human range.

Allowed use:

- Diagnostics by default.
- If repeated eval shows the model is too rhythmically flat, allow a very small
  soft reward that nudges toward the human median/IQR.
- Do not use monotonic "increase sentence length CV" optimization.

### Sentence Final POS Bigram Repeat

Use: no direct reward.

Current audit:

- Low signal.
- human mean: `0.862`
- AI mean: `0.877`
- Cliff's delta: `0.148`
- AI-in-human-IQR rate: `0.605`

Reason:

- Weak separation.
- Overlaps with high-order POS recurrence but adds noise.

Allowed use:

- Diagnostics only.

### Sentence Initial Token Repeat

Use: no direct reward.

Current audit:

- Low signal.
- human mean: `0.388`
- AI mean: `0.424`
- Cliff's delta: `0.209`
- AI-in-human-IQR rate: `0.593`

Reason:

- Surface token starts are heavily affected by dialogue, names, and action beats.
- Strong optimization could discourage legitimate repeated scene rhythm.

Allowed use:

- Diagnostics only.

### Parenthesis Metrics

Use: no direct reward by default.

Reason:

- The current reward reference did not keep parenthesis metrics as meaningful
  scalar metrics.
- Parenthesis usage is sparse and genre/source-format dependent.
- Direct optimization can produce unnatural avoidance.

Allowed use:

- Diagnostics.
- Source-cleaning artifact checks.

### Individual High-Order POS N-Gram Targets

Use: no direct target matching.

Reason:

- Matching specific high-order POS n-grams can be gamed and can reduce variety.
- Sparse n-grams are better used through distributional distance, not as exact
  phrase-like targets.

Allowed use:

- Aggregate JS-distance style scoring.
- Human-vs-AI audit reports.

## Recommended Reward Weight Shape

For future GUI-style reward revisions, use family-level weights rather than many
independent metric weights.

Suggested starting shape:

```text
anti_slop: 0.35-0.45
translationese: 0.25-0.35
comma_overuse: 0.15-0.25
pos_4_6_diversity_repeat: 0.25-0.40
pos_usage_distribution: 0.10-0.20
modifier_shape: 0.05-0.10
sentence_edge: 0.00-0.05
result_contract: hard penalty / gate
collapse: hard penalty / gate
```

Important:

- These are not all normalized the same way in current code. Treat the ranges as
  relative priorities, not literal final CLI weights.
- Do not increase style reward complexity until fixed eval shows stable result
  contract and generation termination.

## Current Sample Diagnosis

Audit target:

- `outputs/202605301522/gui_style_reward_samples (1).jsonl`

Summary:

- rows: `54`
- rewrite: `27`
- generate: `27`
- post-`</result>` rows: `9 / 54 = 16.7%`
- truncated rows: `24 / 54 = 44.4%`
- mean reward score: `-0.415`
- negative reward rate: `98.1%`
- translationese mean: `-0.984`
- anti-slop mean: `-0.809`
- GUI POS score mean: `-0.138`

Interpretation:

- The run is not mainly failing on one subtle style dimension.
- Termination/length problems are still substantial.
- Among style dimensions, anti-slop, translationese, comma behavior, and
  high-order POS repetition/diversity are the meaningful targets.
