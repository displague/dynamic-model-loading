# Per-layer omission and interaction protocol

Declared before new measurements on 2026-09-11. Planned release: v0.3.0. This follows
the original Stage 1 omission study and diagnoses the Stage 2 grouping loss. It does
not reopen the final test set or satisfy the multi-domain held-out quality gate.

## Fixed inputs and execution

Reuse the exact 96-document corpus and popularity/coactivation permutations from
`results/packing-pilot-20260911/pilot`. Verify its manifest, corpus, configuration,
layout hash in the raw calibration row, and passing summary. Verify the pinned
checkpoint file hashes and all token IDs against its manifest before measuring.
Pin the parent manifest and summary digests in the configuration as well as the raw
ledger, so edited metadata cannot silently change the reference checkpoint or results.
The parent ledger SHA256 is
`332d1de1177dc3eb33a87e94a7ea76a3ebbbc82c1dd3a409f54f2a05754ca66d`.

Use Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP32, eager Python instrumentation with
SDPA attention, batch one, 256-token prefixes, seed 1729, four CPU threads, TF32 off,
and the existing Python 3.14 / torch 2.10.0+cu130 environment. The layout calibration
is frozen; do not collect new importance profiles or fit new permutations.

Regenerate dense references locally. Require the same all-weight local grouped
reconstruction limit (relative L2 <= 0.01) and per-document full-model limits
(relative logit L2 <= 0.01 and mean KL <= 0.001) for both layouts before any omissions.
Any failure blocks all sparse probes and preserves its receipt.

## Conditions declared before measurement

For each popularity and coactivation layout, use groups of 32 neurons with 75% of
groups retained. Apply the existing summed-importance hindsight mask to:

1. Each of the 28 layers separately (56 conditions total).
2. All 28 layers together (two matched replication controls).
3. Four fixed contiguous blocks, layers 0-6, 7-13, 14-20, and 21-27 (eight conditions).

This yields 66 conditions, each evaluated on all 16 validation articles. For each
document preserve next-token KL, dense/candidate NLL, relative perplexity, top-1
agreement, and logit error. The 4,080 predicted-token denominator excludes the final
position; weight accounting includes all 4,096 input positions actually processed.
Unmasked FFNs count at full weight volume when reporting the whole-model FFN fraction.
Masking one layer by 25% is about 0.893% hypothetical FFN-weight omission overall,
not 25% model-weight savings. Attention and other fixed weights are not in that fraction.

## Analysis

Publish complete layer and block curves for both layouts, including all negative
differences. Rank individual-layer mean KL descriptively and report the top-four
share of the sum of individual-layer KL. That sum is not the joint-layer KL: report
both and their ratio without treating them as an additive causal decomposition.
The fixed-block experiments are interventions specified here, not groups chosen
after observing a favorable ranking. No learned layer budget is fitted in this run.

Compare the matched all-layer controls against the parent pilot without replacing
either run's rows. Recompute the predeclared paired whole-document bootstrap KL
difference for the all-layer control (2,000 resamples, seed 1729); it is a same-data
replication check, not independent evidence. Do not apply significance labels to
the 28 exploratory layer comparisons or infer generalization from the top-layer ranks.

Retain source/config/protocol/corpus/checkpoint/token/layout hashes, raw rows flushed
before summaries, and immutable exclusive output directories. Reference logits may
spill to local disk outside any timing claims. Save an error receipt on interruption
where possible. No transfer, sparse latency, bounded residency, detector, or predictor
claim follows from these masked dense computations.
