# Are the failed causal selections cheaply repairable?

Prospective analytical refinement study after v0.7.0. Commit and push this protocol,
configuration and apparatus before new pretrained measurements. The dependency
change is recorded in [ADR 0001](adr/0001-selection-and-refinement-gates.md).
No v0.7 file, threshold, raw receipt or nomination is replaced.

## Scope and frozen sample

Use the pinned Qwen2.5-1.5B-Instruct revision and original FP32 environment: Python
3.14.3, torch 2.10.0+cu130, Transformers 5.13.1, SDPA, TF32 disabled, four CPU threads,
seed 1729. Require the v0.7 archive index, exact parent source/input/checkpoint hashes,
selected trace hashes, learned tensors and dense references. Use exactly the first
two development articles in parent corpus order, first 128 token IDs each: 256 input
visits and 254 scored predictions. This is a deliberately bounded reused-data slice,
chosen before these repair measurements; no claim covers all 16 articles.

Keep popularity packing, width 8, 1,120 groups per layer, the 1,008-group initial masks
and 2 GiB equal-layer static FFN cache. Use each of static, recency, EMA and learned.
Replay its saved v0.7 masks as fixed starting points. Corrections change downstream
inputs while these initial masks remain fixed. This is a counterfactual diagnostic,
not a causal corrected-policy rollout or a refit of the selector.

## Controls before repair curves

Save fresh native incremental logits and compare them with the archived prefix
references. Validate every layer's width-8 reconstruction on two calibration FFN
inputs. Validate the all-group popularity path on both articles. Original numerical
limits remain relative L2 <=0.01 and mean KL <=0.001.

Reexecute all four original causal selectors on both prefixes; require their masks
to equal the saved mask prefixes exactly. Preserve their new masks, reference logits
and explicit reproduction receipts before aborting on any mismatch.
Require fixed-mask zero-addition outputs to pass the original numerical gates against
the corresponding fresh causal outputs. Require additive restoration of every
omitted group to pass against native dense outputs for every initial selector/article.
All controls must complete successfully before any repair curve is measured. A
failure stops the diagnostic and retains its complete ledger in a fresh run directory.

## Privileged corrective additions

For every initial selector, run both schedules at additional-group limits
**0, 8, 28, 56, 84, 112**, on both articles. The full grid has 96 document rows.

* Hindsight-ranked: among the initial 112 omissions, add up to the limit in descending
  current abs(z)*down-column-norm group score, breaking ties by packed group index.
* Resident-first: first add every initially omitted group already in the protected
  static hot set; then add up to the limit among remaining cold omissions using the
  same privileged ranking. A zero cold limit still executes resident repairs.

Preserve the current FFN input and its activations. Compute the initial down-projection
contribution and the disjoint corrective contribution separately, then add before
the residual/downstream layer consumes the output. Do not recompute the initial
contribution as part of the correction. The diagnostic can compute dense gate/up and
dense audit output; all of that privileged work is disclosed and is not a claimed
saving. Group-score ranking is a heuristic, not minimization of the omitted output
vector and not an optimal oracle. Recovery need not be monotone.

Archive initial, corrective and final applied bits, omitted-group scores sufficient
to reproduce every ranking decision, initial/corrected local output errors, and
resident/cold added counts. Save per-document PPL/KL/top-1, original denominators,
and token-weighted aggregates. Audit norms remain descriptive, not safety labels.

## Comparisons and accounting

Run the four original causal selectors afresh at one-shot retention fractions
**0.9, 0.925, 0.95, 0.975, 1.0**, both articles: 40 document rows. Their state updates
follow their own approximate paths. Report actual rounded group counts and storage.
The 100% point is the bounded dense-fallback cost/quality endpoint, not a learned
fallback trigger. Initial one-shot controls and complete repaired outputs stay separate.

Charge original selector persistent bytes even though historical masks are replayed.
Repair schedules additionally reserve their own per-layer FP32 down-column norms
and boolean resident maps: 1,034,880 bytes in this model. Recompute the protected hot
set after subtracting those bytes from 2 GiB and reserving one incoming group. The
fresh one-shot comparisons charge their actual selector storage. Compare all paths
with the strongest dense static parent-layout/width baseline given the full 2 GiB.

Replay final union masks with this static policy; also report initial demand, added
cold demand, resident repairs, selected computation volume, and cold preload separately.
Resident additions have zero new weight-transfer demand but nonzero computation.
Predictor initialization, historical trace storage and dense auditor buffers are
explicitly outside this FFN-cache simulation; publish their tensor sizes separately.
This is not bounded process memory, actual transfer, or a runtime measurement.

## Decision and next experiment

Retain relative PPL <=1.01 and >=10% warm simulated savings as a **diagnostic
quality/traffic screen on the completed output**. Report every point and compare
against larger one-shot subsets without post-score row replacement. Hindsight-assisted
points are always ineligible for physical paging. No resident-FFN timing threshold
is applied to this deliberately dense privileged diagnostic.

If a repair schedule recovers quality within the byte allowance, it identifies an
opportunity for a separate prospective causal experiment. That experiment must combine
current input, execution history/change signals, residency and explicitly charged
probes; include safe-to-omit abstention, larger initial subsets and dense fallback.
Initial outputs may fail; the complete causal policy must qualify. If the tested
repairs consume the headroom, report that negative result without declaring all
sequential computation impossible. A separate bounded hardware-cost and second-model/
new-development control may proceed regardless. No final held-out evaluation occurs.

CPU fixtures validate selection, disjoint additive repair, state/hook cleanup,
provenance and corrupt receipts. They are never pretrained or CUDA evidence. The
measurement CLI requires CUDA and the pinned configuration, without fallback.
Bind configuration, protocol and every source file to the clean worktree that owns
the imported apparatus, regardless of the launch directory. Analysis checks archived
bytes against that commit's Git objects, so the repository history is a required
analysis input in addition to the restored parent assets.
