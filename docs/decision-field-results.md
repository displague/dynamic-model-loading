# Decision-field screen: additive effects without a useful acquisition action

The v0.25 screen completed in **118.240 seconds**: 110.121 seconds for the
supervised worker and 8.119 for independent CPU analysis. Source and protocol were
reviewed and pushed at `1c67774f63ac6ee1be4eeeaf620d4d7d9b415c69` before inference.
[Protocol](decision-field-protocol.md), [inspirations and course](information-research-course.md),
[bounded issue #40](https://github.com/displague/dynamic-model-loading/issues/40).

## What was measured

Four-bit draft decode after four eight-bit prefill tokens; four separated layer
sites [1,8,15,22], promoted singly and in all six pairs, plus base and same-history
all-eight controls. Six fit-development and two diagnostic-development documents,
four teacher-forced positions each: 384 full-model intervention branches. No
predictor was fitted. The independent FP32 target supplied audit logits only.

| Result | Fit development | Diagnostic development |
| --- | ---: | ---: |
| Positions | 24 | 8 |
| Base agrees with all-eight / target | 22 | 7 |
| All-eight agrees with target | 24 | 8 |
| Any singleton/pair repairs base disagreement | 0 | 0 |
| Additive prediction agrees with actual pair argmax | 144/144 | 48/48 |
| Aggregate interaction norm / joint-change norm | 6.033% | 6.836% |
| Mean pair full-logit relative-L2 | .002715 | .003601 |

**Hdecision fails; Hjoint passes.** The one diagnostic disagreement is document
`wikitext-validation-2c7c602b5ac557ff`, scored position 2 (seventh consumed corpus
token). Base and every singleton/pair select ID 2016; same-history all-eight and
FP32 target select 14682. The experiment does not show which of the other layers,
larger action sets or interactions would repair that decision.

The positive component is narrow: pair effects were sufficiently additive to
preserve these argmaxes. It does not establish calibrated uncertainty, spatial
dependence, information value, or that additive predictions remain correct near
other decision boundaries. Seven of eight diagnostic positions already agree;
argmax identity is insensitive when margins are large. Correlated positions and
only two documents do not supply an independent-trials significance claim.

## Physical cost and integrity

- All 28 eight-bit numerical outputs exactly match the frozen parent (rel-L2 0).
  All 28 newly constructed packed layers match the parent representation exactly.
- 4,664 actual cold slab loads, 48,141,434,880 H2D bytes, no hits. This includes
  mechanics, prefill and privileged observations. It is research acquisition cost,
  not traffic saved by a policy. A single site promotion costs 20,643,840 bytes.
- Branch wall-time subtotal: 58.147 seconds. Setup, prefill, target audit, storage,
  fingerprints and hashing are additionally included in the worker clock.
- Separate target/draft non-FFN GPU parameter baseline: 7,725,494,272 bytes.
  Resident base/scales: 650,280,960; workspace: 254,607,360; two GPU slabs:
  20,643,840; pinned staging: 10,321,920 bytes.
- Original draft host FFNs: 4,624,220,160 bytes; packed host increments:
  578,027,520 bytes. Python-container metadata: 51,450 bytes (not all process
  overhead); sampled peak RSS: 10,640,404,480 bytes.
- Peak extra CUDA: 959,768,576 bytes (**915.307 MiB**), below the 1024 MiB cap.
  Sampled peak total GPU: 9,325,088,768 bytes; minimum available host:
  6,183,895,040 bytes, across 488 samples.
- Every branch preserves actual prior K/V bytes. Cropped logical KV length and
  underlying retained storage are reported separately; crop is not assumed to
  free the just-created suffix. All scalar, logit and fingerprint readbacks are
  explicitly charged and audited.

There is **no generated acceptance result, speedup, capacity gain, native-pivot
evidence or deployable runtime**. All-eight uses the existing four-bit decode
history; it is not an independently recomputed eight-bit KV trajectory.

## Next question

Stop expansion of this four-region acquisition action set. The next protocol will
test a distinct, finer acquisition/observation geometry near the output decision,
where a paid page observation may avoid a whole-model counterfactual replay.
This changes the action and observation, not Hdecision's published threshold.
Spatial and recursive hypotheses remain open; there is no reason yet to fit a
more elaborate predictor over this failed action set. Milestone 10 stays open.

The raw archive retains complete logits, phase/page/KV receipts, source and fixed
inputs. Compact machine-readable findings are in
[`results/decision-field-20260915`](../results/decision-field-20260915/).
A second CPU-only audit exactly reproduces the complete summary in 22.475 seconds;
no checkpoint inference was repeated. Archive restoration checks every member's
SHA256 and rechecks the raw source bytes.
