# Output-margin sensors: a measured acquisition opportunity, still oracle-assisted

v0.26 passes its two prospective component gates in **145.072 seconds**:
136.362 seconds supervised worker plus 8.711 seconds independent CPU analysis.
Reviewed source was pushed before inference at
`aa8f0d9071b87aac290320dd176432b81ea657e9`.
[Protocol](output-sensors-protocol.md), [inspirations](information-research-course.md),
[issue #41](https://github.com/displague/dynamic-model-loading/issues/41).

## What changed and why it matters

v0.25's four separated whole-layer actions offered no repair opportunity. This
screen instead uses 35 neuron pages in the final FFN, with a deliberately two-bit
resident base and three physical precision increments. Earlier FFNs remain
six-bit during decode. The controlled low-precision region creates a test of
selective acquisition; it is not a claim about a natural deployment baseline.

For this model's gain-only RMSNorm and bias-free readout, a page's correction can
be projected onto a two-token margin. A positive common divisor cannot change
that pair's ordering. The screen validates this application of the RMSNorm
formula, then checks actual full-vocabulary readouts and physical partial loads.
No statistical controller was fitted; no unfetched page was predicted yet.

## Decision headroom

Six fit-development documents supply 96 positions; two reused diagnostic documents
supply 32. All positions are teacher-forced, not generated accepted-prefix tests.

| Diagnostic condition | Matches high-eight final-region draft | Matches FP32 target |
| --- | ---: | ---: |
| Two-bit final-region base | 23/32 | 22/32 |
| Fixed first 17 pages | 28/32 | 29/32 |
| Privileged oracle 17 pages | 30/32 | 31/32 |
| All 35 pages / high-eight | 32/32 | 31/32 |

The oracle repairs **all nine** base/high disagreements; the fixed set repairs
seven. Both introduce **two new disagreements** at positions where base agreed
with high. These are disagreements with the high-precision *draft*, not always
with the target: one change away from high improves target agreement. More
agreement with the draft reference is not automatically more target acceptance.

On diagnostic document 0, base/high differ at seven positions: oracle repairs
seven, fixed five; each introduces one new difference. On document 1, both repair
the two differences and introduce one new difference. Fit-development has 23
base/high differences: oracle repairs 23, fixed ten; each adds one new difference.

**Haction passes.** But the oracle sees every page before choosing its 17, and
uses the high-eight winner. It is not a deployable or causal loading policy. The
next question is whether a few paid observations can identify useful unobserved
pages. The result justifies that short estimation screen, not a long matrix.

## Observation and numerical checks

- All 4,480 projected page margins pass the frozen numerical tolerance.
  Maximum diagnostic absolute error is **4.828e-6**; fit maximum **5.245e-6**.
- All 28 high-eight FFN controls pass against the pinned parent; maximum rel-L2
  **1.882e-7**. The extended integer planes exactly reproduce the parent grid.
- Summing page contributions reconstructs canonical final FFN outputs: maximum
  diagnostic rel-L2 **1.075e-6**, fit **1.373e-6**. Actual fixed/oracle reloads
  are independently checked against those contributions.
- All eight full-model high-eight replays match the direct canonical readout's
  argmax and numerical gate. Entire post-step KV fingerprints remain unchanged;
  prefill and consecutive-frame histories are linked, not merely checked locally.
- Mean paid page observation: **3.532 ms**. Mean full-model high-eight replay:
  **374.273 ms**. **Hobservation passes**, but these are different amounts of
  work/precision; do not call their ratio an inference speedup or an equal-work
  counterfactual speedup. No paired single-page full-model timing was measured.
- The high-eight winner lies outside the base top-two at **3/32** diagnostic
  positions (11/96 fit). A pairwise margin is not a full-argmax certificate.

## Fully charged research work

A page contains **884,736 bytes** (0.84375 MiB) of increments. Seventeen pages
cost 15,040,512 bytes (14.34375 MiB); all 35 cost 30,965,760 (29.53125 MiB).
These are per-action payloads, **not achieved traffic savings**: the collection
loads all observations, then physically reloads both partial controls.

- Earlier-layer cache: 5,454 cold loads / **56,295,751,680 H2D bytes**.
- Final-page cache: 30,801 cold plane loads / **9,083,584,512 H2D bytes**.
- Packed construction H2D: 660,602,880 bytes; input IDs: 1,280 bytes.
  Recorded D2H, including construction checks, vectors, logits, scalars and KV:
  **1,448,081,720 bytes**. Model initialization is additionally inside the worker
  clock; page/copy counters are not a hardware trace of every driver operation.
- Aggregate fixed reload/readout: **7.498 s**; oracle reload/readout: **7.674 s**;
  oracle selection: **.102 s**. Base draft forwards: 36.556 s; target audit:
  3.716 s; canonical high readouts: .292 s. These are research components, not
  a verifier-charged generated throughput comparison.

Actual target plus draft non-FFN parameter storage is **7,725,494,272 bytes**.
Resident base/scales are 639,959,040; workspace 254,607,360; layer pool 20,643,840;
page pool 589,824; pinned staging 10,321,920 + 294,912 bytes. Host charges include
original FP32 draft FFNs 4,624,220,160, original increments 578,027,520, moved
four-bit final base 20,643,840, and new host plane 10,321,920 bytes.

Peak extra CUDA is **907.302 MiB**, below 1024 MiB. Across 686 resource samples,
peak total GPU is 9,325,088,768 bytes, peak process RSS 10,026,598,400, and minimum
available host 6,382,993,408 bytes. The final-region base itself uses 10,321,920
fewer GPU bytes than its former four-bit base, but the sampled total GPU peak is
unchanged from v0.25. **No larger-model capacity frontier was established.**

A second model-free audit exactly reproduces the complete summary in **8.697 s**.
All raw vectors/logits, source, integer planes, receipts and licensing are in the
restoration-verified archive; [compact receipts](../results/output-sensors-20260915/)
retain individual positions and both gate verdicts.

## Next experiment

Both component gates pass. Freeze a short *page-field* estimation experiment:
observe a small paid subset, predict only unobserved decision-sensitive effects,
compare structured-plus-independent uncertainty with independent and shuffled
geometry controls. This is where the spatial-Bayes hypothesis can now be tested.
No online loader, generated acceptance gain, total inference speedup, native
pivot or deployment qualification is established. Milestone 10 remains open.
