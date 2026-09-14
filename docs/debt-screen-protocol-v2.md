# Resident-base, feedback-directed correction acquisition: corrected screen 2

This is a fresh prospective accounting correction, not a selector retune. The
original screen at source 5c0ca048b8ea4dfc663a5d39cc0346b5779cb401 completed its
worker in 138.939 seconds but failed its frozen accounting analysis: measured
baseline allocation was 7,749,224,960 bytes versus 7,725,494,272 known CUDA parameter
bytes, exceeding the assumed 16 MiB margin. Its original protocol, source and raw
receipts remain unchanged and its overall decision remains inconclusive. All
three approximate controls recorded 0/64 accepted proposals; those observations
are known before this correction and are not used to change the candidate.

The only experimental change below is a stricter charge: subtract only the known
CUDA parameter bytes when computing the 768 MiB extra-inference allowance.
Every byte of baseline overhead is now included in that allowance. Record the
actual allocated baseline separately, without assuming a 16 MiB overhead bound.
Validate construction accounting before scored inference. Keep every policy,
workload, numerical tolerance, timing rule and economic criterion unchanged.
Run once in a fresh directory under the same 300-second worker deadline.


Prospective protocol. Freeze and push this protocol, configuration and harness
before inference. Implement within ADR 0004's approximate-draft / dense-verifier
contract and ADR 0006's five-minute screening boundary. ADR 0005 and all prior
negative results remain unchanged. No native worktree, build or patch is authorized
by this screen. Each attempt has a new directory; errors/timeouts are inconclusive.

## Hypothesis and distinction

The old pager omitted unselected FFNs and repeatedly streamed a working set too
large for its cache. This candidate instead computes **all** neurons using a
resident two-bit base and loads paired full-precision pages only as corrections.
For page p, the exact acquired correction is

`delta_p = D_p [silu(G_p x) * U_p x] - Dq_p [silu(Gq_p x) * Uq_p x]`.

Adding every page correction reconstructs the dense FFN up to the fixed numerical
tolerance. Skipping a correction is approximate; independent dense target
verification owns token commitment. CPU catalogues alias the original draft
weights; fetched pages contain full weights, not compressed residual matrices.

The research hypothesis is **feedback-directed output-error acquisition**:
a small resident sketch predicts each page's output correction. Let v_p be its
16-dimensional predicted correction and d=sum(v_p). Acquire the unacquired page
maximizing `2 dot(d,v_p) - dot(v_p,v_p)` (predicted reduction of squared residual
error), provided the maximum is positive. Subtract the *observed*, projected exact
correction from d, recompute priorities, and stop after four pages or when no
positive reduction is predicted. Ties choose the smallest page index. No target
activation, logit, future token or current dense draft output supplies the signal.
The sketch is an estimate, not an omission certificate or monotone-quality promise.

An unchanged-sketch top-four squared-norm ranking is the policy ablation; a
zero-fetch quantized draft is the representation-only control. Improvement versus
dense streaming alone does not establish a useful acquisition policy.

## Representation and frozen sketch

- Same HF Qwen2.5-1.5B-Instruct revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, root `.venv`, FP32 CUDA SDPA,
  four CPU threads, seed 20260914, TF32 off. Verify the earlier checkpoint manifest
  and all model artifact bytes. No downloads or environment changes in the run.
- Keep target dense on CUDA, independently load the draft with non-FFN parameters
  on CUDA and all source FFNs on CPU. No target/draft weight or KV sharing.
- Each gate/up matrix and transposed-down matrix is [8960,1536]. In each row,
  groups of 128 columns use affine min/max quantization to four levels. Scale is
  `(max-min)/3`; a constant group has zero scale and code zero. Rounding is torch
  round-to-nearest-even, clamped to 0..3. Pack four consecutive codes into a byte,
  low-index code in the low two bits. Minima and scales are FP32. This deliberately
  simple quantizer is an apparatus choice, not a competitive PTQ claim.
- Per layer, form an orthonormal input basis V by reduced QR of the transposed
  16 calibration medoids from the **unchanged** v0.20 index. No new inference or
  diagnostic data fits this basis. A single CPU-generated Rademacher output
  projection R of shape [1536,16], divided by sqrt(16), uses seed 20260914 and is
  then copied to CUDA. Freeze actual constructed tensors/hashes before scoring.
- Retain `(G-Gq)V`, `(U-Uq)V`, `D^T R`, `Dq^T R` and V on CUDA. Estimate gate/up
  residuals from xV, apply the nonlinear SwiGLU to the corrected estimates, and
  compute each page's projected difference against the resident-base contribution.
  The 35 page vectors are copied to CPU for deterministic Python-double selection
  using math.fsum. Observed delta_p R is copied after each acquisition; both D2H
  costs and all CPU control/serialization time are charged.
- Page width 256, full gate/up/down payload 4,718,592 bytes; one physical GPU page
  slot and one pinned staging page, released immediately after use. No prefetch or
  persistent full-precision page cache. Four is the maximum corrections per FFN.
- A shared preallocated FP32 three-matrix workspace and explicit uint8/FP32
  unpacking scratch materialize only the current layer. Charge the packed base,
  scale/min metadata, all sketches, workspace, scratch, pages and staging. The
  draft inference allowance is **768 MiB extra CUDA allocated above the known
  target-plus-draft-non-FFN parameter storage (7,725,494,272 bytes)**, including KV/logit/scratch overhead. This
  is a new declared budget, not a claimed improvement at the old 512 MiB budget.
  All four conditions keep the same constructed apparatus resident; unused
  base/sketch storage is still charged to streaming and zero-fetch controls.
  Report setup memory/weight transfers and construction wall time separately;
  setup may use additional scratch, still within joint resource limits.
  The actual pre-construction allocation must not be below the known parameter
  storage. Its entire excess is charged within the extra-inference allowance.
  Audit allocator current/peak allocated/reserved ordering, observed-baseline plus
  resident/workspace lower bounds, and exact logical target/draft KV peaks replayed
  from cache positions. Setup scratch remains separately reported.

## Short workload, controls and receipts

- Reuse the hashed corpus and token IDs in the configuration. First two diagnostic
  documents, first **8 prefix IDs**, **8 output tokens**, one repeat. These are
  previously seen research diagnostics, not held-out evaluation. All four draft
  conditions use the same dense verifier and unchanged four-proposal charging,
  independent KV rollback, EOS set (151645,151643), and reference comparison.
- Conditions: `dense_stream` (all 35 full pages, no base compute), `base` (quantized
  only, no sketch execution or fetch), `fixed` (top four sketch norms), `debt`
  (feedback-directed 0..4 corrections). Document 0 order is that list; document 1
  uses its reverse. Eight scored draft episodes plus two scalar dense references.
- Dense and each draft condition warm once on calibration document 0, prefix 2,
  output cap 4. Compare warmup outputs with their dense reference too. Reset KV
  and page/control state for every episode. No performance-based early stopping
  within the fixed subset; only safety, correctness or the wall deadline can abort.
- Capture first two FFN inputs per layer on diagnostic 0, grouped dense outputs
  and full-model two-position logits. Before scoring, test all-page *correction*
  completion on all 28 FFNs (per-position relative L2 <= .01) and both full-model
  positions (relative L2 <= .01; mean KL <= .001). Also report base/fixed/debt
  local reconstruction error on these 56 fixed inputs. Those local comparisons
  use dense-generated diagnostic inputs, not selectors trained on dense outputs;
  their errors are diagnostic, not generation acceptance or a passed quality gate.
- Retain all constructed packed/sketch tensors, numerical inputs/outputs, source
  snapshots and hashes. Record all page loads/releases and per-layer decisions,
  predicted correction vectors, acquired page IDs, observed projected corrections,
  and every proposal/verification/commit/KV record. Offline analysis independently
  replays page choice and stopping, checks byte charges and exact outputs, and
  reports totals. Archive failed attempts rather than repairing their receipts.
  Charge an outer monotonic episode interval enclosing generation and page/policy/
  round-ledger flushing. Retain inner generation timestamps, prefill/decode and
  target times, cleanup interval and wrapper overhead separately. Reconcile outer
  wall time exactly with its endpoints and timing-component sums within 1e-9 s
  absolute / 1e-12 relative tolerance; reject negative, nonfinite or overlapping
  intervals. Throughput uses that full charged interval, not an unverified total.
- Separate supervisor: fixed 300-second worker wait including imports, provenance,
  construction/serialization, numerical checks, warmups, inference and hashing.
  Preserve timeout/error status; require successful supervisor receipt <=301 seconds
  with <=1 second launch/reaping overhead. Analysis is separately timed. There is
  no automatic expanded-study launch. Joint NVML sampling every 200 ms: GPU used
  <=15,000 MiB, available host >=2,048 MiB, final fail-closed check.

## Advance/stop decision, before seeing any candidate result

All numerical, output, accounting, resource and inventory checks are mandatory.
On the two scored debt episodes combined, **all** the following must hold to be
eligible for a separately frozen expanded study:

1. Emitted accepted tokens / all proposed tokens >=50%.
2. Acceptance fraction is at least 10 percentage points above `base`.
3. Total verifier-charged wall seconds per committed token are at least 5% lower
   than `base` **and** at least 5% lower than `fixed`.
4. Fully charged H2D bytes per committed token are at least 10% below `dense_stream`
   and no higher than `fixed`. Include prefix and discarded proposal traffic.

Use aggregate denominators, not averages of episode ratios. Report every check,
all four controls, setup/warmup costs and raw records even if the candidate fails.
The debt policy must add value beyond quantization and static correction; a native
port is not an escape from failed economics. A negative screen does not disprove
all residual-loading designs. A pass is preliminary, not a 32B result or a reason
to change the native boundary without a separate protocol and justification.

Prediction: full correction reconstructs the reference; the sketch may reduce
local error but its uncertainty and physical acquisition overhead may prevent a
verified-throughput improvement over the zero-fetch base. No parameters will be
retuned on these screen results.

## Bounded antecedent check (2026-09-14)

[DecDEC, originally QDEC](https://arxiv.org/html/2412.20185v2) already selects
input channels by activation magnitude and transfers quantized weight residuals
from CPU, with a fixed channel count and an overlapping CUDA implementation.
Resident low-bit weights plus residual acquisition are therefore **not new**.
Our hypothesis is different: paired nonlinear FFN-page correction selected by
predicted output-error cancellation and updated using the actual acquired
correction. The fixed-sketch and no-correction ablations test that distinction;
custom code alone does not establish global novelty.

[HCInfer](https://arxiv.org/abs/2605.05819v1) uses a compressed GPU backbone with
CPU residual compensation and adaptive rank; [SubSpec](https://arxiv.org/abs/2509.18344v2)
uses low-bit substitutes for offloaded layers and shares resident model state.
This experiment does not claim either idea or their reported gains, does not
share state, and performs correction arithmetic on GPU after synchronous fetches.
This is a bounded antecedent check, not an exhaustive novelty review.
