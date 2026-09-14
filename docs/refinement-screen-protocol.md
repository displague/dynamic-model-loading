# Progressive precision and persistent acquisition: short screen

Prospective protocol, 2026-09-14. Implements the owner's request to test novel
loading hypotheses against the limitations of v0.22, under ADRs 0004 and 0006.
This is not a new native-admission contract and authorizes no llama.cpp changes.

## Questions and departures

The previous two-bit base was unusable as a draft, corrections were full FP32
pages, and no acquisition state survived a token. This experiment instead uses
an embedded 4/6/8-bit representation, physically transfers only additional bits,
and retains useful increments across consumed tokens. All neurons still execute.

Successive refinement is an inspiration, not a novelty claim. Closely related
primary work includes [Any-Precision LLM](https://arxiv.org/abs/2402.10517),
[BitStack](https://arxiv.org/abs/2410.23918),
[AnyBCQ](https://arxiv.org/abs/2510.10467),
[PMPD](https://arxiv.org/html/2410.13461v2), and
[DecDEC](https://arxiv.org/html/2412.20185v2). The experimental departure here is
the combination of an observed inter-precision FFN-output change as an acquisition
signal and a bounded, cross-token admission policy based on prior correction value.
No claim of priority, optimal quantization, or certified neural error is made.

The owner supplied PMPD during implementation, before protocol freeze or model
measurement. It studies phase-aware precision and progressively lowering precision
through generation, using static or prompt-conditioned schedules. Here refinement
goes upward within an FFN execution, and a separate physical cache retains increments
across tokens. PMPD's switching-overhead warning motivates charging all recomputation
and copies. Its high-precision-prefill insight is an important alternative, not
tested by this screen's uniform prefill/decode policy. See the full
[inspiration and departure ledger](refinement-inspirations.md).

## Frozen representation and controller

Use the unchanged local Qwen2.5-1.5B-Instruct snapshot
989aa7980e4cf806f80c7fef2b1adb7bc71aa306 and baseline .venv. All 28 FFNs use
FP32 reference weights, with down transposed into neuron-major form. In each
row's group of 128 weights, encode nearest-even affine eight-bit values 0..255
using minimum and (maximum-minimum)/255. Constant groups reconstruct exactly.
The resident upper four bits reconstruct at the bin midpoint (16*q4+7.5).
Acquire bits 2..3 to add (4*q2-6), then bits 0..1 to add (q2-1.5), all multiplied
by the original scale. Eight bits reproduce that affine eight-bit quantizer,
not FP32. Each refinement slab contains all three projections for one layer:
3*8960*1536/4 = 10,321,920 packed bytes. Scales/minima are resident and shared.

Six conditions, no sweep or post-result replacement:

- `q4`: resident four-bit base only.
- `q6`: always acquire first increment, immediately reusable scratch only.
- `q8`: always acquire both increments, scratch only.
- `adaptive`: always reach six bits; acquire eight bits when
  norm(y6-y4)/(4*max(norm(y6),1e-12)) > 0.02. Scratch only.
- `retained`: identical precision decisions and arithmetic to adaptive, with
  eight persistent slab slots plus the same bypass scratch slot.
- `static`: same adaptive precision and eight slots, permanently admit the first
  increment for layers 0..7, including the first token. No learned ranking.

All conditions compute y4; q6/q8 also compute intermediate outputs. This exposes
controller cost rather than pretending it is free. The geometric error proxy is
heuristic; compare it to norm(y8-y6)/max(norm(y8),1e-12) whenever available.
It is not a downstream/logit bound or a reason to skip dense verification.

For retained, before each consumed draft token choose the eight highest positive
scores (ties: layer, then stage) from prior tokens. Evict others, admit only those
keys, bypass unadmitted loads. At token start decay all scores by 0.5. After each
actual refinement add 0.5*norm(y_new-y_old)/max(norm(y_new),1e-12) to that key.
No target signal enters selection. Cold scores are zero. Refinements and scores
from rejected speculative work remain: they describe paid work, not accepted KV.
KV rollback remains the existing exact-verification boundary. This state may
affect cache hits only, never numerical output or refinement decisions. Reset
cache, scores and counters between documents/conditions, not within a document.

## Bounded workload and controls

Freeze and push implementation/configuration before checkpoint inference. Reuse
the existing token/corpus artifacts from fault-pager-20260914-v1 with their
previous SHA256 identities, first calibration document and first two diagnostic
documents. They are known development data, not held-out confirmation.
Warmup: two prefix tokens and one committed output per condition. Scored: four
prefix tokens, four committed outputs per condition/document, K=4. Reverse the
six-condition order on the second document. One repetition, no automatic retry.
Keep scalar dense target-only references and actual accepted-prefix records.

Before scored generation, save constructed representations and their hashes.
Construct the representation on CPU; charge its host peak and base upload. Snapshot
readback is separately charged and timed, and is not scored serving traffic.
Check each layer's eight-bit reconstruction against an independently computed
affine quantizer on one frozen diagnostic input; relative L2 <=0.01. Retain local
FP32/4/6/8-bit outputs for representation diagnostics, not acquisition signals.
CPU tests require exact code nesting, constant groups, decreasing quantization
interval widths, cache admission/eviction/bypass accounting, rejection handling,
and dense-reference committed IDs. Quantized-model logits need not equal FP32.

Worker deadline 300 seconds includes imports, loading, construction, numerical
checks, warmups, scored inference, snapshots and hashing. Model-free independent
analysis is timed separately. Timeout, mismatch, missing receipts or resource
failure means inconclusive. Never start a long matrix automatically.

## Resources and independent replay

All modes allocate the same base, scratch and nine slab slots. Extra CUDA above
the exact target parameters plus CUDA non-FFN draft parameters must stay <=1024
MiB, including all baseline overhead, KV and temporaries. This is a prospectively
different allowance from v0.22, not a win at its old 768 MiB budget. Enforce total
GPU <=15000 MiB and available host >=2048 MiB. Report allocator allocated/reserved
peaks, sampled total GPU and process/host usage, resident tensors, original host
weights, refinement backing store, pinned staging and controller metadata.
Charge construction H2D/D2H and time separately; scored H2D counts every packed
slab copy, and controller D2H counts every scalar readback. Log each layer's
precision signal and each physical hit/load/eviction. Replay policy, occupancy,
bytes, round commits, timings, numerical controls, resources and source hashes.

This apparatus keeps a dense target already resident on GPU. Logical layout
comparisons are not proof of access to a larger model, and peak VRAM is not
transfer volume. Report standalone draft layout and the fully charged joint
system separately; this is not a comparison against optimized resident GGUFs or
the best stock small draft. Such capacity-frontier work needs another protocol.

## Separate hypotheses and advance rules

Report all conditions and all failures; these diagnostic thresholds are not
universal usefulness gates:

1. H1 representation: q6 accepts >=25% of proposals and >=10 percentage points
   more than q4. q8 is an independent precision-ceiling diagnostic, not a rescue
   nominee if H1 fails.
2. H2 acquisition: adaptive acceptance is at least q6 minus 10 percentage points,
   and H2D per consumed draft token is >=10% below q8. This checks selective
   precision opportunity, not verified speedup or general quality.
3. H3 persistence: retained and adaptive have identical proposals, target commits,
   layer precision decisions and correction observations, and retained reduces
   total scored H2D >=10%. This is cache-policy value independent of model quality.
   Also require static to have identical numerical behavior, and report retained
   versus static traffic/time. **H3 alone is not evidence for value-based ranking.**
   The static control follows an analytical bound identified before measurements:
   each slab is requested at most once per consumed token, so eight resident slots
   can provide at most eight hits on a later token; eight mandatory first-stage
   slabs achieve that bound after their cold fills. No hypothesis that this
   value-ranked cache beats that transfer lower bound is registered. Its overhead
   and loss relative to that control are diagnostic limitations, not surprises
   to hide. More useful ranking requires a workload with nonuniform reuse/value
   or a policy that can change which precision increments are needed.
4. Report whether retained is >=5% faster than adaptive, with all work charged.
   Failure does not erase a real byte or representation result.
5. H4 formula transfer, added before freeze in response to the owner's explicit
   cross-domain request: hypothesize a first-order expansion y(h)=y*+c*h+O(h^2)
   as the quantization grid spacing h shrinks by four. Richardson extrapolation
   then gives yR=y6+(y6-y4)/3. Compute this from the 28 saved local FFN inputs,
   without extra checkpoint inference or any use in the generation policy.
   Pass only if mean local relative-L2 is >=10% below q6 AND at least 21/28
   individual layers are no worse. Also report norm(y8-y6)/norm(y6-y4), the cosine
   between these increments, and each layer's extrapolation error. Rounding may
   violate the stable leading-error assumption. These checks distinguish a useful
   cross-domain transfer from a merely similar-looking formula. H4 cannot promote
   the current generation candidate; a positive result needs its own causal
   whole-model test. The controller's /4 predicts the *next increment*, whereas
   Richardson's /3 estimates the *remaining limit correction* under that model.

Advance only to separately proposed capacity/session investigation if H1 and at
least one of H2/H3 pass; otherwise stop expansion of this configuration. No
held-out quality, long-session, larger-model, exact target omission, inference
speedup over stock, or native-pivot claim follows from this tiny screen.
