# v0.30: physical risk-directed acquisition on generated trajectories

Prospective sixth of six authorized deliveries, issue45. v0.29 nominates risk
after its binary-proxy win. This screen tests full-vocabulary generated proposals,
physical transfers, state and timing. ADRs0004--0006 remain unchanged: no long
suite, native patch or automatic native admission. Source/protocol review, CPU
validation, commit and push precede checkpoint inference. Corrected runs are fresh.

## Frozen workload and artifact

Same pinned Qwen2.5-1.5B-Instruct HF FP32 checkpoint and embedded eight-bit integer
grid as v0.26; config binds checkpoint manifest, corpus/tokens and all representation
files. Frozen predictor is v0.29's static vector/current-direction fit, SHA256
`6bf1319d2d5bc79fd5e9da1af187a3355bfee903a8453f1825825ef9acf3e877` for the parent
prediction artifact. No refitting, diagnostic update or recursive posterior reuse.

Use diagnostic document indices2,3 from the original16-document corpus. They are
outside v0.26's six fit/two diagnostic documents and the subsequent estimator
screens, but the corpus has historical experiments: do not call this a globally
untouched holdout. Four-token prefix, eight generated tokens or earlier EOS
(151643/151645), fixed K4 speculation and greedy dense target. This small prefix
is a capability/cost screen, not16K chat or task completion.

Conditions: doc2 fixed17, risk17, all35; doc3 all35, risk17, fixed17. One repetition.
Separate scalar target-only reference before and after each document; require
identical complete outputs. The target and draft are separate models/storages.
No LM Studio/GGUF substitution. Stock b10919 target-scale baselines remain published
deployment comparisons, not directly comparable timing claims for this1.5B harness.

## Physical mechanism and observations

All28 FFNs use high-eight precision for the first FOUR actual draft calls (prefill).
Afterward earlier27 FFNs use six bits; final FFN has a two-bit resident base and
three host refinement planes per256-neuron page. Allneurons are computed.
Decode computes the base full-vocabulary readout, freezes its top-two direction,
then loads only selected pages. Risk uses the unchanged v0.29 expected actual
pair-decision error rule, receiving one paid scalar projection per page. Fixed
loads first17; all35 loads every page through the same physical apparatus.
Acquired refinements remain in the charged dense workspace for that call. Final
FFN computation reuses them, with NO second load; a new full-vocabulary readout
owns the proposal. The statistical forecast never commits or replaces that readout.

The scalar-observation shortcut is conditional on the separable model/current
direction, as tested in v0.29. Its ancestors are
[Gaussian conditioning](https://gaussianprocess.org/gpml/chapters/RW2.pdf),
[Russell's decision value](https://people.eecs.berkeley.edu/~russell/papers/aij-cnt.pdf)
and the positive-scalar [RMSNorm readout](https://arxiv.org/html/1910.07467v1).
No calibrated risk, full-argmax certificate or global invention is claimed.

After each call, check the actual refined base-pair margin against the sum of paid
scalar effects, including changed RMS denominator (abs1e-4 +rel1e-5). Save scalar
effects, current risk axis, selection scores and full-readout argmax IDs. Source
and neural numerical controls complement ledger replay; the CPU audit does not
independently re-execute every neural forward or reconstruct all vocabulary logits.
All28 full-eight FFN controls compare to the pinned v0.23 parent at relL2<=.01;
integer-plane identity and parent representation equality are required.

## KV and verification contract

Draft and target K/V are separate FP32 caches. The final FFN follows the last KV
write. Hash actual old prefixes before/after append and the complete new draft
cache before/after refinement. Link those hashes through a per-prefix journal.
Reject -> crop BOTH caches via the existing audited generator, check the draft
journal, discard rejected suffix journal entries and consume the fallback unless
finished. Prefill is first-four-call lifetime state, never re-entered after crop.
Statistical prediction resets every call; no rejected-state Bayesian belief remains.
Physical caches use bypass slots/no admissions. Logical KV and actual underlying
storage after cropped views are recorded separately. Tiny-model tests force reject.

Generate every K4 proposal, including the fourth consumed token even if rejected
or output-capped. Record full-readout IDs, actual accepted prefixes, fallbacks,
emitted accepts, complete committed IDs and both cache lengths. Model-free audit
ties proposals to recorded readouts, reproduces verifier commits and reconciles
every call/crop against the scalar target reference. A truncated eight-token
fixture is not a finished answer or agent-task qualification.

## Costs and frozen verdict

.venv Python3.14.3, torch2.10.0+cu130, transformers5.13.1; four torch threads,
SDPA, no TF32, actual CUDA execution. Supervised <=300s worker includes imports,
artifact verification/loading/construction, controls, all references/episodes,
readback/hash logging and raw writes. Analysis is separately timed. Extra CUDA
cap1024MiB above actual known target/non-FFN draft parameters; NVML total<=15000MiB,
host available>=2048MiB. Sample total GPU/RSS and allocator peaks through all phases.

Charge original host weights/increments, base/scales, full FP32 workspace, both
cache pools and pinned staging, predictor arrays (452168B), retained correction
vectors, KV, Python/CPU scratch through process RSS and CUDA peaks. Index is CPU
float64; risk reads6144B current-axis data plus4B per paid scalar. Fixed/all35 also
pay scalar readbacks for numerical auditing but skip the risk controller. Actual
copy/page events and explicit readbacks are counted; these application counters
are not a hardware trace of every framework/driver copy or sampler operation.
Generator/verifier explicit token copies have an opt-in operation ledger, including
full committed-ID readbacks before output truncation; these are independently
reconciled per round and added to runtime copies. Prefix transfer is charged once
per document. All such work is inside the clock. No observation-data or target-weight sharing.

Per-episode charged time includes generate plus final ledger flushing; per-call
policy/observation work and KV hashes are inside it. Setup, wrapper initialization
and final serialization remain separately covered by worker/phase timing. Compare
aggregate conditions, with per-document receipts preserved; no IID confidence
claim from a single counterbalanced pair. All35 is a same-pager full-refinement
control, not a claim to be the best possible dense kernel.

Hfaithfulness: all numerical, scalar-readout, KV, raw/provenance/resource and
committed-output checks pass. Missing/inconsistent receipt is inconclusive/failure.
Hacquisition: risk emitted-accept/proposal rate>=.5 and>=fixed; actual draft page
H2D bytes per emitted accepted token <=.95 times BOTH fixed and all35. All three
must have nonzero accepts; zero is a failed gate, not a missing denominator waiver.
Counts include prefill and rejected work, not only useful pages.
Hruntime: risk charged aggregate time<=both fixed and all35. The resident FP32
target-only clock is reported separately using the better before/after run per doc;
beating pager controls is not automatically an end-to-end target-only speedup.
Only Hfaithfulness+Hacquisition+Hruntime admit a separately frozen expanded
protocol. Otherwise stop this physical candidate; preserve component outcomes.
No target-scale acceptance,32B speedup, larger-model capacity frontier or native
admission is tested here. The six-delivery authorization does not close milestone10.

Run after freeze: `.\.venv\Scripts\python.exe -m dynamic_model_loading.risk_screen --output runs/risk-screen-20260915-v1`.
