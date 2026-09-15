# v0.34: full-vocabulary risk and actual accepted-prefix economics (prospective)

Fourth of six authorized deliveries, [#49](https://github.com/displague/dynamic-model-loading/issues/49).
Return to the one remaining decision-ranking question, separate from v0.33's
positive non-speculative packet loader. Source/protocol review, validation, commit
and push precede every new checkpoint measurement. ADRs0004--0006 remain unchanged:
300-second worker, no failed-screen expansion, no automatic native pivot.

## Changed hypothesis, not a renamed score

v0.30 ranks pages by the probability of a binary base-top-two sign mismatch. A third
vocabulary competitor can invalidate that decision. New hypothesis: rank by the
probability that the actual partial argmax disagrees with the fully refined argmax,
under joint uncertainty about a purchased page and the remaining field. Then
measure actual accepted prefixes using the independent dense target verifier.

Merely multiplying each page score at a fixed token by the same probability that
the prior draft prefix survives cannot change its ranking. Therefore this screen
changes the full-vocabulary event itself; it does NOT claim to optimize a shared
byte budget across future token positions or learn a calibrated target-acceptance
probability. The forecast concerns the fully refined draft, which can itself
disagree with the dense target. Actual verification owns the outcome.

Prediction: full-vocabulary risk may avoid pair-blind errors, but repeated full
readouts are expensive and may lose to loading all pages. The previous geometry
prior is reused, not re-fit or retuned on these diagnostics. Any remaining feature
correlation/model mismatch may still prevent a useful acquisition policy.

## Gaussian decision calculation and attribution

Use v0.29's pinned matrix-normal prior: page means M[p,h], page covariance C[p,q],
diagonal feature variance D[h]. For remaining pages R, define S=sum(delta[R]).
For possible purchase p, Var(delta_p)=C_pp D; Var(S)=sum(C_RR)D;
Cov(delta_p,S)=sum(C_Rp)D. Two shared standard-normal feature draws therefore give
joint samples of delta_p and S. Observe only purchased full correction vectors,
then perform the existing Gaussian conditioning update; no unpurchased labels.

At each of17 purchases, every remaining page gets16 joint samples (eight antithetic
pairs). Compare argmax W*norm_weight*(h+delta_p) to argmax W*norm_weight*(h+S), over
ALL151936 vocabulary IDs. The RMS denominator is positive and common across
vocabulary entries for each state; only that denominator can be omitted. Count
sample disagreements, minimize the count, tie-break on page ID. A finite Monte
Carlo plug-in model is not calibrated risk, an omission certificate or exact EVSI.
Common draws are reused across actions/steps within a call; seed20260918+call.

[Gaussian conditioning and decision loss](https://gaussianprocess.org/gpml/chapters/RW2.pdf),
[RMSNorm](https://arxiv.org/abs/1910.07467), and
[speculative verification](https://proceedings.mlr.press/v202/leviathan23a.html)
provide the antecedents. No invention of those techniques or Monte Carlo is claimed.
The new tested application is full-vocabulary decision-value acquisition coupled
to physical page purchases and real accepted-prefix outcomes, not local FFN L2.

## Frozen artifacts and short workload

Same pinned Qwen2.5-1.5B-Instruct HF FP32 target and independent draft, checkpoint
revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Same embedded precision artifacts,
35 final-FFN pages and v0.29 index, bound by config SHA256. No training or downloaded
substitute. Archive historical corpus/tokens only to preserve dependency provenance;
they are NOT this inference workload. Two new authored prompts are in config.
Tokenize without specials/template and consume first FOUR tokens only. Cap eight
committed tokens or EOS151643/151645, fixed K4 greedy speculation. This is not a
domain task-quality or long-context study; most authored prompt text is unused.

Order: document0 risk17,fullrisk17,all35; document1 all35,fullrisk17,risk17. Separate
scalar target-only reference before and after each document. No discarded warmups.
Risk17 is unchanged binary-pair scoring; all35 is the best prior same-apparatus
acquisition control. No fixed17 rerun is needed to exceed the stronger all35 result.
Both controls run only as controls for the NEW full-vocabulary score on new prefixes.

First four draft calls are high-eight precision prefill. Later earlier27 FFNs use
six bits; final FFN starts at two-bit base and purchases three refinement planes
per selected256-neuron page. All neurons are computed; actual page loads modify
the charged workspace and are not loaded twice for final readout. The full-vocab
controller receives an actual GPU correction vector ONLY after purchase. Its
Gaussian tensors and common draws stay on GPU while scoring; no hidden per-action
host transfer of entire latent batches. All index/draw/index-list H2D and sampled
ID/finite-check/full-vector D2H are explicitly counted.

## Correctness, state and audit

Use the v0.30 target/draft separate-storage and FP32 KV contract. Final readout
refinement follows the last KV write. Actual KV fingerprints link before/after
append, unchanged prior prefixes, before/after refinement and rejection crops.
Reject crops BOTH caches, discards rejected journal suffixes and consumes fallback
unless finished. Neither posterior nor Monte Carlo state persists across calls,
documents or rejected suffixes. First-four-call precision state never re-enters.

Full target greedy IDs must match before/after and across all conditions. All28
eight-bit numerical controls must match pinned parent at relL2<=.01. The existing
physical scalar/readout identity still checks actual selected vector effects with
abs1e-4+rel1e-5; full-vector projection audit uses the same tolerance. A new final
full-vocabulary readout, never a sampled/predicted token, supplies each proposal.
Record proposed, target-predicted, accepted/emitted prefixes, fallback and stop
reason, all extra fourth-token work, cache crops and complete committed output.

Audit all source/config/artifact/tokenizer bindings, raw inventory, physical loads,
proposal/readout links, verification commits, KV journals and costs. For Monte
Carlo, record every action's sampled full-vocabulary ID pairs, chosen page and
purchased vector. Independently reconstruct scores/ties/action availability,
Gaussian updates, vector/scalar consistency and all byte counts. This audit does
NOT independently rerun the entire vocabulary head for every latent sample;
source binding and tiny numerical tests support that execution claim. Do not call
the posterior calibrated or the sampler a verified mathematical certificate.

## Costs and gates

Existing .venv Python3.14.3/torch2.10.0+cu130/transformers5.13.1, actual CUDA SDPA,
TF32 off; four PyTorch AND NumPy/OpenBLAS threads, with BLAS DLL hash in manifest.
Worker includes imports, artifact checks/loading/construction, controls, inference,
all scoring and raw writes. Audit timed separately; no automatic full matrix.
Extra CUDA<=1024 MiB above actual target/non-FFN draft parameters; total sampled
GPU<=15000 MiB, host available>=2048 MiB. Charge prior/index tensors, sample logits,
workspace, original host weights/increments, both pinned pools, retained corrections,
KV backing storage after crop, rejected work and raw arrays via allocator/RSS.
Keep construction/setup costs distinct from conditional per-episode economics.

Hfaithfulness: all mandatory reference/numerical/KV/audit/resource controls pass.
Hacquisition: fullrisk accepts>=50% and no smaller fraction than EITHER control;
its charged H2D per accepted token is at least1% lower than EACH control. Charged
H2D includes physical weights, Monte Carlo inputs and generator/verifier transfers;
D2H is reported separately and its latency is charged. Inclusive cross-multiplication.
Hruntime: charged episode wall no greater than EITHER control. All gates required
for nomination, with no rounding waiver or post-hoc repeat.

The new1% bar is prospective, NOT a waiver of v0.30. With saturated eight-token
acceptance and four high-precision prefill calls, removing18 of35 final-page loads
per decode step offers only about2.6% total weight-H2D headroom; the old5% bar is
not a useful screen for this fixed workload's saturated case. This stricter cost
accounting includes new controller traffic and asks whether any material saving
survives. It does not turn a1% saving into a deployable achievement. Failure stops
this full-vocabulary ranker; no further ordering of these pages is automatically
queued. Packet capacity/stress work remains a separate course direction.
