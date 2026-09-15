# v0.26 output-margin sensors: prospective mechanism screen

ADRs 0004--0006 apply. Review and push clean source/config/protocol before
checkpoint inference; use a fresh output directory. Five-minute worker includes
all imports, source/model hashing, construction, checks, collection and raw
hashing. Analysis separately timed. Errors/timeouts/incomplete receipts are
inconclusive. No native patch or long matrix follows a short screen.

## Departure from v0.25, and mathematical antecedent

v0.25 found no useful action among four separated whole-layer promotions. Stop
that action set. This experiment changes both unit and observation: 35 physical
neuron pages in the final FFN, and a projected output-decision measurement without
a whole-model counterfactual replay. It is not another predictor over v0.25 data.

For this pinned Qwen, the final readout is bias-free and RMSNorm has only a
learned gain: L(h)=W diag(gamma) h / sqrt(mean(h^2)+epsilon). For two *base-draft*
tokens u,v, let a=(W_u-W_v)*gamma. Then the sign of their margin after a page
correction delta is the sign of a'(h+delta), because the divisor is positive.
This is our application of the [RMSNorm formula, Zhang/Sennrich 2019](https://arxiv.org/html/1910.07467v1),
also checked against installed Qwen2RMSNorm and its bias-free lm_head. It is not
invention of normalization, an approximation to all layers, or a certificate that
no third token wins. Validate FP32 numerical error and full-vocabulary outcomes.

The cheap observation can support the [sensor-placement / spatial-Bayes course](information-research-course.md).
No Bayesian model or online acquisition policy is fitted in this release. Full
page observations and the oracle set are privileged measurement tools, not savings.

## Fixed representation and deliberate challenge

Same pinned 1.5B HF artifact, separate FP32 target and draft non-FFN parameters,
SDPA, CUDA, four CPU threads, no TF32, measured .venv versions as v0.25.
All original CPU FFNs remain charged. Reconstruct/compare the v0.23 embedded
group-128 representation against its fixed inventory. Other 27 draft FFNs use
eight-bit prefill and six-bit decode, with cold whole-slab transfers.

Only the final FFN changes: split each packed upper-four-bit base into upper two
and next two bits. GPU two-bit reconstruction is (64*q2+31.5)*scale+minimum.
Three host two-bit refinements add (16*r0-24)*scale, (4*r1-6)*scale,
(r2-1.5)*scale. Together these reconstruct the same eight-bit integer grid as
v0.23, with potentially different FP32 rounding. Move the original final four-bit
base to CPU; do not leave an uncharged GPU copy. Check all 28 high-eight numerical
outputs against the pinned same-input parent, rel-L2<=.01, and exact integer-plane
identity before inference.
The config separately freezes SHA256 of the three original final-layer base arrays;
the new archive's copied parent arrays cannot bless their own changed bitplanes.

Page = 256 adjacent neurons, all three projections, across three refinement
planes: 35 pages, each 884,736 payload bytes plus resident min/scale metadata.
Pinned staging and two fixed GPU page slabs hold one two-bit plane at a time;
there are no admissions/hits. The earlier-layer cold slab cache remains separate.
This intentionally low-precision final region is a *controlled challenge* to expose
decision headroom, not a natural deployment baseline or proof of larger-model
access. Its two-bit base saves 10,321,920 resident bytes relative to its own
four-bit base; full measured allocations determine actual process memory.

## Trajectories and observations

First six calibration and first two diagnostic documents of pinned parent corpus;
four eight-bit prefill tokens then next sixteen teacher-forced corpus tokens.
96 fit-development and 32 diagnostic-development positions, no held-out claim.
Prefill is explicitly controlled by the worker, not inferred from a rollback-prone
counter. Separate target cache teacher-forces the same tokens for audit only.

At a decode position run the draft once with its final two-bit FFN. Capture its
final FFN input, pre-FFN residual R and pre-RMS hidden vector h=R+y_base; select
base top-two token IDs. Use R+y_new for canonical readouts rather than numerically
recovering R by subtracting y_base from h. For
each physical page in index order, fetch the three increments, compute its local
contribution difference delta_s and projected margin effect a'delta_s / RMS(h).
All other neurons still execute; they are not treated as exact zero.
Record full delta vectors and basis/gain/readout rows, base hidden state, actual
two-row readout results, page events, timings, readbacks and allocations.

After all pages, compute canonical dense eight-bit final FFN output using the
refined workspace. Compare it with base output plus summed page deltas. Save
base, canonical high-eight, fixed-first-17-page and oracle-17-page full logits,
and independent target logits. The oracle chooses the 17 largest signed page
contributions favoring the high-eight winner over the base winner; ties use index.
It uses unfetched observations and is NOT a permitted online controller. A fixed
set has the same 17-page payload. Reset the final workspace to two bits and
physically reload each selected set to compute its canonical mixed-precision FFN
and full readout. Thus both partial controls are actual loaded actions, but their
payload is NOT traffic saved by an online policy: collection first paid for all
35 observations, then pays for both 17-page replays. Keep these costs separate.

At the first decode position of each document, additionally crop to the pre-step
length and replay the full draft with final high-eight. Compare full logits with
the direct canonical readout (rel-L2<=.01 and same argmax), and require the *entire*
post-step KV fingerprint to equal the original two-bit call. This works only
because the final FFN follows the last attention/KV update. Old/prefix bytes also
remain unchanged. Report logical and underlying KV storage separately. Charge
replays, readouts, transfers, fingerprints and all retained host observations.
Bind the first frame to the recorded prefill fingerprint, and each later starting
fingerprint to the preceding frame's ending one. Separately time high readout,
oracle selection, fixed reload/readout and oracle reload/readout, with their CPU
readbacks and page-ledger flushes; retain total frame time as the containing clock.

## Frozen gates

Integrity requires all representation/numerical/KV/resource/provenance checks.
Projection prediction uses the actual corrected RMS divisor. Across every page,
absolute error against the direct two-row readout must be <=1e-4+1e-5*abs(actual).
Also require grouped-all-page final FFN rel-L2<=.01 against canonical dense output.

- Hobservation: projection and full-replay identity checks pass, and aggregate
  time for one page observation is lower than aggregate mean full-model replay
  time. This is measurement-mechanism evidence, not inference throughput.
- Haction: at least four of 32 diagnostic positions differ between base and
  canonical high-eight; oracle-17 repairs at least two of those, with no more
  new disagreements on previously matching positions than fixed-first-17.

Report fit and diagnostic results, fixed/oracle disagreements, high-eight vs
target, and candidate-vocabulary misses separately. A pass permits a new short
spatial/recursive-estimation protocol on this observation, not policy nomination.
Failure stops this action/observation combination; it does not waive v0.25 gates.

Extra CUDA<=1024 MiB above actual target+draft non-FFN parameter storage;
total GPU<=15000 MiB; available host>=2048 MiB. Include original host weights,
CPU four-bit final base, new host plane, all old increments, resident two-bit base,
shared min/scales, both cache pools/staging, workspaces, observations and KV.
Frozen source, full raw vectors/logits, controls, timings, explicit stage transfer
and readback ledgers are archived. Independent CPU analysis reconstructs signed
projections, oracle selection, full-logit metrics, all denominators, inventories,
physical loads, phase resource/timing and gate decisions before publication.
