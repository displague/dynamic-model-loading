# Calibrated error acquisition and high-precision prefill: prospective short screen

Under ADRs [0004](adr/0004-fault-pager-research-track.md),
[0005](adr/0005-native-pivot-evidence-boundary.md) and
[0006](adr/0006-screen-before-performance-matrix.md). Freeze and push reviewed
source, this protocol and `configs/bayes-screen.json` before checkpoint inference.
A correction requires new source and a fresh directory, retaining failed receipts.
This is a new hypothesis, not a changed tolerance for v0.23. No native pivot or
long suite is authorized by this screen. Inspirations: [ledger](bayes-inspirations.md).

## Question and fixed model

Can covariance over observed draft states predict the magnitude of an unfetched
6-to-8-bit FFN correction, with calibrated uncertainty useful for acquisition?
Does eight-bit prefill improve subsequent six-bit draft faithfulness independently?
Do not extrapolate a vector correction direction: v0.23 did not support that premise.

Unchanged Qwen2.5-1.5B-Instruct HF revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP32 target and separate draft non-FFN
weights, `.venv`, Python 3.14.3 / torch 2.10.0+cu130 / transformers 5.13.1, CUDA,
SDPA, no TF32, four CPU threads. Reuse the v0.23 group-128 embedded 4/6/8-bit
quantizer; reconstruct and compare every packed tensor against its frozen archive.
Archive new constructed tensors too, so replay is self-contained. Check all 28
eight-bit FFN outputs against the parent's saved same-input outputs, rel-L2 <= .01.
Eight-bit is not the FP32 target. All FFN neurons execute at every precision.

## Covariance hypothesis and fitting (no diagnostic tuning)

For current draft input x and outputs y4, y6, label
`z = log(max(||y8-y6|| / max(||y6||,1e-12),1e-8))`.
The target never supplies features or labels. Features are a seeded 16D
Rademacher projection of x/||x|| (CPU generator seed 20260915, signs /4),
log(max(||y6-y4||/max(||y6||,1e-12),1e-8)), and log(max(||y6||,1e-12)).
This is a nonlinear GP on 18 features derived from 1536D states, not a full residual
field posterior, ensemble Kalman filter, or online Bayesian session assimilation.

Fit one CPU float64 GP per layer on first eight calibration documents, four rows
per document. Each document starts from a fresh independent draft KV cache: consume
first eight corpus tokens using eight-bit FFNs, then next four corpus tokens with
six-bit outputs. Compute the eight-bit counterfactual at each decode FFN solely
for its label, returning y6 so later states really follow that draft path.
These are draft-state **teacher-forced**, not own-generated, training trajectories.

Fit-only feature mean/std (population std floor 1e-6); standardized covariance
`C=.8 U' U/n + .2 I`; whiten with its lower Cholesky factor. RBF kernel amplitude1,
length squared = torch median off-diagonal squared whitened distance (floor1e-6).
Fit-only z mean/population std (floor .1). Fixed observation-noise variance .1,
Cholesky jitter1e-8. Posterior mean and predictive sd include observation noise:
`mu=zbar+s*k'*(K+.1I+1e-8I)^-1*zstd`,
`sd=s*sqrt(max(1.1-k'*(K+.1I+1e-8I)^-1*k,1e-12))`.
No hyperparameter sweep or online updates.

Next nine calibration documents use the identical 8+4 trajectory. For each layer,
one score per whole document is the maximum of four `(z-mu)/sd` values. Take
order ceil((9+1)*.9)=9, clipped below at zero. Upper prediction is mu+q*sd.
The constant control uses fit-only layer z mean/std and the same block procedure.
Fewer than nine blocks must give an infinite bound, not a false finite 90% bound.
The nominal split-conformal interpretation requires exchangeable documents; this
fixed corpus and adaptive generation do NOT demonstrate that assumption. Report
empirical coverage only, never a target correctness or generation coverage guarantee.
Freeze fitted tensors and calibration records before diagnostic evaluation.

Two first diagnostic documents each supply four labeled shadow decode positions
after eight-bit prefill, for 224 rows / 56 layer-document blocks. The diagnostics
are reused development material, not held-out qualification. Actual generation
does not fetch skipped eighth-bit increments for counterfactual labels.

## Six paired conditions and KV contract

One repetition, first two diagnostic documents, eight prefix tokens, four committed
tokens, K=4. Order below on first document; reverse on second. Warm all six with
two prefix tokens / one committed token on first fit document. Keep scalar target
greedy references. Every candidate's committed IDs and stop reason must match.

| Condition | Prefill FFN bits | Decode rule |
| --- | --- | --- |
| q6 | 6 | Always 6 |
| p8d6 | 8 | Always 6 |
| p8d8 | 8 | Always 8 |
| p8mean | 8 | Acquire 8 if GP mean log-error > log(.06) |
| p8gp | 8 | Acquire 8 if calibrated GP upper log-error > log(.06) |
| p8constant | 8 | Acquire 8 if calibrated constant-layer bound > log(.06) |

All use eight fixed persistent stage-zero slabs, layers 0..7, plus one bypass
slab; same cache control, avoiding the v0.23 cache-ranking confound. Every decoded
FFN first computes y4 and y6. Adaptive modes pay feature/readback and prediction
cost. High prefill controls omit unused predictions. No borrowed target KV or
shared target/draft weights. K/V tensors remain FP32 in all modes. Precision here
means FFN weight precision, not KV quantization or target arithmetic precision.

Only first N actual consumed draft calls are prefill. The last prefix's saved
logit (high precision except q6) predicts the first proposal. Later decode and
fallback calls never re-enter prefill after rollback. Both independent caches
crop to base+accepted on rejection. Accepted mixed-precision KV remains; rejected
suffix KV is discarded, fallback is consumed once unless finished. Changing
precision does NOT recompute old KV. Reset cache, consumption counter and prefix
fingerprint for each episode; no calibration state survives except frozen GP.
Hash all actual prefix K/V bytes at prefix completion, after every subsequent
draft step and after every crop; require unchanged SHA256. Charge CPU readback,
hashing, feature copies, acquisition, controller and ledger flushing to wall time.

## Gates, separately reported

Integrity/resource failure, error, timeout or incomplete matrix is inconclusive.
These fixed screening criteria are not significance tests:

- Hcal: GP mean log-error MAE <= .9 * constant-mean MAE on 224 shadow rows, and
  >=90% of 56 layer-document blocks covered by GP upper bounds (all four rows).
- Hunc: GP upper bound has strictly fewer false-safe shadow decisions than mean
  prediction, and skips at least one shadow row. False-safe means predicted
  log-error <= log(.06) but observed log-error > log(.06); report denominators.
- Hprefill: p8d6 accepted/attempted fraction >= q6 + .10, and total physical H2D
  per committed token <= .9 * p8d8. Traffic includes prefill and rejected work.
- Hpolicy: p8gp acceptance >= p8d8 - .10, H2D/committed <= .9 * p8d8, and
  H2D/committed <= .95 * p8constant with acceptance >= p8constant - .10.

Only Hprefill OR (Hcal AND Hunc AND Hpolicy) warrants a new expanded protocol;
otherwise stop this candidate. Keep all four verdicts, individual documents,
precision counts, predictive errors/coverage, accepted-prefix ledgers, consumed
tokens, H2D/committed, H2D/consumed, memory and end-to-end timing. A useful prefill
control alone is not a Bayesian acquisition success. No larger-model access is
tested; the dense reference fits already. Capacity remains a separate open aim.

## Physical budget, replay and time

Five-minute supervised worker includes imports, model hashing/loading,
construction, parent equality checks, all calibration/shadow inference, fit,
snapshot, warmups, scalar references, generation and final raw hashing. CPU
independent analysis separately timed. Record phase peaks before resetting the
allocator. Extra CUDA <=1024 MiB above actual target+draft non-FFN parameters
(7725494272 B); retain overhead, weights, projection, scratch, cache and joint KV
in that allowance. Total GPU <=15000 MiB; available host >=2048 MiB. Charge
original host FFNs, all packed host increments, pinned staging, GP/feature tensors,
Python metadata and observed process RSS; memory isn't inferred from traffic.

Archive raw features/labels, GP tensors, both physical/precision ledgers, prefix
fingerprints, verification rows, resource samples and frozen source. Independent
CPU analysis verifies inventories, reconstructs GP fit/calibration using separate
linear algebra, replays predictions, acquisition and cache events, KV crop event
sequence, target identity, memory/timing and gates. CPU tests cover covariance
behavior, insufficient calibration, missing/tampered receipts and forced rejection.
