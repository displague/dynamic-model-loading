# Bounded CUDA capture and target-path diagnostic

Prospective apparatus supplement to the [operation-offload protocol](verification-offload-protocol.md),
under ADR 0003 and issues #27/#28. Commit and push the fixture, harness and analysis
before these additional diagnostic requests. The short timing matrix already uses
its frozen source at `3f44ac2`; no new diagnostic is inserted into those timings.

## CUDA-copy capture

Use the installed Nsight Systems 2025.5.2 CLI on the unchanged stock b10919 server,
with the existing target/0.5B artifacts, ngl44/t24, f16 KV, 4096 context, K16. Capture
two fresh processes, threshold32 then threshold8. Each executes the same 32-token
library warmup and 32-token code-cache request as the mechanism diagnostic.

Enable only CUDA/NVTX tracing, CUDA graph-node activity, no CPU sampling or context
switch tracing. Record the profiler executable hash/version, exact arguments,
environment, owned target PID, raw report/export, HTTP receipts, log intervals,
resource samples and both Unix and monotonic request timestamps. All profiling
times are diagnostic and excluded from the stock timing frontier. Trace includes
startup; do not count initial loading as verification traffic.

Use the SQLite session UTC epoch plus activity timestamps to align copies/kernels
with the measured HTTP windows, if this metadata is available and consistent.
Report copy-kind bytes/counts and kernel activity within each request. These windows
include prefill, drafting and verification; sums of overlapping durations are not
critical-path time. Distinguish startup and ordinary request traffic. A whole-request
H2D difference alone is not an isolated verifier timer or a contiguous-layer transfer.
Keep actual activity/copy granularity and repeat patterns where observable.

Stock scheduler assignments already show whether host-weight operations move to
CUDA. Physical bytes require actual captured activities. An exported report with no
CUDA activity is a capture failure, not zero GPU work. This installed profiler may
be incompatible with the newer CUDA runtime; retain failure details and do not
install/rebuild a runtime to rescue this bounded capture. Stop the capture branch
if it cannot produce usable activity with these settings.

Owned process trees run without visible windows. Sampling covers startup through
the final request; profiler export/shutdown is excluded. Its process-specific memory
and I/O fields concern the wrapper, while total device and host fields retain the
15000 MiB / 2 GiB bounds. Never interpret wrapper I/O as target I/O. Stop/export
the named session, request shutdown of that exact owned session if still running,
and retain exit/errors. Complete capture qualification requires both expected
requests, successful stop/export and wrapper exit, and an independently passing raw
resource trace. Partial CUDA activity remains distinguishable from a complete usable
diagnostic. Retain exact log byte intervals for acceptance reconstruction.

## Aligned target-only numerical factors

The frozen [eight-prefix fixture](../data/target-path-diagnostic.json) contains all
four v0.14 historical baseline rebuild mismatches, plus generated-position32 controls
for code-parser, code-sql, data-audit and topic-transition. Selection uses only
published v0.14 data, and the input source hashes are recorded. The selected
divergence stratum is intentionally enriched; it cannot estimate a population flip
rate. The four ordinary positions are fixed controls, not assumed non-divergent.

Every condition is target-only Q4_K_M, context4096, f16 KV, batch256, target batch
threads24, explicit operation threshold32, neutral greedy sampler, no context shift.
Run the following five fresh processes once, in the listed order, with all eight
prefixes in fixture order:

| Factor | GPU layers including output | Decode threads | Microbatch | KV construction |
|---|---:|---:|---:|---|
| prefill | 45 | 16 | 256 | entire aligned prefix in one request |
| microbatch | 45 | 16 | 1 | entire aligned prefix, one-token microbatches |
| incremental-reference | 45 | 16 | 256 | original prompt prefill, then supplied continuation one token at a time |
| threads | 45 | 24 | 256 | same incremental construction |
| placement | 44 | 16 | 256 | same incremental construction |

For incremental construction, request one prediction after the original prompt,
then extend the input by exactly one **frozen historical token** per request using
`cache_prompt=true`. Intermediate predictions are retained in raw HTTP receipts but
do not choose the supplied continuation. Require native `cache_n=prefix_length-1`
and `prompt_n=1` after each extension. First requests must have zero cache hits.
Reset between cases. This validates cache-reuse accounting and aligned conditional
decisions; it is not a bytewise comparison with the original continuous-request KV.
If the stock path does not satisfy that accounting, retain the failure and do not
quietly substitute a full-prefix replay for incremental state.
The analyzer must reproduce every expected intermediate request and cache count
from its raw HTTP request/response, not merely trust a stored pass flag or final
distribution. Missing intermediate requests fail the state-accounting audit.

At each final prefix request exactly one token and all 152064 pre-sampling
probabilities (`n_probs=152064`, `post_sampling_probs=false`). Other requests have
`n_probs=0`. Preserve raw HTTP bodies and token-ID-indexed float32 vectors. The native
API returns **log-softmax values, not raw logits**; zero probability is encoded as
`-FLT_MAX`. Require complete unique vocabulary IDs, finite encoded values and the
one-token output count. Compare the emitted ID with the encoded argmax set and
retain failure as evidence, not a license to alter decoding.

The final diagnostic token/probability requests are not timing observations. Reporting
full probabilities can change synchronization and output cost. No equality to the
old n_probs=0 performance path is presumed. The top-two log-probability difference
approximates the corresponding raw logit gap up to float softmax/log serialization;
absolute raw-logit infinity distance is unavailable through this API.

Compare threads and placement to incremental-reference, prefill to
incremental-reference, and microbatch to prefill. Report both top-two gaps, emitted
argmax changes, full reconstructed probability infinity distance and probability
margin. The elementary margin > 2*distance condition is checked in **probability
space**, with that representation named; do not label it a raw-logit bound. Report
selected and ordinary strata separately, plus matching/nonmatching historical
decisions. No general flips-per-1000-tokens estimate follows from eight chosen points.

## State-correctness boundary and stopping

The earlier stock-source inspection establishes how acceptance, rollback and bonus
handling are intended to work. Reconciled events and these target-only cache checks
do not independently prove every speculative verifier's prefix/mask/KV correctness.
b10919 does not return target probabilities for each accepted speculative token.
Keep that exact limitation in #27 instead of calling this a complete state audit.

Do not force the original sustained identity failures to pass by redefining their
reference. This bounded diagnostic can identify specific execution-path effects and
remaining ambiguity. It is not a library-version sweep, proof of universal near-tie
behavior or authorization for a custom runtime. After one run, report its findings
alongside the completed stock frontier and choose any follow-up from measured need.
