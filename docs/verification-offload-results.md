# Stock verification offload and target-path evidence

The stock b10919 environment setting `GGML_OP_OFFLOAD_MIN_BATCH=8` moves eligible
host-weight verification operations to CUDA and raises the short-context K16
request rate from **11.6805 to 19.7385 emitted tokens/s**. The more relevant gain
over the newly repeated default-threshold K4 configuration is **11.8%**
(17.6585 tokens/s). The prospective 25-token/s forecast is not reached. No runtime
patch, new draft representation, learned controller or model download was required.

These are observed rates on generated continuations that can differ across
execution paths. The original v0.14 sustained-identity failures remain failures.
A separate target-only diagnostic reproduces the four selected historical
rebuild mismatches through prefix/KV construction, but does not independently
validate all speculative target state. Issue #27 therefore remains open.

## Protocol, execution and resource contract

The [main protocol](verification-offload-protocol.md) and timing harness were
committed and pushed as `3f44ac2` before inference. The
[numerical/profiler supplement](target-path-and-profile-protocol.md) was frozen
at `f884517`; one prospective profiler lifecycle/logging correction was frozen
at `f4621be`. All native measurements use clean, detached source checkouts.
The date suffix `20260913` is a run label; the raw UTC timestamps, not that label,
define the actual execution dates. No successful timed observation was replaced.

The unchanged runtime is llama.cpp b10919, source
`d3146f2b56c2db4711ac8391871c9e529d1946d7`, Windows x64 CUDA 13.3. The target is
official Qwen2.5-32B-Instruct Q4_K_M (19,851,336,384 bytes across five shards);
the resident draft is Qwen2.5-0.5B-Instruct Q8_0 (675,710,816 bytes). Required
model and executable/DLL hashes are checked per fresh process against the
[catalogue](../configs/stock-speculation-artifacts.json).

The machine has an Ultra 9 275HX, approximately 32 GiB host RAM and an RTX 5080
Laptop GPU with 16 GiB VRAM. All runs retain 200 ms total-device, host-available,
process and environment receipts. The frozen sampled limits are at most 15,000
MiB total GPU usage including desktop, and at least 2 GiB host available, including
startup and inter-request gaps. Sampling can miss short peaks and does not prove
absence of WDDM eviction or physical storage reads. No tests, reviews, compression,
downloads or unrelated native model runs overlap the timed matrix.

Short conditions hold target GPU layers at 44 **including output**: 43 GPU
transformer layers and 21 CPU transformer layers. Target decode/batch threads
are 24/24; draft threads are 8/24, with the draft fully on GPU. Context capacity
is 4096, batch/microbatch 256, both KV caches f16. Actual prompts are 73–103 tokens.
Six frozen sustained prompts generate up to 256 tokens with EOS respected.
Every condition runs twice in reversed configuration order and recorded shuffled
prompt orders, with a separate 32-token warmup per fresh process.

Each child explicitly clears inherited LLAMA/GGML/CUDA overrides and records
`LLAMA_TRACE`, `GGML_SCHED_DEBUG`, and the operation threshold. The knob is read
at backend initialization, so every change uses a fresh process. Disabled-offload
uses `--no-op-offload`; that also affects prefill and cannot isolate verification
by whole-request subtraction.

| Condition | Emitted/request s | Emitted/native decode s | Native decode steps/s | Mean prefill s/request | Mean native decode s/request |
| --- | --- | --- | --- | --- | --- |
| cpu-k4 | 17.6585 | 18.3544 | 18.2827 | 0.541 | 13.948 |
| cpu-k8 | 16.3155 | 16.8801 | 16.8141 | 0.517 | 15.166 |
| cpu-k16 | 11.6805 | 11.8697 | 11.8234 | 0.342 | 21.567 |
| offload-k8 | 19.0336 | 19.6160 | 19.5394 | 0.393 | 13.051 |
| offload-k16 | 19.7385 | 20.5345 | 20.4543 | 0.493 | 12.467 |
| disabled-k16 | 11.3563 | 12.1670 | 12.1195 | 1.495 | 21.041 |

## Backend assignment and physical movement

Three verbose diagnostics precede the timing matrix. In the first actual target
verification graph with 16 proposed tokens plus the current target position,
all **21 cold-layer FFN up projections** are assigned to CPU at threshold 32,
all 21 to CUDA at threshold 8, and all 21 back to CPU with operation offload
disabled. CUDA split inputs explicitly include copied host-weight tensors.
The pinned build compiles out scheduler reason labels; missing `1.off` text
is not evidence of absent offload. Graphs can be reused, and these assignment
counts are not executed-operation counts or physical bytes.

Nsight Systems 2025.5.2 captures actual CUDA copies and kernels from the unchanged
CUDA 13.3 binary. The first capture completed both HTTP requests and exported
activity, then failed its immediate post-shutdown target-exit check; native logs
also were not forwarded by the profiler wrapper. The owned job cleaned up the
target. That entire failed attempt remains available as partial evidence.
The preregistered correction routes native output through stock `--log-file`
and records verified-owned-process cleanup after export. Both fresh corrected
captures complete and pass the independent CUDA/resource/window audit.

| Threshold | Request | H2D bytes | H2D copies | D2H bytes | Kernels | Summed H2D ms |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | calibration-explanation | 6246614648 | 1351 | 204709888 | 97388 | 123.431 |
| 32 | code-cache | 6242370176 | 744 | 94030336 | 42814 | 122.559 |
| 8 | calibration-explanation | 68744509048 | 4701 | 355454976 | 103888 | 1356.149 |
| 8 | code-cache | 24997028480 | 1749 | 145627648 | 44764 | 491.410 |

The code fixture's large-copy histogram repeats the same tensor-size pattern
four times at threshold 8 versus once at threshold 32: 79,626,240-byte copies
occur 204 versus 51 times, and 116,121,600-byte copies occur 48 versus 12 times.
This is tensor-granularity movement, not one magically preassembled layer payload.
The smaller verification tails can remain below the threshold. Both profiled
code requests have six recorded acceptance events, 60 attempted draft tokens and
25 accepted draft tokens; maximum K alone would not predict their movement.

UTC session metadata aligns SQLite activities with HTTP intervals; measured wall
and monotonic interval durations agree within the declared tolerance. Startup
loading is outside those windows. The windows include prefill, drafting and target
verification. Summed copy/kernel durations can overlap and are not exposed stalls
or isolated cycle latency. Profiler and verbose timings are excluded from every
throughput ranking. There is no import of v0.9 bandwidth or v0.12 schedule estimates
as measured costs of this loop.

## Populated long-context control

At the frozen long placement, threshold8/K16 reaches **13.6047 emitted tokens per
native decode second**, versus 4.8013 for default K4 and 3.4409 for target-only.
Its whole-request rate is **3.0026 tokens/s**, 42.1% above K4's 2.1123, because
each request also pays approximately 33 seconds of prefill. Native decode averages
9.409 seconds for 128 emitted tokens, versus 26.660 for K4. The native decode-step
rate, excluding the first emitted token, is 13.4984 versus 4.7638 steps/s.
All **16 scored long requests** generate identical IDs to their two local target
references, including both repeats. This bounded observed agreement does not
establish an independently verified speculative-state implementation.

The prospective long configuration uses exactly 16,384 input tokens, 18,432
capacity and a 128-token continuation cap. Both target and draft KV are q8_0.
Target placement is 38 GPU layers including output, or 37 GPU and 27 CPU
transformer layers. This is a separate frozen feasible allocation, not a search
for the best long-context placement. Target-only uses that same placement and
precision as its local numerical control; it is not compared for identity to
the historical f16-KV reference.

The first target process freezes two tokenized inputs before generation. Synthetic
labelled engineering records fill the middle of the documented chat format;
only filler tokens are trimmed. The final code-cache or data-audit instruction
and generation suffix remain intact. All later conditions reuse the exact token
file. This tests populated cache and execution cost, not long-context reasoning
quality. A large capacity flag alone is not counted as a long-context observation.

| Condition | Emitted/request s | Emitted/native decode s | Native decode steps/s | Mean prefill s/request | Mean native decode s/request |
| --- | --- | --- | --- | --- | --- |
| target | 1.8018 | 3.4409 | 3.4140 | 33.824 | 37.200 |
| cpu-k4 | 2.1123 | 4.8013 | 4.7638 | 33.927 | 26.660 |
| cpu-k16 | 1.6546 | 2.9246 | 2.9018 | 33.583 | 43.766 |
| offload-k16 | 3.0026 | 13.6047 | 13.4984 | 33.201 | 9.409 |

The long target's q8_0 KV allocation is 2448 MiB at the declared capacity:
**1415.25 MiB on GPU and 1032.75 MiB on CPU**. Nominal all-layer KV payload is not
the amount of GPU weight capacity displaced. Complete target/draft allocation
lines are in the accompanying tables.

## Bounded numerical characterization and remaining state obligation

The frozen eight-prefix fixture includes all four v0.14 baseline rebuild
mismatches and four ordinary generated-position-32 controls. Five target-only
processes vary full-prefix prefill, one-token microbatching, incremental prefix
construction, decode threads, or one GPU layer of placement. Every comparison
holds the supplied token prefix fixed. Incremental requests begin with the
original prompt, then add one frozen historical token at a time; intermediate
predictions do not choose the supplied trajectory.

| Target-only factor | Requests | Final historical matches | All historical matches | Unique supplied-prefix matches |
| --- | --- | --- | --- | --- |
| prefill | 8 | 4/8 | 4/8 | 4/8 |
| microbatch | 8 | 6/8 | 6/8 | 6/8 |
| incremental-reference | 477 | 8/8 | 477/477 | 358/358 |
| threads | 477 | 8/8 | 477/477 | 358/358 |
| placement | 477 | 7/8 | 473/477 | 355/358 |

| Reference | Candidate | Stratum | Flips/positions | Max probability infinity distance |
| --- | --- | --- | --- | --- |
| incremental-reference | threads | selected historical divergence | 0/4 | 0.000000000 |
| incremental-reference | threads | fixed ordinary position | 0/4 | 0.000000000 |
| incremental-reference | placement | selected historical divergence | 1/4 | 0.014353655 |
| incremental-reference | placement | fixed ordinary position | 0/4 | 0.006212476 |
| incremental-reference | prefill | selected historical divergence | 4/4 | 0.047704415 |
| incremental-reference | prefill | fixed ordinary position | 0/4 | 0.014610095 |
| prefill | microbatch | selected historical divergence | 2/4 | 0.037716540 |
| prefill | microbatch | fixed ordinary position | 0/4 | 0.012519165 |

All 1,447 raw diagnostic requests are preserved, including intermediate decisions.
The analyzer checks every HTTP request/response, exact expected prefix sequence,
hash and native cache counter. Incremental extensions report `cache_n=length-1`
and `prompt_n=1`; first requests have zero cache hits. Forty final requests return
all 152,064 serialized log-softmax values. Their token-indexed float32 vectors
reproduce from raw bodies; all 40 emitted tokens equal their returned argmax,
and none uses the zero-probability sentinel.

The API exposes log-softmax, not raw logits. Top-two log-probability gaps approximate
raw logit gaps up to serialization; absolute raw-logit infinity distance is not
available. The report uses reconstructed probability infinity distance and the
probability-space margin condition, with no observed bound violation. The four
selected incremental-reference top-two gaps are 0.02300, 0.01318, 0.05302 and
0.03559. Small margins alone do not validate state or establish small perturbations.

This establishes a concrete prefix-construction explanation for the four selected
historical rebuild reversals without requiring speculation. It does not settle
all 38 historical configuration/prompt divergences. The selected sample cannot
support a population flips-per-1000 rate. Nor does the native cache accounting
compare every KV byte, causal mask or position to an independent implementation.

The pinned acceptance function compares proposed IDs to target decisions and
truncates at first rejection, with replacement/bonus and suffix handling in the
server. Reconciled counters and source inspection support that procedure. However,
b10919's speculative token-probability path has a TODO; these stock API receipts
cannot independently audit each accepted token against a correctly populated
verifier state. **Do not call this a 100% independent speculative argmax/KV audit**
or relabel historical cross-path identity failures as passes.

## Delivery, reproduction and next decision

The stock setting is a useful short-context candidate and changes which fixed
draft length wins. Its value must include actual memory placement, context,
discarded proposals and all request work. Verification is not assumed flat, and
no universal confidence threshold or calibrated per-CPU-layer slope follows.
The profiler supports the proposed movement mechanism; it does not provide
separate draft/verify/coordination critical-path timers.

Issue #28 records the completed bounded setting/context study; #27 retains the
independent speculative-state obligation. The stock milestone stays open.
Representation #25 remains deferred for opportunity cost, with its bandwidth
argument conditional on the specified full payload. Adaptive #26 remains optional
and should compare complete costs of (draft length, backend, context). No custom
shared-resident runtime is selected by this release.

See [all per-request, repeat, acceptance and allocation tables](verification-offload-tables.md).

The raw archive contains **9,617 files** and is **229,600,806 bytes** compressed.
Every member was extracted into a fresh directory and checked against its original
SHA256 and size; original files were rehashed after extraction. All **five analyses**
(timing, numerical, failed capture, corrected CPU capture and corrected CUDA capture)
then reproduced byte for byte from the restored source and receipts. The inventory
and reanalysis receipt are retained beside the compact result JSON files.

The completed main matrix has **23 groups and 114 requests**: 23 warmups, three
mechanism requests, 72 scored short requests and 16 scored long requests. There
are no main-matrix failures or unrun conditions. Including the five numerical
processes and three profiler attempts gives 31 native process attempts and 1,567
inference requests; health, tokenization and template calls are not inference
requests in that count. The main matrix peaks at **14,721.12 MiB** total GPU-used,
with minimum host available **8.787 GiB**. Both corrected captures and all five
numerical runs also pass their complete sampled resource traces.

Offline analysis independently reconstructs canonical acceptance from exact native
log byte ranges, reconciles counters, verifies completion payloads against frozen
inputs/decoding, and compares stored responses to raw HTTP bodies. The final
release candidate must pass the complete **250-test** CPU fixture suite in both
torch 2.10.0+cu130 and 2.12.0+cu130 on a clean checkout; actual final-tip test and
review receipts are release assets. No model inference is repeated for publication.

To reproduce after extracting the raw archive into a workspace, use the archived
`runs/reproduce-verification.py` with `--archive verification-v0150-raw` and
`--analysis verification-final-analysis-v1`, arranging the extraction at
`runs/restore-verification-v0150-raw`. Copy the archived `runs/reproduce-verification.py`
and `runs/verification-final-analysis-v1` into the workspace's own `runs` directory
as well; those provide the runner and original analysis copies that its comparison
expects. Use the documented Python environment with NumPy and psutil. It uses the included analysis-source snapshot
and compares all five regenerated JSON files to both restored and original
analysis copies. The publication also retains the exact invocation receipt.
The tracked `scripts/archive_verification.py` creates and verifies an archive of
explicit receipt paths; models and environments are deliberately outside this bundle.
