# Stock operation-offload threshold and populated-context control

Prospective follow-up to v0.14 under [ADR 0003](adr/0003-verified-speculation-boundary.md).
Commit and push this protocol and its executable harness before inference. Historical
v0.14 identity failures, raw records and timing definitions remain unchanged.

## Question and fixed substrate

Does lowering the stock CUDA operation-offload threshold make speculative target
verification more economical by moving host-weight operations to CUDA? The reviewer
forecast of approximately 25 emitted tokens/s is recorded before measurement. It is
a forecast, not a nomination threshold or a target for further tuning.

Use the unchanged b10919 executable and DLLs, source
`d3146f2b56c2db4711ac8391871c9e529d1946d7`, Qwen2.5-32B-Instruct Q4_K_M target
and Qwen2.5-0.5B-Instruct Q8_0 draft from the existing
[artifact catalogue](../configs/stock-speculation-artifacts.json). Rehash every
required file per fresh process. No model acquisition or runtime rebuild is needed.

The pinned CUDA source reads `GGML_OP_OFFLOAD_MIN_BATCH` during backend registration,
defaulting to 32. MUL_MAT uses `op->ne[1]` as its operation batch size. The scheduler
can assign eligible host-weight operations to CUDA. Its source labels that branch
`1.off`, but **reason labels are compiled out** in the pinned build. Compare actual
backend/source assignments; never interpret the missing reason tag as zero offload.
This is an operation preference, not a contiguous-layer-copy guarantee. Maximum
draft length does not establish every operation's actual batch shape.

The historical harness clears inherited LLAMA/GGML/CUDA settings. The new harness
likewise clears them, case insensitively, then explicitly sets and records
`LLAMA_TRACE=1`, `GGML_OP_OFFLOAD_MIN_BATCH=8|32`, and `GGML_SCHED_DEBUG=0|2` in the
child process only. The parent and machine environment remain unchanged. Record the
exact command, source/protocol/input/catalogue hashes, removed override names and
effective values. Use fresh output directories and fresh owned processes throughout.

## Short-context matrix

Hold target placement at **44 GPU layers including output** (43 transformer layers,
21 CPU transformer layers), target threads 24, target batch threads 24, draft
threads 8, draft batch threads 24, draft fully GPU resident. Use 4096 context,
256 batch/microbatch, f16 target/draft KV, flash attention, no context shift,
`--load-mode none`, lazy loading off and cache RAM zero, matching the v0.14 family.

| Condition | Threshold | Maximum draft K | Operation offload |
|---|---:|---:|---|
| cpu-k4 | 32 | 4 | enabled |
| cpu-k8 | 32 | 8 | enabled |
| cpu-k16 | 32 | 16 | enabled |
| offload-k8 | 8 | 8 | enabled |
| offload-k16 | 8 | 16 | enabled |
| disabled-k16 | 8 | 16 | disabled with `--no-op-offload` |

Names beginning `cpu` describe the hypothesized verification path; assignments
must establish whether it is true. Operation offload changes apply to prefill too.

First run three **mechanism diagnostics**, in the order cpu-k16, offload-k16,
disabled-k16. Each uses a 32-token library-explanation warmup and 32-token code-cache
request, `GGML_SCHED_DEBUG=2`, verbosity 6. Retain graph/split assignments, exact log
intervals, acceptance events and native decode shapes. Graph assignment counts are
not executed-operation counts or physical copy bytes; graphs can be reused and
printed sizes are rounded. Attempt a separate bounded Nsight CUDA-copy/kernel
capture if stock diagnostics cannot quantify movement. Capture failure is retained;
absence of a supported profiler must remain an explicit traffic limitation. Neither
verbose nor profiled measurements enter throughput rankings.

Then run all six conditions twice with scheduler debug zero and verbosity 4. The
first order is cpu-k4, offload-k16, cpu-k8, disabled-k16, offload-k8, cpu-k16; the
second reverses it. Each fresh process has the same 32-token warmup and six sustained
v0.14 prompts with their original 256-token caps. Within each process shuffle the
six prompts using seed `20260913 + repeat`. Use the original rendered token file
whose SHA256 is `184a3734aabc39f524820ec8f63a9b2fd097a5cb5c56c2f1706a304cef345829`.
No task rewriting, altered cap or forced EOS suppression after observing output.

## Populated long context

Run target-only, cpu-k4, cpu-k16 and offload-k16 twice. The first order is target,
cpu-k4, offload-k16, cpu-k16; the second reverses it. Use **18432 context capacity,
exactly 16384 input tokens, 128-token output cap**, target placement **38 including
output**, and **q8_0 target and draft K/V**. Other settings stay fixed. This separate
placement is predeclared to accommodate KV/workspace growth; it is not an optimum
search or a measurement of fourteen displaced layers. If it fails the common memory
budget, preserve that failure and stop this matrix pending a fresh allocation protocol.

The first target process prepares two deterministic inputs, code-cache and data-audit.
Insert synthetic labelled engineering records into the middle of the documented
chat template. Tokenize header, filler and suffix separately; trim only filler to
make exactly 16384 tokens, retaining the entire final request and generation suffix.
The resulting token sequence, rather than an assumption about retokenizing concatenated
text, defines the input. Preserve all tokenization HTTP receipts and construction
counts before generation. Freeze that file for all later runs. A short 32-token
warmup precedes the two long requests. This is a controlled populated-context workload,
not a long-context reasoning benchmark or proof of useful 16K agent behavior.

The target-only q8_0-KV run is the local numerical reference. Do not require it to
reproduce f16-KV output. Report GPU and CPU KV allocations separately, actual prompt
and output lengths, prefill, decode and whole-request times. Two long prompts and two
repeats are a bounded sensitivity control, not a comprehensive working configuration.

## Measurements, failures and interpretation

Greedy decoding, neutral penalties, fixed sampler/seed, single request, EOS respected;
retain emitted IDs, stop reasons, raw HTTP bytes, native timings, acceptance events,
checkpoint replay flags, server logs and allocation records. Reconcile events with
native draft counters before reporting accepted-prefix survival. Do not infer a
survival curve from aggregate acceptance percentage. Record actual verifier shapes
where native logs expose them; distinguish those records from inferred K+1 bounds.
Verbose logs repeat acceptance in a debug `new n_tokens` message; keep that raw
message but exclude it from canonical acceptance-event counts.

Report per-prompt and aggregate **emitted tokens / measured request seconds** and
native prompt/decode timing separately. Native decode timing is runtime-reported,
not an independent timer around verification. Inter-acceptance log timestamps may
describe cycle intervals but do not isolate draft/verify/control components. Do not
import v0.12 scheduling estimates or v0.9 bandwidth as measured new cycle costs.

Use the v0.14 common sampled bounds: total device usage including desktop <=15000
MiB, host available >=2 GiB, 200ms NVML/host/process sampling. Fail closed on missing
telemetry or any observed violation, including startup and inter-request intervals.
Sampling can miss short peaks and does not establish absence of paging. Native
allocation logs and host I/O changes supplement, not replace, those caveats. Keep
CPU tests, reviews, compression, downloads and unrelated benchmark processes out of
timed runs. Do not modify user services or power settings.

An in-budget mechanism or timing failure is retained; do not silently choose a
different placement, threshold or thread count. A failed configuration can be omitted
from later scored requests only with its failure and reason explicitly recorded.
Retries require a fresh output directory and a documented apparatus/external cause.
Never replace slower or divergent successful observations.

Compare threshold variants at matching placement before any best-configuration
claim. The offload-disabled control affects prefill and decode, so whole-request
differences alone do not isolate verification. A speed change with no confirmed
backend mechanism remains unexplained. A successful stock path can make custom
representation work unnecessary; no source patch or shared-resident runtime follows
automatically. Keep #25 deferred, #26 contingent on measured K/backend opportunity.

## Bounded fidelity work alongside performance

#27 has two distinct obligations: (1) accepted tokens must follow target decisions
on the intended prefix, positions, mask and target KV, with correct suffix removal,
replacement/bonus and stopping; (2) aligned-prefix numerical effects of threads,
placement and batch shape must be characterized separately. The v0.14 rebuilds do
not preserve original KV history, so their agreement cannot settle (1).

The stock acceptance function compares each proposed ID with the sampled target
decision and truncates on first rejection. Source inspection and reconciled counters
support that procedure, but cannot independently validate every target logit or KV
entry. In particular, b10919's speculative token-return path has a probability
reporting TODO. Do not call counter consistency a 100% independent argmax audit.
A separate bounded observer protocol must precede additional state/numerical
measurements; this performance experiment can finish with #27 explicitly open.

Keep the fixed historical target-only output reference. Distinguish procedural
greedy behavior under a declared execution path from cross-path token identity.
Near ties alone do not exonerate an implementation: compare perturbation magnitude
and aligned non-divergent positions as well as selected first-divergence cases.

The full-resident-weight draft bandwidth argument is conditional on reading that
whole payload per token; it does not bound substantially smaller representations.
The later stopping decision concerns incremental **complete cycle** cost per added
committed token, including threshold crossings. No universal confidence cutoff is
adopted here.
