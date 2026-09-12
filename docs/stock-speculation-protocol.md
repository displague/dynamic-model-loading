# Stock heterogeneous speculation: prospective protocol

Status: prospective. Freeze this document, artifact catalogue, authored workloads
and measurement harness in a pushed clean commit before target inference. Artifact
downloads, source inspection and binary help are preparation, not scoring.
Placement and thread calibration are explicitly separate from evaluation and are
retained. Follow [ADR 0003](adr/0003-verified-speculation-boundary.md).

Implementation clarification before any target inference (2026-09-12): the stock
server's decode interval excludes the first generated token, which comes from
prefill. Report both the declared emitted-token/decode-time ratio and the stock
decode-step rate using `predicted_n - 1`, plus total request output/time. Never
attribute the prefill token to a verification cycle. The smoke user message keeps
the v0.11 one-line-answer prefix; all new runs use the authored workload system
message and the target's embedded chat template. This is a target-relative smoke
comparison, not a cross-release 1.5B utility rescore. No measurements preceded
these clarifications.

Stock-placement clarification before inference: `--spec-draft-ngl all` places the
draft transformer and output layers on the GPU; the stock loader keeps input
embeddings on the CPU. Charge and report that actual split. Here, resident drafting
means resident draft transformer/output computation, not zero CPU allocations or
an entirely GPU-resident model file. Explicit CUDA device arguments and parsed
offload counts must confirm the requested placement; CPU backend fallback cannot
qualify as a feasible GPU configuration.

## Question and fixed substrate

Does a draft with GPU-resident transformer/output layers improve committed-token throughput for an offloaded
Qwen2.5-32B-Instruct Q4_K_M target, after charging target residency displaced by the
draft? The primary comparison is each best complete measured configuration under
the same total resource allowance. This is a bounded configuration frontier, not
proof of an optimum across all kernels, placements or models.

Use official llama.cpp **b10919**, commit
`d3146f2b56c2db4711ac8391871c9e529d1946d7`, Windows x64 CUDA 13.3 release binaries
and matching runtime DLLs. Verify release SHA-256, executable/DLL hashes, version,
help and checked-out source. Preserve all effective arguments and relevant
environment variables. No synthetic acceptance, patched runtime or modified target.

The artifact catalogue pins official Qwen Q4_K_M target shards, official 0.5B/1.5B
Q8_0 drafts, and bartowski's IQ2_XS 32B compressed control. Q8_0 limits an additional
draft-quantization confound for the small models. Download into a single explicit
local directory and verify every file against repository LFS hashes. No conversion
or merge is needed for the split target. IQ2_XS is independently quantized and
shares no allocations with the target. If storage or runtime host/GPU memory makes
it infeasible, retain that resource result; do not substitute another target after
seeing acceptance. Small-draft measurements remain independently useful.

## Resource and placement calibration

One request, context 4,096, batch 256, microbatch 256, flash attention on, F16 KV for
both models, CPU target sampling, default quantized kernels, mmap loading and lazy
tensor reads off. Set fitting off and record explicit GPU layer counts. Disable
prompt caching across requests and reset each request's logical context. Do not
alter unrelated services or power settings. Record GPU driver, CPU topology,
available physical/commit memory, clock/temperature/power observations and process
memory. Background load is a limitation rather than an invisible constant.

Use a common **15,000 MiB total device-used** budget including the observed desktop
baseline, with at least **2 GiB host available physical memory** during each measured
request. These are operating limits, not evidence of the minimum achievable memory.
Poll resource use at 200 ms where available. Report sampled peaks and sampling
limits, not guaranteed instantaneous peaks. Record process read/write and page-fault
counters plus system available RAM; fault counts alone cannot distinguish hard
faults or prove absence of storage reads.

For target-only and each draft representation, keep all draft layers on the GPU.
Begin target placement at 48 layers (small/no draft) or 14 (IQ2_XS); increase or
decrease one layer at a time to find the largest placement passing startup and the
calibration request within both budgets. Log every failure, effective allocation
and candidate. Use K=16 to reserve the largest measured speculative shape; do not
retune residency separately after observing each evaluation K's speed. Compare the
largest feasible placement and four fewer target layers on the calibration prompt
only. For both, compare target CPU thread counts 8, 16 and 24 (batch threads 24;
draft CPU threads 8). Two calibration requests per setting; choose lowest median
request decode time, ties within 1% favor more residency then fewer CPU threads.
This finite search defines 'best measured placement'; it is not a global optimum.

The calibration prompt is a separate authored sustained explanation in
`data/stock-speculation-workloads.json`; cap 128 new tokens. No evaluation prompt
contributes to allocation or thread selection. Before evaluation, write a frozen
selection receipt. If an evaluation exceeds a budget or errors, retain its failure
and exclude it from qualifying speed claims; a repaired run requires an explicit
amendment and a fresh output directory.

## Matrix and workload

1. Target only at its selected placement/thread count.
2. Target + 0.5B Q8_0 draft, K=4/8/16, selected shared placement.
3. Target + 1.5B Q8_0 draft, K=4/8/16, selected shared placement.
4. Feasible target + IQ2_XS 32B non-sharing draft, K=4/8/16.
5. Target only at each distinct draft configuration's target placement/thread
   count. These controls isolate residency/CPU-schedule changes; the primary
   baseline remains the best target-only configuration.

Use `--spec-type draft-simple`, `--spec-draft-n-max K`,
`--spec-draft-n-min 0`, `--spec-draft-p-min 0`,
`--spec-draft-ngl all`, and `--no-spec-draft-backend-sampling`. K is a maximum:
EOS, remaining context and remaining generation budget can shorten a draft.
Target-only uses `--spec-type none`. Verify all spellings in pinned help.

Six authored sustained workloads (three code, three prose/data reasoning), each
capped at 256 generated tokens, supply throughput measurements. The old twenty
v0.11 balanced tasks are correctness smoke fixtures, each capped at 64; they do not
measure steady-state decoding. Do not score the new target against the 1.5B Gate A.
Use the target's embedded chat template with a generation prompt, record rendered
text and token IDs, and pass those same IDs to every candidate. Record full tokenizer
vocabulary/merge/special-token fingerprints for all exact artifacts and the stock
compatibility check; family names alone are insufficient.

Greedy decoding (`temperature=0`, no repetition/frequency/presence penalty), fixed
seed 20260912, no grammar, no added stop strings, native EOS enabled. Return raw
generated token IDs. Compare actual output lengths, prompt lengths, context position
and stop reason. A cap is a maximum, not forced generation. Do not suppress EOS to
manufacture steady-state work.

Run the selected target-only reference first and preserve it. Then run three
repetitions of the sustained matrix: a seeded shuffle of configuration order for
repeat 1, reversed for repeat 2, another seeded shuffle for repeat 3. Shuffle prompt
order with the same recorded generator. Each fresh server gets an unscored warm-up
on the calibration prompt; preserve its output and cost separately. Run smoke
fixtures once per configuration, separate from the timed sustained repetitions.
Record startup, warm-up, prefill and decoding separately. Startup is not guaranteed
storage-cold because the OS file cache is not forcibly evicted.

## Telemetry and correctness

Set stock `LLAMA_TRACE=1`, timestamps and plain logs to retain each accepted/attempted
draft count. Capture every HTTP request/response, return tokens, timings and metrics,
server log ranges, resource samples and source/config/artifact fingerprints.
Keep the same telemetry enabled in target-only and draft runs; its cost is included.
No CPU tests, code review, compression or downloads run during scored GPU timing.

Parse each final verification event once; checkpoint-replay events remain separately
classified. Check sums against stock counters. Report accepted-prefix survival
P(A>=i), attempted lengths, verification rounds, rejected work and committed output.
End-of-request truncation means 'accepted+one target token' can exceed actually
emitted tokens; retain both. Do not infer a distribution from an aggregate acceptance
percentage. If logs/counters are insufficient, explicitly report which curve is
unavailable. Log timestamps bound intervals but do not isolate draft, verification
and coordination component times; do not label them measured component timings.

The reference is the declared target-only greedy path. Compare every candidate's
generated IDs and stop reason against that reference for the same input/cap.
Also compare placement-matched controls and target-only repeats. When a path first
differs, replay the common prefix plus a one-token decision using target-only at
both placements and the candidate configuration, with top probabilities enabled
in this separate untimed diagnostic. This isolates some placement/prefix effects;
it cannot reproduce every original verification-batch kernel shape. A mismatch
remains unresolved unless evidence actually identifies its cause. Do not change
the reference or claim identical greedy output when observed sequences differ.

Keep compact JSONL records, output IDs and strings, scalar probabilities/timings and
small diagnostic reservoirs. Do not spill full vocabulary logits by default or
claim exact KL from top-k records. No stochastic distribution-preservation result
is claimed by a greedy test.

## Analysis and decision

Publish all cases, failures and resource exclusions. Compute aggregate committed
tokens/s as total emitted tokens divided by total measured decode seconds; also
report per-prompt and per-repeat figures, request latency, prefill, time-to-first
token where exposed, and total request time per emitted token. Report EOS/capped
mix and changing output lengths alongside speed. For mismatching outputs, timing
is a workload observation, not an exact-output acceleration claim.

Use episode-level paired summaries across the six sustained prompts, retaining
repeat spread; this small pilot is not a population confidence claim. Optimize no
test thresholds after scoring. There is no mandatory 2x/3x gate. A candidate is a
credible practical improvement only if output fidelity, resource feasibility and
paired timing evidence support that statement together.

The subsequent resource model uses actual cycles or measured request aggregates:
sum(time)/sum(committed tokens), with explicit boundaries. For hypothetical cycle
cost C, measured baseline seconds/token t_b and desired speedup S, the required
expected committed length is S*C/t_b. Hypothetical substitutions remain sensitivity
analyses. Do not scale 1.5B acceptance to 32B or reuse v0.12 reconstructed latency.
Results choose the next experiment; selected-row CPU work and custom shared-state
drafting remain optional. Retain negative outcomes and publish a bounded conclusion.

Primary implementation references: [pinned speculative docs](https://github.com/ggml-org/llama.cpp/blob/d3146f2b56c2db4711ac8391871c9e529d1946d7/docs/speculative.md),
[pinned server interface](https://github.com/ggml-org/llama.cpp/blob/d3146f2b56c2db4711ac8391871c9e529d1946d7/tools/server/README.md),
[pinned verification loop](https://github.com/ggml-org/llama.cpp/blob/d3146f2b56c2db4711ac8391871c9e529d1946d7/tools/server/server-context.cpp).
