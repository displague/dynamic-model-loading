# v0.37: retention saves transfers but loses wall time

The frozen short screen completed in34.209s including27.595s worker and6.614s
audit. A second raw audit matches exactly without rerunning inference. All checked
candidate logits are bit-identical to their resident references, including the
declared warmups. All48 scored outputs per condition match, with independent
fresh KV state per episode. Faithfulness/resources/acquisition pass; runtime fails.

| Scored condition | Tokens | Weight + metadata H2D | End-to-end wall |
|---|---:|---:|---:|
| Resident FP32 reference | 48 | 0 during generation | 0.483145s |
| Demand-only packets | 48 | 4,317,841,200 B | 1.891826s |
| 1024-row/layer LRU | 48 | 2,330,056,432 B | 4.241058s |

Retention saves46.0365% H2D but takes2.2418 times the packet wall. It finds243,218
cache hits among526,566 observed required rows. Decode H2D falls from2,848,852,200
to860,023,336 B; prefill is slightly more expensive due to cache index work.
The192MiB cache is allocated to both conditions and is NOT a VRAM saving.

Charged forward-call time (prefill plus decode, excluding between-call/receipt
work) is1.593093s for packets versus1.944361s retained. Thus serialization is a
substantial extra cost but removing it would not turn these forward totals into
a retention speedup. No profiler establishes individual kernel bottlenecks.
The end-to-end criterion stays unchanged; no favorable subset replaces it.

## Resources and scope

Host outgoing weights1,610,612,736 B; GPU cache201,326,592 B; workspace67,108,864 B;
device packet67,174,400 B and equally sized pinned packet. GPU non-outgoing
parameters3,652,419,584 B. Host maps and Python receipts are included in process
RSS, with peak12,820,336,640 B and minimum host availability5,163,614,208 B.
Candidate episodes pass4800MiB sampled/allocator checks. The reference phase uses
6500MiB allowance; peak whole-worker NVML5,864,787,968 B includes that phase.
Initial full-GPU reference loading explicitly prohibits a CPU-first capacity claim.
Original reference parameters5,263,032,320 B are copied H2D then D2H at transition;
candidate construction copies3,652,419,584 B H2D. Setup/warmups stay in raw receipts.

Three new authored32-token prefixes and16-token continuations are a minutes-scale
screen, not long context, task quality, a scale result, optimized Q4 comparison or
statistical significance. Stock32B target-only/small-draft profiles remain different
deployment baselines, not numerically comparable checkpoints. No native pivot.

## Decision and next hypothesis

Stop this LRU implementation as a latency candidate; do not expand its matrix.
The acquired-byte signal is real, but cache management and its charged records do
not repay the saved movement here. A different causal early-transfer implementation
may test waiting/overlap with exact observed demand completion, under a new protocol.
No Bayesian field follows automatically from observed temporal reuse.

[Protocol](retention-screen-protocol.md), [course](anticipation-research-course.md),
[compact evidence](../results/retention-screen-20260915/summary.json).
[LLM in a Flash](https://arxiv.org/abs/2312.11514) supplies windowing/bundling
antecedents; [DejaVu](https://proceedings.mlr.press/v202/liu23am.html) motivates the
separate anticipation question. No invention of caching or sparse algebra claimed.

639 CPU tests pass. The pre-inference review's exact-threshold floating-point bug
was fixed and regression-tested BEFORE measurement without changing the gates.
Raw archive70,387,510 B,151 members, SHA256
`4c7b55de52096c9a27bfbc39bd7cde3f5180c96c2be3e924f078316ee95d9bdd`.
Only bounded issue52 completes; this is delivery1 of the authorized4--8, and
milestone10 remains open.
