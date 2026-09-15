# v0.41: packets retain decode economics but lose prefill gates

The 49.901-second screen (44.283 s worker, 5.618 s audit) preserves exact checked
FP16 logits, including warmups, and 48 scored greedy tokens per condition under
the same 4800 MiB CPU-first boundary. Overall acquisition/runtime and decode gates
pass; BOTH prefill gates fail. Decision: stop uniform long-prefix expansion,
not erase the surviving decode result. Independent raw replay matches exactly.

| Phase / metric, summed over three scored episodes | Dense stream | Packet |
|---|---:|---:|
| Prefill outgoing H2D | 5,033,164,800 B | 3,469,199,688 B |
| Prefill forward-call wall | 0.740460 s | 1.083512 s |
| Scalar decode outgoing H2D | 75,497,472,000 B | 2,513,089,216 B |
| Scalar decode forward-call wall | 10.416656 s | 2.221702 s |
| Whole-episode outgoing H2D | 80,530,636,800 B | 5,982,288,904 B |
| Whole-episode wall, including final records | 11.262525 s | 4.187751 s |

Prefill uses 68.9268% of dense-stream bytes and 146.3295% of its call time: a 31.07%
traffic saving does not meet the 50% bar and does not prevent 46.33% slower prefill.
Decode uses 3.3287% of stream bytes and 21.3284% of its call time. Whole packets
save 92.5714% outgoing H2D and 62.8169% episode wall, but that positive total does
not turn the failed prefill gates into passes. Phase call timing excludes final
episode serialization; the whole-episode denominator charges it.

Packet scored episodes finish 1.307912/1.426948/1.452891 s; TTFT
0.345108/0.404136/0.360626 s. Dense TTFT 0.233449/0.240511/0.269859 s. All packet
episodes pass the 5 s bar. Stream warmup 5.374834 s and packet warmup 1.470636 s are
preserved; the useful-latency gate concerns scored packet episodes, not the stream
warmup. CPU load/placement 7.663518 s; cold shared-worker-entry-to-first DENSE logit
17.201395 s, not packet-first cold TTFT. Supervised wall additionally includes
module startup and workload-integrity precheck.

## Memory and correctness

Peak sampled global GPU 4,763,783,168 B (about 4543 MiB); peak allocator reserved
4,211,081,216 B. Both controls have that same footprint and fit 4800 MiB. Max KV 527
positions/172,687,360 B, consistent FP16 throughout. No hidden preconstruction CUDA
allocation/current/peak, no full 2.7B GPU preload, unchanged 3,625,472,000 B resident
non-outgoing parameters. Host packed/original-layout copies, workspace, packets,
full-prefix activations, readbacks, finite checks, metadata and records charged.
Whole-worker peak RSS 9,369,878,528 B; host minimum 7,024,128,000 B. No unique access,
optimized Q4, physical-small-card OOM or fully resident 2.7B reference claim.

Inputs are 512-token prefixes of concatenated KNOWN archived WikiText records,
not held-out natural long documents. Records/groups and literal config texts were
frozen before generation. Continuations remain capped at 16; all reference IDs/
stops and checked logits match exactly. This does not establish task quality.
Baseline-first order and a single screen remain limitations. v0.40 used different
texts: this is NOT a paired 32/512 context-only causal contrast. The prefill/decode
asymmetry measured WITHIN this 512-token screen is the actionable observation.

## New hypothesis, not expansion of the failed uniform policy

Use dense contiguous outgoing transfers for multi-token prefill, then exact-row
packets for scalar decode, keeping the same host banks/workspace and FP16 KV.
This deliberately spends more prefill bytes to avoid union readback/gather/scatter.
It must be measured against always-packet and always-stream on the same frozen
workload; it is not a claimed win from adding the best historical timings together.
A separate short protocol can test that phase-switched physical grain. Do not
extend this failed uniform-prefill candidate into an hours-scale matrix.

[Protocol](context-screen-protocol.md),
[compact receipts](../results/context-screen-20260915/summary.json).
[LLM in a Flash](https://arxiv.org/abs/2312.11514),
[PowerInfer](https://arxiv.org/abs/2312.12456), and
[OPT](https://arxiv.org/abs/2205.01068) remain the physical-grain/sparse-model
antecedents. Neither set unions nor phase specialization are claimed inventions.

657 baseline CPU tests pass; historical v0.37-v0.40 summaries replay unchanged.
Source review approved, independent raw replay exact. Restored/hash-checked raw
archive `context-v041-raw.zip`: 64,918,727 B, 142 members, SHA256
`190a02821b42d59a53a706d0045f667caae841124d178095849c8e9ae53df270`.
Complete only issue 56/delivery 5; milestone 10 remains open. Final clean-tip full
suite attaches to the release. No native pivot or optimized-low-bit comparison.
