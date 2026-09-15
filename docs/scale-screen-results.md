# v0.40: larger CPU-first FP16 packets pass the short acquisition screen

The34.122-second screen (29.931s worker,4.191s audit) passes every frozen component
gate. Original OPT2.7B FP16 starts CPU-only and runs under4800MiB without a full
GPU model preload. Packets save95.2499% outgoing H2D and64.6241% measured episode
wall versus direct contiguous FP16 streaming. All checked full-vocabulary logits,
including warmups, are bit-identical to the dense streamed reference;48 scored
greedy tokens per condition match. A second raw replay reproduces the entire summary.

| Scored FP16 condition | Weight + index H2D | Wall,48 tokens | Peak sampled GPU |
|---|---:|---:|---:|
| Direct contiguous dense stream | 80,530,636,800 B | 6.487486s | 4,530,999,296 B |
| Exact observed-row packets | 3,825,246,984 B | 2.295006s | 4,530,999,296 B |

Three new authored32-token prefixes with16-token generation cap. Stream TTFTs
0.129209/0.128431/0.131604s; packet TTFTs0.100843/0.085685/0.086818s. Packet episode
walls0.800805/0.728631/0.765569s, all below the5s useful-latency bar. Dense episodes
2.052867/2.072080/2.362540s. These are baseline-first, single-screen observations,
not randomized steady-state confidence intervals or complete agent tasks. They
measure a base language model's capped continuation, not instruction-following quality.

Declared stream warmup2.707457s and packet warmup0.896350s remain archived. CPU
load/placement setup5.598647s. Cold worker-entry-to-first dense-reference logit
13.274175s includes artifact hashing but excludes Python module startup; process
startup is included in supervised worker wall. This is NOT a measured packet-first
cold TTFT, because the reference phase comes first. No hidden first-use discard.

## What the memory result means

Full unique FP16 parameters5,303,193,600B exceed the4800MiB allowance by270,028,800B
before KV/runtime. Both offloaded controls actually run inside it: peak global
sampled GPU4,530,999,296B (about4321MiB), peak allocator reserved3,978,297,344B.
Current AND historical peak CUDA allocated/reserved bytes are all zero before
construction. Only3,625,472,000B of non-outgoing parameters move H2D; tied embeddings
remain tied. There is no fully resident2.7B GPU reference or return-to-CPU phase.

Outgoing host weights1,677,721,600B plus an equal original-layout host copy support
a strong stream baseline and are charged to BOTH conditions. CUDA workspace
52,428,800B; CUDA/pinned packet52,510,720B each; no retained row cache. FP16 KV is
327,680B per position (47 positions maximum). Whole-worker peakRSS9,541,406,720B,
minimum available host memory8,068,919,296B. All construction/reference/warmup/scored
phases pass the same memory cap and host floor. Copies, CPU gathering, indices,
activity/logit readback, staging/scatter, resource sampling and records are charged.

The capacity gain belongs to outgoing offload. Packet's contribution is cheaper
acquisition at that same footprint, not uniquely enabling a model streaming cannot
run. The actual GPU is a16GiB RTX5080 Laptop with an imposed4800MiB limit, not a
physical4.8GiB-card OOM test. Full FP16 residency exceeding this imposed budget
does NOT imply optimized Q4/Q8 cannot fit or run faster.

## Scope, provenance, and next question

Reference is dense CONTIGUOUS STREAMED FP16 on2.7B. The corresponding direct-stream
representation matched resident FP16 at1.3B in v0.39; fully resident2.7B equality
was not independently measured. No FP32 quality, optimized quantization, long
context, SSD I/O, target-scale speculation, Qwen32B or deployment claim. Same
baseline .venv and actual CUDA execution; pinned model/file hashes and package,
arithmetic, memory and source receipts are in the archive. No native change.

The four-delivery course now has an answer beyond its FP32 artifact: this exact
acquisition mechanism survives FP16 and2.7B CPU-first scaling against a strong
same-precision stream control. It does not beat warm FP16 when residency is feasible
(v0.39), and its LRU/early-forecast variants lost economics (v0.37/v0.38).

Before native work, the remaining high-value questions are a qualified optimized
low-bit comparison on the same sparse checkpoint and durability under larger
prefill/context/block unions. Each needs a new frozen minutes-scale protocol;
passing this screen is not that validation. ReLU specialists/decision rankers were
not closed by the historical Qwen-only v0.31/v0.34 studies. Preserve that distinction,
but do not manufacture additional releases by rerunning the failed LRU/forecast.
Four completed deliveries satisfy this course; milestone10 remains open.

[Frozen protocol](scale-screen-protocol.md),
[compact evidence](../results/scale-screen-20260915/summary.json).
The inspirations remain [OPT](https://arxiv.org/abs/2205.01068),
[PowerInfer](https://arxiv.org/abs/2312.12456), and
[LLM in a Flash](https://arxiv.org/abs/2312.11514). The research contribution here is
the implemented physical-acquisition screen and its measured boundaries, not a
claim to have invented observed sparsity, offload or packet transfers.

654 CPU tests pass; exact v0.37/v0.38/v0.39 historical replays remain unchanged.
Required source review and the final no-preload strengthening are approved.
Raw `scale-v040-raw.zip`:30,125,254B,141 members, SHA256
`640b0a5aeee48543e0302518fa2e11219dcebf335e126b4ce649caa644235b95`.
Every archive member was restored/hash checked. No checkpoint weights redistributed.
Complete only issue55/delivery4; final clean-tip full-suite receipt attaches to release.
