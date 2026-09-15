# v0.33: compact exact-active-row packets pass the short acquisition screen

Third delivery of the v0.31--v0.36 course, [#48](https://github.com/displague/dynamic-model-loading/issues/48).
The corrected screen passes both frozen gates while every checked logit and
emitted token matches the original dense reference exactly. This is a measured
physical-loading gain against the declared streaming controls, not a faster
fully resident model, useful capacity frontier, novel kernel or native qualification.

| Condition | Tokens | Weight + index H2D | Transfers/extents | Episode wall |
|---|---:|---:|---:|---:|
| Dense streaming | 16 | 25,769,803,776 B | 384 | 2.142871s |
| Sparse 128-row extents | 16 | 22,376,611,840 B | 2,178 | 2.088562s |
| Compact row packet | 16 | 1,635,580,200 B | 384 | 0.656515s |
| Restored resident | 16 | 0 B | 0 | 0.168545s |

**Htraffic passes:** 93.6531% below streaming and 92.6907% below sparse, including
1,595,688 B of row-index metadata. Frozen requirement was >=50% against EACH.
**Hruntime passes:** 69.3628% and 68.5662% lower wall, against >=20% for EACH.
Packet is 3.26 times as fast as this dense-streaming control, but still about
3.90 times slower than the restored resident. A two-prompt/eight-token-per-prompt
screen does not establish general speedup, long-context behavior or task quality.

## What actually changed

The first projection is always fully computed. Exact observed ReLU activity chooses
which outgoing rows matter for the current forward. CPU gathers active rows and
their int64 indices into ONE contiguous pinned header/payload packet. One H2D copy
transfers it; GPU scattering places rows into the original dense weight workspace.
The original linear operator executes with its bias. All other weights/computation
remain dense. No zero sentinel, prediction, approximation policy, training, new
sampler, CUDA kernel or llama.cpp modification is involved.

Sparse and packet see the same 367,698 active neuron-token observations out of
9,043,968. Physical grain, not a different selector, changes traffic. Packet prefill
H2D is 757,171,600 B; decode 878,408,600 B. Discovery costs 9,044,352 B D2H in
each sparse condition. Summed packet TTFT is 0.212298s versus streaming0.281803s
and sparse0.347012s. Initial resident wall0.762484s includes first-use initialization
and is not a steady-state speedup denominator.

The experiment draws on [LLM in a Flash's acquisition/bundling](https://arxiv.org/abs/2312.11514),
[PowerInfer's sparsity-aware execution](https://arxiv.org/abs/2312.12456), and
[Deja Vu's contextual sparsity](https://proceedings.mlr.press/v202/liu23am.html), with
departures detailed in the protocols. Packetization, gather/scatter and ReLU
sparsity are established techniques; no global algorithmic novelty claim is made.

## Failed attempt and prospective correction

Initial source
[`a111921`](https://github.com/displague/dynamic-model-loading/commit/a1119214c711ad7439adfebfb16d8c0b00f2ca2c)
froze the [first protocol](row-packet-screen-protocol.md). Its only run,
`runs/row-packet-screen-20260915-v1`, stopped after18.089s and six of ten episodes.
All four completed hybrid ID sequences matched, but the second prompt's packet
reached relative logit L2 **1.30455e-5**, beyond the fixed1e-5. No complete economic
gates or restoration ran. The failure remains **inconclusive**, never a pass.

Saved-logit diagnosis identified a numerical mismatch with the reference operator:
the initial harness used a different contiguous orientation and separate bias add.
The [corrected protocol](row-packet-corrected-protocol.md), reviewed and pushed at
[`03e1414`](https://github.com/displague/dynamic-model-loading/commit/03e1414686b1a5bbf1ca05c2312732b3ee64c5f8),
retains the SAME model, prompts, order, precision, thresholds and timeout but uses
the original contiguous weight orientation and F.linear for ALL three conditions.
No tolerance widening or selected replacement row. The two prompts are explicitly
known/reused after v1, not a globally untouched holdout.

Corrected run `runs/row-packet-screen-20260915-v2`: worker19.110770s + independent
audit6.065795s = **25.176565s**. All six hybrid episodes and both restored repeats
have zero per-position full-logit error and identical argmax/IDs/stop reason.
Including the failed worker, these two attempts plus the corrected first audit
take43.265402s; diagnostic/review work is separate. Second model-free audit exactly
reproduces the complete corrected summary without checkpoint inference.

The corrected strided control copy can allocate a temporary contiguous GPU buffer
and rearrange on-device, as [PyTorch2.10's copy implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/aten/src/ATen/native/cuda/Copy.cu)
shows. These costs remain in measured clocks and allocator peaks. H2D counters
count explicit logical payload, not PCIe wire traffic or every framework D2D copy.
The comparison is this declared PyTorch mechanism, not the best possible native
or pretransposed streaming runtime. That stronger-baseline limitation remains.

## State, memory and reproducibility

Each episode has its own KV, with16-token prefill and scalar decode. Maximum KV
9,043,968 B. No speculative acceptance or reused cross-episode cache is claimed.
The independent audit reconstructs histories, exact activity-derived rows/extents,
indices, byte totals, serialization, clocks, resource bounds, source and tokenizer
bindings. It does not re-execute every neural forward.

Original parameters5,263,032,320 B; hybrid GPU parameters3,652,419,584 B; original
host outgoing weights1,610,612,736 B. Shared GPU workspace67,108,864 B; stream
pinned staging67,108,864 B; extra pinned packet and GPU packet EACH67,174,400 B.
Total pinned buffers134,283,264 B. All conditions carry these costs even when
packet scratch is unused. Hybrid allocation3,796,271,104 B but reservation remains
5,435,817,984 B. Full resident startup/restoration still prevents a hard-budget
capacity claim. Nor is FP32 the strongest possible low-memory deployment baseline.

Construction H2D5,263,032,320 B; conversion D2H1,610,612,736 B in0.678267s;
restore H2D1,610,612,736 B in0.499728s. All ten episodes separately total
49,781,997,656 B explicit H2D and34,177,584 B D2H. 68 samples: total GPU peak
5,990,617,088 B, sampled RSS peak9,047,523,328 B, minimum host available
9,807,609,856 B. Allocator peaks and all resource limits pass.

**579 CPU tests pass**, with independent initial-source and correction reviews.
[Compact receipts](../results/row-packet-screen-20260915) preserve the failed
supervisor/decision as well as corrected measurements. Release final-clean-tip XML
and validation are separate assets. Main raw archive `packet-v033-raw.zip`:
32,528,344 B,138 members, SHA256
`c295f0953136c3489da23a9e08a0f8d83e5e18a3d2d74e3efb6f9f2c26e41bbd`.
It includes corrected raw data AND nested `packet-v033-first-failed.zip` plus its
receipt:12,356,618 B,114 members, SHA256
`968b370f2fb5851f95af4767a787dd0ff5287af32efb54a6d67828a700c48933`.
Both archives independently decompressed/hash checked; source files rechecked.
Full pinned OPT weights remain external dependencies under their own license.

## Next

Nominate compact exact-active-row transport for a separately frozen capacity or
stress test, not an immediate native port or long suite. Three course deliveries
remain: accepted-prefix decision risk and useful bounded capacity require their
own protocols and stronger-baseline comparisons. Complete only #48; milestone10
remains open. This positive does not erase failed Qwen selectors, specialist heads,
v0.32's frozen miss or this release's first numerical failure.
