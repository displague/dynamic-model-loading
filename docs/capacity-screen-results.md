# v0.35: useful FP32 packet loading below full-residency capacity

Fifth of six deliveries, [#50](https://github.com/displague/dynamic-model-loading/issues/50).
The [prospective screen](capacity-screen-protocol.md) passes all four gates in
23.033s: worker17.076s plus independent audit5.957s. A second exact audit takes
6.003s without inference. This nominates a separately frozen stronger comparison,
not a long matrix, native pivot, larger checkpoint or deployable runtime.

## What a prompt actually did

Load the pinned OPT1.3B checkpoint on CPU, keep outgoing FFN weights there, and
move only the remaining parameters to GPU. For each of two known16-token prefixes,
compute the first projection fully, observe exact ReLU zeros, gather active
outgoing rows and indices into one packet, copy to GPU, scatter and run the original
dense F.linear. Generate eight tokens with a fresh KV cache per episode. No learned
omission, speculative draft or different output model is used.

All32 generated tokens across four episodes and EVERY checked full-logit value
match the published dense FP32 reference exactly. References were imported as
CPU-only arrays bound to v0.33's published archive and source; no full resident
reference model was loaded on GPU in this process. These are reused known prefixes,
not new linguistic holdout or task-quality evidence.

| Condition | Tokens | Weight + index H2D | Episode wall |
|---|---:|---:|---:|
| Direct contiguous dense streaming | 16 | 25,769,803,776 B | 2.646471s |
| Exact-active-row packets | 16 | 1,635,580,200 B | 0.736909s |

Packet saves93.6531% H2D and72.1550% aggregate wall (3.59x speed ratio), passing
the50% byte/20% wall gates. Its two episodes take0.456s and0.281s, below the5s bound;
the whole worker is below60s. The stream episodes take1.648s and0.999s. First-use
and order effects remain: one counterbalanced pair provides no confidence interval
or steady-state speed guarantee. No discarded warmup or post-hoc repetition.

Unlike v0.33's strided-copy control, dense streaming now retains a pre-arranged
original-layout host copy and issues one contiguous H2D directly into the original
workspace. It pays no per-call transpose. The extra host layout is charged to both
conditions. Packet's saving survives this stronger same-precision control.

## Capacity rather than just lower traffic

Declared ceiling4,800MiB =5,033,164,800 B. Full FP32 parameters alone require
5,263,032,320 B (about5,019.2MiB), so cannot be fully resident under this contract.
The worker starts with ZERO allocated/reserved CUDA tensor bytes after CPU model
loading; it never performs a full-GPU load or restoration.

Non-fc2 parameters3,652,419,584 B plus workspace67,108,864 B and GPU packet67,174,400 B
produce3,786,702,848 allocated bytes after construction. Maximum allocated
3,805,911,552 B and reserved3,827,302,400 B stay below the imposed allocator ceiling.
Total-device NVML peak is4,380,004,352 B (4,177.1MiB), also below4,800MiB.
Checks start before model loading and span construction/inference/completion.
NVML is sampled at0.2s and boundaries, not an exhaustive driver-memory trace.

This demonstrates useful bounded FP32 access on the tested workload. Ordinary
dense offloading ALSO fits; the acquisition contribution is cheaper/faster use
under the same contract, not the act of fitting alone. This is an artificial
ceiling on a16GiB GPU, not a test on a physical4.8GiB card, actual full-model OOM,
larger checkpoint, SSD-backed host tier or a general capacity frontier.

Host packed outgoing weights1,610,612,736 B plus an equal-sized original-layout
copy remain resident. Pinned stream67,108,864 B and pinned packet67,174,400 B total
134,283,264 B. Maximum KV backing storage equals logical9,043,968 B; no rejected
suffix or shared target/draft cache exists in this non-speculative experiment.
Peak sampled RSS10,591,019,008 B; minimum available host7,993,212,928 B. CPU startup
sources and temporary packing costs are included in resource sampling.

Setup4.227s includes model loading and3.201s packing/movement; construction H2D
3,652,419,584 B, D2H0 B. Setup is outside conditional episode times but inside the
17.076s worker, which also includes imports, artifact checks and raw writes. Cold
process is not cold disk I/O; the filesystem cache was not flushed. All episode
explicit copies total27,405,384,712 B H2D and15,479,904 B D2H. Packet metadata alone
is1,595,688 B; exact-activity discovery reads back9,044,352 B. Its prefill payload
is757,171,600 B and decode878,408,600 B. CPU gather/GPU scatter are timed.

## Validation and interpretation

604 CPU source tests pass. Mandatory review also verified complete tiny-OPT cached
outputs, tied parameters, full-shape meta byte counts, external reference bindings,
the5s/60s inclusive gate boundaries and unchanged v0.32/v0.33 replay. Its separate
test invocation passed50 cases with one fixture blocked by temporary-directory
permissions; the normal51-case focused and604-case full runs passed. This is an
environment limitation, not a suppressed failure. Final clean-tip suite receipts
are attached separately to the release.

Raw `capacity-v035-raw.zip`:12,105,512 B,124 members, SHA256
`c546d19fe7156786b3a6ec9f5351482b4b4f11ac2bd53e529dc4ab3d1d2bafdb`.
Every member was decompressed/hash checked; raw sources rechecked. The archive
contains the external dense-reference arrays and published binding inventory, not
checkpoint weights. [Compact receipts](../results/capacity-screen-20260915).

[LLM in a Flash](https://arxiv.org/abs/2312.11514) and
[PowerInfer](https://arxiv.org/abs/2312.12456) motivate acquisition and heterogeneous
capacity; packetization and CPU offload are known techniques. This measured
outgoing-only application is not an invention of them or a reproduction of their
full systems. The [PyTorch allocator](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
limits caching allocations, while separate sampling checks total-device usage.

One delivery remains. The next question is whether ordinary resident FP16 gives
the same short greedy outputs with lower memory/latency. Until that comparison,
do not equate this FP32 capacity result with novel access unavailable through
ordinary precision reduction. Complete only #50; milestone10 stays open.
