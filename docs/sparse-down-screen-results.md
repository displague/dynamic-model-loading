# v0.32: exact-zero acquisition saves bytes, narrowly misses the latency screen

The second delivery in the v0.31--v0.36 course implements actual outgoing-weight
loading after exact ReLU activity discovery. [Protocol](sparse-down-screen-protocol.md)
was reviewed, committed and pushed as
[`12a9822`](https://github.com/displague/dynamic-model-loading/commit/12a9822a58bab6e3581c4d3c6f9f50f28cf062bf)
before the only checkpoint run, `runs/sparse-down-screen-20260915-v1`.
Worker 19.625s + independent audit 4.506s = **24.131s**.

| Condition | Emitted tokens | Outgoing-weight H2D | Load extents | Episode wall |
|---|---:|---:|---:|---:|
| Resident initial reference | 16 | 0 B | 0 | 1.316781s |
| Dense streaming | 16 | 25,769,803,776 B | 384 | 2.072192s |
| Exact-zero 128-row pages | 16 | 22,048,407,552 B | 2,286 | 1.970661s |
| Resident after restoration | 16 | 0 B | 0 | 0.177691s |

**Htraffic passes: 14.4409% saved. Hruntime fails: 4.89965% below streaming,
against the frozen inclusive 5% requirement.** No repeat was used to scrape the
threshold and no long matrix follows. The 0.10 percentage-point miss is not a
statistically established latency failure: two eight-token episodes cannot settle
a population speedup. The compound screen remains failed as registered.

First-use initialization is included, not discarded; initial resident timing is
therefore not a fair steady-state speedup denominator. The restored resident is
far faster than either offloaded path. This is not faster inference when the model
already fits fully resident. Sparse summed TTFT is 0.328087s versus streaming's
0.289371s; the candidate did not improve every phase.

## Mechanism and faithfulness

All first projections, attention, embeddings, biases and head are computed and
resident. Original fc2 weights alias host-packed storage; selected contiguous
extents really copy through pinned staging to one shared GPU workspace. Only exact
observed zero activations justify skipping corresponding outgoing weights.
The matrix multiply remains dense PyTorch; no predictor, sparse CUDA kernel or
llama.cpp change. This departs from v0.6's retrospective sparsity simulation.

Sparse observation sees 369,908 nonzeros in 9,043,968 neuron-token observations,
about 95.91% zeros. That does **not** translate into 95.91% transfer savings:
128-row page union and prefill largely fill the load units. Prefill transfers
3,219,128,320 B versus streaming's 3,221,225,472 B. Decode transfers
18,829,279,232 B versus 22,548,578,304 B. Discovery adds 9,044,352 B D2H.

All four hybrid episodes and both restored repeats match original greedy IDs and
stop reasons. Maximum per-position full-logit relative L2 is 8.54746e-6, below the
fixed 1e-5; every argmax matches. Dense restored repeats have zero error. This is
finite-tolerance numerical faithfulness, not bit-identical hybrid floating point
or a theorem for arbitrary prompts. OPT is pretrained, not instruction-tuned;
eight-token continuations do not demonstrate task quality or long-form usefulness.

Each episode creates independent KV; 16-token prefill then scalar decode. Maximum
KV is 9,043,968 B. No speculation, accepted-prefix claim or cross-episode KV reuse.
Independent CPU replay verifies source blobs, tokenizer, raw inventory, serialized
activity-derived page/extent decisions, copies, histories, logits, KV and clocks.
It does not independently recompute every neural forward. Second audit reproduces
the complete summary exactly without new inference.

## Costs and limitations

Model parameters 5,263,032,320 B, registered buffers zero. Hybrid GPU parameters
3,652,419,584 B; host-backed outgoing weights 1,610,612,736 B; GPU workspace and
pinned staging each 67,108,864 B. CUDA allocation at hybrid boundary is
3,729,096,704 B, but allocator reservation remains 5,366,611,968 B after loading
the resident reference. Consequently this run is **not** a hard-budget capacity
frontier: startup/restoration and retained allocator reservation still exceed the
hybrid parameter footprint. All original host storage, workspace, staging, KV
and sampled process memory are charged; no invisible second target exists.

Construction H2D 5,263,032,320 B; conversion D2H 1,610,612,736 B in 0.625858s;
restoration H2D 1,610,612,736 B in 0.497722s. Across all eight episodes, explicit
copies total 47,818,212,800 B H2D and 21,915,456 B D2H, excluding those separately
reported setup copies. These are logical payload bytes, not PCIe wire counters.
68 resource samples record total GPU peak 5,919,313,920 B, sampled RSS peak
8,909,803,520 B and minimum host availability 9,604,304,896 B. All recorded CUDA
high-water marks and resource bounds pass; raw tensor sizes accompany episodes.

Pre-inference review corrected inclusive boundary arithmetic, overlapping workspace
replay and missing allocator-peak enforcement. **566 CPU tests pass.** Review
receipts, XML and compact results are in
[`results/sparse-down-screen-20260915`](../results/sparse-down-screen-20260915).
The release includes full raw tensors, sources and independent audits. Archive
`sparse-down-v032-raw.zip`: 14,744,011 B, 124 members, SHA256
`c59a65a806c057485fd292187a30bba20da667d18b6785fecc555e48986544ac`;
all members decompressed/hash checked and raw sources rechecked. OPT weights remain
an explicitly pinned external dependency under their own license.

## Next hypothesis

Stop expansion of this 128-row/many-extent candidate, not exact-zero research.
Change acquisition grain and transport: compact actual active rows into one packet
per layer, paying CPU gather, metadata, transfer, scatter and workspace costs.
This tests packing against fragmented copies, not a new neighborhood predictor or
a loosened v0.32 gate. It needs a separate reviewed short protocol. No native port
or longer matrix is authorized by this result. Complete only issue #47;
milestone 10 and four releases in this course remain open. Inspirations are
explicitly attributed in the [protocol](sparse-down-screen-protocol.md).
