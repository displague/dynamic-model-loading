# v0.40: CPU-first OPT2.7B FP16 packets under 4800 MiB

Separately frozen short scale screen following v0.39's surviving physical/footprint
components. Reviewed source/config/artifact manifest/tests committed and pushed
before inference. One supervised <=300s worker, unchanged .venv, no long matrix,
native patch, predictor, cache retention, or new quantization implementation.

## Question, antecedents, and prospective prediction

Can exact observed-zero packets accelerate a larger ReLU checkpoint from CPU-only
startup, under a budget below its full FP16 parameter footprint? Expect faithful
outputs, >=50% fewer H2D bytes and >=20% less wall than direct contiguous dense
FP16 streaming. Expect BOTH offloaded controls to fit. That is not uniquely enabled
access: packets would improve the latency of an access route streaming also enables.

[OPT](https://arxiv.org/abs/2205.01068),
[PowerInfer](https://arxiv.org/abs/2312.12456), and
[LLM in a Flash](https://arxiv.org/abs/2312.11514) supply sparse-activation/hot-cold
and physical transfer antecedents; no priority claim for sparsity or packets.
The mechanism still observes exact ReLU zeros AFTER fully computing fc1, fetches
every required outgoing row, and computes original dense fused fc2. No statistical
omission or speculative verification. No acceptance numbers are applicable.

## Artifact, arithmetic, and honest memory boundary

Original facebook/opt-2.7b revision905a4b602cda5c501f1b3a2650a4152680238254,
32 ReLU layers, hidden2560, FFN10240, vocabulary50272. Pinned file hashes/bytes in
`results/opt27-artifact-20260915/manifest.json`; downloading and meta-only parameter
counting preceded this protocol, not inference. Do not redistribute original weights.
FP16 weights AND KV, SDPA, four CPU threads, TF32 off, FP16 reduced-precision
reduction explicitly enabled/recorded, no precision changes or mixed KV state.

Full unique FP16 parameters:5,303,193,600B, exceeding4800MiB=5,033,164,800B before
KV/runtime. CPU outgoing weights:1,677,721,600B. Other CUDA parameters:3,625,472,000B.
Shared CUDA workspace52,428,800B, CUDA packet52,510,720B, equal pinned packet, zero
retention cache. Additional original-layout host weights1,677,721,600B support a
strong stream baseline and are charged to BOTH. KV327,680B/position, maximum47
positions. Preserve tied embeddings; count unique moved parameters and all buffers.

CPU-only load verified with zero allocated/reserved CUDA bytes before construction.
Only non-outgoing parameters move to GPU. Enforce allocator fraction4800MiB/device
total from startup and sample global NVML <=4800MiB across ALL phases, including
construction, dense reference, first use and warmups. Host available >=2048MiB.
The physical test GPU is16GiB: this is an imposed budget, not a real4.8GiB-card OOM
experiment. No fully resident2.7B GPU preload, no hidden target-shaped reference.

## Workload, reference, and accounting

Config fixes three NEW authored32-token prefixes (prose/code/arithmetic), generation
cap16, plus the known separate32-token warmup. Freeze before token generation.
Start CPU load, construct host banks/non-outgoing CUDA placement, run stream doc3
warmup, then stream docs0/1/2 as stable dense target-only greedy references. Run
packet doc3 warmup, then packet docs0/1/2. Reset KV/row state every episode.
Archive/check packet warmup against stream warmup too. All generated IDs and stops
must agree; every checked full-vocabulary logit relativeL2 <=1e-5 (fixed previous
contract). This is equality to dense STREAMED FP16 on2.7B, not to an independently
measured fully resident2.7B model. v0.39 qualified this direct-stream representation
against resident FP16 at1.3B. Do not infer FP32 task quality or deployment quality.

Baseline-first phase order is disclosed, not randomized inference about steady
state. Score exactly three episodes per condition; exclude only declared warmups
from warm totals. Report setup, cold worker-entry-to-first-logit, both warmups, total
worker+audit time, scored per-episode TTFT/wall, allocator/global GPU/host peaks.
Cold worker-entry clock includes artifact hashing but not Python module startup;
supervised worker wall includes process startup. Retain first use, no post-hoc
discard/replacement. All copies, indices, readback, staging, gather/scatter, finite
checks, recording and boundary samples charged to wall. Primary H2D is outgoing
weight+index bytes; explicit token/logit transfers archived separately in all modes.
No SSD I/O claim: weights are in host RAM for episodes, not page-faulted from disk.

## Frozen gates and stopping

- Hfaithfulness/resources: full raw/provenance/copy/KV audits, all packet IDs/stops
  and checked logits agree, all allocator/global samples obey4800MiB and host floor.
- Hcapacity: actual CPU-first construction/episodes within that budget, full FP16
  parameter count exceeds budget, no hidden GPU preload. Applies to both controls.
- Hacquisition: packet scored H2D <=50% dense stream.
- Hruntime: packet scored episode-wall sum <=80% dense stream.
- Huseful_latency: each scored packet32+16-token episode <=5s.

Any timeout, correctness or acquisition/economics failure stops this candidate,
preserving all receipts in a fresh run directory. Passing this three-prompt screen
is component evidence only, not an expanded validation or native-pivot gate.
Optimized Q4/Q8 comparison remains OPEN: none is installed/qualified for this
checkpoint in the measured baseline. Smaller FP16 footprint is not proof packets
beat compression. A following course must freeze that comparison and/or a longer
context/batch-union durability screen separately. Do not publish unmeasured wins.
