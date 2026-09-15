# v0.35: CPU-first packet loading under a declared 4800MiB GPU ceiling

Fifth of six authorized research deliveries, [#50](https://github.com/displague/dynamic-model-loading/issues/50). This is a separately frozen short
follow-up to the passing v0.33 packet screen, not an expanded matrix or native pivot.

Hypothesis: exact-active-row packets retain useful transfer/latency savings over
direct contiguous dense streaming while constructing and running an FP32 model
whose full parameter payload exceeds the permitted GPU budget. Capacity alone is
not novel: ordinary dense offload also fits. The acquisition benefit must survive
that control. Prediction: both fit; packet saves bytes and wall, while ordinary
lower precision could still dominate both. That stronger deployment comparison
is a separate sixth delivery, not a success claim here.

## Construction and strengthened control

Pinned OPT1.3B revision3f5c25d0bc631cb57ac65913f76e22c2dfb61d62, same artifact hashes,
FP32 SDPA, four CPU threads, TF32 off, unchanged .venv. Load ONLY on CPU. Pack fc2
outgoing rows on the host and move only non-fc2 parameters, biases and buffers to
CUDA, retaining tied Parameter objects. No full resident model on GPU at startup,
no restore-to-GPU step and no resident-reference inference in this worker.

Keep the corrected original-layout F.linear and exact activation certificate from
v0.33. One pinned/device packet contains only actual active outgoing rows and
int64 indices; finite stale workspace rows are harmless ONLY for observed exact
zero activations. No learned omission or quantization. KV starts fresh per episode.

Strengthen dense streaming: retain an additional host copy in the original weight
layout, copy it into the existing staging viewed in that layout, and issue one
contiguous H2D into the original-layout workspace. No strided H2D or per-call
framework transpose. Charge the extra1610612736 host bytes to BOTH conditions.
Both retain packet scratch and streaming staging. This deliberately favors a
stronger control over the previous framework-layout disadvantage.

Static FP32 parameters5263032320 B exceed4800*2^20=5033164800 B before KV/workspace.
Non-fc2 CUDA parameters3652419584 B +67108864 workspace +67174400 packet can fit.
Host outgoing packed rows1610612736 B plus the extra layout of the same size;
pinned stream67108864 B plus packet67174400 B. Construction conversions, temporary
copies, host source, buffers, KV, readbacks and raw writes remain charged.

Set the CUDA caching allocator fraction to4800MiB divided by actual device total
before model allocations. Independently enforce allocated/reserved high-water
marks and total-device NVML samples<=4800MiB from before loading through completion;
sample every0.2s and at boundaries. NVML is sampled, not a proof against every
sub-sample non-allocator transient. Allocator limits do not cap all driver memory.
Host available>=2048MiB; supervisor hard kill at300s includes imports/loading/raw
writes. This is an imposed budget on a16GiB laptop, NOT a physical smaller GPU,
an observed full-model OOM, SSD access or a larger checkpoint.

## Workload and independent reference

Reuse the exact two known v0.33 prefixes, first16 plain tokens, cap8 new tokens or
EOS2, no chat template. This is not fresh linguistic holdout. The new object is
construction/budget/transport, not task quality. Order doc0 stream,packet; doc1
packet,stream. No discarded warmups. All four episode clocks include raw writes;
initialization is separate and included in the total worker clock.

Freeze the v0.33 corrected run's resident episode0/1 tensors and receipt hashes
and tokens. Copy them as external CPU-only references, bound to the published
archive inventory and source commit. No reference checkpoint pass in this worker.
Compare every emitted greedy ID/stop reason and per-position full-logit relativeL2
<=1e-5, unchanged. Record complete actual model inputs, KV length/storage, full
logits, exact activity, physical row/packet/copy clocks and startup device inventory.
Audit new source against Git blobs, reference files against the pinned parent,
independent tokenization, physical contents/costs, all resource caps and receipts.

## Prospective gates and stopping

Hfaithfulness: complete reference, numerical, state, physical and provenance checks.
Hcapacity: CPU-first construction, full FP32 parameter payload>declared ceiling,
both controls complete within allocator AND sampled total GPU ceilings.
Hacquisition: packet weight+index H2D<=50% of direct-stream H2D.
Hruntime: packet aggregate episodewall<=80% of direct-stream, each packet episode
<=5s, and entire supervised worker<=60s (including startup, controls and writes).
Inclusive comparisons. One counterbalanced pair is not a confidence interval.
All required to nominate, no repeat to cross a bar. A failure stops this screened
candidate or triggers a separately documented correctness correction in a fresh
directory, never a long suite or altered tolerance. No automatic native admission.

## Antecedents

[LLM in a Flash](https://arxiv.org/abs/2312.11514) motivates tier-aware acquisition
and useful capacity; [PowerInfer](https://arxiv.org/abs/2312.12456) motivates sparse
activity and heterogeneous residency. This outgoing-only packet implementation
does not invent sparsity, packing or CPU offload and does not reproduce their
full systems. [PyTorch2.10 allocator documentation](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
defines the allocator-only ceiling; separate NVML sampling owns total-device checks.
