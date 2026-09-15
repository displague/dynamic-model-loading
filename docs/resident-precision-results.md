# v0.36: FP16 fits and matches tokens, but first-use latency misses the gate

Sixth/final authorized delivery, [#51](https://github.com/displague/dynamic-model-loading/issues/51).
The [frozen comparison](resident-precision-protocol.md) completes in31.048s:
worker25.024s plus independent audit6.024s. Both reported gates are false, for
different reasons. This is a completed valid measurement, not an invalid run.

## Observations before interpretation

Ordinary resident FP16 matches ALL16 own-trajectory greedy tokens and both stop
reasons of the immutable dense FP32 reference. No expected tokens were forced.
All artifact, source, resource, state, full-vocabulary and copy audits pass.

| Episode | Tokens | Wall | First-token latency |
|---|---:|---:|---:|
| Document0, first use | 8 | 13.067391s | 11.104314s |
| Document1, subsequent use | 8 | 0.061606s | 0.008005s |

Hordinary_access fails ONLY its <=5s per-episode latency requirement. The overall
worker remains below60s, and capacity/output parity pass. The first two forward
calls account for almost all first-episode time (about11.10s and1.91s). No profiler
was run, so kernel compilation, autotuning, driver setup or another mechanism is
not established as the cause. Preserve this first-use stall; no discarded warmup,
post-hoc repeat or relaxed bound. One later fast episode is not a warmed benchmark.

Hfp32_numerical separately fails the unchanged1e-5 full-logit relativeL2 criterion.
Document0 errors range0.000615--0.001256; document1 range0.001698--0.005117. All16
predictions remain aligned because no IDs diverge. This means numerical change,
NOT demonstrated task-quality damage. FP16 was an ordinary representation control,
not an attempt to invent a precision method or certify generally exact FP16.

## Capacity and charged work

Unique resident parameters2,631,516,160 B, all FP16, tied embeddings preserved;
registered buffers0 B. Parameters require half the FP32 payload. Actual FP16 KV
backing storage equals logical storage, reaching4,521,984 B. No draft, pager,
precision switching, verification cache or hidden FP32 target model is resident.

Peak total sampled GPU3,406,925,824 B, peak allocated2,646,135,808 B and reserved
2,854,223,872 B all stay below4,800MiB =5,033,164,800 B. Startup begins with zero
CUDA tensor allocation. Peak sampled RSS3,365,748,736 B; minimum available host
15,109,443,584 B. Total-device sampling is nominally0.2s plus boundaries, not an
exhaustive driver-memory trace; the allocator has its own enforced limit.

CPU load plus full-model H2D setup2.302s, construction H2D2,631,516,160 B. Across
both episodes, H2D is368 B of token inputs, with NO runtime weight acquisition;
D2H3,217,776 B includes actual input readbacks and all full-vocabulary logits.
Native FP16 logits are cast exactly to FP32 for the raw files; the cast, readbacks,
KV inspection and raw writes are timed. The13.129s episode sum excludes setup,
while the25.024s supervised worker includes imports/loading/checks/inference/writes.
Cold process does not imply cold filesystem I/O; no cache flush was performed.

No paired speedup versus v0.35 is claimed. Those packet times came from a separate
process and different first-use/order history. The current experiment instead
tests its own frozen useful-access threshold, and that compound gate fails.

## Validation, raw artifacts and course conclusion

622 CPU tests pass, with18 focused checks rerun after a receipt-label clarification.
Mandatory source review checked32 selected CPU/meta cases, full-shape allocation,
half-KV output, immutable reference loading, divergent-prefix alignment and CLI
failure propagation. The temporary-directory fixture was omitted only in the
restricted reviewer invocation and passed in the normal full suite. A second
model-free audit matches every result field exactly in6.079s. Final clean-tip
full-suite validation accompanies the release separately.

Raw `resident-v036-raw.zip`:6,410,382 B,115 members, SHA256
`acbe26d172b01687a2cdaa7901e7d573a7b078696ddbd49cf68c1b4451e1730b`.
Every member was decompressed/hash checked and raw sources rechecked. External
FP32 reference arrays and source bindings are included; checkpoint weights are
not redistributed. [Compact receipts](../results/resident-precision-20260915).

The [OPT checkpoint](https://huggingface.co/facebook/opt-1.3b) and ordinary FP16
provide the control, not a claimed invention. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)
provides the precision caveat; [allocator documentation](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
defines the allocator-only cap. [LLM in a Flash](https://arxiv.org/abs/2312.11514)
motivates the useful-capacity comparison behind the packet research.

The six-delivery [course summary](residency-course-results.md) distinguishes the
physical acquisition result from its limits. Ordinary FP16 demonstrably fits and
matches these outputs; its first-use latency misses our bar. Neither fact proves
that the packet path offers uniquely available access or beats a warmed FP16
runtime. The positive result remains exact FP32 outgoing-weight acquisition on
sparse OPT, under a declared budget and stronger same-precision streaming control.

Complete #51 and the authorized six deliveries, NOT milestone10 or the overall
research question. No further sweep, long matrix, larger download or native pivot
starts automatically. A larger-model/strong low-memory frontier remains unmeasured.
