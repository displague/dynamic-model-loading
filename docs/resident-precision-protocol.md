# v0.36: does ordinary resident FP16 already provide the bounded access?

Sixth and final delivery of the authorized course, [#51](https://github.com/displague/dynamic-model-loading/issues/51), not closure of milestone10.
The v0.35 packet screen passed an FP32 capacity/latency contract. Test a stronger
ordinary representation control before claiming access unavailable without novel
loading. This is a deployment-comparison boundary, not a new precision algorithm.

Prediction: resident FP16 fits comfortably below4800MiB and probably reproduces
the same short greedy continuations with useful latency. Its logits need not meet
the FP32 numerical tolerance. If so, packet loading retains its measured exact-
FP32 acquisition result, but does NOT establish uniquely available useful access.
If FP16 differs, that alone is not evidence of worse task quality or a reason to
waive approximation standards; report the changed outputs and stop extrapolation.

## Frozen short screen

Same pinned OPT1.3B checkpoint, tokenizer and license; only representation changes
to standard resident FP16 parameters and FP16 KV. No weight pager, quantizer,
sampler modification, training or llama.cpp work. Use existing .venv, four CPU
threads, SDPA, TF32 off and default recorded FP16 reduced-precision reduction=true.
Load on CPU at FP16 and then move the full model to GPU. Actual unique parameter
payload must be2631516160 B with no registered buffers, half the FP32 payload.

CPU-only dense FP32 references are the same immutable v0.33 arrays, tied to the
published commit and archive inventory. Same two known16-token prefixes, document0
then1, fresh per-document DynamicCache, greedy cap8 or EOS2. No warmup discarded,
repeat or candidate replacement. Run its own trajectory even if it differs from
FP32; do not force the expected tokens. Log full50272-vocabulary values after an
exact FP16-to-FP32 cast for readable artifacts, actual inputs, outputs, stop, KV
logical/backing bytes/dtype and all transfers/clocks. Cast/readbacks/writes are timed.

Allocator fraction4800MiB/device-total before allocations; current/high-water
allocated and reserved<=4800MiB and sampled totalNVML<=4800MiB from before loading
through completion. Sample at0.2s and boundaries; hostavailable>=2048MiB. Same
imposed16GiB-device allowance, not physical smallGPU, OOM or exhaustive drivertrace.
Record CPU loading and H2D construction separately, while supervisor300s includes
imports/loading/checks/inference/writes. No overlapping model processes or suite.

## What the gates mean

Hordinary_access requires complete artifact/source/physical/state/resource audits,
all16 own-trajectory greedy IDs and stop reasons equal FP32, <=5s per episode and
<=60s total supervised worker. Passing means ordinary FP16 provides this same
tested output within the declared useful budget; novel-only capacity is NOT
demonstrated. Failing leaves that question unresolved, not a general quality loss.

Separately report Hfp32_numerical: identical IDs and every aligned full-logit
relativeL2<=1e-5. This is NOT an entrance gate for finishing the FP16 comparison,
because measuring ordinary precision's numerical change is its purpose. Where
trajectories diverge, compare logits only while consumed prefixes still match,
including the prediction at first divergence, then stop numerical alignment.
Never call different-context logit differences a precision error.

v0.35 packet timings may be shown ONLY as historical context: no paired speedup
ratio, runtime-win gate or causal timing claim from these separate processes and
different first-use order. Current FP16 useful latency is independently gated.
Static memory and identical observed greedy outputs suffice for this bounded
access counterexample, without a best-kernel claim. No task-quality, longcontext,
larger-checkpoint or generally exact FP16 claim follows sixteen tokens.

Archive full receipts and independent model-free audit, finish code review and
final clean-tip tests, then publish the sixth prerelease. End the authorized course
without automatically opening another sweep, long matrix or native pivot.

## Attribution

[OPT's original model](https://huggingface.co/facebook/opt-1.3b) supplies the shared
checkpoint; ordinary half precision is a control, not our invention.
[PyTorch2.10 numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)
explains precision-dependent results; [allocator documentation](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
defines the allocator-only limit. Separate NVML samples cover observed total usage.
The packet path's [LLM in a Flash antecedent](https://arxiv.org/abs/2312.11514)
motivates testing useful tiered access against an appropriate representation control.
