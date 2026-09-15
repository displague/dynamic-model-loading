# v0.37: exact-active-row retention, frozen short screen

Prospective protocol under ADRs0004--0006; implementation/config/tests must be
reviewed, committed and pushed before the sole worker starts. A <=300s supervisor
includes imports, weight verification, loading, all warmups and generation. No
timeout, missing receipt or incorrect output can pass; correction gets a fresh run.

Question: can 1024 retained outgoing rows per OPT layer reduce total acquisition
and wall time relative to demand-only packets? Expect nonzero reuse but uncertain
net speed: LRU work, index transfers and device gathers may exceed avoided H2D.
This is the retention/windowing antecedent from
[LLM in a Flash](https://arxiv.org/abs/2312.11514), not invention of LRU or sparsity.
It precedes an independent anticipation hypothesis; no learned field is fitted.

## Frozen apparatus and controls

`configs/retention-screen.json` fixes original OPT1.3B HF revision
3f5c25d0bc631cb57ac65913f76e22c2dfb61d62 and all original artifact hashes from the
tracked ReLU manifest. Baseline .venv, FP32, SDPA, four CPU threads, no TF32.
Three newly authored prose/code/arithmetic prefixes, first32 encoded tokens,
16 generated greedy tokens maximum; EOS2 honored. This is not task-quality proof.

Run one separate32-prefix/16-token resident warmup, then all three resident
references. Move model to CPU and release allocator cache, construct host outgoing
weights and shared original-contiguous GPU workspace, then move only other weights
to GPU. Run one separate warmup for packet then retained. Finally measure fixed
order doc0 packet/retained, doc1 retained/packet, doc2 packet/retained. Warmups are
retained and charged separately, never dropped after seeing their times. References
and warmups are not the scored numerator. No cross-document or cross-episode cache
state. Prefill and decode traffic/latency are reported separately.

The packet control and retained candidate reserve the same192MiB row cache. Only
retained uses it; report that reservation explicitly, not as a memory saving.
At each call observe exact nonzero activation union. Hits are scattered from GPU
cache BEFORE inserting misses, which may reuse their slots. Fetch misses in one
weight/index packet, scatter to original-layout dense workspace and retain final
LRU owners. Cached rows are keyed by layer AND neuron. Fully compute fc1 and dense
fused fc2 as before. Finite checks, activity readback, host LRU, CPU gathering,
index H2D, GPU gathers/scatters and JSON/tensor receipts are charged to episodes.
Unloaded workspace rows are finite stale values, not an inactivity signal.

References may use6500MiB; candidate episodes must stay below4800MiB measured by
allocator and sampled NVML. Host availability >=2048MiB throughout. Full reference
GPU loading occurs in this process: NO CPU-first capacity claim for this screen.
Charge original host weights, workspace, pinned/device packet, row cache, host
maps via process RSS, temporary device gathers, complete K/V storage and copies.

## Frozen gates and audit

- Hfaithfulness: every candidate/reference own-trajectory ID and stop agrees;
  every checked full-vocabulary logit relativeL2 <=1e-5 (zero also reported).
- Hacquisition: summed scored retained weight+metadata H2D <=90% of packet.
- Hruntime: summed scored retained end-to-end wall <=95% of packet.
- Hresources: declared candidate/sample/host limits pass; complete receipts.

All four are required to nominate this retention candidate. Report every result,
including warmups, not a favorable subset. No statistical significance claim on
three pairs. Independent raw replay reconstructs each active set, LRU transition,
index/payload bytes, greedy token/KV history, resource extrema and source hashes.
One successful short screen is not an expanded evaluation or native admission.
