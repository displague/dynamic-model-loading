# v0.39: exact-row packets at FP16 with strong resident and streaming controls

Prospective representation screen, not expansion of the failed retention/forecast
candidates. Source/config/tests reviewed, committed and pushed before one <=300s
supervised worker. All baseline packages stay in .venv. Same pinned OPT1.3B original
HF artifact, FP16 weights AND KV, SDPA, four CPU threads, TF32 off, FP16 reduced-
precision reduction explicitly enabled and recorded. No mixture of FP32/FP16 KV.

## Questions and predictions

Does exact observed-zero outgoing loading remain numerically faithful at FP16,
save traffic/wall versus direct dense FP16 streaming, and reduce measured memory
relative to ordinary resident FP16? Expect YES to physical acquisition/memory and
NO to beating warm resident FP16 in latency. These are separate gates: don't use
an FP32 streaming baseline to claim competitiveness against ordinary precision.

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)
motivates a same-precision reference, not an assumption of equality with FP32.
[OPT](https://arxiv.org/abs/2205.01068) supplies pretrained ReLU architecture;
[LLM in a Flash](https://arxiv.org/abs/2312.11514) supplies transport-grain/reuse
antecedents. No new quantization algorithm, native kernel or sparsification training.

## Workload and phase contract

`configs/precision-packet-screen.json` fixes the SAME three v0.37 authored32-token
prefixes and16-token cap plus separate warmup. These prompts are known, not a new
holdout. First run a declared resident FP16 warmup, then fresh resident references
for all three documents. Return model to CPU, release allocator cache, pack FP16
outgoing rows, retain a separate original-layout host copy for a strong contiguous
stream control, and move only other parameters to CUDA. Declare separate stream
then packet warmups. Scored order doc0 stream/packet, doc1 packet/stream, doc2
stream/packet. KV and row state reset at EVERY episode. No persistent row cache.

First-use and all warmup receipts are retained separately and charged to total
worker time; scored primary warm totals exclude only the prospectively declared
warmups. Resident references precede hybrid runs: disclose this phase/order limit,
not a randomized steady-state confidence interval. v0.36 is not rerun or revised.
FP16 output is compared against original resident FP16, not presented as FP32 task
quality. Full reference GPU startup means NO CPU-first or uniquely enabled access
claim here, even when candidate footprints are smaller. Both ordinary FP16 and
packets fit the4800MiB candidate allowance. Reference phase allowance6500MiB.

Stream copies every outgoing tensor through a contiguous original-layout pinned
view directly to the contiguous GPU workspace, preserving fused F.linear. Packet
gathers observed nonzero rows, transfers their FP16 bytes and int64 indices, and
scatters into the same original-layout dense workspace. Fully compute fc1. Finite
checks, unions across prefill tokens, readback, CPU gather, index copies, scatter,
temporary allocations, JSON/tensor writing, and extra original-layout host copy
are charged. No trace-predicted bytes or ignored cache/precision overhead.

## Gates and interpretation

- Hfaithfulness/resources: all IDs/stops and every checked full-vocabulary logit
  match resident FP16 within1e-5 relativeL2; full memory/copy/source audits pass.
- Htransport: packet H2D <=50% of dense stream AND wall <=80% of dense stream.
- Hfootprint: maximum scored packet sampled GPU use <=85% of maximum scored
  resident FP16 sampled GPU use, with packet episodes <=5s. Allocator and global
  NVML peaks are reported separately; require at least one sample per episode.
  A fixed resource sample before each episode finishes is charged to every mode.
- Hresident_speed: packet wall <=95% of warm resident wall, reported independently.

Faithfulness/resources/transport/footprint may nominate a separately frozen SHORT
larger-model capacity experiment, even if resident-speed fails: the question then
becomes the memory boundary, not speed against a model that already fits. No gate
opens an hours-scale matrix or native pivot. If component gates fail, stop this
representation. Q4/Q8 optimized deployment comparison remains unmeasured; ordinary
FP16 is a stronger comparison than historical FP32, not the entire precision frontier.
