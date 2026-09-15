# v0.42: switch acquisition grain between prefill and scalar decode

Sixth delivery within the owner's four-to-eight allowance. This is a new physical
policy motivated by v0.41's prefill/decode asymmetry, not expansion of its failed
uniform-prefill policy. Review and commit/push source, configuration, workload and
gates before one supervised worker capped at 300 seconds. No long matrix follows.

## Mechanism and prospective prediction

Use the same CPU-first OPT-2.7B FP16 model, two host weight layouts, dense CUDA
workspace, and FP16 KV throughout. For an outgoing projection with more than one
input token, transfer the entire original-layout outgoing tensor contiguously.
For a scalar input, observe exact ReLU zeros and transfer the required rows as a
packet. No hidden precision change, statistical omission, predictor, retained row
cache, or extra model. The first projection and outgoing arithmetic remain dense.

In this frozen workload, the rule means dense prefill and packet decode. It avoids
prefill union readback, CPU gathering, indices and scatter, while knowingly buying
more prefill bytes. Record the actual physical branch on every projection call.
Prediction: at least 10% lower prefill call time and 5% lower whole-episode time
than always-packet, with no more than 30% extra outgoing bytes. v0.41 implies about
26.1% extra bytes if its trajectory repeats; that calculation is a prospective
cost expectation, NOT a measured hybrid speedup. All gates remain fixed afterward.

This is ordinary phase specialization applied to this physical loader, not a claim
to invent prefill/decode separation. [LLM in a Flash](https://arxiv.org/abs/2312.11514)
motivates physical transfer grain, [DejaVu](https://proceedings.mlr.press/v202/liu23am.html)
sparse execution scheduling, and [OPT](https://arxiv.org/abs/2205.01068) the ReLU
model. Its local novelty is the implemented, measured acquisition policy and
explicit tradeoff against both uniform policies. No llama.cpp tuning or patch.

## Frozen paired workload and execution

Reuse EXACTLY v0.41's known archived WikiText concatenations and 512-token prefixes,
with 16-token greedy continuations and a separate warmup input. The configuration
embeds literal texts and the original corpus Git/hash identity; reconstruct and
audit them. These are known diagnostic texts, not a new holdout. All three modes
use the same reference IDs, stop conditions and FP16 cache rules.

CPU-only startup with zero current AND historical peak CUDA allocation/reservation
before construction. Dense contiguous streaming supplies the stable target-only
reference: stream warmup, then documents 0, 1 and 2. Declare packet and phase-switch
warmups afterward. Scored order: document 0 packet/phase, document 1 phase/packet,
document 2 packet/phase. There is one episode per document/mode, not a confidence
interval from repeated timing. Alternation balances candidate order imperfectly;
the dense reference remains baseline-first. All KV and row state reset per episode.

The original model revision, baseline .venv, four CPU threads, SDPA, TF32-off and
explicit FP16 reduced-precision arithmetic remain unchanged. No full GPU preload.
Both current/peak allocator and sampled global GPU stay below 4800 MiB across ALL
phases on the 16 GiB device; host available memory stays above 2048 MiB. Maximum
KV is 527 positions, or 172,687,360 B. Charge the same 3,625,472,000 B of non-outgoing
CUDA parameters, 52,428,800 B workspace, 52,510,720 B CUDA and pinned packets each,
and two host outgoing layouts of 1,677,721,600 B each to every mode. No memory
advantage over either offloaded control is presumed.

Check all generated IDs/stops and every recorded full-vocabulary logit against
dense streamed FP16, including both candidate warmups. Fixed relative-L2 tolerance
is 1e-5; report exact equality separately. Cache dtype is always FP16, and actual
logical/storage bytes must agree. No speculative draft is involved.

## Gates and accounting

- Faithfulness/resources/CPU-first construction must pass all inherited audits.
  Each scored packet or phase-switch episode must finish within 5 seconds.
- Hprefill_gain: phase-switch prefill forward-call sum <= 90% of packet prefill.
- Hwhole_episode_gain: phase-switch whole wall sum <= 95% of packet whole wall.
- Htraffic_tradeoff: phase-switch outgoing H2D <= 130% of packet outgoing H2D.
- Hstream_baseline: phase-switch outgoing H2D <= 20% of dense streaming AND whole
  wall <= 50% of dense streaming. Do not claim that defeating packets alone proves
  useful access against a stronger same-precision baseline.
- Hphysical_split: phase-switch prefill bytes equal dense-stream prefill bytes;
  phase-switch scalar-decode bytes equal the paired always-packet decode bytes.

Outgoing H2D includes weight and index bytes. Archive explicit token/logit copies,
raw activity masks only when actually observed, selected rows, branch identities,
KV, per-call clocks, per-episode TTFT/wall, setup, warmups and resource samples.
Independent audit must count prefill and decode bytes exactly once. Dense prefill
must not incur or fabricate a packet activity readback merely to populate a record.

Phase call times include acquisition/compute/readback; final episode serialization
is outside phase calls but inside whole wall. Metadata, gather/scatter, finite
checks, branch dispatch and records are charged. Report extra bytes versus packets
even if latency improves. Cold shared-worker-to-first DENSE logit is not a measured
packet-first or phase-first cold TTFT; supervised wall additionally includes module
startup and workload-integrity checks. Preserve every declared first-use episode.

Any timeout, correctness/resource failure or failed economics gate stops this
candidate. Preserve fresh raw receipts; no relaxed threshold or longer matrix.
A pass is a short physical-policy result, not optimized-Q4/Q8 competitiveness,
natural long-context validation, unique access, deployment or native-pivot evidence.
After this sixth delivery, leave those broader frontiers open rather than fabricate
seventh/eighth releases by renaming a failed variant.
