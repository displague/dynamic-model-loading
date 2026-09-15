# v0.42: phase-switched acquisition buys more bytes for lower latency

The 40.671-second screen (35.306 s worker, 5.364 s audit) passes every frozen
component gate. Dense outgoing transfers during prefill, followed by exact-row
packets during scalar decode, improve latency at the same measured GPU footprint.
All checked FP16 logits match the dense-streamed reference exactly, including both
candidate warmups; all 48 scored greedy tokens per condition agree. Independent
raw replay reproduces the entire summary exactly.

| Scored condition | Outgoing weight + index H2D | Prefill call time | Whole-episode time |
|---|---:|---:|---:|
| Dense contiguous stream | 80,530,636,800 B | 0.517263 s | 6.927632 s |
| Always-packet | 5,982,288,904 B | 0.865208 s | 3.223919 s |
| Dense prefill / packet decode | 7,546,254,016 B | 0.501669 s | 2.545918 s |

Compared with always-packet, the phase switch buys 26.1433% more outgoing bytes
and saves 42.0176% of prefill call time and 21.0303% of whole-episode time. The
extra bytes are an explicit cost, not a traffic-saving claim against packets.
Compared with dense streaming, it saves 90.6293% of outgoing H2D and 63.2498% of
episode wall time. These comparisons use the live controls in this run, not timings
borrowed from v0.41 or a best-of-historical reconstruction.

The physical split is verified: phase-switch prefill transfers 5,033,164,800 B,
exactly matching dense prefill. Its decode transfers 2,513,089,216 B, exactly
matching always-packet decode. No prefill activity readback is fabricated for the
dense branch. The same host banks and workspace serve both phases; FP16 KV is
neither converted nor rebuilt at the transition.

## Separate the gains from recording overhead

The whole-episode improvement is 0.678001 s across three paired documents.
Summed forward-call time falls from 2.606082 s to 2.257160 s, a 0.348921 s or
13.3887% reduction. The remaining 0.329080 s of improvement is outside those call
intervals and includes reduced final recording/episode overhead. Forward calls
themselves still include in-call recording; this is NOT a trace-free kernel or
production-runtime benchmark. Removing instrumentation requires a new protocol.

Decode call time does not improve: 1.755492 s for the phase switch versus
1.740874 s for always-packet. The useful change is prefill acquisition/handling,
not a claimed new decode algorithm. Fewer transferred bytes alone would have
ranked the slower uniform-packet policy first on this workload.

Scored phase-switch episodes finish in 0.859260, 0.834590 and 0.852069 s. Their
TTFTs are 0.170605, 0.166336 and 0.174357 s, versus paired packet TTFTs of 0.303323,
0.288293 and 0.281268 s. All candidate episodes satisfy the 5-second gate. Declared
warmups remain visible: stream 2.887195 s, packet 1.212476 s, phase switch 0.860404 s.
CPU load/placement is 5.869322 s. Cold shared-worker-to-first DENSE logit is
13.085643 s, not phase-first cold TTFT; supervised wall includes module startup
and workload-integrity precheck as well.

## Memory, provenance and limits

All three conditions peak at 4,763,783,168 sampled GPU bytes (about 4543 MiB),
below the imposed 4800 MiB allowance. Peak allocator reservation is 4,211,081,216 B.
CPU-first checks show zero current and historical CUDA allocation/reservation
before construction. Non-outgoing CUDA parameters remain 3,625,472,000 B; workspace
52,428,800 B; CUDA and pinned packet buffers 52,510,720 B each. Two outgoing host
layouts of 1,677,721,600 B each are charged to all modes, with no row cache.
Maximum FP16 KV is 527 positions, or 172,687,360 B. Whole-worker peak RSS is
9,674,321,920 B and minimum available host memory is 7,668,584,448 B.

The original OPT-2.7B checkpoint and baseline .venv remain fixed: Python 3.14.3,
PyTorch 2.10.0+cu130, CUDA build 13.0, Transformers 5.13.1, Safetensors 0.8.0,
Hugging Face Hub 1.19.0 and NumPy 2.3.5. Actual CUDA execution, arithmetic flags,
source snapshots, file hashes, copies, branch identities and clocks are archived.

These are three known archived WikiText concatenations, 512-token prefixes and
16-token caps, with alternating candidate order and a baseline-first streamed
reference. They are not a new holdout, a timing confidence interval, natural long-
context validation or task-quality evaluation. The device has 16 GiB; this is an
imposed budget, not a physical small-card OOM experiment. All three controls fit.
There is no unique-access win, independent fully resident 2.7B reference, FP32-
quality guarantee, optimized Q4/Q8 comparison, native patch or deployment claim.

## Course conclusion

Six releases now distinguish reuse, anticipation, representation, capacity, prompt
union and acquisition phase. The LRU and early-forecast extensions lost economics.
FP16 packets survived a larger CPU-first model, but uniform packets lost prefill
gates. The phase-switched implementation addresses that measured failure and wins
its short paired screen by deliberately trading bytes for less handling work.

Next work needs qualified optimized low-bit baselines, instrumentation-overhead
controls, and broader workload/context validation, each separately frozen with a
minutes-scale subset first. Those are open frontiers, not claims this release
settles. ReLU specialist/ranker variants also remain genuinely unmeasured; historical
Qwen-only studies do not close them. Stop this course at six completed deliveries;
milestone 10 remains open and no native pivot follows automatically.

[Protocol](phase-screen-protocol.md),
[compact receipts](../results/phase-screen-20260915/summary.json).
[LLM in a Flash](https://arxiv.org/abs/2312.11514),
[DejaVu](https://proceedings.mlr.press/v202/liu23am.html), and
[OPT](https://arxiv.org/abs/2205.01068) are the physical-grain/scheduling/architecture
antecedents. Phase specialization is not claimed as an invention; this is a new
implemented and measured policy within the repository's physical-loading program.

All 662 baseline CPU tests pass; v0.37-v0.41 raw summaries replay unchanged. Source
review is approved, the repeat raw audit is exact, and every ZIP member was restored
and hash checked. Raw `phase-v042-raw.zip`: 79,147,215 B, 160 members, SHA256
`97ff97efbbb79923c79f12a60ef09666c7dc4dba4caff1b512c3bdc2efb39f19`.
Complete only issue #57 and delivery 6. Final clean-tip full-suite evidence attaches
to the prerelease; no model weights are redistributed.
