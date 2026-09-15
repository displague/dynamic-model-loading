# v0.38: early forecasts overlap fc1 but do not accelerate generation

The31.987-second supervised screen (25.795s worker,6.192s audit) preserves exact
logits and greedy outputs in every compared episode, including warmups. All48
scored tokens per condition match. A second raw replay is identical; no rerun.

| Scored condition | Weight + metadata H2D | Wall for48 tokens |
|---|---:|---:|
| Resident FP32 | 0 during generation | 0.487845s |
| Observed demand-only packet | 4,266,763,400 B | 1.951097s |
| Identical predictions, late copy | 5,850,946,000 B | 2.158978s |
| Early copy before fc1 | 5,850,946,000 B | 2.210470s |

Faithfulness/resources and Hoverlap pass; Htraffic and Htiming fail. Early copies
increase H2D37.1284% over packets (frozen maximum25%) and wall13.2937%. They are
also2.385% slower than the identical late-prediction control. Stop this candidate.

Of381,529 forecast rows,188,336 are useful and193,193 are wasted. Early/late byte
identity and actual-row demand completion are independently replayed. Predictions
never authorize skipping a nonzero contribution. The first call has no history.

Early forecast-copy event time90.8247ms overlaps its fc1 bracket for85.7176ms,
94.3770%. This is event-region overlap, including dispatch gaps, NOT profiler proof
of concurrent arithmetic kernels. Exposed event-wait sums are only1.7982ms early
versus1.8540ms late. The late control can overlap other work after fc1, including
activity discovery; it is not globally serialized inference. A favorable overlap
counter therefore did not buy end-to-end time. No kernel bottleneck is inferred
without profiling, and the original frozen timing gate is retained.

## Memory and limitations

No weight cache. The same extra8,396,800 B forecast device buffer and equally sized
pinned buffer are allocated to all controls. Base device packet67,174,400 B,
workspace67,108,864 B, host outgoing weights1,610,612,736 B, resident non-outgoing
parameters3,652,419,584 B. Host predictive lists are in RSS. Whole-worker peakRSS
12,956,065,792 B; minimum host availability5,010,710,528 B. Candidate episodes pass
4800MiB, while full-GPU reference setup uses6500MiB allowance and drives overall
peak NVML5,864,787,968 B. No CPU-first capacity claim. Initial reference parameters
5,263,032,320 B move H2D then D2H at transition; candidate construction moves
3,652,419,584 B H2D. Setup, first use and declared warmups remain in raw records.

Three new32-token prefixes and16-token continuations are not task quality, long
context, larger-model deployment or a precision frontier. Event instrumentation,
gathering, readback, scatter, false positives and records are paid. No long matrix,
field fit, native patch or stock-flag sweep follows this failed timing candidate.

## Next direction and antecedents

Move to the separately frozen lower-precision packet/strong resident comparison,
then larger sparse-model capacity. Do not re-label these forecasts as a successful
loader because their transfer events overlap. A better prior would still need an
affordable observation/scheduling mechanism and measured end-to-end benefit.

[DejaVu](https://proceedings.mlr.press/v202/liu23am.html) motivates asynchronous
prediction; [PowerInfer](https://arxiv.org/abs/2312.12456) motivates locality and
heterogeneous placement. The experiment is not their reproduction or a priority
claim. [Protocol](early-screen-protocol.md), [receipts](../results/early-screen-20260915/summary.json).

646 CPU tests pass and v0.37 replay remains identical after shared-harness
parameterization. Source and endpoint-accounting reviews approved before inference.
Raw archive77,091,994 B,169 members, SHA256
`33fb7a8a40986274354dd9464639dffcf122bc7210ac459c4dc328bf7c4b207c`.
Only issue53 and delivery2 of the authorized4--8 complete; milestone10 stays open.
