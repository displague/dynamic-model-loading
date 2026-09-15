# v0.38: causal early acquisition versus identical late predictions

Prospective short screen under ADRs0004--0006, reviewed/committed/pushed BEFORE
inference. Same pinned original OPT1.3B FP32 artifact and baseline .venv as v0.37;
three NEW authored32-token prefixes, at most16 generated tokens. Exact configs in
`configs/early-screen.json`. One <=300s supervised worker, including imports,
model loading, hashing, reference episodes, warmups and every check. No long matrix.

## Hypothesis and prediction

Prediction may be useful to schedule transfers BEFORE exact activity is known,
without authorizing omitted contributions. Last call's active rows for the SAME
layer, first1024 in ascending neuron order, are the entire causal signal. No Qwen
geometry, learned model, current target observation or cross-document state.
Expected: some event-region overlap, but limited fc1 duration, CPU gather work and
false-positive fetches may prevent net speed. H2D may increase; that alone is not
the verdict. This is a different acquisition-timing hypothesis from v0.37 LRU.

[DejaVu](https://proceedings.mlr.press/v202/liu23am.html) motivates asynchronous
prediction; [PowerInfer](https://arxiv.org/abs/2312.12456) supplies locality and
heterogeneous-residency antecedents. No invention of prefetch or CUDA streams.

## Three controls and correctness

- packet: fetch only currently observed active rows, no early weights or cache.
- late: form last-call predictions after fc1 completes, transfer them, then fetch
  every currently active row not predicted. This deliberately pays false positives.
- early: identical predictions and packet representation, but launch the transfer
  on a separate CUDA stream in the fc1 pre-hook. Then execute original fc1/ReLU;
  wait for the forecast packet, scatter it and fetch every missing required row.

All outgoing matmuls remain original-layout dense fused F.linear. First projection
fully computed. Finite stale rows contribute only where observed activation is
zero. False positives waste bytes; false negatives stall for exact demand fill.
Use separate pinned/device forecast and demand packets; reuse a forecast buffer
only after its completion event and outgoing consumer complete. No retained weight
cache. Forecast buffers (8,396,800 B each at1024 rows) are charged to ALL controls.
Explicit weight/index/activity/token/logit copies, CPU selection/gather, CUDA event
creation/synchronization, scatter, raw recording and maps/RSS all count.

Resident reference phase and hybrid transition follow v0.37. Separate resident
warmup then three references; hybrid warmups packet/late/early. Scored order:
doc0 packet/late/early, doc1 early/packet/late, doc2 late/early/packet. Clear KV and
predictive history at every episode. New resident references establish outputs;
none of these short prefixes is a held-out task-quality claim. Reference GPU limit
6500MiB, candidate4800MiB, host floor2048MiB. Full reference GPU startup means NO
CPU-first capacity claim. Baseline resident FP16/Q4 competitiveness remains separate.

## Events, gates and stop rules

Record CUDA-event forecast-copy duration and its overlap with the fc1 event-bracketed
region, plus exposed host event-wait time. Event regions include dispatch gaps;
they are NOT profiler proof of concurrent arithmetic kernels. Both early and late
pay event instrumentation. No inference-time speculative token acceptance is claimed.

- Hfaithfulness/resources: all candidate IDs/stops agree, every checked full-vocab
  logit relativeL2 <=1e-5; all resource/provenance/raw accounting checks pass.
- Htraffic: early and late have identical acquisition bytes, and early weight plus
  metadata H2D <=125% of demand-only packets over the scored episodes.
- Htiming: early total wall <=95% of BOTH packet and late controls.
- Hoverlap: sum event-region overlap >=10% of early forecast-copy event time,
  with positive duration. This is an apparatus condition, not a stand-alone win.

All gates required to advance this candidate. Equality comparisons preserve the
declared thresholds. Freeze before results; failures/timeouts stop expansion, and
any corrected experiment uses a fresh directory. Report raw warmups, all three
conditions, wasted forecasts and byte/latency/memory separately. No native pivot.
