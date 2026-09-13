# Stock thresholds and attention-resident verification

The attention-resident configuration reaches **20.568 native decode steps/s**
after the populated16K prefixes, versus15.295 for its fresh whole-layer control.
That is34.5% more decode throughput and15.6% more emitted IDs per complete request.
Mean128-token request time falls39.827→34.451s. All four paired long outputs match
exactly; all eight placement outputs match their frozen historical references.

These are two reused development prompts and two reversed repeats on this laptop,
not a general workload benchmark or an independently reproduced GPU result.
The target is Qwen2.5-32B-Instruct Q4_K_M with a Qwen2.5-0.5B-Instruct Q8_0 draft,
stock b10919, q8_0 KV for both, context capacity18432 with16384 populated tokens,
K16, p_min0, target threads24/24, draft8/24, batch/microbatch256, greedy decoding,
and one active request. The current driver is616.92; v0.14 preparation recorded
616.64. Performance comparisons here use fresh matched controls.

## Threshold comparison

The [original protocol](stock-placement-protocol.md) held ngl38 including output
and every other setting fixed. All12 long outputs match the historical references.

| Threshold | Native steps/s | Mean complete request(s) | Mean prefill(s) |
|---:|---:|---:|---:|
| 8 |14.439|40.228|31.421|
| 4 |15.130|39.871|31.468|
| 2 |15.405|39.818|31.553|

Threshold2 wins the frozen pooled decode-time rule; threshold4 falls outside the
1% tie band. The decode gain over8 is6.7%, while whole-request throughput improves
only1.0%. This is useful incremental tuning, not the main new result. No confidence
cutoff or K expansion was tested. The effective p_min remains0.

## Placement and complete cost

The separately registered [fixed comparison](fixed-attention-protocol.md) uses
threshold2 for both layouts. The control places37 transformer layers plus output
on GPU. The candidate assigns all64 transformer layers plus output to GPU, then
overrides the first32 layers' gate/up/down FFN tensors to host storage with stock
`--n-cpu-ffn 32`. The draft stays unchanged.

| Item | Whole-layer control | Attention resident /32 host FFNs |
|---|---:|---:|
| Target GPU model buffer(MiB) |10939.06|10661.48|
| Target host model buffer(MiB) |7986.95|8264.53|
| Target GPU KV(MiB) |1415.25|2448.00|
| Target CPU KV(MiB) |1032.75|0|
| Draft GPU model /KV(MiB) |500.84 /114.77|500.84 /114.77|
| Sampled total GPU peak in timed runs(MiB) |13846.39|14534.39|
| Native decode steps/s |15.295|20.568|
| Mean prefill(s) |31.513|28.254|
| Mean complete128-token request(s) |39.827|34.451|
| Emitted IDs /complete request second |3.214|3.715|

Both layouts meet the same15000MiB total sampled GPU allowance and2GiB minimum
available host RAM. Their actual allocations differ. This study does not search
for a new globally optimal whole-layer placement; the control is the best
previously measured long-context whole-layer configuration. The candidate moves
1032.75MiB of target KV onto GPU while reducing resident target model bytes by
277.58MiB. Other runtime buffers contribute to the measured total. Draft host
model storage is137.94MiB in both layouts.

Each layout records88 verification cycles across its four scored requests. Output
agreement is observed on these trajectories; it does not close #27's historical
short-context discrepancies or establish universal numerical/state correctness.
Native generation steps exclude the first token supplied by prefill. The emitted
ID and complete-request denominators are kept separate in the [full tables](stock-placement-tables.md).

## What the backend evidence establishes

Both corrected mechanism processes record82 target graph dumps with complete
192-projection FFN coverage. The whole-layer control has37 or64 target attention
assignments on CUDA, depending on the graph. The candidate has all64 on CUDA in
every recorded target graph. Its first32 FFNs have0 or96 CUDA projection assignments
per graph, consistent with operation offload applying to eligible batches while
single-position work can remain on CPU. Actual allocated model buffers are CUDA_Host,
not merely requested CPU overrides; every selected gate/up/down override is present.

These are backend assignments, not executed-operation counts, actual H2D bytes,
or isolated CPU-attention/copy timers. Some control graphs also place all attention
on CUDA. The results therefore support the deliberate attention/KV placement; they
do not validate an assumed180ms CPU-attention residual or an overlap speedup forecast.
All measurements use stock runtime controls; no custom kernel or scheduler was added.

## Preserved stops and fresh source boundaries

1. Source`ffdbba1` completed the threshold comparison and36/34/32-host-FFN allocation
   diagnostics. Their GPU peaks were13586.39,14076.39 and14534.39MiB. The30-host-FFN
   attempt stopped during startup at15004.39MiB, just4.39MiB above the experimental
   cap. This is not physical OOM or a general infeasibility result. The original
   stage produced no allocation decision or scored placement comparison.
2. Source`1cb43bd` prospectively fixed32 from its completed trial and used a fresh
   root. Both native mechanism processes completed, but the byte-identity audit
   rejected the candidate snapshot's LF→CRLF conversion before timing started.
   Preserve that attempt and its separately reproduced audit failure.
3. Source`3103f0d` copies candidate bytes exactly, with an LF/CRLF regression test,
   and reruns the entire fixed comparison in a third root. All six corrected
   processes complete. No old row, hash, limit or failure is rewritten or waived.

The delivery retains18 native process attempts and44 inference requests:12 threshold
and8 placement requests are scored; the remaining requests are warmups or allocation/
mechanism diagnostics. The startup-bound attempt makes no inference request. The
original and first fixed attempts remain separate from the corrected comparison.

## Scope and next use

The known-layout retained-prefix result is already published in v0.16. The new
attention-resident layout has earned a fresh retained-prefix comparison, which is
the remaining part of #31. Its changed placement has not yet been tested on that
continuing-conversation fixture,32K contexts, arbitrary prompts or sampling.
The128-token cap can leave incomplete answers; no agent-utility score is claimed.
Keep #27 and the broader stock-runtime milestone open. #25/#26 and custom overlap
remain deferred; the next practical measurement should test the new layout's
continuing-turn latency and state reuse before another model or controller.
