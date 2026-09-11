# Transfer amplification and bounded-cache trace protocol

Declared before new measurements. Planned delivery: v0.4.0, issue #9, original Stage 2.
This measures analytical selection traces and simulated traffic, not a sparse runtime.

## Inputs, grid, and correctness

Keep the Qwen2.5-1.5B-Instruct checkpoint, FP32, Python 3.14 / torch 2.10.0+cu130,
16 WikiText development articles, 80 calibration articles, and numerical limits from
the packing pilot. Require the pinned corpus and regenerated layout digests to match
that pilot. Recompute all four layouts from calibration only with unchanged seeds and
grouping settings. Save calibration mean importance, exact layouts, source and input
receipts before omission probes. Do not access a final test set.

Test native, random, popularity, and coactivation layouts at group widths 8, 32, 128
and retained group fractions 0.5, 0.75, 0.85, 0.9, 0.95, 0.99: 72 conditions. This
extends the preexisting grid prospectively; original rows and thresholds remain
unchanged. Reuse the original numerical gate and block all probes if any check fails.
Every condition measures all 16 articles with the existing teacher-forced masked
dense computation. Preserve all KL, NLL, relative PPL, and top-1 results.

## Trace semantics and amplification

Record the actual selected group mask at every token and FFN. Store packed bits with
explicit shape, group width, neuron count, document ID, layout, retention, and hashes.
Quality belongs to that applied group mask, including effects of earlier omissions.
No trace may be attached to a different condition's quality metrics.

The framework computes a document in prefill order. For trace simulation, traverse
the captured masks in token-major, then layer-major order. This is a hypothetical
batch-one decode access order reconstructed from teacher-forced prefill; it does not
establish identical masks under incremental decode arithmetic or closed-loop generation.
Within a layer, consume already resident requested groups before loading misses in
ascending group-index order. The complete requested group set is already known to
this hindsight analysis. No predictor cost is hidden in an execution-speed claim.

Also record two geometric diagnostics from the same intermediate importance scores:
the fraction of top individual-neuron selections covered by the applied group mask,
and the bytes of all groups covering those top neurons divided by their individual
weight bytes. The latter is neuron-cover amplification, not actual fetched volume
and not a separately quality-evaluated mask. Extra contributions need not preserve
the individual-neuron output. Charge short tail groups by their true neuron count.

## Cache policies and budgets

Simulate 256 MiB, 512 MiB, 1 GiB, and 2 GiB FFN budgets. Reserve one maximum-size group
slot within each budget for incoming/bypassed data; the remainder is retained cache.
Report unusable remainder bytes and the reserved slot separately. The fixed non-FFN
parameter allocation comes from actual model storage. KV, host staging, and runtime
workspace are unmeasured here, so these budgets do not bound total process memory.
Slots are maximum-group sized; a short tail transfers only its actual bytes but still
occupies one slot. This implementation requires homogeneous layer dimensions.

Policies: no retained cache, LRU, a static global hot set ranked by calibration mean
group importance per byte, and a static hot set distributing slots evenly across
layers and ranking within each layer by that same statistic. Use stable ID tie breaks.
Static pages are preloaded at the start of each cold document, and preload traffic is
charged even if a page is never selected. Non-hot requests bypass the retained cache.
Warm replay reuses that static preload; it does not charge it again. LRU starts empty
for the cold document and preserves final state for one warm replay. No-cache has no
reuse. Report per-document and aggregate bytes, demand hits/misses, and preload bytes.

Compare each policy with its dense all-group access baseline at the same cache budget.
For the development screen, the strongest dense static baseline may choose any layout
and group width in the declared grid, independently of the sparse condition. Preserve
both this strongest comparison and each policy's matched-layout/width dense baseline.
Static traffic is simulated; no PCIe transfers are performed. LRU may use a proven
shortcut: if every repeated layer visit is separated by more distinct other-layer
groups than cache slots, it cannot hit. Validate the trace's layer counts and group
sizes before applying that sufficient condition. Preserve its numeric bound. For
small traces, verify against an explicit LRU simulator; no optimal-cache claim is made.

## Development screen and reporting

Prospectively screen the new grid for relative PPL <= 1.01 and at least 10% fewer warm
simulated bytes than the strongest dense static baseline at the same budget. Report
KL/top-1 and per-document behavior regardless; PPL alone is not task equivalence.
This is a provisional feasibility screen for a causal-predictor experiment, not a
deployment gate, a prediction of speedup, or a reclassification of prior failures.
Passing it still leaves the multi-domain held-out gate and all measured-runtime gates
open. If no condition passes, investigate independent controls rather than assuming a
predictor or pager will fix the missing quality/traffic opportunity.

Publish every condition, budget, policy, and cold/warm result. Select any subsequent
prototype condition only by this declared rule: passing condition with lowest warm
simulated traffic, then lowest KL, then layout name/group width/retention/budget.
Use policy name as a final tie break if all those quantities tie.
Publish the choice and freeze its next-stage protocol before new measurements.

Large packed-bit traces may be GitHub release assets rather than Git objects. Keep
their hashes, per-document quality, and complete replay/analysis receipts in Git;
retain the exact source and an immutable local copy. Preserve incomplete runs and
interruption receipts. Do not claim physical transfer speed, bounded residency, or
cache optimality from this study.
