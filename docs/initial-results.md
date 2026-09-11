# Initial apparatus findings — 2026-09-11

The code runs on native Windows with the RTX 5080 Laptop GPU. This milestone produced
a preserved BF16 numerical failure and a successful separate FP32 control. It has
not demonstrated a memory-saving or faster runtime.

## Evidence

- [BF16 run and failure](../results/initial-20260911/bf16/report.md)
- [BF16 raw ledger](../results/initial-20260911/bf16/results.jsonl)
- [Accumulation-setting diagnostic](../results/initial-20260911/precision-diagnostic.json)
- [FP32 control and complete diagnostic table](../results/initial-20260911/fp32/report.md)
- [FP32 raw ledger](../results/initial-20260911/fp32/results.jsonl)
- [FP32 model/source/environment manifest](../results/initial-20260911/fp32/manifest.json)

These receipts are archived in version control under `results/initial-20260911/`.
Original runs remain under ignored `runs/`. Layout arrays are preserved losslessly as
`layouts.json.gz`; decompress before checking the raw ledger's layout SHA256.

Both runs used Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 248 calibration tokens, and 469 diagnostic
input tokens across six authored documents. Quality metrics cover 463 next-token
predictions. This is far below a research evaluation sample; no confidence interval
or task-accuracy conclusion is warranted.

## Correctness outcome

All 84 local grouped-reconstruction checks (28 FFNs x three layouts) passed in BF16.
However, all six documents failed the full-model numerical gate under each reordered
layout. Native order reproduced the reference exactly. The worst relative logit L2
was 0.02959 for random order and 0.02468 for popularity order, above the preset 0.01
limit. The worst mean KL was 0.001528, above the preset 0.001 limit. The harness blocked
all BF16 sparsity probes.

On one diagnostic document, disabling BF16 reduced-precision accumulation reduced
random-layout relative L2 from 0.01814 to 0.01484; that still failed the same limit.
The diagnostic does not establish the sole source of the drift.

The separate FP32 control passed all local and full-model checks with unchanged
tolerances. Maximum relative logit L2 was 2.85e-6 and maximum mean KL was 5.68e-8.
This supports the permutation/indexing implementation and points to numerical
sensitivity in BF16. It does not establish a satisfactory BF16 execution policy.

## Grouping diagnostic

All FFNs were masked together using already-computed activation importance. This is
teacher-forced analytical evaluation, not a causal selector or sparse implementation.
The following subset uses the calibration-popularity layout:

| Neurons/group | Groups retained | Actual selected weight fraction | KL(dense to masked) | Relative perplexity | Top-1 agreement |
|---:|---:|---:|---:|---:|---:|
| 1 | 75% | 75.00% | 0.002765 | 1.0090 | 96.98% |
| 32 | 75% | 75.00% | 0.1860 | 1.0959 | 75.38% |
| 128 | 75% | 75.71% | 0.2565 | 1.0747 | 72.35% |
| 1 | 50% | 50.00% | 0.03942 | 1.0380 | 89.20% |
| 32 | 50% | 50.00% | 0.6680 | 1.5639 | 60.48% |
| 128 | 50% | 50.00% | 0.8708 | 2.1643 | 57.02% |

These points expose a packing problem: much of the individual-neuron benefit is lost
when whole groups must be selected. Popularity grouping improves over native and
random layouts at group sizes 32 and 128 here, but does not close the gap. This does
not prove that co-activation grouping would succeed or fail; it has not been tested.

Larger groups do not order every quality metric monotonically: 128-neuron groups at
75% have better corpus perplexity but worse KL/top-1 than 32-neuron groups. Those
metrics measure different effects. Keep all of them rather than selecting the most
flattering metric.

Single-neuron results also vary slightly across physical orders. Selection is
discontinuous and later selectors receive perturbed hidden states. The current
results do not isolate the source of this sensitivity; do not claim exact mask
invariance across full inference merely because all-weight outputs nearly agree.

## Memory and timing boundaries

Unique model parameter storage was 3,087,428,608 bytes in BF16 and 6,174,857,216 in FP32.
FFNs account for 2,312,110,080 and 4,624,220,160 bytes respectively. These are actual
parameter sizes, not the local GGUF sizes from LM Studio.

The recorded dense-baseline CUDA allocated-memory peaks were about 3.13 GB in BF16
and 6.24 GB in FP32. They exclude non-PyTorch allocations and are not a bounded-cache
experiment or a claim about the peak over every later analysis operation.

Warm median prefill across these short documents was about 24.5 ms in BF16 and 48.6 ms
in FP32. Median incremental decode step times were about 21.6 and 20.1 ms respectively.
The decode timings are only 16 steps of one sequence, without controlled thermals or
replication; do not infer that FP32 is faster. Raw per-step times remain in the ledger.

## Next experiment

Before building a pager, resolve the BF16 numerical policy and expand the data beyond
the smoke fixture. Add layer-by-layer omission diagnostics and co-activation packing
using calibration data only. Then quantify cache reuse and real group transfer costs.
The decisive target remains quality versus transferred bytes under an explicit cache
budget, with all costs charged. This run does not yet measure that frontier.
