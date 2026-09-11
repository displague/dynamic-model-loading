# Apparatus run

Status: **smoke_diagnostics_completed**.

This is a smoke test on a small authored corpus. It is not a held-out quality benchmark,
a bounded-memory runtime, or evidence of an inference speedup.

## Correctness

All-group reconstruction and layout checks passed: True.

Tolerances were copied into this run before measurement. Raw per-document and per-layer
results are in `results.jsonl`; model, source, corpus, and environment provenance are in `manifest.json`.

## Hindsight diagnostics

All FFNs are masked together. Gate/up activations are computed densely before selection.
Selected bytes are a hypothetical no-cache weight volume, not measured transfers.

| Layout | Neurons/group | Groups retained | Weight fraction | KL | Relative perplexity | Top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| native | 1 | 50% | 0.500 | 0.037702 | 1.0423 | 89.633% |
| native | 1 | 75% | 0.750 | 0.0028268 | 1.0078 | 96.544% |
| native | 32 | 50% | 0.500 | 0.98023 | 2.3985 | 53.780% |
| native | 32 | 75% | 0.750 | 0.31612 | 1.3043 | 74.082% |
| native | 128 | 50% | 0.500 | 1.521 | 4.2654 | 46.436% |
| native | 128 | 75% | 0.757 | 0.36338 | 1.2279 | 70.410% |
| random | 1 | 50% | 0.500 | 0.038767 | 1.0335 | 89.201% |
| random | 1 | 75% | 0.750 | 0.0028004 | 1.0081 | 97.408% |
| random | 32 | 50% | 0.500 | 1.0344 | 2.3728 | 54.212% |
| random | 32 | 75% | 0.750 | 0.28546 | 1.2457 | 73.434% |
| random | 128 | 50% | 0.500 | 1.5977 | 4.4650 | 44.708% |
| random | 128 | 75% | 0.757 | 0.4231 | 1.4497 | 67.171% |
| popularity | 1 | 50% | 0.500 | 0.039419 | 1.0380 | 89.201% |
| popularity | 1 | 75% | 0.750 | 0.002765 | 1.0090 | 96.976% |
| popularity | 32 | 50% | 0.500 | 0.66798 | 1.5639 | 60.475% |
| popularity | 32 | 75% | 0.750 | 0.186 | 1.0959 | 75.378% |
| popularity | 128 | 50% | 0.500 | 0.8708 | 2.1643 | 57.019% |
| popularity | 128 | 75% | 0.757 | 0.25651 | 1.0747 | 72.354% |

No selection predictor, co-activation clustering, cache simulation, weight transfer runtime,
or omission-error detector is implemented. These measurements do not establish their feasibility.
