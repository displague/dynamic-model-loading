# Apparatus run

Status: **packing_pilot_completed**.

Purpose: `exploratory_wikitext_packing_pilot_not_confirmatory_evaluation`.

This is an exploratory diagnostic. It is not a confirmatory held-out quality benchmark,
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
| native | 1 | 50% | 0.500 | 0.030682 | 1.0226 | 91.618% |
| native | 1 | 75% | 0.750 | 0.00215 | 1.0003 | 97.696% |
| native | 32 | 50% | 0.500 | 0.78252 | 2.0548 | 62.157% |
| native | 32 | 75% | 0.750 | 0.20459 | 1.1846 | 80.098% |
| native | 128 | 50% | 0.500 | 1.2819 | 3.3948 | 52.426% |
| native | 128 | 75% | 0.757 | 0.34835 | 1.3676 | 73.873% |
| random | 1 | 50% | 0.500 | 0.030873 | 1.0228 | 91.912% |
| random | 1 | 75% | 0.750 | 0.0021534 | 1.0000 | 97.721% |
| random | 32 | 50% | 0.500 | 0.79618 | 2.1064 | 61.789% |
| random | 32 | 75% | 0.750 | 0.20479 | 1.1663 | 79.681% |
| random | 128 | 50% | 0.500 | 1.3389 | 3.4865 | 52.157% |
| random | 128 | 75% | 0.757 | 0.34628 | 1.3587 | 73.284% |
| popularity | 1 | 50% | 0.500 | 0.030622 | 1.0219 | 91.765% |
| popularity | 1 | 75% | 0.750 | 0.0021726 | 1.0001 | 97.770% |
| popularity | 32 | 50% | 0.500 | 0.57749 | 1.6948 | 68.211% |
| popularity | 32 | 75% | 0.750 | 0.15412 | 1.1287 | 83.431% |
| popularity | 128 | 50% | 0.500 | 0.8038 | 2.0871 | 61.593% |
| popularity | 128 | 75% | 0.757 | 0.24444 | 1.2252 | 78.824% |
| coactivation | 1 | 50% | 0.500 | 0.030532 | 1.0210 | 91.887% |
| coactivation | 1 | 75% | 0.750 | 0.00217 | 1.0000 | 97.770% |
| coactivation | 32 | 50% | 0.500 | 0.678 | 1.8519 | 64.853% |
| coactivation | 32 | 75% | 0.750 | 0.17936 | 1.1676 | 81.127% |
| coactivation | 128 | 50% | 0.500 | 0.98416 | 2.5802 | 57.819% |
| coactivation | 128 | 75% | 0.757 | 0.2637 | 1.2423 | 76.642% |

No selection predictor, cache simulation, weight transfer runtime,
or omission-error detector is implemented. These measurements do not establish their feasibility.
