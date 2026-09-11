# Apparatus run

Status: **correctness_gate_failed**.

This is a smoke test on a small authored corpus. It is not a held-out quality benchmark,
a bounded-memory runtime, or evidence of an inference speedup.

## Correctness

All-group reconstruction and layout checks passed: False.

Tolerances were copied into this run before measurement. Raw per-document and per-layer
results are in `results.jsonl`; model, source, corpus, and environment provenance are in `manifest.json`.


No selection predictor, co-activation clustering, cache simulation, weight transfer runtime,
or omission-error detector is implemented. These measurements do not establish their feasibility.
