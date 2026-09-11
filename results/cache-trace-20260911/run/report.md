# Apparatus run

Status: **packing_pilot_completed**.

Purpose: `development_cache_trace_simulation_not_measured_transfers`.

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
| native | 8 | 50% | 0.500 | 0.37576 | 1.3732 | 72.083% |
| native | 8 | 75% | 0.750 | 0.085004 | 1.0696 | 87.181% |
| native | 8 | 85% | 0.850 | 0.034926 | 1.0324 | 91.422% |
| native | 8 | 90% | 0.900 | 0.019024 | 1.0228 | 93.627% |
| native | 8 | 95% | 0.950 | 0.0060565 | 1.0033 | 96.544% |
| native | 8 | 99% | 0.990 | 0.00068386 | 0.9993 | 98.701% |
| native | 32 | 50% | 0.500 | 0.78252 | 2.0548 | 62.157% |
| native | 32 | 75% | 0.750 | 0.20459 | 1.1846 | 80.098% |
| native | 32 | 85% | 0.850 | 0.093461 | 1.0730 | 86.446% |
| native | 32 | 90% | 0.900 | 0.05172 | 1.0424 | 89.314% |
| native | 32 | 95% | 0.950 | 0.021307 | 1.0169 | 93.407% |
| native | 32 | 99% | 0.993 | 0.0021046 | 1.0020 | 97.794% |
| native | 128 | 50% | 0.500 | 1.2819 | 3.3948 | 52.426% |
| native | 128 | 75% | 0.757 | 0.34835 | 1.3676 | 73.873% |
| native | 128 | 85% | 0.857 | 0.16046 | 1.1404 | 82.132% |
| native | 128 | 90% | 0.900 | 0.10272 | 1.0773 | 85.417% |
| native | 128 | 95% | 0.957 | 0.034653 | 1.0156 | 92.181% |
| native | 128 | 99% | 1.000 | 0 | 1.0000 | 100.000% |
| random | 8 | 50% | 0.500 | 0.38084 | 1.4212 | 71.716% |
| random | 8 | 75% | 0.750 | 0.087731 | 1.0723 | 86.250% |
| random | 8 | 85% | 0.850 | 0.035227 | 1.0333 | 92.108% |
| random | 8 | 90% | 0.900 | 0.018944 | 1.0143 | 94.314% |
| random | 8 | 95% | 0.950 | 0.0066975 | 1.0022 | 96.740% |
| random | 8 | 99% | 0.990 | 0.00068552 | 1.0005 | 98.995% |
| random | 32 | 50% | 0.500 | 0.79618 | 2.1064 | 61.789% |
| random | 32 | 75% | 0.750 | 0.20479 | 1.1663 | 79.681% |
| random | 32 | 85% | 0.850 | 0.10238 | 1.0756 | 86.275% |
| random | 32 | 90% | 0.900 | 0.055204 | 1.0352 | 89.314% |
| random | 32 | 95% | 0.950 | 0.022281 | 1.0069 | 93.137% |
| random | 32 | 99% | 0.993 | 0.0021059 | 1.0015 | 97.868% |
| random | 128 | 50% | 0.500 | 1.3389 | 3.4865 | 52.157% |
| random | 128 | 75% | 0.757 | 0.34628 | 1.3587 | 73.284% |
| random | 128 | 85% | 0.857 | 0.15696 | 1.1416 | 80.417% |
| random | 128 | 90% | 0.900 | 0.10207 | 1.0755 | 84.804% |
| random | 128 | 95% | 0.957 | 0.035172 | 1.0242 | 91.250% |
| random | 128 | 99% | 1.000 | 4.5267e-08 | 1.0000 | 100.000% |
| popularity | 8 | 50% | 0.500 | 0.32171 | 1.2851 | 74.853% |
| popularity | 8 | 75% | 0.750 | 0.073388 | 1.0507 | 87.574% |
| popularity | 8 | 85% | 0.850 | 0.030553 | 1.0210 | 91.936% |
| popularity | 8 | 90% | 0.900 | 0.015765 | 1.0090 | 94.314% |
| popularity | 8 | 95% | 0.950 | 0.0056058 | 1.0060 | 96.618% |
| popularity | 8 | 99% | 0.990 | 0.00055898 | 0.9994 | 98.775% |
| popularity | 32 | 50% | 0.500 | 0.57749 | 1.6948 | 68.211% |
| popularity | 32 | 75% | 0.750 | 0.15412 | 1.1287 | 83.431% |
| popularity | 32 | 85% | 0.850 | 0.071906 | 1.0537 | 88.039% |
| popularity | 32 | 90% | 0.900 | 0.039528 | 1.0238 | 91.275% |
| popularity | 32 | 95% | 0.950 | 0.017153 | 1.0140 | 93.897% |
| popularity | 32 | 99% | 0.993 | 0.0014766 | 0.9993 | 98.309% |
| popularity | 128 | 50% | 0.500 | 0.8038 | 2.0871 | 61.593% |
| popularity | 128 | 75% | 0.757 | 0.24444 | 1.2252 | 78.824% |
| popularity | 128 | 85% | 0.857 | 0.11195 | 1.0781 | 85.196% |
| popularity | 128 | 90% | 0.900 | 0.070102 | 1.0495 | 88.137% |
| popularity | 128 | 95% | 0.957 | 0.025138 | 1.0117 | 93.064% |
| popularity | 128 | 99% | 1.000 | 4.4153e-08 | 1.0000 | 100.000% |
| coactivation | 8 | 50% | 0.500 | 0.35194 | 1.3679 | 74.069% |
| coactivation | 8 | 75% | 0.750 | 0.077583 | 1.0633 | 86.838% |
| coactivation | 8 | 85% | 0.850 | 0.036064 | 1.0260 | 90.980% |
| coactivation | 8 | 90% | 0.900 | 0.018574 | 1.0139 | 93.725% |
| coactivation | 8 | 95% | 0.950 | 0.0062269 | 1.0044 | 96.814% |
| coactivation | 8 | 99% | 0.990 | 0.00061204 | 1.0010 | 98.971% |
| coactivation | 32 | 50% | 0.500 | 0.678 | 1.8519 | 64.853% |
| coactivation | 32 | 75% | 0.750 | 0.17936 | 1.1676 | 81.127% |
| coactivation | 32 | 85% | 0.850 | 0.076564 | 1.0713 | 87.892% |
| coactivation | 32 | 90% | 0.900 | 0.045855 | 1.0386 | 90.270% |
| coactivation | 32 | 95% | 0.950 | 0.018407 | 1.0197 | 93.284% |
| coactivation | 32 | 99% | 0.993 | 0.0019264 | 1.0017 | 98.186% |
| coactivation | 128 | 50% | 0.500 | 0.98416 | 2.5802 | 57.819% |
| coactivation | 128 | 75% | 0.757 | 0.2637 | 1.2423 | 76.642% |
| coactivation | 128 | 85% | 0.857 | 0.12147 | 1.1072 | 84.730% |
| coactivation | 128 | 90% | 0.900 | 0.07672 | 1.0624 | 87.525% |
| coactivation | 128 | 95% | 0.957 | 0.026949 | 1.0236 | 92.304% |
| coactivation | 128 | 99% | 1.000 | 4.4374e-08 | 1.0000 | 100.000% |

No selection predictor, cache simulation, weight transfer runtime,
or omission-error detector is implemented. These measurements do not establish their feasibility.
