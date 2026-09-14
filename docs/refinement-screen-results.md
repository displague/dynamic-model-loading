# Progressive precision: useful eight-bit proposals, failed selective policy

The [prospectively frozen screen](refinement-screen-protocol.md) completed once:
81.182 seconds for the supervised worker, 12.362 seconds for independent analysis
(93.543 seconds combined). A second model-free audit reproduced the entire summary
in 12.136 seconds. No timeout, rerun, threshold change, long matrix or native build.
Source freeze: `3e0f585286ee87e0d424783b9f776a04eee1d98e`.

The runtime physically acquires packed two-bit increments above a resident
four-bit FFN base. It can reach six or eight bits without retaining all those
weights on GPU. This is implemented research apparatus, not llama.cpp tuning.
All 12 scored candidate outputs and six warmups match the dense target reference.

## Generation and physical traffic

Two known development documents, four prefix and four committed output tokens
each, K=4, one repeat, reverse condition order on document two. Every aggregate
row below emits the same eight target-verified tokens. Times include prefill,
verification, rejected draft work, controller, physical copies and ledger flushes.
Construction, snapshots, warmups and model-free analysis are separately charged.

| Condition | Accepted / proposed | Wall seconds | Increment H2D GiB | Cache hits |
|---|---:|---:|---:|---:|
| Dense target only | n/a | 0.271 | 0 | 0 |
| Four-bit base | 6/16 (37.5%) | 2.958 | 0 | 0 |
| Six bits, stream increments | 5/20 (25%) | 8.574 | 8.3441 | 0 |
| Eight bits, stream increments | 8/8 (100%) | 6.841 | 8.6133 | 0 |
| Adaptive precision, stream | 8/8 (100%) | 6.886 | 8.6133 | 0 |
| Adaptive + value-ranked retention | 8/8 (100%) | 6.877 | 7.9884 | 65 |
| Adaptive + fixed retention | 8/8 (100%) | 6.823 | 7.5366 | 112 |

Document-level acceptance is preserved: q4 is 3/8 on each; q6 is 3/8 then
2/12; q8/adaptive/retained/static are 4/4 on each. These tiny denominators do not
qualify general draft faithfulness or task quality. The previous v0.22 workload
and memory allowance differ; this is not a paired speedup over that release.

The adaptive rule requests eight bits on **448/448 scored FFN executions**. Its
precision selection offers no traffic advantage over fixed eight-bit streaming.
Six bits transfers 275.625 MiB per consumed draft token, eight bits 551.25 MiB;
different accepted prefixes cause different numbers of consumed tokens (31 vs 16).
Do not interpret the similar aggregate traffic as equal per-step economics.

Value-ranked retention saves **7.2545%** versus streaming, below the frozen 10%
screen threshold, and uses **5.9949% more traffic** than fixed retention. Its
0.1262% wall-time reduction versus streaming fails the frozen 5% timing check;
one repetition does not establish a significant speed difference. The fixed
cache saves **12.5%**, attaining the preregistered access-pattern bound:
2 documents * 7 post-cold tokens * 8 retained slabs = 112 hits.
Numerical values, precision decisions, proposals and verification records are
identical across all three adaptive cache conditions in the screen. Those scored
conditions have no rejected proposals; a separate forced-rejection CPU test
validates placement-only state retention after discarded speculative work.

This separates a real reuse benefit from an unproven value-ranking benefit. High
correction magnitude is not the same objective as minimizing future reloads on
this access pattern. No candidate beats the resident target-only reference.

## Formula transfer: why Richardson fails here

The owner's formula-to-other-domains method led to a prospective fourth test:
interpret 4/6/8-bit outputs as a numerical-refinement sequence. If
`y(h) = y* + c*h + O(h^2)` with a stable leading coefficient and fourfold spacing
reduction, `y6 + (y6-y4)/3` should cancel the leading error. This is conditional,
not a theorem about quantized FFNs. Sources and departures are recorded in the
[inspiration ledger](refinement-inspirations.md), including PMPD supplied by the
owner, Richardson, Any-Precision, BitStack, AnyBCQ and DecDEC.

The 28 frozen local inputs give:

| Output representation | Mean relative-L2 error against FP32 |
|---|---:|
| Four bits | 0.203799 |
| Six bits | 0.0498965 |
| Eight bits | 0.0121657 |
| Richardson from four/six bits | 0.0794774 |

Extrapolation worsens mean error by **59.28%**, and improves **0/28** layers.
The mean successive-increment norm ratio is **0.249340**, close to the assumed
quarter, but their mean directional cosine is only **0.071428**, not close to
one. The magnitude pattern resembles numerical refinement while the vector
alignment does not support a stable leading error term. That is evidence against
this fixed extrapolation formula, not against every multi-fidelity estimator.
The next-increment norm proxy underestimates the observed normalized change in
253/448 eight-bit scored executions (56.47%); it is not a certified upper bound.

These are local dense-input diagnostics. The extrapolated output was **not** used
in generation, fitted to the data, or promoted after the results were observed.
Possible future directions include statistically calibrated uncertainty instead
of fixed-direction correction, or PMPD-inspired high-precision prefill to isolate
KV contamination. Both require distinct causal controls and another protocol;
neither is an achieved result here.

## Memory is separate from traffic

| Component | Bytes | MiB |
|---|---:|---:|
| Resident four-bit codes and scales/minima | 650,280,960 | 620.15625 |
| Unpack, FP32 projection and refinement scratch | 254,607,360 | 242.8125 |
| Eight persistent slots plus one bypass slot | 92,897,280 | 88.59375 |
| Total explicitly allocated refinement apparatus | 997,785,600 | 951.5625 |
| Pinned host staging | 10,321,920 | 9.84375 |
| CPU refinement backing store | 578,027,520 | 551.25 |

The joint system additionally has 6,174,857,216 B of target GPU parameters,
1,550,637,056 B of draft non-FFN GPU parameters, and 4,624,220,160 B of original
draft FFN CPU parameters. Nothing is shared. Peak extra CUDA above the known GPU
parameter sum is **990.6348 MiB**, inside the prospectively declared **1024 MiB**
allowance, including baseline overhead, KV and temporaries. This is not the old
768 MiB allowance. All modes allocate the same apparatus, including unused slots.
Actual logical KV peaks, controller object sizes and allocator peaks remain in
the episode receipts. These are allocation-controlled rather than optimized
minimum-memory baselines.

All 463 resource samples pass: maximum total GPU usage **8967.098 MiB**, minimum
available host **7223.277 MiB**, peak process RSS **8823.762 MiB**. The runtime
can compute the eight-bit representation with nonresident refinements under its
extra-CUDA cap. This does **not** show access to a larger model than an optimized
quantized runtime, a long-session advantage, or a better capacity/quality frontier.
The dense 1.5B reference already fits this GPU; target/draft joint memory is paid.

CPU representation construction takes 4.873 seconds and uploads 650,280,960 B;
snapshotting takes 1.961 seconds and reads back 650,280,960 B. No FP32 FFN source
is uploaded during construction. Warmups total 13.815 seconds. Inference H2D
figures reconcile instrumented real tensor-copy payloads, not a simulated page
volume; they are not independent bus-level profiler measurements.

## Decisions, validation and reproduction

- H1: **fails**. Six bits reaches 25% acceptance but does not beat q4 by 10 points.
- H2: **fails**. Adaptive precision preserves the tiny sample's eight-bit output
  but saves no acquisition traffic versus eight-bit streaming.
- H3: **fails its threshold**. Retention saves real traffic, but 7.25% is below
  10%, and the fixed control does better.
- H4: **fails**. Fixed Richardson extrapolation worsens every local layer.

Overall decision: **stop expansion of this configuration**. The physical
incremental representation, eight-bit proposal result, fixed-cache benefit and
failure explanations remain useful evidence. No broader impossibility claim,
native-admission waiver, long matrix or stock flag sweep follows.

The source-freeze suite passes 402 CPU tests in 35.96 seconds. Review strengthened
peak-memory checks in all phases, independent packed-artifact reconstruction,
and forced-rejection state isolation. Independent eight-bit quantizer comparison
has maximum relative-L2 1.01322e-6; independent archived-bit-plane reconstruction
has maximum 1.46284e-6. Both are below the unchanged 0.01 numerical limit. These
packing checks do not assert that quantization is FP32-equivalent.

From the prepared baseline environment and preserved parent/checkpoint artifacts:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_progressive_precision.py tests/test_refinement_screen.py
.\.venv\Scripts\python.exe -m dynamic_model_loading.refinement_screen --output runs/refinement-screen-<fresh-name>
```

To audit without checkpoint loading after extracting the release archive under
`runs` (keep `supervisor.json` beside `worker`):

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.refinement_screen --analyze --output runs/refinement-screen-20260914-v1/worker
```

The [compact receipts](../results/refinement-screen-20260914) retain summaries,
episode records, manifest/file identities, allocation, numerical receipts, archive
inventory, reviews and validation. Text copies may normalize line endings; the
release archive preserves exact original bytes, all constructed tensors, policy
and physical ledgers, source snapshot, corpus/token IDs and licensing metadata.
`refinement-screen-v023-raw.zip`: 1,259,987,941 bytes, 169 members;
SHA256 `da6b5cf78d59d52d1663e172c2e3055b278bd51b678e4868cf46f6fb082fcc17`.
Every member was decompressed and SHA-checked, and raw sources rechecked.

This completes bounded [issue #38](https://github.com/displague/dynamic-model-loading/issues/38),
not the [novel-loading milestone](https://github.com/displague/dynamic-model-loading/milestone/10).
