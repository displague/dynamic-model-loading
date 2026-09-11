# Research plan and delivery history

This roadmap reconstructs the staged course in the final critical review of the
[original research conversation](https://chatgpt.com/share/6aa40208-727c-83e9-a91b-b2ce031c96eb).
It was recorded retrospectively on 2026-09-11. The original experiment protocols were
written before their measurements; this roadmap is not a retroactive preregistration.

The question is whether a dense FFN's useful computation survives physical grouping,
can be selected cheaply from causally available information, and can be executed with
less real data movement under a bounded memory budget. Corrective refinement is a
separate hypothesis. A residency miss is observable; an omitted important group can
silently damage quality without causing a residency miss.

## Original stages and present evidence

| Stage | Deliverable and exit gate | Current evidence / remaining work |
|---|---|---|
| 0. Reproducible reference | Dense and physically repacked outputs agree under declared tolerances; record timing, memory, and provenance. | FP32 passes. BF16 permutation fails; canonical down-projection order is exact but costly. Practical BF16 policy and proposed ReLU positive control remain open. |
| 1. Hindsight sparsity envelope | Individual-neuron and per-layer/multi-layer omission curves on whole-document and domain splits; useful sparsity survives unseen inputs. | Smoke, a 16-article Wikipedia slice, and 66 per-layer/joint conditions exist. Sensitivity is distributed on this slice. Multi-domain held-out quality and task outcomes remain unmeasured. |
| 2. Physical packing and cache traces | Quality versus transferred bytes and transfer amplification at 256/512 MiB and 1/2 GiB FFN-cache budgets. | All 72 declared conditions and 1,152 traces completed. Popularity/width 8/90% retention is the only condition passing the prospective development screen; best warm traffic saving is 15.54% at 2 GiB. This is reused-data simulation; held-out quality and physical transfers remain open. |
| 3. Causal prediction | Static hot-set, recency, EMA, and learned-selector quality/bytes/cost curves; savings pay for prediction, including approximate closed-loop inference. | The frozen Stage 2 ordering nominates popularity/width 8/90%, with the 2 GiB equal-layer static cache. Predictor measurements have not started. Document feature availability and audit omitted computation independently. |
| 4. RAM-to-VRAM execution | Explicit bounded slots, pinned staging, actual skipped reads/computation; beat the strongest same-budget baseline after all costs. | Not started. Compare dense streaming, static/recency/activation-aware caches, and CPU/GPU execution splits. |
| 5. Corrective refinement | Silent-failure risk versus added bytes/latency; improve on choosing a larger initial subset. | Not started. Define unsafe omission first, retain the FFN input, add missing contributions before downstream commitment, and include bounded dense fallback. |

The original longer-term milestone follows these stages: repeat on a 7B-class model
and another family, then study quantization, heterogeneous groups, and limited
lookahead independently. Low-rank residuals and training for block structure are
representation experiments, not assumed runtime improvements. SSD/NVMe experiments
must measure actual storage reads and cache behavior. MTP, Bayesian state, conversational
memory, and an always-on service remain deferred.

## Releases are experiment deliveries

| Release | Scope | Finding |
|---|---|---|
| [v0.1.0](releases/v0.1.0.md) | Reference apparatus, original BF16 failure, separate FP32 control | Correct paired packing in FP32; grouping loses much of individual-neuron quality. |
| [v0.2.0](releases/v0.2.0.md) | Larger packing pilot and arithmetic diagnosis | Tested co-activation heuristic loses in every grouped setting; canonical BF16 order is exact. |
| [v0.3.0](releases/v0.3.0.md) | Per-layer omission sensitivity and interaction diagnosis | Top-four layers account for 20.8% / 19.6% of single-layer KL sums; all-layer controls reproduce exactly. |
| [v0.4.0](releases/v0.4.0.md) | Applied-mask traces, transfer amplification, and bounded-cache simulation | One of 72 conditions passes the development screen; best simulated warm saving 15.54%, relative PPL 1.008978. |

No completed delivery establishes an inference speedup or achieved memory reduction.
The stage milestones remain open where their full gates have not been met. The
machine-readable [GitHub plan](planning/github-plan.json) records issue scope and
milestone definitions; [GitHub](https://github.com/displague/dynamic-model-loading/milestones)
tracks live status.

## Evaluation and publication rules

Preserve the original 0.01 relative-L2 and 0.001 KL numerical gates. These are packing
correctness limits, not general sparse-quality equivalence. Any task/PPL quality
tolerance must be declared before its evaluation; the conversation's illustrative
1% PPL / one percentage-point task limits and 20% runtime improvement are proposed
management gates, not adopted success claims or predictions.
The v0.4 protocol prospectively adopts aggregate relative PPL <=1.01 and >=10% warm
simulated traffic reduction for a development feasibility screen only. It does not
adopt or satisfy deployment/task/runtime gates; 9/16 articles at the nominated
condition individually exceed 1% PPL increase.

Charge fixed weights, FFN cache, KV, predictor, staging, workspaces, and other GPU
allocations separately. Selected weight volume is neither transferred bytes nor cache
capacity. Report prefill and decode separately, and evaluate cold/warm conditions,
topic switches, context length, and batching when their stage is reached. Include
optimized quantized resident models and smaller-model utility comparisons later.

Every release retains a detailed Markdown note, annotated tag, GitHub release, raw
receipts, validation, and an explicit remaining-work statement. Completed phases are
pushed. Commit messages explain findings to collaborators rather than narrating an
interactive session. Historical protocol/source snapshots are immutable.

PyTorch 2.12.0+cu130 is present in the local uv cache for CPython 3.13. The measured
baseline is CPython 3.14 / PyTorch 2.10.0+cu130; keep it for the layer study. A separate
environment control must validate a version change before combining its measurements
with this baseline. The [official wheel index](https://download.pytorch.org/whl/cu130/torch/)
also lists a CPython 3.14 build if a same-interpreter comparison is required.
