# Acquisition lifetime and anticipation: four to eight deliveries

The owner authorized four to eight additional research releases on 2026-09-15.
Initial sequence: bounded row retention, causal anticipation with exact demand
completion, stronger warmed precision comparisons, then larger sparse-model
capacity. Four deliveries are the commitment; up to four distinct follow-ups may
be added when measured evidence supplies a concrete hypothesis. Negative screens
remain completed deliveries, not successful gates. No long matrix or native pivot.

Correct the model assignments: v0.31 specialists and v0.34 vocabulary risk both
used Qwen, not OPT/ReLU. Those outcomes do not close their ReLU analogues. Nor does
observed activity make prediction useless before the observation is available.
Known zero contributions certify outgoing omission only AFTER the fully computed
first projection. Predictions here schedule transfers; missing required rows must
arrive before computation. Wrong predictions may waste time/bytes, never authorize
an unverified omission. Historical protocols and notes remain immutable.

## Hypotheses and antecedents

- Retention: consecutive observed active sets may overlap enough to repay bounded
  cache bookkeeping. [LLM in a Flash](https://arxiv.org/abs/2312.11514) uses windowing
  and row-column bundling. Retention and packets themselves are established ideas;
  our candidate is an outgoing-only exact arithmetic implementation and accounting.
- Anticipation: useful work might overlap transfers scheduled from past activity.
  [DejaVu](https://proceedings.mlr.press/v202/liu23am.html) motivates predictive,
  asynchronous execution; [PowerInfer](https://arxiv.org/abs/2312.12456) motivates
  neuron locality and heterogeneous residency. Our predictions must not stand in
  for observed-zero correctness. A better fit is not automatically better loading.
- Precision: resident FP16 probably beats FP32 packets when both fit and warmed.
  A new paired protocol must retain initial use separately, not erase v0.36's stall.
  Lower precision has its own reference; FP32 drift and target-token agreement
  remain separate. No claim to beat optimized Q4 from a naive PyTorch quantizer.
- Scale: OPT-2.7B is a candidate larger ReLU artifact, not a promised successful
  deployment. Inspect and pin its original HF artifact before inference. Compare
  real feasible low-memory baselines; disclose any missing quantized comparison.

Stock 32B target-only and small-draft results remain practical deployment controls,
not numerically interchangeable models. ADRs 0004--0006 and the baseline .venv stay.
Charge setup, warmups, CPU/GPU scratch, metadata, KV, cache/displaced residency and
all transfers. Unchanged bytes with lower exposed waiting can be an acquisition
win; predictor accuracy or a synthetic overlap ceiling cannot be called that win.

Each delivery receives a separate frozen protocol and <=300-second supervised
worker. Failed correctness/resources/timeouts stop that candidate. An expanded
study requires a separately frozen protocol even after a passing short screen.

## Completed course: v0.37-v0.40

- [v0.37 retention](retention-screen-results.md): 46.04% fewer H2D bytes, 2.24x wall.
  Reuse exists; this 192 MiB LRU implementation loses economics.
- [v0.38 early acquisition](early-screen-results.md): 94.38% event-region overlap,
  but 37.13% more bytes/13.29% more wall than demand packets. Stop this forecast.
- [v0.39 precision](precision-packet-results.md): FP16 packets save 94.41% H2D and
  52.23% wall versus dense streaming; warm resident FP16 is about 4x faster and fits.
- [v0.40 scale](scale-screen-results.md): CPU-first OPT-2.7B FP16 under 4800 MiB,
  95.25% less H2D/64.62% less wall than FP16 streaming, all checked logits exact.
  Both offloaded controls fit; full resident FP16 parameter bytes exceed allowance.

Four deliveries are complete, not four successful methods. This supplies a larger
same-precision acquisition result, not optimized-Q4 competitiveness or native
admission. Stop the failed retention/forecast variants. Leave qualified low-bit
baselines, context/block-union durability and the genuinely unmeasured ReLU
specialist/ranker variants open; new studies require new protocols. No additional
release is fabricated by renaming one of the same failed candidates. Milestone 10
remains open and the practical stock 32B comparison is unchanged.

## Owner-requested continuation within the remaining allowance

After v0.40 the owner explicitly requested continuation. The fifth delivery is a
new [512-token context-union screen](context-screen-protocol.md), justified by the
surviving larger FP16 acquisition result. It separates prefill union economics
from scalar decode, uses known archived diagnostic text, and freezes all source/
selection/gates before one minutes-scale worker. This is not automatic expansion
of the earlier failed retention/forecast methods or full context validation.

[v0.41 completed](context-screen-results.md): exact outputs and bounded memory,
but both prefill gates fail while decode/whole-episode gates pass. This permits
one distinct sixth hypothesis: switch physical acquisition grain by phase, dense
prefill and observed-row scalar decode. It is unmeasured until separately frozen
and run; no best-of-historical-time recombination is evidence of speedup.
