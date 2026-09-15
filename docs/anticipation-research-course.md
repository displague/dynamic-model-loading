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
not numerically interchangeable models. ADRs0004--0006 and the baseline .venv stay.
Charge setup, warmups, CPU/GPU scratch, metadata, KV, cache/displaced residency and
all transfers. Unchanged bytes with lower exposed waiting can be an acquisition
win; predictor accuracy or a synthetic overlap ceiling cannot be called that win.

Each delivery receives a separate frozen protocol and <=300-second supervised
worker. Failed correctness/resources/timeouts stop that candidate. An expanded
study requires a separately frozen protocol even after a passing short screen.
