# Fault-pager research context

This context note was written while the already-frozen CUDA matrix ran on
2026-09-14. It changes no protocol, selection rule, tolerance or result. It is a
bounded antecedent check, not an exhaustive novelty survey or a priority claim.

Selective execution, weight acquisition and speculative verification are not new
individually. The contribution under test here is the repository's specific causal
medoid-index acquisition policy and its measured physical behavior under explicit
residency, transfer and target-verification accounting.

## Relevant antecedents

- [LLM in a flash](https://arxiv.org/abs/2312.11514v3) studies flash-to-DRAM
  loading, reuse of previously active neurons and bundled row/column reads. Our
  experiment measures CPU-to-GPU copies, not flash-storage acceleration.
- [PowerInfer](https://arxiv.org/abs/2312.12456v2) partitions frequently active
  neurons onto GPU and computes cold neurons on CPU, using activation prediction
  and sparse operators. Our selected draft instead acquires selected host pages
  for GPU execution; a separate dense target owns the commitment decision.
- [DynamicInfer](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b9762d1f15c68057275fa18d384c1a8-Abstract-Conference.html)
  already studies runtime-dependent neuron scheduling, hierarchical caches and
  activation-aware prefetch with overlap, evaluated on ReluLLaMA and Prosparse.
  This study is on the pinned SwiGLU checkpoint and synchronous transfers. Merely
  adding a predictor, cache or overlap would not establish a new research claim.
- [SubSpec](https://arxiv.org/abs/2509.18344v2) constructs low-bit substitutes for
  offloaded layers and shares resident layers and KV. Our study neither implements
  that sharing nor treats self-drafting or offloaded verification as an invention.
- [HCInfer](https://arxiv.org/abs/2605.05819v1) places a compressed backbone on
  GPU and offloads residual compensation to CPU with adaptive rank and asynchronous
  execution. [Deputy](https://aclanthology.org/2026.findings-acl.991/) dynamically
  chooses full, low-rank and skipped computation. A future residual or low-rank
  proposal must state its departure from these mechanisms explicitly.

## Interpretation boundary

These papers' reported speedups are not comparable measurements for this run:
artifacts, model transformations, baselines, hardware and correctness contracts
differ. None substitutes for target-scale evidence in this repository.

The current release tests a custom research implementation, not stock flag tuning.
That fact alone is neither a global novelty claim nor evidence of acceleration.
After the full matrix, a follow-up must identify a measured failure mechanism, a
specific acquisition-policy departure, and a fresh prospective comparison. Reusing
known mechanisms under a new name or moving slow Python code to llama.cpp would
not answer the original research question.
