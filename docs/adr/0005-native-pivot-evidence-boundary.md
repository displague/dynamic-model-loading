# Require physical-policy evidence before a llama.cpp patch

Status: accepted, 2026-09-14. Complements ADR 0004.

## Decision

Implement and measure the fault-pager protocol first in the existing PyTorch
research apparatus. Do not create a llama.cpp fork, worktree, patch, or custom
native build merely to reproduce the experiment in a more optimized substrate.

A native pivot is permitted only after the completed PyTorch delivery supplies all
of the following, evaluated once on the frozen fault-pager protocol without candidate
replacement:

1. dense-completion page arithmetic passes its frozen numerical gates; every paged
   generation's committed IDs and stop reason equal its dense target-only reference;
2. across all twelve scored generation episodes (four prompts by three repetitions),
   the prospectively nominated 512 MiB related-token-prefetch draft accepts at least
   50% of proposed draft tokens;
3. that 512 MiB prefetch policy has at least 10% lower fully charged draft-page H2D
   bytes per proposed token than its equal-budget, selection-matched 512 MiB eager
   selected-draft control. The dense target-only reference and all three paged controls'
   verifier-charged end-to-end committed tokens/s, total wall time,
   target-verification cost, host/device residency, and acceptance records must be
   published regardless of whether this predicate passes;
4. a concrete reason that the remaining question depends on llama.cpp—for example,
   an interaction with its quantized tensor representation, GGML scheduling,
   CUDA kernels, KV implementation, or stock speculative loop that PyTorch cannot
   faithfully test; and
5. a new, pushed native protocol with a pinned source revision, separate local
   worktree/build, fixed target artifacts, stock-baseline comparison and the full
   memory/resource accounting required for the native runtime.

An uncompetitive PyTorch wall time alone is not evidence against an otherwise
eligible acquisition policy: it may be an apparatus limitation. Nor is it evidence
to patch llama.cpp. Conversely, a result that fails the explicit eligibility
predicate remains a result; it does not get rerun as a native patch in search of a
different answer.

## Consequences

- Keep the existing pinned stock b10919 binary and its measured profiles unchanged
  as practical baselines.
- Keep any eventual native source/builds in isolated local worktrees with their own
  receipts. They must never substitute for or overwrite the stock baseline.
- Do not claim a PyTorch pager is competitive with llama.cpp, or that a native patch
  is warranted, before the stated evidence boundary is satisfied.
