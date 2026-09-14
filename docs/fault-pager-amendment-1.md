# Fault-pager implementation details, before inference

Written before checkpoint inference for this track. This supplements the original
protocol without changing its model, policy, page count, budgets, tolerances, four
proposals, or admission thresholds. Source and this amendment are pushed before
calibration. The original document remains a historical record.

## Concrete execution and boundaries

- Use the root Python 3.14.3 / torch 2.10.0+cu130 / transformers 5.13.1 environment,
  four CPU threads, seed 20260914, FP32, SDPA, TF32 disabled. Hash checkpoint files
  against the original packing manifest. No downloads or substitutions.
- Calibration uses the first 129 plain-text token IDs without special tokens from
  each of the 80 calibration documents. Collect positions 0--127, so every one has
  a next-token label. Mechanics uses the first 128 IDs of all 16 diagnostic rows.
  Generation uses the first 32 IDs of the first four diagnostic documents, native
  EOS and a 64-new-token cap. Index medoids/rankings and selected source positions
  are persisted before diagnostic scoring. Farthest-first traversal excludes
  already selected positions even when all remaining distances are zero.
- Keep an independent dense GPU target and CPU-backed draft FFNs; draft attention,
  embeddings, output head and norms stay on CUDA. Source tensors are shared only
  with their own draft catalogue views. No target/draft weights are shared. Charge
  all actual allocations. This is an artificially constrained draft experiment;
  the resident 1.5B target is not the offloaded 32B baseline.
- Page tensors are packed gate/up/down-transpose into one contiguous FP32 payload.
  One pinned host page is staging; a cache miss performs a real H2D copy. Eager mode
  releases after each page; LRU/prefetch retain within the stated payload budget.
  Synchronous CUDA events time copies, synchronized wall times include gathering,
  lookup, control, logging and computation. This implementation tests synchronous
  prefetch; it makes no overlap claim. Record logical bytes and actual copied bytes
  separately, along with staging/controller memory. No current omitted activation
  is read. All selected arithmetic uses increasing page index in all controls.
- For dense-completion mechanics, apply each page to the whole 128-token tensor;
  this diagnostic is untimed for claims. Selected drafting processes one token at
  a time, including its 32-token prefix; every layer chooses 27 pages independently.
- At the end of each layer's token visit, enqueue its current lookup's top two
  pages for the next token. Before that next token, process all pending entries in
  increasing layer order and ranking order. A prefetched page evicted before demand
  use gets a cancellation receipt. Reset the queue on rejection; retain paid
  physical residency. Clear cache/queue/KV at every new episode. The queue is causal
  to the next token; no rejected-token controller state supplies future lookups.
- Keep separate target and draft DynamicCaches. Verify four proposals with one
  dense target batch. On rejection crop both caches to the matching prefix, then
  consume the target fallback as ordinary context. On full acceptance commit four
  tokens (no bonus); consume the fourth in the draft before its next proposal.
  Record cache lengths and every proposal/prediction/accepted prefix/fallback.
- Always pay for four proposed tokens, including work after a draft EOS and beyond
  the remaining output allowance. Commit at most the remaining allowance and stop
  on target EOS. Report algorithmically accepted and actually emitted accepted
  counts separately; ADR 0005 acceptance uses the latter divided by all proposals.
  Batched target numerical differences remain failures relative to the fixed scalar
  greedy reference, never waived or silently replaced.
- Run dense reference first (one warm-up on calibration row 0, then three repeats
  of the four prompts). For each of the six candidate configurations warm up on
  that same 32-token calibration prefix with the same 64-token cap before its first
  scored repeat. Recreate an empty episode cache after warm-up. There are 72 scored
  candidate episodes, four documents for each condition/repetition, in the original
  seeded condition order; documents keep original order. All warm-ups are retained.
- Sample total device memory and host available memory at 200 ms plus phase/episode
  boundaries. Token/round loops only check a latched error, without unequal extra
  samples. Join sampling and check a final boundary before recording completion.
  Report both CPU/GPU index tensors and Python catalogue/cache/queue metadata
  (`sys.getsizeof`, deduplicated, excluding separately charged tensor storage);
  retain metadata samples at layer boundaries. These are Python object sizes,
  not an estimate of the host allocator's complete footprint; RSS is also sampled.
  Missing telemetry, startup/resource failure or bad mechanics stops
  that run with raw evidence. Output disagreement is recorded across the unchanged
  complete matrix and blocks a qualifying gain. Each run is exclusive, snapshots
  sources and inputs, and refuses dirty/unpushed source. No measurements precede
  this amendment.

## Interpretation

The experiment can establish physical transfer/correctness behavior and compare
its three Python acquisition policies. Historical stock 32B target-only and small
draft receipts remain the practical baseline, but their different weights, precision
and execution substrate prevent a causal speed comparison against these 1.5B rows.
The frozen ADR 0005 predicate is necessary, not sufficient, for a native pivot.
