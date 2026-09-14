# Fault pager: prospective physical-runtime protocol

Status: prospective, written 2026-09-14 before implementation or inference.
This experiment is governed by [ADR 0004](adr/0004-fault-pager-research-track.md).

## Question

Can an explicit, bounded GPU cache of physical SwiGLU FFN pages make a causally
selected approximate draft useful after a dense target verifies every committed
token? This is a physical-loader feasibility experiment. It does not claim a 32B
outcome, a sparse-compute speedup, or a qualified agent-utility result.

## Fixed substrate

- Checkpoint: `Qwen/Qwen2.5-1.5B-Instruct` revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP32.
- Interpreter/environment: the root `.venv`; record actual Torch, CUDA, driver,
  device, and source hashes. CUDA is required; no CPU fallback is a scored run.
- Input corpus and token IDs: the 16 rows whose `split` is `diagnostic`, in their
  recorded order, from `runs/qwen15b-packing-pilot-20260911/corpus.jsonl`
  (full-file SHA-256
  `5a100c930dae2532232c83d50af75f67b0d8c1e4c5365fdf4c14f450c1eb0b31`).
  The run copies those rows and the rendered token IDs before inference. No
  calibration, prompt, or token substitution after a result is seen.
- Sequence mode: a teacher-forced 128-token page-mechanics check, then greedy
  speculative generation of 64 tokens from the first four diagnostic documents.
  Batch one and no KV cache in the mechanics check make every layer visit and page
  access attributable. The draft proposes exactly four tokens per round; the dense
  target evaluates that proposed prefix, commits its matching prefix plus its next
  greedy token on a rejection, and starts the next round from the committed prefix.
  Generation records and charges its own target and draft KV.
- Page: contiguous, paired gate/up rows and the matching down columns for 256 FFN
  neurons. A page includes all three tensors and records its exact byte size. The
  final short page retains its true width.
- Cache budgets: 128 MiB and 512 MiB aggregate page-cache payload. Page-table,
  metadata, source tensors, staging, model-resident non-FFN weights, workspaces,
  and allocator overhead are reported separately and never counted as page payload.
- Feasibility budget: every scored configuration, including the desktop baseline,
  dense target, draft non-FFN weights, target/draft KV, page payload, page table,
  staging, workspaces and allocator overhead, must remain at or below 15,000 MiB
  sampled device-used memory and leave at least 2 GiB available physical host memory.
  The source tensor store, process working set, pinned/pageable staging and all
  duplicate model copies are charged and reported even when they are outside the
  page-payload subtotal. An allocation counts once only when the implementation
  proves shared storage; otherwise target and draft copies are charged separately.
  A resource breach is retained as a failed configuration and cannot support a
  physical-improvement claim.

## Implementations and controls

All paths use the same page format, source tensor dtype, input IDs, model
non-FFN weights and eager dense reference.

1. **Dense target reference:** normal dense FP32 CUDA model, greedy decoding.
2. **Dense-completion page check:** transfer and accumulate every page in increasing
   page order. It validates page arithmetic, page-table accounting, and demand-miss
   handling against `grouped_forward`; it is not a throughput candidate.
3. **Eager selected draft:** a physically partitioned draft transfers its selected
   pages in increasing order and immediately releases them. There is no GPU cache.
4. **LRU selected draft:** an explicit `(layer, page)` residency map serves selected
   page hits; a missing selected page synchronously stages host-to-device and evicts
   least-recent pages until it fits.
5. **Related-token prefetch draft:** the same LRU cache, but at the end of a layer
   visit it queues that layer's two highest-ranked pages from the prior token's
   side-index lookup. Tie order is increasing page index. The queue is processed
   before the draft's next token; an item already resident is a counted hit and an
   item evicted before use is a counted cancelled prefetch. Prefetch work is charged
   even if later unused.

Draft page selection uses a frozen, prompt-conditioned side index, not current
gate/up scores. Before the scored runs, execute the dense target teacher-forced on
the 80 `calibration` rows of the corpus above for their first 128 predicted tokens.
For each layer, collect the FP32 FFN input vector and dense page-importance vector
`sum_page(abs(silu(gate)*up) * norm(down_column))` for every position. Construct 16
medoids per layer by choosing the earliest vector first, then repeatedly choosing the
vector with greatest minimum squared-L2 distance to existing medoids (earliest
position breaks ties). Each medoid stores its own 35-page importance ranking,
descending with increasing page index for ties. Persist the index, all source row and
position identifiers, and its SHA-256 before any scored draft request.

At each draft FFN call, its input vector is already available from the preceding
attention/norm computation. Choose the nearest stored medoid by squared L2 (lowest
medoid index breaks ties) and execute the highest 27 of 35 pages (the ceiling of 75%
of the 8,960/256 pages) in that stored ranking. The side index therefore requires no
current gate/up projection, no read of an unselected page, and no target output at
selection time. Its FP32 centroids, rankings and lookup time are charged as
controller memory/work. The target model does not provide activations or logits to
the draft, and target outputs never choose pages. After verification, however, every
committed target token (including a fallback token) becomes ordinary causal context
for both models before the next proposal round. Every selected-page miss is recorded;
unselected draft pages are an intentional approximation, not a fault.

## Correctness contract

Page accumulation uses FP32 and calls the original SwiGLU activation. The
dense-completion page check must reconstruct each FFN relative to `grouped_forward`
within relative L2 `<= 0.01` for two recorded inputs per layer. It must also satisfy
the existing full-model relative-logit L2 `<= 0.01` and mean KL(dense || pager)
`<= 0.001` limits for every diagnostic document.

The selected draft is intentionally approximate and is not judged by those dense
logit gates. Instead, its committed output must equal the dense target-only greedy
reference IDs and stop reason for every generation prompt. The implementation must
record every proposal, dense target verification result, accepted prefix, rejected
draft token, and fallback target token. A draft page miss is not an error: it is a
required, counted acquisition event. Missing page data, oversized cache, failed
transfer, or a committed ID not verified by the target is an implementation failure.

## Measurements

Run the dense target reference first and retain its IDs before any draft measurement.
For each of the six `(eager, LRU, prefetch) x (128 MiB, 512 MiB)` configurations,
run one unscored warm-up then three scored repetitions. For repetition `r` in
`0, 1, 2`, construct that six-item list in the order just stated, shuffle it with
Python `random.Random(20260914 + r).shuffle`, and run the resulting list serially.
Record the realized order. Synchronize CUDA before each wall-clock boundary. Record
raw JSONL before summaries with, at minimum:

- page identity, logical request (`prefetch` or `demand`), outcome (hit, load,
  eviction, cancelled prefetch), bytes, CUDA/wall transfer interval and cache state;
- per-layer and per-document timing, H2D bytes, demand/prefetch hit rates,
  evictions, selected/completed pages, and token count;
- GPU allocated/reserved/peak memory, host process memory, page-table/index/staging
  byte accounting, device-used/available-host samples, duplicate/shared-storage
  evidence, and source/checkpoint/config/corpus/token/source hashes;
- dense-completion logits metrics; draft and committed IDs; proposal/acceptance
  records; EOS/length; and all errors.

Compute H2D bytes from the page ledger, not from an inferred selected fraction.
Report median per-token wall time, total H2D bytes/token, and demand misses/token.
Keep prefetch bytes distinct from demand bytes. CUDA event timings are component
observations only; synchronized wall time is the end-to-end measure.

## Decision rule

This release establishes only what the receipts support.

- A pager is **correctly implemented** only if dense completion passes the stated
  numerical checks, cache accounting stays in budget, and every committed draft
  token has target-verification provenance.
- A prefetch draft is a **measured physical improvement** only if, at the same
  budget and equal prompts, its committed IDs match the target reference and it
  improves both aggregate H2D bytes/committed token and median synchronized
  end-to-end wall time relative to the selection-matched LRU draft. Target
  verification, rejected draft work, target/draft KV, cache, source and staging
  costs are included. No post-score threshold or candidate swap is allowed.
- The experiment is a negative result if acceptance is too low to pay for the
  verifier, prefetch has no benefit, or either paged draft loses to eager selected
  streaming. Such a result constrains this page format, model, cache budgets and
  policy; it does not settle all dynamic-loading designs or 32B behavior.

No target-scale build, learned router, omission policy, custom CUDA kernel, or
llama.cpp patch follows automatically. A subsequent protocol must choose based on
the full raw results and charge the target's actual placement and memory.
