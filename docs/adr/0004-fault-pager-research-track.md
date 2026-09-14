# Restore physical dynamic-loading research as the primary track

Status: accepted, 2026-09-14. Supersedes ADR 0003's *priority decision* at the
project owner's direction; it does not alter any historical result, protocol, or
release.

## Decision

The stock llama.cpp profiles remain the fixed practical baseline. They are not the
primary research outcome. The project will again investigate novel, physical
dynamic loading for models whose FFN weights cannot remain GPU resident under the
declared budget.

The first new implementation is a page-indexed FFN **draft** runtime, outside
llama.cpp. It keeps an explicit host-backed page catalogue and bounded GPU page
cache, records each hit, eviction, transfer, and demand miss, and executes only
causally selected pages in the draft. The unmodified dense model verifies every
proposed token before commitment; it is the correctness boundary. A causal
prompt-conditioned side index may order/preload pages, but may not inspect current
target activations or use target outputs to choose draft pages. A dense-completion
page mode separately validates page arithmetic and miss handling, but is not
presented as a byte-saving policy.

The frozen Qwen2.5-1.5B-Instruct FP32 artifact is the first implementation
substrate because its tensors and numerical reference are already reproducible.
Its use is a bounded runtime feasibility test with an artificially constrained FFN
residency budget, not a claim of target-scale success. A 32B experiment requires
positive physical evidence from this implementation plus a separate target-scale
protocol; it must not be inferred from a small-model trace.

## Consequences

- Do not call stock flags, host FFN placement, or speculative draft length a novel
  loading policy. They remain comparison baselines.
- Retain v0.2--v0.12 failures. This is a distinct physical pager and exact fallback
  contract, not a revision of their selection criteria or a retroactive success.
- Freeze the page format, cache budgets, preload policy, corpus, target artifact,
  numerical tolerances, telemetry, and comparison harness before inference.
- Charge host source weights, page-table/index memory, pinned staging, GPU cache,
  all transfers, synchronization, and displaced target residency. Do not equate
  selected volume with transferred bytes.
- Compare the pager with an equally partitioned eager-streaming draft, a
  selection-matched no-cache draft, and the dense target-only reference. Publish a
  negative result if verifier-charged committed throughput or bytes do not improve.

## Rationale and limitation

ADR 0003 appropriately stopped spending the primary engineering budget on the
tested fixed-group causal omission/repair design after it failed its declared
quality and economics gates. It did not establish that a page catalogue, causal
preload signal, or demand-completion mechanism cannot work. The stock studies
answer a practical deployment question; this track answers the original
acquisition question.

Exact dense FFNs require every contribution for an exact output. Consequently, a
page cache alone may have poor reuse when its budget cannot retain a whole working
set. The novel page-selection policy therefore operates only in the expendable
draft; target verification makes the final committed sequence exact. Acceptance,
rejected draft work and the dense verifier cost decide whether that contract is
useful.
