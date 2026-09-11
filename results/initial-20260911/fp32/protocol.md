# First apparatus protocol

Written before the first checkpoint measurements on 2026-09-11. This is a development
protocol, not a preregistration of a confirmatory research result.

## Question and scope

Can paired FFN weights be physically reordered and recombined correctly? Once that
gate passes, how does a hindsight importance heuristic behave when its choices are
restricted to contiguous groups? The first run uses Qwen2.5-1.5B-Instruct, BF16, batch
one, at most 128 tokens/document. Attention stays unchanged. Everything is resident.

The supplied corpus is ten short, authored documents: four calibration and six
diagnostic. It checks the apparatus. It cannot establish generalization, task
accuracy, statistical significance, or a deployable quality tolerance. Plain text is
tokenized without a chat template. Corpus next-token NLL is not instruction-following
accuracy. No confirmatory held-out set is selected or scored in this milestone.

## Fixed correctness gates

- Every tested FFN's grouped all-neuron reconstruction: relative output L2 <= 0.01.
- Every diagnostic document under every configured full layout: relative logit L2
  <= 0.01 and mean KL(dense || repacked) <= 0.001.
- Any failed gate blocks all sparsity probes in that run. Do not relax thresholds
  after observing a failure. Preserve the failure receipt and diagnose it separately.

All layers are checked using the first two dense FFN input vectors of one diagnostic
document. Grouped reconstruction uses width 128 in the default run and FP32 summation
of BF16 contributions. This limited numerical check is supplemented by FP32 unit
tests across group widths, short tail groups, and tiny real model architectures.
The full-model correctness pass checks physical reordering, not end-to-end grouped
kernel execution. That larger validation remains necessary before a grouped runtime.

## Analytical probes

Score each already-computed intermediate neuron with abs(z_j) * norm(W_down[:, j]).
Sum those scores within a physical group; retain the highest-scoring groups per
token, rounding group count upward. Compare native ordering, a seeded random
permutation, and descending calibration-average importance. Popularity grouping is
not co-activation clustering. Use group widths 1, 32, 128 and retained group fractions
0.5, 0.75. All FFNs are masked simultaneously; later selections see perturbed hidden
states, but the input token sequence remains teacher-forced.

Both gate and up projections are computed densely to obtain z. The down projection
is also a dense masked multiplication. Therefore this is hindsight analysis and has
no sparse latency, memory-residency, or hardware-traffic claim. It is not an oracle:
the heuristic ignores contribution cancellation and downstream sensitivity.

For hidden size d, r neurons, and b bytes/weight, a complete SwiGLU group occupies
3*d*r*b bytes. Report the sum of selected group sizes over every evaluated token
position as hypothetical no-cache selected weight volume. Account short tail groups
using their actual width. This quantity is neither actual transfers nor a cache
capacity. NLL/KL exclude the last position (no observed next token), while selection
accounting includes every position actually evaluated, including the last one.

Keep per-document KL, relative perplexity, top-1 agreement, and logit error in the raw
ledger. Aggregate NLL/KL weighted by predicted tokens, and derive relative perplexity
from aggregate NLL differences. No sparse quality pass/fail is assigned in this smoke
run. Calibration data alone determines popularity ordering.

## Measurement and provenance

Use an immutable checkpoint revision and record hashes of its files, source files,
config, corpus, and token IDs. Save the exact layouts. Create a new run directory;
never overwrite one. Flush raw JSONL rows before writing a summary. An interrupted
run retains its partial ledger and an error receipt where possible.

Measure dense warm prefill with a shape-specific warmup, three synchronized wall-clock
repetitions, and no KV cache. Separately measure 16 incremental greedy decode steps
with KV; report individual times, and do not stop at EOS. These are instrumentation
baselines, not comparisons against optimized LM Studio/Ollama kernels. Loading time
includes checkpoint hashing and setup; it is not cold-disk time-to-first-token.

Record allocated and reserved CUDA memory and their peaks, the total unique model
parameter bytes, and FFN parameter bytes. Framework allocator figures are not total
process VRAM. GPU temperature/power and device-wide used memory are snapshots only;
power/thermal state and other applications are not experimentally controlled here.

## Subsequent research gates

1. Expand calibration and use separate whole-document/domain development and held-out
   corpora. Define quality criteria and confidence intervals before final evaluation.
2. Add one-layer ablations, co-activation packing, transfer amplification, and explicit
   trace simulation with bounded caches. Label heuristic bounds honestly.
3. Compare causal prediction against static popularity, recency, and EMA, including
   closed-loop generated sequences and abrupt changes.
4. Measure real RAM-to-VRAM traffic under a bounded FFN cache and charge all other
   allocations. Compare dense streaming and CPU/GPU splits at equal budgets.
5. Test local additive correction before downstream commitment against simply choosing
   a larger initial subset. Residency misses and silent selection errors stay separate.

Weight caching does not supply conversational memory. Bayesian state, MTP, SSD
paging, low-rank residuals, and an always-on service remain later branches.
