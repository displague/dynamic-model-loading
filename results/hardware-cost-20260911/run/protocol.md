# Bounded cost characterization for width-8 FFN acquisition

Prospective hardware microbenchmark, issue #18. Commit and push apparatus and protocol
before measurement. This does not change v0.7's frozen resident-FFN affordability rule
or authorize a model pager. Its purpose is to expose the cost of the primitives that
a future complete selection-and-refinement policy would use.

Use the existing Python 3.14.3 / torch 2.10.0+cu130 CUDA environment, FP32, four CPU
threads, seed 1729, TF32 disabled. Serialize with all other project GPU experiments.
Qwen-shaped groups have three width-8 projections at hidden size 1,536: 147,456 bytes
per group. Test exactly 1, 8, 32, 128, 512 and 1,120 groups. These are synthetic seeded
weights and one-token inputs, not a pretrained-model benchmark.

Keep a full 1,120-group pageable CPU pool, a pinned copy and a fixed seeded permutation
of group IDs. Preallocate pinned staging, GPU group-major storage and canonical
gate/up/down matrices for each size. Allocation, pool initialization, index selection
planning and the initial pageable-to-pinned pool copy are startup costs outside the
steady-state timing. Initialize and reuse a CUDA event pair before timing. Report
allocated tensor bytes and GPU allocator peaks through the end of all repetitions
separately. Generate the pool, permutation and input in that order with a local CPU
generator seeded at 1729. Preserve the exact pool SHA-256, permutation and input;
independent analysis must reproduce this recipe. Ranking uses the separately seeded
global CPU generator, also 1729.

Measure six workloads, in this fixed order, at every size:

1. Preaggregated pinned-to-GPU copy into existing group-major storage.
2. Preaggregated pageable-to-pinned staging followed by the same H2D copy.
3. Scattered pageable group gathering into pinned staging, followed by H2D.
4. Individual pinned group copies to the corresponding GPU slots, including dispatch.
5. Resident canonical grouped FFN computation: gate, SiLU, up product, down projection.
6. Scattered gathering, staging, H2D, group-major-to-canonical GPU packing and FFN output.

There are three warm repetitions and ten measured repetitions per workload/size.
Synchronize before and after each repetition. Preserve wall elapsed time and CUDA
event elapsed time around the entire submitted interval. The CUDA interval can include
host submission gaps; it is not exclusively kernel or PCIe execution time. Do not sum
isolated workloads to assert overlap or exposed critical-path latency. No overlap is
introduced in the combined path, so that path is a serialization baseline only.

Before timing each workload/size, verify its copied payload or numerical output against
the same selected source weights. Require exact copied values and <=0.01 relative-L2
for FFN output against a CPU FP32 reference. Any failed integrity check stops all
subsequent timings, retaining the raw ledger. Every workload's repetition grid and
median must be reproducible from the ledger. Report effective payload bytes per second
only for workloads that actually transfer; never label it physical link saturation.

Separately measure ranking on 32 token fixtures x 28 layers x 1,120 groups, retaining
1,008 groups. Compare the existing stable descending argsort with unsorted top-k plus
boolean scatter. Scores are seeded permutations of 0..1,119, ensuring uniqueness.
Require identical masks on all fixtures before timing. Use the same three warm/ten
measured repetitions and both clocks. Record a separate all-equal-score tie comparison;
top-k's tie choices are not a replacement for the original deterministic group-index
rule. This isolates ranking, not the complete learned/recency/EMA selector.

Preserve source, configuration, environment, every integrity check and raw timing,
actual tensor sizes, checksums and all medians. CPU fixtures can validate accounting
and reduction logic but cannot supply CUDA timing. Do not claim model quality, bounded
whole-model residency, tokens per second, or an inference speedup. A future runtime
must measure the actual complete critical path against its strongest same-budget
baseline; neither passing nor failing v0.7's 10% rule settles that comparison.
