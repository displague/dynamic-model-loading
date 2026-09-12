# Move the correctness boundary to target verification

Status: accepted, 2026-09-12. Implements the project owner's converged review after v0.12.

## Decision

Stop further development of the current Qwen2.5-1.5B FP32, fixed-group,
per-token selective-execution and local-repair path as the primary implementation
strategy. Retain every protocol, failure, raw result, analyzer and correctness
fixture. This is an engineering priority decision, not an impossibility result for
sparse inference, nonlinear predictors or omission detectors.

The primary next hypothesis is that a resident draft can amortize execution of an
offloaded target through stock heterogeneous speculative decoding. Optimize draft
representation, target residency and draft length together under measured GPU and
host-memory constraints. Compare against both target-only offload and existing
small-draft speculation. A stock configuration that serves the practical objective
is a successful outcome; a novel runtime is not required.

## Basis and limits of the evidence

| Evidence | Decision-relevant finding | Limit |
|---|---|---|
| v0.2 packing | Width-one hindsight omission retains much more quality than the tested physical groups. | Neither causal selection nor a fixed resident subset. |
| v0.4 traces | The protected hot set saves traffic; LRU has zero hits at the tested capacities. | Reuse distance exceeds capacity under this schedule; this does not establish absence of locality. |
| v0.7 causal selection | All four tested selectors fail quality; adaptive eager implementations also fail the declared cost screen. | Random projection plus ridge does not bound nonlinear predictor quality or fused implementation cost. |
| v0.8 privileged repair | A narrow hindsight-assisted repair opportunity exists. | Frozen starting masks, privileged ranking, two prefixes and different memory accounting prevent a matched recovery-fraction comparison with v0.12. |
| v0.9 hardware primitives | Dynamic gathering and acquisition granularity materially affect cost. | Primitive timings are not model latency. |
| v0.12 own-trajectory repair | At 92.5% retention partial evidence improves relative PPL from 1.112546 to 1.089487, but no policy qualifies both quality and economics. | Four reused prefixes; not a universal sparsity result. |
| v0.12 detector | Dense fallback occurs on 96.610% of layer visits, with relative PPL 1.011613 and negative warm saving. | The detector labels local relative FFN error above 0.05, not validated task failure. |
| v0.12 schedule | Full completion's reconstructed cost also beats the declared dense schedule. | Batching confounds omission benefit; do not carry reconstructed latency constants into the pivot. |

The historical reports and their frozen decisions remain unchanged. The v0.11
12/20 task result and failed 14/20 Gate A are historical small-model fixture
qualification results. They do not govern correctness research for a new exact
target. Target utility remains a separate product question; parameter count does
not answer it.

## Execution contract

Use Qwen2.5-32B-Instruct Q4_K_M as the first fixed target, with pinned artifacts,
stock llama.cpp and a 4,096-token context capacity. Its five official shards total
19,851,336,384 bytes, exceeding this GPU's 17,094,901,248-byte reported physical
capacity before KV and workspaces. Verify the artifact and device receipts before
measurement; file size is not an allocation estimate.

Keep a stable target-only greedy reference. Compare generated token IDs and replay
the first differing decision on an identical prefix. Separate algorithmic
correctness, state/rollback correctness and numerical agreement. Do not redefine
the reference after a mismatch. Sampling would require its appropriate rejection
algorithm and is outside the first benchmark.

Measure committed output per total cost, including discarded draft work, target
verification, coordination, displaced target residency, each model's KV and
workspaces. Shared allocations count once only if the implementation shares them.
Short scalar tasks are smoke tests; sustained code and prose workloads measure
decoding. Retain EOS and record actual sequence lengths. A separate compressed
32B draft is a non-sharing control, not a formal bound on shared self-drafting.

Use actual target-scale measurements to choose subsequent work. A selected-row CPU
kernel, 1.5B representation study, stronger predictor or shared-resident runtime is
optional and requires evidence of a bottleneck it could address. This decision
does not authorize a custom shared-resident runtime merely by accepting the pivot.

## Planning consequences

Close #11 and #16 as not pursued under the superseded design. Close remaining
old-scope #19 work as not planned, retaining its title, original description and
links to completed experiments. Track speculative confidence/length control in a
new issue; reuse instrumentation and state interfaces, not fitted coefficients or
thresholds. Defer remaining old-path work without claiming its hypotheses passed.

Create separate stock-benchmark, measured-cycle/resource, representation-survival
and adaptive-control issues. Only the stock benchmark is the immediate commitment.
These are alternatives informed by evidence, not another mandatory gate ladder.
ADR 0001 and ADR 0002 remain the contracts for reproducing their historical studies;
their physical-paging dependencies no longer govern the primary program.

## Related work

[PowerInfer](https://arxiv.org/html/2312.12456v2) already combines hot GPU neurons
with cold CPU execution and evaluates SiLU-family models. Its reported gains are
context from specific systems, not a ceiling or a prediction for this laptop.
[Dovetail](https://arxiv.org/abs/2412.18934) directly studies GPU drafting with CPU
verification and the cost of speculative depth.
[SubSpec](https://arxiv.org/abs/2509.18344) studies low-bit substitutes and shared
state/components for offloaded targets. No unoccupied-niche claim is made.
[Speculative decoding's distribution-preservation result](https://proceedings.mlr.press/v202/leviathan23a.html)
does not by itself validate this implementation's numerical and state behavior.

The [stock protocol](../stock-speculation-protocol.md) specifies the bounded next
experiment and its provenance, workload, placement and fidelity records.
