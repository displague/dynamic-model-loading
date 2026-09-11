# ADR 0001: Gate the complete acquisition policy, not its first pass

Status: accepted on 2026-09-11, following the project design review supplied by the
project owner after v0.7.0. This is a prospective dependency change. It does not
amend any published experiment, failure, threshold or release.

The original linear roadmap required a causal one-shot selector to qualify before
any refinement experiment. That ordering prevents testing whether partial execution
can supply evidence for economical corrective additions. The first-pass and complete
policy hypotheses need different gates.

Analytical refinement feasibility may now start from failed selections. Preserve the
FFN input and add missing contributions before downstream commitment. Privileged
repair diagnostics measure opportunity only; hindsight assistance is neither an
optimal oracle nor a causal deployable policy. Fixed masks replayed on a corrected
trajectory are explicitly counterfactual.

A complete causal selection-and-refinement policy must demonstrate its own quality,
acquired-byte and charged-cost frontier before physical paging or refinement runtime.
Its initial subset need not pass alone. Larger one-shot subsets, resident cores and
dense fallback are required comparisons. Residency misses and silent omissions stay
separate; local norms do not define task failure.

Keep v0.7's <=10% resident dense-FFN timing screen as its historical decision rule.
It is not a universal necessary condition for paged inference. A separate bounded
hardware characterization measures staging, grouped execution and transfers without
building a pager. A future runtime gate concerns measured exposed critical-path cost,
including acquisition, scheduling, refinement and transfers, against the strongest
same-budget baseline. Never infer end-to-end speedup by adding unoverlapped timings.

Separate supporting tracks may proceed: one small second-model/new-development
control, and a bounded BF16 study against a common higher-precision reference with
prospective downstream criteria. Neither overwrites the FP32 mathematical controls
or BF16 failures. Full model sweeps, representation changes and services remain gated
by specific evidence described in the research plan.

The current implementation tests opportunity first, then causal availability of a
repair signal under a separate protocol. A negative privileged heuristic is evidence
against that repair schedule, not an impossibility result for sequential computation.
