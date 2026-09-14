# Screen candidates before long performance matrices

Status: accepted, 2026-09-14, implementing the project owner's instruction to
verify on a minutes-scale subset before hours-scale performance measurement.

## Decision

Future acquisition-policy studies begin with a prospectively frozen short screen:
CPU correctness tests, a bounded physical CUDA subset, raw verification and transfer
receipts, and explicit advance/stop rules. A supervisor terminates the measured
worker after five minutes, including imports, model loading and checks. A timeout,
missing receipt or correctness/resource failure cannot pass. Setup, warmups and
screen analysis are distinguished from scored episode times.

The first screen qualifies this workflow using the already measured v0.20 pager,
not a newly invented policy. Its known negative result and reused diagnostics make
this a retrospective workflow check, not held-out confirmation. Its exact new
workload is frozen in the [screen protocol](../fault-screen-protocol.md).

No screen automatically launches a long suite. A pass means only that a candidate
may receive a separately frozen expanded protocol; the remaining diagnostic
documents and longer generations have not been qualified. Future materially
different representations require their own controls and frozen screening criteria.
An acquisition-policy signal is necessary; faster apparatus alone is insufficient.

## Boundaries

ADR 0004's novel-loading mission remains primary. ADR 0005 and the historical
v0.20 full-matrix contract remain unchanged: a short screen cannot satisfy the
native pivot predicate, waive a failure, or authorize a llama.cpp worktree/patch.
Published protocols/results and failed runs are never replaced by a favorable
subset. Report negative screens and stop the expensive matrix for that candidate.
