# Qualified utility and causal evidence

Status: accepted, 2026-09-11. Implements the review following v0.10.

Preserve v0.10 as the plain-text exact-output control. Qualify Qwen's documented
chat interface independently of analytical repair. OPT remains an architectural
sparsity control. Utility requires successful dense behavior; dense agreement and
task correctness are separate endpoints.

Gate A compares native library generation, incremental dense execution, and fully
retained instrumented/repacked execution with identical inputs and decoding.
Compare logits on identical prefixes, including when free-running tokens diverge.
Old tasks are debugging fixtures. Freeze the interface and scoring before fresh
balanced development evaluation; never replace failed rows or retune old scores.

Gate B requires initial and corrective decisions on the controller's own corrected
trajectory. Allow one corrective acquisition round per FFN, followed by commit or
bounded dense completion. Distinguish detection from acquisition. Calibration may
use dense labels at approximate/corrected states; evaluation queries and persistent
state must see only permitted observations. Compare partial-evidence repair with
equally resourced larger one-shot selection, resident-first predetermined additions,
and dense completion. A stronger predictor alone does not demonstrate repair value.

Gate C charges the complete batch action: prediction, observation, exploration,
resident computation, acquisitions, gathering, staging, packing, dispatch and
fallback. Report rounds and fallback frequency alongside bytes and overhead.
Prepared sparse payloads cannot be free when the comparator pays preparation.
Synthetic serialized hardware costs are sensitivity bounds, not exposed stalls.

Gate A blocks utility claims only. Gates B and C govern a bounded physical-runtime
nomination; physical paging remains deferred until their evidence qualifies.
The three gates do not close broader held-out or generalization milestones.
