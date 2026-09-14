# Collaboration and publication

Keep research stages, release deliveries, and satisfied gates distinct. The original
research plan and current state are in `docs/plan.md`; protocols must precede new
measurements. Preserve negative results, raw receipts, source snapshots, and fixed
tolerances. A corrected experiment uses a fresh run directory.

Complete a code review and relevant validation before committing. At each completed
delivery, commit and push to the configured remote, create an annotated release tag,
publish the GitHub release, and update the associated issues and milestones. Use
issues and milestones directly; do not create pull requests for this workflow.

Write commit messages for future collaborators: explain the research question,
implementation, measured findings, validation, limitations, and resulting next step.
Maintain `docs/releases/<version>.md` as the source for both the annotated tag body
and GitHub release notes. Use absolute GitHub links there so the same text works in
all three locations. Research releases are prereleases until a deployable runtime is
established. Do not mark an entire research milestone complete because one experiment
or release completed. Do not rewrite published history without explicit authorization.

Keep `.venv` as the measured baseline unless an environment-change protocol calls for
another interpreter. Record actual package versions and CUDA execution. Available
LM Studio/Ollama formats are deployment comparisons, not interchangeable HF weights.

The primary program is novel physical dynamic-loading research under ADR 0004,
with the native-pivot evidence boundary in ADR 0005. Stock target-scale heterogeneous
speculation under ADR 0003 remains the practical comparison baseline. Retire the tested
1.5B FP32 per-token paging/repair path as an engineering priority decision; preserve
its protocols and failures without implying impossibility. Historical Gate A/B/C
contracts remain reproduction rules, not entrance gates for the new program.

Freeze target artifacts, workloads, selection rules and harness before inference.
Charge draft memory and displaced target residency together. Compare against the
best measured target-only and stock small-draft configurations. Keep a stable
target-only greedy reference, actual accepted-prefix records and compact receipts.
Model-dependent acceptance and runtime economics must be measured on the target.
No custom shared-resident runtime follows automatically from the pivot; measured
bottlenecks choose the next experiment. Close superseded tasks as not planned,
keeping historical issue bodies and completed experimental deliveries intact.
