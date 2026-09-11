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

After v0.7, analytical refinement may start from failed one-shot selections. Gate
physical paging and refinement runtime on the complete causal acquisition policy,
not necessarily its first pass. Privileged and frozen-mask repair diagnostics are
explicitly ineligible for deployment. Follow docs/adr/0001-selection-and-refinement-gates.md;
preserve historical protocols and keep hardware-cost and limited generalization
controls separate from runtime claims.
