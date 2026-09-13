# Compaction request classification correction

Keep both earlier attempts. At `3983a3e`, manual mode passes five distinct Reads,
the Read cap produces explicit truncation followed by a successful targeted read,
and `/compact` emits a manual boundary. The automatic case still fails its verdict.

Inspection of retained requests shows why: the pinned client's compactor retains
tool schemas for prompt-cache sharing. It appends a text-only summary instruction
instead of removing the tools. The scripted API incorrectly treated those requests
as normal reads and returned tool calls. This also corrects amendment 1's attribution
to fresh configuration: the first attempt did issue compaction requests, but the
harness did not recognize them. Naming an explicit auto window remains a disclosed
test setting, not evidence that it was necessary to reach the compactor.

Recognize the pinned summary instruction in the final user text block, excluding
tool-result text, and return a text summary. Record actual summary requests and
compaction boundaries. The auto case is an expected-failure control: native exit
0 or 1 is acceptable only with an observed boundary or thrashing diagnostic, no
timeout/setup exception. Manual/read/command cases still require normal completion.

Freeze this correction before the third run. No model calls, production placement
changes or private transcript replay are introduced. Earlier failed aggregate
verdicts remain unchanged. The result is client control-flow validation only.
