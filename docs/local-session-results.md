# Local API and coding-client integration

The unchanged stock b10919 server supports the declared streaming and function-tool
round trips through Chat Completions, Responses and Messages. Claude Code2.1.260
completes the corrected one-file task through actual Read/Edit calls. Codex0.154.0
connects but does not complete the edit: its file-read command is rejected by CLI
policy in both attempts. Use the [local launcher guide](local-interactive.md) with
this distinction. No native rebuild, API proxy or new model was required.

| Check | First frozen run | Corrected run |
|---|---|---|
| Chat Completions: streaming / call / result |3/3|3/3|
| Responses: streaming / call / result |3/3|3/3|
| Messages: streaming / call / result |3/3|3/3|
| All four profile startup/text checks |4/4|4/4|
| Codex changes only calculator.py correctly |fails|fails|
| Claude changes only calculator.py correctly |fails; collector diagnostic error|passes,2 tool calls|
| Aggregate all-checks verdict |fails|fails|

The first source is db262c0be119698ea20b374036b931037b184b9c and the corrected
source is2916eeada0970c6d517db609ab43f75f73081477. Both were reviewed and pushed
before inference; their raw directories are local-session-20260913-v1 and-v2.
The [original protocol](local-session-protocol.md) and
[prospective amendment](local-session-amendment-1.md) preserve the distinction.
The aggregate failure is intentionally retained even though the usable Claude
path and every API/profile check pass in the corrected run.

The task begins with `def add(a, b): return a - b`. Success requires exactly the
same function with `a + b`, actual tool events, clean client exit, and no additional
workspace files. The API fixture forces one function with key alpha, supplies a
known result, and checks the subsequent stream. These are transport and a single
existing-file edit checks, not general task-solving, autonomous tool selection,
interactive permission-UI validation or target-output equivalence benchmarks.

In the first run, Codex requests a nested shell invocation and returns instructions
after rejection. Claude invents `/path/to/calculator.py`; restricted mode rejects
that path. Its string-valued permission diagnostic exposes a collector bug, while
the actual file stays unchanged. The amendment supplies real workspace/file paths,
explains direct PowerShell syntax, puts the intended workspace-write flag on exec,
and guards object-valued messages. No sandbox bypass or automatic escalation is
introduced. In the corrected run Codex's ordinary Get-Content request is still
rejected; Claude reads the correct file, edits it, and finishes with no denial.

All four configurations use the same stock binary,18432 capacity, q8_0 KV,
flash attention, batch256 and pinned artifacts. The measured configuration retains
all attention/KV and has32 host FFNs, threshold2/K16. Whole-layer keeps ngl38 at
threshold2/K16; stock-speculative uses ngl38,threshold32/K4; target-only uses ngl38
with no draft. The last is placement-matched, not globally optimized offload.
Interactive additions are alias dml-qwen32b, explicit Jinja and greedy/seed0 defaults.
Clients may override sampling; no performance ratio is derived from these checks.

| Corrected profile | Peak sampled GPU MiB | Resource check |
|---|---:|---|
| measured |14447.10|passes|
| whole-layer |13743.10|passes|
| stock-speculative |13709.10|passes|
| target-only |13063.10|passes|

Every process stays within the earlier15000MiB GPU and2GiB available-host bounds.
Those are sampled observations, not a promise for arbitrary interactive prompts or
additional desktop workloads. All eight native processes receive requested shutdown.

The installed Claude init event advertises Read/Edit, with no MCP servers, skills
or plugins. The launcher requests Read/Edit/Write; no Write or command execution is
qualified by this fixture. Claude reports unknown-model32000 max-output metadata
despite the2048 environment setting; the short result does not prove enforcement.
Its dollar estimate is not a measured local cost. Codex warns about unknown model
metadata and host-skill context; those warnings and its policy rejection are not
hidden by treating a zero CLI exit as a successful edit.

The pinned Responses adapter rejects previous_response_id and skips non-function
tool definitions. The observed Codex failure does not isolate either as its cause.
No full-default Codex/Claude tool ecosystem is claimed. In-process retention remains
the v0.18 result; arbitrary harness prefix reuse and complete target/draft disk
restart are unqualified. The historical nine short replay misses remain in#27.

The raw archive preserves API bodies, complete SSE bytes, client events, invocation
settings, generated files, native logs, resource samples and source snapshots from
both runs. Machine-specific Codex state databases stay local; they are not required
by the model-free analyzer. Restored analysis and final-tip validation accompany
the release. The optimization queue stays paused. Upstream proposals and the blog
are subsequent work, not part of this delivery.
