# Local interactive integration protocol

Status: prospective. No new optimization or performance experiment. ADR 0003 and
the v0.18 decision to pause tuning remain in force. The owner now requests direct
interactive use with Codex and Claude Code; upstream work and a blog are later work.

Keep b10919 d3146f2b56c2db4711ac8391871c9e529d1946d7, the pinned target/draft,
18432 capacity, q8_0 KV, flash attention, batch256 and the recorded thread settings.
`local_session.py` reuses the measured argv constructors. All four profiles use
the unchanged stock binary: measured (ngl65,32 host FFNs,threshold2,K16), whole-layer
(ngl38,threshold2,K16), stock-speculative (ngl38,threshold32,K4), and placement-matched
target-only (ngl38,threshold32,no draft). Target-only is not a new optimized baseline.
Interactive additions are a stable model alias, explicit Jinja and greedy defaults;
clients may override sampling. No 64-token server cap is imposed on interactive use.

Before inference, review and push this protocol, launchers and smoke harness. Run
from that clean source in a fresh directory. Verify all native/GGUF hashes. Capture
argv, effective process environment, native logs, API bodies/status/stream events,
client versions/output/exit and generated files. Never archive cloud credentials.
The server binds only loopback. Child-only settings leave user credentials and
configuration intact. Codex gets a dedicated local configuration home without copied
auth; Claude uses bare/restricted mode and an explicit local placeholder API key.

First start the measured profile. Test Chat Completions, Responses and Messages in
both streaming text and a two-request function-tool round trip. The tool is
`read_probe`, with required argument `key="alpha"`; return `PROBE_4821` as its
result. Success requires a parsed call with that argument and a final answer
containing the returned value. Tool choice is forced for the first request and
disabled or unavailable for the result request. A clean successful terminal stream
event is required; errored or truncated streams do not pass through matching text.
These test transport/schema, not autonomous skill.

Then test installed Codex 0.154.0 and Claude Code 2.1.260 serially in separate fresh
toy directories. Each sees `calculator.py` containing `def add(a, b): return a - b`.
Ask it to inspect and change only this file to addition and summarize. Allow ten
minutes per CLI, retain failures and timeouts, and check the resulting AST is exactly
the original single function with `a + b`. Tool events and local-file change are
required, with no additional workspace files; an ungrounded final claim is not a pass.
Codex's smoke invocation disables inherited project-document context for this toy
task; interactive invocation retains the chosen project's instructions. No general agent-capability claim
follows. Claude's declared tool subset is Read/Edit/Write and its short system prompt
is a local integration choice; no claim about full default plugins/MCP/bash.

Finally start the three comparison profiles serially, checking health, effective
placement and one short Chat Completions request. No rate comparison is derived from
these checks. Startup is screened against the earlier15000MiB GPU and2GiB available
host bounds; record resources through completion. API errors do not trigger model,
quantization or native-code changes. Any necessary corrected harness/configuration
gets a reviewed, pushed amendment and fresh run root; preserve the first attempt.

Routes alone do not establish compatibility. The pinned Responses adapter rejects
previous_response_id and skips non-function tools; explicitly report such limitations.
No proxy or silent tool removal is introduced. Historic nine short replay misses,
capped-task limitations and unqualified full target/draft disk restore remain visible.
