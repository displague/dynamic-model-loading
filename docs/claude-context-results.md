# Claude small-context recovery

The local launcher now defaults Claude Code to explicit manual compaction. This
avoids an incompatible automatic reserve without changing the server's truthful
18432-token capacity. Restart the Claude client; the running llama.cpp server can
stay up. Use `/context` and `/compact` around 8000 tokens, or save a short handoff
and `/clear` if summarization fails to recover space.

## Why the reported session thrashed

Read-only metadata from the reported session contains six automatic compactions.
Their estimated pre/post sizes were 6746/2420, 3517/3138, 4415/3075, 5890/3115,
4522/3366 and 4954/3609 tokens. There were no Read tool calls; the largest serialized
tool result was 421 characters. Private conversation and project content are not
included in the release. This does not support blaming an oversized file read.

The installed Claude Code 2.1.260 binary subtracts the configured output reserve,
then a fixed 13000-token compaction buffer:

`18432 - 2048 - 13000 = 3384`

The percentage override takes a minimum with that threshold, so increasing it
cannot recover the lost window. The manual setting preserves the model window
and native limits; it does not pretend that the server accepts a larger context.
These are version-specific source observations. The default output value shown
in unknown-model metadata (32000) differs from the actual configured request budget.

## Installed-client check

All calls went to a scripted loopback Messages endpoint, using fresh isolated
Claude configuration directories. No model inference, GPU benchmark, cloud model
request or private-project replay occurred. The client reports synthetic token
usage supplied by the fixture; its displayed costs are not inference charges.

| Attempt | Frozen source | Outcome |
|---|---|---|
| 1 | `e5142b7` | Four failed harness verdicts; preserve all receipts |
| 2 | `3983a3e` | Three manual cases pass; automatic comparison fails |
| 3 | `a2fb21c` | All four declared control-flow checks pass |

In the final run, automatic mode emits two automatic boundaries at 6019 reported
tokens. Manual mode completes five distinct content-bearing Reads, with zero
automatic boundaries. Explicit `/compact` emits a manual boundary and receives a
text summary. The oversized Read is **trimmed**, returning 168 of 1501 lines with
`truncatedByTokenCap=true`; a separate two-line Read succeeds. This is not a claim
that the whole file was read. Normal requests carry `max_tokens=2048`.

The automatic control establishes compactor activity at this small window. It does
not reproduce the full three-rapid-refill breaker or qualify its five-read task;
only four distinct content-bearing Read results survive in its recorded requests.

The first harness assumed compaction removed tool schemas and that excess Read
output would be rejected. Neither assumption held. Cache-sharing compaction retains
tool definitions and appends a text-only summary instruction. Repeated reads of an
unchanged file return cache reminders. The amendments preserve and explain those
failures; they also correct the initial attribution to fresh client configuration.
The named 100000 auto-window in the final tests is capped at the actual 18432 model
window and is a disclosed test setting, not a production context increase.

Independent review corrected false passes on denied Reads, separated full-read cap
evidence from targeted-read success, isolated test configuration and covered setup
cleanup. The full final-candidate suite is required before publication.

## What changed for use

- Child-only `DISABLE_AUTO_COMPACT=1` selects manual recovery by default.
- Remove `--disable-slash-commands`, which prevented `/compact`. Bare/restricted
  mode, explicit Read/Edit/Write selection and strict MCP remain.
- Configure a 2048-token Read budget, one concurrent read-only tool operation,
  and instructions to request small ranges and avoid unchanged rereads.
- Keep output budget 2048, thinking disabled and the real 18432 context.
- `-ClaudeCompaction auto` remains an explicit comparison option with the known
  automatic-budget limitation. No global client settings or native runtime changes.

Manual recovery was exercised through the actual CLI's streaming print interface.
This is not an interactive-terminal rendering test, a live-model summary-quality
test or a qualification of long task completion. The 8000-token operating point is
conservative guidance, not a measured universal safe boundary. Large user inputs,
multiple tool results and lengthy summaries still need context management.

The 230-member raw archive restores with matching hashes; all three result ledgers
match byte-for-byte. Fresh test configuration homes remain local. The archive
excludes private session content and the extracted proprietary source chunk.

See [protocol](claude-context-protocol.md), [first correction](claude-context-amendment-1.md),
[classifier correction](claude-context-amendment-2.md),
[compact receipts](../results/claude-context-20260913/analysis.json), and
[interactive guide](local-interactive.md). Issue #33 is the bounded workaround;
#27, the broader milestone and the paused optimization queue retain their status.
