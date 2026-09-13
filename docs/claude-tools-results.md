# File-tool access and rejected edits

The reported session had32 Edit calls and one Read. Native results report seven
successful Edits and one successful Read, with25 rejected Edits:13 ambiguous
matches, seven missing matches, four attempts to create an existing file and one
invalid JSON input. This is an aggregate diagnosis, not task-completion scoring.
Private source, tool arguments and conversation content are excluded from releases.

There were two different problems. `--bare` exposed Read/Edit even when the
launcher requested Write; this was present before v0.19.1. Edit itself could
create new files and replace exact existing text. The repeated failures were
argument-validation errors, not missing write permission. Empty `old_string`
requests creation; it cannot append to or overwrite an existing nonempty file.

v0.19.2 replaces bare mode with safe mode, retaining restricted workspace access,
the explicit three-tool list, strict MCP, child-only local settings, manual
compaction, truthful18432 context and2048 output/read budgets. Safe mode disables
customization discovery without removing Write. Interactive permission prompts
remain enabled. The minimal prompt explains creation, exact-match edits, reading
after a rejection and stopping after two failed corrections. These instructions
do not enforce a retry cap or establish that Qwen will reliably follow them.

## Frozen installed-client checks

Claude Code2.1.260, SHA256
`ab5fe6cb800a9faf3df1cc0f6bcb71f097296b302a16b576ca906446a22fea71`,
ran against a scripted loopback Messages endpoint. Generated workspaces and fresh
test-only configuration homes isolate the user's project and preferences. The
fixture uses scoped print-mode acceptEdits; interactive permission UI is not tested.
No model inference, GPU timing or server restart occurred.

| Case | Result |
|---|---|
| Bare control at `266b616` | Exactly Read/Edit advertised. Read, Edit creation and unique replacement all pass; exact file contents verified. |
| Safe candidate at `266b616` | Exactly Read/Edit/Write advertised. All8 prescribed success/rejection results pass, including creation and overwrite, intermediate Read/creation contents, and unchanged bytes after each rejected Edit. |
| First context rerun at `266b616` |3/4 criteria pass; automatic control fails because the synthetic classifier misses genuine summary requests followed by system-role telemetry. Its manual boundary alone does not establish an intended summary response. |
| Corrected context rerun at `0808769` |4/4 unchanged criteria pass after a prospective classifier amendment. |

The corrected automatic control produces three compact boundaries at6041 reported
tokens and exits1 with the native thrashing warning. This passes the declared
negative-control criterion; it is not successful automatic task completion.
The manual loop completes five distinct successful Reads without auto boundaries;
the Read budget trims168 of1501 lines before a successful two-line Read; `/compact`
receives a scripted summary and records one manual boundary. Synthetic token
counts and summaries do not qualify live-model summarization or editing behavior.

The first failed context attempt remains intact. The amendment skips only trailing
system-role messages before applying the existing final-user summary classifier;
it does not search older turns or tool results. Launcher and verdict thresholds
are unchanged between the two context attempts.

## Receipts and disposition

The raw archive contains276 members (541850 bytes), SHA256
`2147574e92ad81a6acf7b31863bc64bdfe63a92ff376e74956328ec182ee8805`.
Every restored member matches its hash. Aggregated result ledgers reconstruct
byte-identically and both file-tool verdicts recompute from restored requests.
Source snapshots cover both frozen commits. Fresh CLI state directories and
private project/session contents are excluded.

Independent review and full-suite validation in both retained environments govern
publication; the release carries their final-candidate receipts. Complete#34 as
bounded client compatibility work. Keep#27 and the broader milestone open and the
optimization queue paused. No native runtime, model, precision, placement, Codex
configuration, upstream PR or blog change is implied.

- [Protocol](claude-tools-protocol.md)
- [Prospective correction](claude-tools-amendment-1.md)
- [Compact receipts](../results/claude-tools-20260913)
- [Launch and recover](local-interactive.md)
- [Official CLI reference](https://code.claude.com/docs/en/cli-reference)
