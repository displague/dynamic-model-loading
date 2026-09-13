# Recognize compaction instructions before trailing system telemetry

The first checks at source `266b616c739e4d79d8a12dd36f21e87b7a97498b`
passed both native file-tool cases. The context rerun passed three of four
declared criteria; its automatic case failed. Preserve
`claude-tools-20260913-v1` and `claude-tools-context-20260913-v1` unchanged.

Safe mode sends a system-role total-token-budget reminder after the compaction
instruction. The old bare-mode classifier only accepted a final user message,
so it answered genuine summary requests with scripted Read calls instead of a
summary. The first manual boundary therefore also does not qualify an intended
summary response, despite passing that existing boundary criterion.

Change the synthetic classifier to skip trailing system-role messages and require
the same explicit summary instruction in the final remaining user text block.
Do not search older user turns or tool-result text. Test that an intervening
assistant turn prevents classification. This is a harness correction, not a
change to the launcher, native client, compaction budgets or pass criteria.

Review and push this amendment before rerunning all four context cases in a fresh
`claude-tools-context-20260913-v2` directory. The passing file-tool cases need no
repeat because neither their launcher nor their fixture changes. Retain the
failed context attempt; do not describe its false classifier as a client failure.
