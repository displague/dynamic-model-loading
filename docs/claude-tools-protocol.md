# Claude file-tool availability and rejected-edit recovery

Prospective client check under ADR 0003. Freeze this protocol, launcher and harness
in a reviewed, pushed commit before execution. Preserve every attempt in a fresh
directory. No model, server, precision, context, permission bypass or optimizer
changes. Do not read or modify the user's coding project during the fixture.

The reported session contains successful Edit writes alongside rejected requests:
missing/ambiguous old strings and empty old strings against existing files.
Those are argument-validation failures, not evidence of removed Edit permission.
Claude Code 2.1.260 bare mode also omits Write despite an explicit tool list. Its
Edit can create a file with an empty old string, but not overwrite an existing one.

Candidate: replace `--bare` with `--safe-mode`, retain `--restricted`, explicit
Read/Edit/Write, strict MCP, child-only local credentials, truthful 18432 context,
2048 output/read budgets, manual compaction and interactive permission prompts.
Explain exact-match editing, creation versus overwrite, targeted recovery and a
two-correction stopping instruction. Prompt instructions are not an enforced
retry limiter and cannot guarantee the local model will obey them.

Run installed Claude 2.1.260 against a scripted loopback Messages endpoint, with
fresh test-only configuration homes and print-mode acceptEdits scoped to generated
fixture directories. Each case has a 120-second deadline. Record the client hash,
source commit, complete requests/responses, actual advertised tools, tool results
and snapshots of generated files. No private session or source code is archived.

1. Bare control: expect exactly Read/Edit; Read an existing file, create a new file
   through Edit with an empty old string, and make one unique replacement.
2. Safe candidate: expect exactly Read/Edit/Write. Read the existing file, then
   deliberately reject empty, ambiguous and absent old strings without changing
   the file. Apply a unique replacement. Write a new file, Read it, then overwrite
   it through Write. Require every prescribed success/rejection by tool-use ID,
   unchanged bytes after each rejection and exact final bytes/file inventory.
3. Rerun the four existing `claude_context_check.py` cases with the new launcher.
   Require all existing criteria unchanged, including explicit manual compaction.

The scripted endpoint supplies tool arguments; these are native tool/control-flow
checks, not a claim that Qwen now reliably chooses edits or recovers from errors.
No new GPU inference is required. The earlier live-model Read/Edit success and
the reported failures retain their original scope. A future real coding task is
the place to assess residual model behavior, without another optimization sweep.

Use relevant unit validation and independent code review before commits; run the
full suite on the final clean candidate in both retained environments. Archive
and restore synthetic receipts with exact member hashes. Publish v0.19.2 as a
client compatibility prerelease; keep #27 and the broader milestone open.

Sources: [CLI flags](https://code.claude.com/docs/en/cli-reference),
[environment variables](https://code.claude.com/docs/en/env-vars).
