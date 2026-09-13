# Integration amendment 1: explicit workspace and diagnostic parsing

Prospective follow-up after the frozen db262c0 run, before any corrected inference.
Preserve `runs/local-session-20260913-v1` in full. All three API text/tool round trips
and all four profile startup/text checks passed. Neither installed client changed
the toy file. Codex's nested PowerShell invocation was rejected by policy; Claude
invented `/path/to/calculator.py`, received a real restricted-path denial, and asked
for help. Its string-valued diagnostic also raised an AttributeError in the collector.
No pass is inferred from either client's zero exit code or prose claim.

Keep the native runtime, models, sampling, API checks, profiles, policy boundaries,
versions and toy task semantics unchanged. Supply the exact workspace in Claude's
minimal system prompt and the exact file in both tasks. Give Codex a PowerShell
syntax hint: code directly in exec_command, no nested shell invocation, omit the
sandbox_permissions argument as the never-approval tool contract requires. Put the
already-intended `-s workspace-write` on the `exec` subcommand in the noninteractive
invocation, where its help documents the option; the interactive root keeps its
own flag. Do not disable sandboxing or automatically approve escalation.
Parse tool events only from object-valued messages, preserving diagnostic strings.

Review and push these changes, then run the same matrix from clean corrected source
in a fresh v2 directory. Report the two runs separately. This is transparent client
setup debugging, not a new quality benchmark or evidence against the target model.
