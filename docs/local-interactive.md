# Run the measured configuration locally

Claude Code 2.1.260 requires **manual compaction** with this small context window.
The launcher now defaults to that workaround; see [context recovery](claude-context-protocol.md).
Restart the Claude client to load the new settings; the existing model server can
stay running. The old conversation is not automatically resumed or erased.

```powershell
.\local.ps1 claude -Workspace C:\Users\displ\Documents\test-project
```

Use `/context` to check usage and `/compact` around 8000 tokens, before a large
tool exchange. If the summary fails or remains too large, save a concise handoff
and use `/clear` for a fresh conversation. Avoid repeated whole-file reads.
This requires active context management; it does not increase server capacity or
qualify unlimited coding sessions. `-ClaudeCompaction auto` explicitly restores
automatic compaction for comparison, including its known small-window problem.

**Start with Claude Code for file editing.** Its Read/Edit tool loop passed the
[local integration check](local-session-results.md) on the measured profile.
Codex connects through Responses, but its file-read command was rejected by CLI
policy in both noninteractive trials; that coding path remains unqualified.
The [protocol](local-session-protocol.md) and [setup correction](local-session-amendment-1.md)
precede their respective runs.
The native runtime is the already-installed stock Windows CUDA13.3 b10919 build:
`runs/llama-b10919/llama-server.exe`. No patched build or new model download is needed.
Exact artifact hashes are in [the catalog](../configs/stock-speculation-artifacts.json).

From the main checkout, use two PowerShell terminals:

```powershell
# Terminal 1: keep this server alive across turns. Ctrl+C stops it.
.\local.ps1 serve

# Terminal 2: verified limited Claude file-editing configuration.
.\local.ps1 claude -Workspace C:\path\to\project

# Optional Codex configuration; coding-tool qualification did not pass.
.\local.ps1 codex -Workspace C:\path\to\project
```

The browser chat UI is also at <http://127.0.0.1:8080>. Run one GPU server at a time.
New sessions write argv, process environment, logs and resource samples below
`runs/local-sessions/`. Model and binary hashes are verified before each launch.
Use `-PrintCommand` to print the complete native argv and environment without loading.

```powershell
.\local.ps1 serve -Profile measured -PrintCommand
.\local.ps1 serve -Profile whole-layer
.\local.ps1 serve -Profile stock-speculative
.\local.ps1 serve -Profile target-only
```

| Profile | Target placement | Process threshold | Draft |
|---|---|---:|---|
| measured | `-ngl 65 --n-cpu-ffn 32` |2|0.5B, K16|
| whole-layer | `-ngl 38` |2|0.5B, K16|
| stock-speculative | `-ngl 38` |32|0.5B, K4|
| target-only | `-ngl 38` |32|none|

All profiles use stock llama.cpp. The third is the best tested default-threshold
long-context draft configuration; the fourth is placement-matched, not globally
optimized target-only inference. Switching profiles requires stopping the first
server and launching a fresh process. The threshold is never set machine-wide.

Common settings preserve18432 context capacity, q8_0 K/V for both models, flash
attention, batch/microbatch256, one slot,24 target threads and8 draft threads.
The measured draft has p_min0 and backend sampling disabled. Interactive additions
are alias `dml-qwen32b`, explicit Jinja, temperature0 and seed0 defaults. A client
may override sampling. No64-token server response cap is imposed; fit prompts,
tool schemas, new input and requested output inside the capacity. These interactive
requests are not the earlier frozen performance workload.

The launcher uses a dedicated local Codex configuration home under
`runs/local-client-state/codex`, without copying cloud authentication. Codex uses
the custom Responses provider at `http://127.0.0.1:8080/v1`. Claude uses Messages
with `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, a local placeholder key, thinking
disabled, a2048 output-budget environment setting, bare/restricted mode and requested
Read/Edit/Write tools. The installed Claude2.1.260 advertises **Read and Edit** in
the actual init event, and those are the two tools exercised. Its metadata still
reports the unknown-model32000 default output limit; this short check does not
establish enforcement of the2048 environment setting. Keep requested turns small
enough for the server's18432 capacity.
These are deliberately small local client configurations, not full default
plugins, MCP integrations or arbitrary command execution. Interactive permission
prompts remain enabled. The smoke used print mode with scoped edit permission;
interactive UI behavior and arbitrary coding tasks are not a separate qualification.
No user-wide client configuration is edited. Claude's displayed dollar estimate is
client bookkeeping, not measured local inference cost or a cloud API charge.

The pinned Responses conversion is a compatibility wrapper: it rejects
`previous_response_id`, and skips non-function tool definitions. A route responding
does not qualify every coding harness tool. All three API routes passed the declared
streaming/function-result checks in both runs. The Codex failure is a client-policy
observation, not proof that one of these API limitations caused it. Its unknown-model
metadata and inherited host-skill warnings are also retained in the receipts.

The measured16K configuration reaches20.57 native decode steps/s on v0.17's
sustained fixture; v0.18's five retained capped turns take23.985s with0.623s mean
TTFT. These are different workloads. Four of five retained answers hit64 tokens,
histories differ after extraction, and nine older short replay discrepancies remain
open. In-process prefix retention is measured; full target-plus-draft restore after
restart is unqualified. See [the configuration record](stock-long-context-configuration.md).

Sources: [pinned server APIs](https://github.com/ggml-org/llama.cpp/blob/d3146f2b56c2db4711ac8391871c9e529d1946d7/tools/server/README.md),
[pinned Responses conversion](https://github.com/ggml-org/llama.cpp/blob/d3146f2b56c2db4711ac8391871c9e529d1946d7/tools/server/server-chat.cpp),
[Codex custom providers](https://learn.chatgpt.com/docs/config-file/config-advanced),
[Claude gateway configuration](https://code.claude.com/docs/en/llm-gateway),
[Claude environment variables](https://code.claude.com/docs/en/env-vars).
