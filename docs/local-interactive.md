# Run the measured configuration locally

Integration checks are pending under [the prospective protocol](local-session-protocol.md).
The native runtime is the already-installed stock Windows CUDA13.3 b10919 build:
`runs/llama-b10919/llama-server.exe`. No patched build or new model download is needed.
Exact artifact hashes are in [the catalog](../configs/stock-speculation-artifacts.json).

From the main checkout, use two PowerShell terminals:

```powershell
# Terminal 1: keep this server alive across turns. Ctrl+C stops it.
.\local.ps1 serve

# Terminal 2: choose a coding client and the directory it may work in.
.\local.ps1 claude -Workspace C:\path\to\project
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
disabled, a2048 output budget, bare/restricted mode and Read/Edit/Write tools.
These are deliberately small local client configurations, not full default
plugins, MCP integrations or arbitrary command execution. Interactive permission
prompts remain enabled. No user-wide client configuration is edited.

The pinned Responses conversion is a compatibility wrapper: it rejects
`previous_response_id`, and skips non-function tool definitions. A route responding
does not qualify every coding harness tool. Complete compatibility results will
be recorded here after the declared checks; do not infer support from the route list.

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
