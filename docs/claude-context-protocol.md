# Claude small-context recovery protocol

This is a client regression check under ADR 0003, not a new inference benchmark.
Freeze this protocol and harness before exercising the installed CLI. Use a fresh
run directory on every correction. The running user-owned llama.cpp server and
private project are not test targets.

## Observed failure and source diagnosis

Read-only metadata inspection of the reported Claude Code 2.1.260 session found
six automatic compactions, at estimated pre-compaction sizes of 6746, 3517,
4415, 5890, 4522, 4954 tokens. No Read calls preceded those compactions;
the observed tool results were small edit responses. Do not publish the private
transcript or project content.

The installed binary's embedded JavaScript subtracts the configured output reserve
(2048 here) from the true context window (18432), then subtracts a fixed 13000
tokens. This gives a proactive threshold of 3384. Its percentage override takes
the minimum with this value, so it cannot raise the threshold. These are version-
specific implementation observations, not a promise about future Claude builds.

## Intervention

Provide explicit manual/auto compaction modes; default this small-window launcher
to manual. Set child-only `DISABLE_AUTO_COMPACT=1`, keep manual `/compact`, and
preserve the true context and native context-blocking guard. Never pretend that
the server has a larger window. Set the file Read output budget to 2048 tokens,
serialize read-only tool execution, and request targeted ranges. Preserve current
permissions, bare/restricted mode, model artifacts, placement, KV and decoding.

Manual operation requires the user to check `/context`, compact around 8000
tokens, and save a concise handoff/start fresh if compaction does not recover
space. This is a workaround for an incompatible automatic reserve, not a repaired
automatic controller or a guarantee that arbitrary tasks fit.

## Bounded validation

1. Unit checks cover child-only environment isolation, both modes, retained true
   context/output budgets and unchanged server argv. Existing relevant tests run.
2. A loopback scripted Messages endpoint drives the actual installed Claude CLI
   with synthetic responses and synthetic reported token usage. No GPU inference,
   cloud endpoint, credentials, real project content or task-quality scoring is used.
3. At reported 6000 input tokens, issue five Read tool calls on a tiny fixture.
   Compare auto and manual modes. Auto must emit a compaction boundary or the
   thrashing diagnostic; manual must complete all five reads with zero automatic
   boundaries. Record full synthetic requests, SSE replies, stdout and stderr.
4. Check an oversized Read is rejected under the configured 2048-token budget and
   a targeted Read succeeds. Check `/compact` is still available in manual mode.
   Report any inability to exercise an interactive command from print mode as a
   limitation rather than passing it from source inspection alone.
5. Each subprocess has a 120-second deadline and an owned process lifetime. Store
   the installed version/executable digest, source commit, exact command and child
   overrides, case outcomes and exceptions. A failed or missing case fails the
   corresponding claim. Do not substitute model-generated evidence for scripted
   responses, or call this a live-model compaction/task-completion test.

Publish the workaround and bounded receipts as v0.19.1 after review and validation.
Keep #27 and the broader milestone open; no optimization experiment follows.

References: [environment controls](https://code.claude.com/docs/en/env-vars),
[CLI reference](https://code.claude.com/docs/en/cli-reference).
