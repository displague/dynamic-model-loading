# Continuing turns on resident attention and host-backed FFNs

Prospective protocol, 2026-09-13. Finish the retained-prefix scope of issue #31
under ADR 0003. Commit and push this protocol, harness and fixtures before inference;
measure from a clean detached checkout of that pushed commit. No result yet.

## Question and fixed comparison

Does the v0.17 placement benefit survive the original v0.16 continuing-conversation
fixture? Compare two fresh configurations under the same total resource allowance:

| Layout | Target GPU layers including output | First host FFNs | Threshold | K |
|---|---:|---:|---:|---:|
| whole | 38 | 0 | 2 | 16 |
| attention | 65 | 32 | 2 | 16 |

Keep b10919 and the exact executable/DLL/model hashes in
`configs/stock-speculation-artifacts.json`: Qwen2.5-32B-Instruct Q4_K_M target,
Qwen2.5-0.5B-Instruct Q8_0 draft. Target threads24/24; draft8/24, all draft layers
on CUDA. Both contexts18432, both KV q8_0, flash attention on, batch/ubatch256,
p_min0, min draft0, greedy target with neutral penalties and EOS respected.
One slot, fit off, no context shift, no RAM prompt cache, no runtime patches.
The fixed32 allocation is taken from the completed v0.17 result, not reselected.
This is a common-allowance comparison of specified layouts, not a global allocation optimum.

Use `.venv` for orchestration; native CUDA performs inference. Hash all artifacts
before each fresh server; record actual Python/packages, arguments and environment.
Clear inherited LLAMA/GGML/CUDA overrides; set LLAMA_TRACE1, threshold2 and scheduler
debug0 explicitly. Use verbosity4 for both timed layouts. No new graph profiler:
v0.17 contains the override/graph evidence. Require actual CUDA_Host model buffers
and exact KV allocations in every new startup receipt. Target KV must be2448MiB
all on CUDA for attention, versus1415.25MiB CUDA and1032.75MiB CPU for whole;
draft KV114.77MiB CUDA. Do not mistake requested host overrides for actual buffers.

Hard sampled bounds remain total GPU <=15000MiB and host available >=2GiB.
This is 15000MiB, not15GiB. Sample about200ms from startup through completion;
require exact count/extrema, monotonic times, <=2s sample gaps and request coverage.
A failed startup, HTTP, resource or provenance check stops the matrix without
replacement or configuration retreat. Preserve the failed attempt. A correction
requires fresh source/protocol and a fresh run root.

## Original-cap bridge, not a new usability benchmark

Use unchanged `data/committed-replay.json` long-code-cache prompt (16384 IDs) and
`data/continuing-agent-turns.json`. Initial response cap128, then five cap64 turns:
code, extraction from tool-shaped input, arithmetic, copying and topic change.
EOS stays enabled. No 256-token variant, 32K extension or positional-policy change.
Keep all capped/incomplete/wrong answers. These are latency and agreement results,
not task-success scores or autonomous tool execution.

Reuse the original conversation construction explicitly through
`continuing_agent.agent(configuration=..., server_factory=...)`. Append actual
emitted IDs, close the assistant message once, and tokenize only the authored
suffix. Preserve raw suffix requests/responses. Abort without trimming if
prompt + output cap + K exceeds capacity.

Run in this exact serial order, each cell a fresh native process:

1. attention repeat1 retained, then its reset pair;
2. whole repeat1 retained, then its reset pair;
3. whole repeat2 retained, then its reset pair;
4. attention repeat2 retained, then its reset pair.

Each process has the same excluded short-code-cache32-token warmup. There are
8 processes, 48 scored requests and8 warmups. Reset always follows retained; that
ordering limitation remains explicit. No tests, reviews, archives, downloads or
other CPU/GPU-heavy tasks run concurrently with native measurement.

Retained executes the initial request with cache_prompt=false and extensions true.
Reset replays the exact six prompt arrays from its completed paired retained run,
using cache_prompt=false every time. Its responses never change later inputs.
Bind copied input records to the original request bytes. Different layouts may
develop different histories; report their actual prompt and output equality.

## Receipts and analysis

Preserve raw SSE bytes, event arrival timestamps, all request payloads, generated
IDs/content, stopping reasons, native logs/accepted prefixes, manifests, attempts
and resource samples. TTFT is the first token-bearing event, with first nonempty
text separately reported. Separate initial cold requests from five continuing turns.
Report per-turn timing and pooled emitted IDs / total HTTP time, mean five-turn
time, native generation steps/s excluding the prefill-supplied first token,
prefill/decode durations, accepted proposals/cycles, peak memory and failures.

Analysis reconstructs the original conversation and reset binding from raw records,
checks the exact matrix/source/configuration/order, startup buffers, finite native
timing and integer counters, sample coverage, raw SSE and acceptance reconciliation.
Native prefill+decode must fit within the HTTP interval with the existing100ms
receipt-clock allowance; this is not a model-quality tolerance. Requests <=1800s,
fresh-process interval <=3600s. A compact pass flag is insufficient.

Target cache_n/prompt_n are measured and sum to actual input length; all reset
cache_n must be0. Retained cache misses remain outcomes. Record newly appended
client tokens separately from re-evaluation of the last generated position.
The pinned draft-simple process receives successful target batches; target prompt_n
therefore gives source-derived draft forwarded prompt positions. It is not an
independent draft KV reuse counter or a timer, and it excludes neither proposal
work nor catch-up cost. Keep that interface limitation explicit.

Report retained/reset equality, cross-layout prompt/output equality, and repeat
agreement. Do not call differing-input comparisons identical-input acceleration.
There is no new task utility threshold, independent replay or numerical exception.
v0.16's256/256 long and1527/1536 short replay remains unchanged; #27 stays open.

The pinned disk slot save/restore handlers serialize ctx_tgt only. This experiment
qualifies in-process retention; it cannot qualify restoration of the target/draft
pair across restart. Preserve the relevant native source with the release archive.

Archive all raw receipts and exact source snapshots; actually restore and reproduce
analysis byte-for-byte before release. Complete independent code review and full
clean-candidate validation, then commit/push/tag/prerelease and update #31 without
closing the broader milestone. Stop the optimization queue after reporting this
bounded result. Larger contexts/caps, K24, new drafts, q4 KV, adaptive control,
overlap and physical pagers require separate measured motivation and protocols.
