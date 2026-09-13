# Resident attention preserves the continuing-turn benefit

The fixed attention-resident / 32-host-FFN configuration completes five retained
continuing turns in **23.985 seconds**, versus **33.110 seconds** for a fresh
whole-layer control. Mean token-bearing TTFT is **0.623 versus 0.776 seconds**.
Both use threshold2/K16/p_min0 and the original 128/64-token caps. The placement
benefit survives prefix retention in this fixture; the outputs do not establish
task completion or universal target equivalence.

This comparison includes **different conversation histories**. Initial, code and
extraction prompts match across layouts; extraction outputs differ, changing the
three later prompts. The pooled 38.0% emitted/request throughput increase is a
conversation-level comparison, not identical-input acceleration across every turn.
The same-input, same-output code turn takes 4.504 s versus 6.153 s. Full per-turn
identity and timing are in the [tables](attention-agent-tables.md).

## What ran

Reviewed and pushed [source 5b7dcbd](https://github.com/displague/dynamic-model-loading/commit/5b7dcbd8ace3357b263536f0a38b88badfec1050)
froze the [protocol](attention-agent-protocol.md), explicit conversation/server
adapters and analyzer before inference. Eight fresh native processes ran serially
in two reversed layout repeats, each retained conversation followed by its reset
pair. There were 48 scored requests and 8 excluded short warmups, with no failed
attempts, corrections or replacement rows. Timed work had no concurrent reviews,
tests, downloads or archive compression.

The artifacts remain stock b10919, native commit
`d3146f2b56c2db4711ac8391871c9e529d1946d7`, Qwen2.5-32B-Instruct Q4_K_M and
Qwen2.5-0.5B-Instruct Q8_0. Target threads 24/24, draft 8/24, q8_0 KV both models,
capacity 18432, initial populated 16384, flash attention on, batch/ubatch256,
greedy decoding, EOS respected, fit/context-shift/RAM prompt cache off.
`.venv` Python 3.14.3 / torch 2.10.0+cu130 orchestrated; the stock Windows CUDA 13.3
binary executed the models. Driver616.92 and the full environment are recorded.

Whole-layer placement uses38 GPU layers including output. Attention placement uses
65 including output and `--n-cpu-ffn 32`. Actual CUDA_Host model buffers and exact
KV allocations passed every startup. Target KV is 2448 MiB entirely CUDA in the
candidate; whole placement splits1415.25 MiB CUDA /1032.75 MiB CPU. Draft KV is
114.77 MiB CUDA. These allocation receipts do not add executed operation counts,
H2D traffic or component attention timers to v0.17's graph evidence.

## Retention and cold requests

| Layout / mode | Five continuing turns | Mean token TTFT | Emitted/request second |
|---|---:|---:|---:|
| Whole retained |33.110 s|0.776 s|8.608|
| Attention retained |23.985 s|0.623 s|11.883|
| Whole reset |201.514 s|34.601 s|1.414|
| Attention reset |172.973 s|30.569 s|1.625|

Each reset uses its retained pair's **exact six prompt arrays**, never its own
responses as later history. All 24 reset requests report zero reused tokens.
The attention retained repeats take 23.993 s and 23.977 s; whole repeats33.166 s and
33.053 s. Retention avoids repeated long-prefill cost on both layouts. The initial
cold requests remain much slower: mean34.574 s for attention and 40.596 s for whole
in the retained processes. Mean continuing TTFT is not an every-turn guarantee:
the attention extraction turn, with 821 authored input tokens, takes about 1.903 s
to its first token-bearing event.

Across ten continuing requests per layout, retained target counters report171844
reused and 1988 evaluated prompt positions; resets evaluate all 173832. Equal counts
do not imply equal prompt IDs across layouts. Newly appended suffix lengths and
evaluated counts are reported separately in the tables. Draft forwarding counts
are source-derived from the successful target batches delivered to draft-simple,
not an independent measurement of draft KV reuse, residency or catch-up time.

## Output and memory limits

All 24 within-layout/mode repeat pairs reproduce their output IDs. Cross-layout
retained outputs agree on 6/12 requests; prompts agree on 6/12. Retained/reset output
agreement is 8/12 for whole and 10/12 for attention. The extraction and arithmetic
whole-layout reset differences and attention topic-turn difference are preserved.
There is no numerical waiver and no new independent replay. Historical #27 remains
open with 256/256 long and 1527/1536 short agreement from the earlier layout.

Both retained layouts emit285 IDs over five turns in each repeat; four turns hit
the64-token cap. Copying stops at EOS with 29 IDs. Extraction starts writing code
instead of finishing the requested extraction; arithmetic runs out of room before
its answer. All outputs remain in the raw receipts. This release qualifies a
continuing-turn performance fixture, not an agent's task-completion capability.

Sampled total GPU peaks are 14447.10 MiB for attention and 13759.10 MiB for whole,
under the frozen15000 MiB allowance. Minimum host availability is 12.647 GiB.
The candidate uses688 MiB more sampled GPU memory. No placement was reoptimized
to consume remaining headroom. Sampled bounds do not prove absence of eviction or
storage faults between observations. A15000 MiB allowance is not a15 GiB allowance.

## Reproduction and disposition

The [compact receipts](../results/attention-agent-20260913/README.md) and v0.18.0
raw asset preserve all manifests, native logs, SSE bytes and arrival times,
resource traces, prompts, outputs, measured source, native source excerpts and
protocol review records. The raw ZIP is 3370935 bytes, SHA 256
`1a297d72a74eb253f4c6bbf66dec5d67786caf333cbd0a2be84968ff43a9bd21`.
All 1418 members (32621641 uncompressed bytes) were actually restored and hashed.
Re-running the analyzer without model loading reproduces byte-identical analysis:
`56263be853457fe51e2ed7f3909f0d3f06e644d646b7e26d7cefc3c5e67d542d`.

This completes #31's bounded retained-layout scope. Use the measured stock
configuration with the disclosed limits; stop the optimization queue here.
The broader milestone and #27 remain open; #25/#26 remain deferred. No 32K,
larger output cap, draft/model/controller/overlap/pager expansion follows.

The [configuration guide](stock-long-context-configuration.md) distinguishes
in-process retention from disk restart. Pinned slot persistence saves/loads only
`ctx_tgt`; it does not serialize the separate draft context in those handlers.
Complete target/draft restart behavior remains unqualified.
