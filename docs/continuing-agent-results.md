# Continuing turns benefit from retained state and operation offload

The useful stock configuration survives a continuing-conversation test. After the
initial 16K request, threshold 8/K16 takes **34.67 seconds** for five continuing turns,
versus **79.98 seconds** with the best previously tested default-threshold draft
setting, threshold 32/K4. Mean client time to first token is **0.766 seconds**.
Both retain the prefix. Their actual inputs match on all 12 requests across two
repeats; output IDs match on 10/12, with only the final topic-change answer differing.
This is a measured same-input configuration comparison, not universal identical
greedy output or an agent-competence claim.

Against its own exact-prompt reset control, threshold 8/K16 reduces the five-turn
time from **199.10 to 34.67 seconds**. Mean TTFT falls from **34.087 to 0.766 seconds**.
Prefix reuse is responsible for avoiding repeated long prefill. Operation offload
still matters after that reuse: continuing emitted/request throughput is **8.221**
IDs/s versus **3.513** for retained threshold 32/K4, approximately **2.34x**. This
metric includes incremental prefill and is not interchangeable with v0.15's 13.50
native decode steps/s on a different, sustained 128-token workload.

The independent target replay also provides a useful limit. **All 256 long-context
committed IDs match**, while **1,527/1,536 short-context IDs match**. Nine mismatches
occur across four of the six short trajectories. Their cache and independent argmax
checks pass. The full declared agreement test therefore fails, and #27 remains open.
No margin/distance waiver, changed reference, omitted position or numerical
exoneration is applied. This bounded audit is complete; it does not commission
another broad numerical campaign or block the separate stock placement work.

## Experiment and resource accounting

The [protocol](continuing-agent-protocol.md), executable and fixtures were reviewed,
committed and pushed as `17859f7` before new inference. Model execution uses the
unchanged b10919 Windows CUDA 13.3 binary, official Qwen2.5-32B-Instruct Q4_K_M target
and Qwen2.5-0.5B-Instruct Q8_0 draft. The orchestrator remains Python 3.14.3 in `.venv`;
installed PyTorch 2.10.0+cu130 is not used for these native model runs. Package and
native artifact identities are retained.

The conversation configuration keeps 38 target GPU layers including output,
target threads 24/24, fully GPU-resident draft layers with threads 8/24, context 18432,
q8_0 K/V for both models, batch 256/ubatch 256, and `p_min=0`. No new runtime, model,
allocation search, confidence cutoff or controller is introduced. The two replay
processes use target only at matching short/long placements and precisions.

All **10 native processes and 1,848 inference requests** complete: 1,792 independent
one-token replay requests, 48 conversation requests and 8 untimed short warmups.
There are no failed native attempts, skipped rows or substituted outputs. The
review found and corrected six receipt-validation defects before inference. Final
analysis reconstructs source identities, matrix coverage, raw SSE token coverage,
stopping, cache accounting, acceptance counters and resource coverage rather than
trusting compact success flags.

Peak sampled total GPU use is **14,038.395 MiB**, below the 15,000 MiB bound. Minimum
available host memory is **12.548 GiB**. Sampling includes startup and has its count,
extrema, gaps and request coverage checked. This is not proof that no paging or
eviction occurs between samples. All native model processes are closed after use.

## What the conversation actually exercised

The initial code prompt contains 16,384 tokens. Continuing input lengths are 16550,
17435,17546,17661 and17724. Native target counts report 16511,16613,17498,17609 and
17691 reused positions, respectively; new evaluation counts are 39,822,48,52 and33.
Every reset request reports zero reused tokens. The 822-token extraction append
therefore retains its real incremental-prefill cost rather than being called a
zero-cost cache hit. These counts repeat across both settings and repeats.

This is one scripted conversation with code, extraction from tool-shaped records,
arithmetic, copying and a topic change. It uses the model's own previous outputs,
then supplies exactly those prompts to a separate reset process. There is no
autonomous tool execution. Each continuing turn has a 64-token cap. Four of five
retained offload answers reach that cap; some end before satisfying the request,
including extraction and arithmetic. Only three of five retained default answers
reach the cap. All generated text is [published unedited](continuing-agent-generations.md).
No utility score or successful completion of all five tasks is claimed.

All 24 within-configuration repeat comparisons reproduce output IDs. Across the
retained/reset comparisons, 18/24 full requests agree, including 14/20 continuing
turns. Reset and retained paths can differ despite identical supplied prefixes.
Likewise, the last retained topic-change response differs between configurations
(60 IDs/default versus 64/offload). Reported throughput uses actual output counts,
and the complete-turn comparisons disclose those differences. The first four
continuing retained outputs agree across configurations.

## Draft state: measured observations and source-derived accounting

The stock API exposes target cache/new counts and attempted/accepted draft totals;
it does not expose an independent draft KV-reuse counter. The pinned
`server-context.cpp` calls `common_speculative_process` after a successful target
batch, and `draft-simple::process` decodes that same batch on the draft context.
The shared `common_memory::seq_rm` wrapper trims both contexts. Source snapshots
and hashes are archived.

Accordingly the newly evaluated target prompt positions also identify positions
forwarded into the draft prompt-processing path **by source derivation**. This is
not an independently measured draft-cache counter. Proposal generation and the
explicit verification catch-up path remain additional work, included in latency
and represented by the accepted-prefix records. A target cache hit is not reported
as proof of zero draft work or validated draft KV contents.

## Interpretation and next work

The retained-prefix fixture converts the cold-prefill concern into a measured
interaction benefit. It also preserves two cautions: a short output cap limits
task conclusions, and cross-path token equality remains conditional. The evidence
supports useful stock runtime settings on this fixture without redefining #27.

Close bounded [conversation #29](https://github.com/displague/dynamic-model-loading/issues/29)
as an experiment completed; keep the stock milestone and fidelity issue open.
Next, [#30](https://github.com/displague/dynamic-model-loading/issues/30) tests 2/4/8
thresholds at fixed K16, followed by
[#31](https://github.com/displague/dynamic-model-loading/issues/31)'s attention-resident /
host-FFN placement under the same resource budget. The pinned runtime already has
`--n-cpu-ffn`; verify actual host buffers and attention/backend assignments before
interpreting that experiment. Do not assign an unexplained cycle residual to CPU
attention or promise a multiplier before measuring it. Other drafts, learned
adaptive K, overlap changes and optional mask-union work remain deferred.

[Every request and replay discrepancy](continuing-agent-tables.md) Â·
[Protocol](continuing-agent-protocol.md) Â·
[Machine-readable analysis](../results/continuing-agent-20260913/analysis.json)
