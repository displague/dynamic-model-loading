# Reproduce the useful v0.15 long-context configuration

Stock b10919 (`d3146f2b56c2db4711ac8391871c9e529d1946d7`), pinned Windows CUDA13.3
artifacts in `configs/stock-speculation-artifacts.json`. Qwen2.5-32B-Instruct Q4_K_M
target plus Qwen2.5-0.5B-Instruct Q8_0 draft, RTX5080 Laptop16GB, Ultra9 275HX.

The measured configuration reaches **13.50 native decode steps/s** after a populated
16,384-token prefix. This is **2.83x** the best tested default-threshold draft setting;
whole-request throughput improves **42.1%**, with roughly33s prefill and9.4s decode
for128 emitted IDs. Two prompts and two repeats; all16 long requests agree with
their placement-matched target-only references. Independent replay remains separately
tracked. This is a fixed-placement comparison, not an optimized target-only frontier
or a claim about continuing-agent latency. See [results](verification-offload-results.md).

The exact generated argv and child environment are retained in the v0.15 raw assets.
Use `scripts/verification_offload.py`'s long/offload-k16 command constructor as the
source of command-line settings rather than silently adding server defaults:

| Resource / execution setting | Value |
|---|---|
| Target GPU layers |38 including output (37 transformer layers GPU,27 CPU)|
| Target threads / batch threads |24 /24|
| Draft |fullyGPU; threads8 / batch24|
| Context capacity / populated prefix |18432 /16384 tokens|
| KV for both models |q8_0 K and V|
| Logical batch / microbatch |256 /256|
| Operation threshold |`GGML_OP_OFFLOAD_MIN_BATCH=8` in fresh child only|
| Draft maximum / minimum / p_min |16 /0 /0|
| Other |flash attention on, fit off, load-mode none, cache-ram0, no context shift|

The threshold is read during backend startup. Do not set it machine-wide. The harness
clears inherited LLAMA/GGML/CUDA settings, records the replacement values and hashes
every executable/DLL/GGUF before loading. Its measured resource bounds are total
sampled GPU<=15000MiB and available host>=2GiB, including desktop use.

This configuration streams cold target tensors for eligible verification blocks.
It does not select semantic neuron pages. Lower thresholds, attention-resident
placement and retained prefix state are prospective experiments, not claimed gains.
