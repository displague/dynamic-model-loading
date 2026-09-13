# Stock thresholds and attention-resident placement

Prospective protocol, 2026-09-13. This protocol, harness and decision rules must be
reviewed, committed and pushed before inference. Follow ADR 0003. Issues
[30](https://github.com/displague/dynamic-model-loading/issues/30) and
[31](https://github.com/displague/dynamic-model-loading/issues/31) are bounded
experiments, not an authorization for a custom runtime. v0.16 already measured
independent replay and retained-prefix interaction; its failures remain unchanged.

Question: can stock threshold or tensor placement improve the populated16K
configuration after charging all attention, KV, FFN, draft and workspace memory?
CPU attention is a hypothesis about part of the cost, not a measured residual timer.

Use the pinned b10919 binary and catalog target Qwen2.5-32B-Instruct Q4_K_M plus
Qwen2.5-0.5B-Instruct Q8_0. Reuse exactly the two 16,384-token prompts in
`data/committed-replay.json`. Maximum output 128; respect EOS. Same neutral greedy
decoding and special-token policy as v0.16. K16, explicit p_min=0, target threads
24/24, draft threads8/24, q8_0 KV for both, context18432, batch/microbatch256,
one active slot, flash attention on, fitting/context shifting off. Process-local
environment overrides only; clear inherited LLAMA/GGML/CUDA controls and record
their names and the effective replacements. No new draft confidence dimension.

All native processes run serially, on clean pushed source. No tests, reviews,
archives or downloads run concurrently with timings. Each process has the same
short-code-cache 32-token warmup, excluded from scores but retained in receipts.
Every long request has cache_prompt=false. Raw SSE and arrival timestamps, native
logs, complete argument/environment/artifact/source identities, acceptance events,
and sampled resources are retained. Every failure stays in its original directory;
a corrected protocol uses a new root. No replacing unfavorable rows.

1. **Threshold:** ngl38 including output, thresholds8/4/2 at K16. Repeat1 order
   8,4,2; repeat2 order2,4,8. Both prompts per process; reverse prompt order in
   repeat2. Six processes, twelve scored requests. Select minimum pooled native
   predicted_ms divided by actual emitted IDs, over all four requests per threshold.
   Within 1% of the minimum, choose the largest threshold. Also report native step
   rate, complete request time, prefill, output identities and individual results.
   This is selection on development fixtures, not held-out superiority evidence.

2. **Allocation:** at that threshold, ngl65 (64 transformer layers plus output)
   and stock `--n-cpu-ffn N`, N=36,34,32,30 in that order. Each receives warmup
   and long-code-cache with a 32-token cap. Verbosity6 exposes actual tensor
   overrides. Require all first-N gate/up/down tensors to use CUDA_Host, all target
   q8_0 KV on CUDA (2448 MiB), draft KV114.77 MiB. Sampled GPU total must not exceed
   15000 MiB, host available at least2 GiB. For allocation eligibility reserve a
   further200 MiB (peak<=14800). Stop on the first completed ineligible allocation;
   choose the smallest N among previously eligible attempts. Preserve every
   attempted candidate. A native failure aborts the stage without an allocation
   decision; it remains unscored and cannot establish infeasibility or select a
   previous candidate. Investigating or correcting it requires a new run/protocol.
   This brackets a feasible split within four declared points;
   it does not find the globally optimal allocation. No eligible split ends this
   branch without patching the runtime. Allocation timings do not rank candidates.

3. **Mechanism:** fresh verbose scheduler-debug2 processes for ordinary whole-layer
   placement and the selected host-FFN placement, same threshold, warmup plus
   long-code-cache32. Separate graph dumps at node-number resets; identify target
   graphs by all64 attention layer IDs in cache inputs. Report FLASH_ATTN and
   FFN operation backend assignments, including CPU tails. Allocation plus all64
   target attention assignments on CUDA establishes the intended decomposition.
   Graph assignments are not execution counts, transfer bytes or isolated time.
   If the intended decomposition is absent, report it and stop timed placement work.

4. **Placement:** if feasible and mechanism confirmed, fresh whole-layer and
   selected host-FFN processes, two repeats and both128-token prompts. Repeat1
   whole then FFN-host; repeat2 FFN-host then whole, reversed prompt order. Four
   processes/eight scored requests. Apply the common15000MiB/2GiB resource limits.
   Compare pooled costs and individual failures without changing thresholds or
   allocation. Keep v0.15 frozen greedy outputs and cross-placement agreement as
   diagnostics, never waive differences via a logit-margin inequality. A speedup
   does not prove universal target-output identity or useful task completion.

Output-limited tails can shorten actual proposals. Record each attempted/accepted
prefix; attempted+1 is a source-derived verification width, not direct graph-shape
telemetry. No p_min acceptance interpretation. Resource sampling cannot prove an
unsampled allocation maximum or absence of transient spill. Native step counts,
emitted IDs, whole-request time and cycles are distinct denominators.

The complete ledger and decisions must reproduce from archived raw files without
model weights. Reject missing/extra matrix rows, changed fixtures/configurations,
unbound decision files, SSE/token/counter inconsistencies and uncovered resource
intervals. Native failures are retained but abort scoring/selection; final analysis requires
all attempted native runs to have completed. They are not feasibility observations
and cannot select a previous candidate. No automatic rerun is permitted.
Require actual model-buffer allocation records as well as requested overrides;
pinned-host allocation failure may fall back to an ordinary CPU buffer. Bind an
attempt record before artifact preflight, including failures. Validate observed
serial process and phase order. Native counters must be integral and emitted counts
must agree; summed native prefill/decode cannot exceed client request time by more
than100ms of receipt-clock allowance. Native generation rate uses predicted_n-1
per request, because the first emitted token comes from prefill. This accounting
allowance is not a numerical/model-quality tolerance.
No new numerical tolerance or task gate is introduced. A new candidate can justify
a separately frozen retained-prefix comparison; this delivery does not silently
append K24, another draft,32K contexts, learned control or a scheduler patch.
