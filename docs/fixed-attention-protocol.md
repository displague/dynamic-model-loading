# Fixed attention-resident comparison after the allocation-stage stop

Prospective follow-up, 2026-09-13; commit and push before any new inference.
Preserve the original stock-placement protocol and root unchanged.

Under source ffdbba1, thresholds8/4/2 completed twelve scored requests. The frozen
rule selected2 (15.405 native steps/s versus14.439 at8). The completed36/34/32
host-FFN trials passed their buffer/KV/resource checks. The30-host-FFN trial stopped
at startup under the15000MiB guard. The original stage therefore produced **no
allocation decision or scored placement result**. Its failed attempt remains
diagnostic only. This is not an inference that all smaller allocations are infeasible.

Freeze32 host FFNs as one concrete candidate based on its completed trial, whose
receipts are hashed in `configs/fixed-attention-candidate.json`. This is a new
prospective comparison, not a retroactive pass or automatic retry of the failed30
trial. No claim of globally optimal placement is intended.

Use the same pinned native artifacts, threshold2, K16, p_min0, two original16K
prompts, q8_0 target/draft KV, context18432, target24/24 and draft8/24 threads,
batch/microbatch256, neutral greedy settings and15000MiB/2GiB common bounds.
The whole-layer control remains ngl38 including output; the candidate is ngl65
with `--n-cpu-ffn 32`. Do not retune either on the new results.

In a **fresh root**, run two scheduler-debug2 mechanism processes: whole-layer,
then fixed32. Each has the original short32-token warmup and long-code-cache32.
Require actual CUDA_Host allocations, full target KV on CUDA, all64 target
attention assignments on CUDA, complete192 FFN projection coverage, and the
14800MiB reserve in the fixed32 mechanism run. Missing decomposition ends timed
work; native failure aborts scoring. No silent row replacement or failure waiver.

Then four fresh timed processes: repeat1 whole-layer then fixed32; repeat2 fixed32
then whole-layer. Each has the same warmup and two128-token long requests; reverse
prompt order in repeat2. Reuse the validated raw-SSE, acceptance, exact input,
resource-coverage, native-timing and process-order audits. Bind the fixed candidate
file and this protocol alongside the measured source. Report pooled and individual
prefill, native decode, request time, resource use and historical/cross-placement
output identity. Generation steps use predicted_n-1 per request. No margin waiver.

Both layouts share a total resource limit, not necessarily identical allocations.
The old control is the best previously measured long-context whole-layer setting;
this study does not sweep whole-layer residency for a new global optimum. Pinned
host buffers and graph assignments are not measured copy bytes or isolated CPU
attention timing. The current driver is616.92; v0.14 preparation recorded616.64,
so performance claims use these fresh controls. These reused development prompts
and two repeats do not establish workload-wide superiority or agent utility.

Retained-prefix behavior on the known layout is already measured in v0.16. A
changed layout earns a separately frozen conversation comparison if its measured
benefit warrants it. K24, other drafts,32K contexts and custom kernels remain out
of this delivery. Archive both the aborted original stage and this follow-up,
keeping their source commits, protocols and outcomes distinct.
