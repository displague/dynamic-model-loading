# v0.41: 512-token union stress after the short scale screen

Owner requested continuation after the first four deliveries; this fifth delivery
uses the remaining four-to-eight allowance. It is a separate minutes-scale
durability diagnostic, not an expanded validation matrix or native admission.
Reviewed source, config, source-text selection and gates committed/pushed before
one <=300s supervised inference worker. Same baseline .venv, original pinned
OPT2.7B FP16 weights/KV,32 layers, SDPA/four threads/TF32 off as v0.40.

## Distinct question and frozen prediction

For one token, outgoing acquisition uses the observed nonzero set A_t. For a
prefill block B, a single physical packet must serve the union U_B=union(A_t).
Set union grows monotonically when tokens are added to a fixed activation trace;
that does not imply its wall cost or new trajectories are monotonic. The question
is whether increasing the prefix from32 to512 tokens consumes the acquisition
advantage, even while token-by-token decode retains it.

Predict: exact outputs/resources remain feasible, decode still saves >=90% H2D
and >=50% forward-call wall versus streaming, but prefill's <=50% traffic and
no-slower-call gates may fail as the union grows. Overall episode savings may
survive because the16-token continuation includes15 scalar decode forwards.
Report these outcomes separately: do not hide a prefill loss in a decode total.

[LLM in a Flash](https://arxiv.org/abs/2312.11514) motivates physical-grain/window
effects, [PowerInfer](https://arxiv.org/abs/2312.12456) hot/cold sparse activations,
and [OPT](https://arxiv.org/abs/2205.01068) the original ReLU architecture. Set-union
accounting is ordinary mathematics, not a new sparsity claim. The candidate is
unchanged exact observed-zero loading; this study tests its operating boundary.

## Workload selection and reference contract

Use existing archived WikiText corpus hash
5a100c930dae2532232c83d50af75f67b0d8c1e4c5365fdf4c14f450c1eb0b31, at
`results/relu-control-20260911/run/corpus.jsonl` from commit
4cff9a312a355f8eff596044875d871a92189664. Join records[0,1,2], [3,4,5], [6,7,8]
with two newlines for scored inputs; [9,10,11] is separate warmup. Config embeds
the literal texts and provenance; audit reconstructs them from that Git object.
Take exactly the first512 OPT tokens per input, no special tokens; generate up to
16 greedy tokens, stop on EOS. Selection is by fixed record order, not scores.
These are KNOWN historical calibration texts concatenated across record boundaries,
not a new holdout or a natural long-document benchmark. No text is repeated to pad.
All input token counts must suffice before inference; no replacement after results.

CPU-only model load; no CUDA allocated/reserved/current/peak bytes before candidate
construction. Only non-outgoing parameters move to CUDA. Stable target-only greedy
reference is contiguous dense STREAMED FP16, not full resident2.7B. Run stream
warmup then docs0/1/2; packet warmup then docs0/1/2. Reset KV at each episode. Same
FP16 arithmetic, separate phase-first-use accounting, no speculative draft. All
checked packet logits must match the corresponding dense reference at relativeL2
<=1e-5, with exact IDs/stops; includes warmups. No FP32 quality claim.

Every phase stays under4800MiB sampled global GPU and allocator limit on the16GiB
device; host available >=2048MiB. Max KV527 positions*327,680B=172,687,360B.
Same3,625,472,000B non-outgoing CUDA parameters,52,428,800B workspace,
52,510,720B CUDA and pinned packets, two1,677,721,600B host weight layouts.
Temporary full-prefix activations/logits, finite checks, union/readback, gather/
scatter, indices, tensor/JSON records and memory boundary samples are charged.
Dense control has a contiguous original-layout host-to-workspace path. No cache,
prediction, omission, native patch, compression or hidden preload is introduced.

## Separate gates and reporting

- Inherit v0.40 faithfulness/resources/CPU-first capacity gates. Both controls must
  fit; no unique-access claim. Each scored packet episode must finish <=5s.
- Whole-episode Hacquisition: packet outgoing weight+index H2D <=50% of stream.
  Hruntime: packet scored wall sum <=80% of stream. Full recording is in wall.
- Hprefill_traffic: packet prefill outgoing H2D <=50% of stream prefill.
  Hprefill_runtime: packet prefill forward-call sum <=stream prefill-call sum.
- Hdecode_traffic: packet decode outgoing H2D <=10% of stream decode.
  Hdecode_runtime: packet decode forward-call sum <=50% of stream decode-call sum.

The first forward consumes512 tokens; remaining forwards consume one each. Report
all per-phase absolute bytes/call times/ratios plus per-episode TTFT/wall, setup,
warmups, cold worker-entry-to-first DENSE logit, total worker+audit time and full
resource charges. Phase call times include acquisition/compute/readback, but final
episode serialization is not assigned to one phase; whole wall still charges it.
They are different timing denominators, not interchangeable end-to-end savings.
The cold worker clock starts inside the shared model worker after the workload-
integrity precheck; total supervised wall includes that precheck and module startup.

Any correctness/resource failure or timeout ends the candidate and preserves the
fresh run. A failed component stops uniform long-prefix expansion even if another
component passes; no long matrix follows. A phase asymmetry may motivate a NEW
short physical-grain protocol, not retuning this screen's bars or extending it.
Any overall pass remains a diagnostic, not long-context validation or optimized
Q4/Q8 competition. v0.40's short prompt results and all negatives remain intact.
