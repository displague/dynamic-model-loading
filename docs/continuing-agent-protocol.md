# Independent committed-token replay and continuing conversations

Prospective protocol, 2026-09-13. Implements ADR 0003 and the post-v0.15 review.
Commit and push this protocol, fixtures and executable harness before inference.
No result exists yet. Use a clean detached checkout of that pushed commit.

## Questions and fixed substrate

Does a second, incrementally populated target agree with the committed trajectories
of the useful v0.15 configuration? Does retained prefix state improve continuing
turn latency, and does threshold8/K16 retain its advantage over threshold32/K4?
These are independent questions. Replay discrepancies do not silently become passes
and do not stop the separately declared performance fixture.

Use the unchanged b10919 executable/DLLs and GGUF hashes in
`configs/stock-speculation-artifacts.json`. The target is Qwen2.5-32B-Instruct Q4_K_M;
the speculative draft is Qwen2.5-0.5B-Instruct Q8_0. Model execution is stock native
CUDA; `.venv` Python orchestrates requests and records its version. No new package,
model, runtime patch, target precision, thread search or allocation calibration.
Every child clears inherited LLAMA/GGML/CUDA overrides and records its own environment.
`p_min=0`, min draft length0, greedy target, neutral penalties, EOS respected,
no context shift, no RAM prompt cache, fit off, batch256/ubatch256, one slot.
Keep sampled total GPU memory <=15000MiB and host available >=2GiB; 200ms samples
include startup. A violation fails that run without replacement. These samples do
not establish absence of paging/eviction between observations.
Analysis requires exact sample count/extrema, monotonic timestamps with gaps no
larger than2s, and coverage through the last request (within2s of the final sample).
Reject missing source identities, duplicate/misnamed matrix entries and non-finite
or over1800s request durations. Replay has a600s per-decision bound. Retain HTTP
error status, headers and body before failing; EOS never permits exceeding the cap.

## Independent replay (#27)

`data/committed-replay.json` freezes all six short and both populated16K continuations
from repeat1 threshold8/K16 in v0.15; the source file identities are included. They
were selected because they constitute the practical configuration, not because an
independent replay is known to pass. Repeated identical outputs are not scored twice.
The six short trajectories contain1536 tokens; the two long trajectories256.
No historical source is changed. This set is not all v0.14/v0.15 configurations.

Run one fresh target-only process per context, in order long then short. Short:
ngl44 including output, threads24/24, ctx4096, f16 KV. Long: ngl38 including output,
threads24/24, ctx18432, q8_0 KV. No draft; threshold32. Use original full prompt on
the first decision (`cache_prompt=false`), then extend with **recorded** committed
tokens one at a time (`cache_prompt=true`). Request exactly one greedy token with
two pre-sampling log-softmax entries. Score before appending the recorded token,
even following a mismatch. Check native cache_n=prefix_length-1 and prompt_n=1
on extensions. Reset between trajectories. The independent computation does not
reuse speculative KV, and its reported argmax must agree with its emitted token.

Check every emitted ID including any replacement/bonus/EOS IDs, without claiming
the trace labels each role independently. All frozen trajectories stop at their
128/256 output cap; validate that cap and absence of an earlier EOG. No extra token
is demanded beyond a limit stop. Later agent turns may stop at EOS and are audited
under their recorded native stop reason. Any independent mismatch remains a failed
agreement observation: there is no margin/distance exception. Top-two output is
diagnostic only, not full-vocabulary KL or a bound on historical verifier logits.

Report counts and first/all discrepancy positions by trajectory and context. Close
#27 only to the narrower agreement claim if the declared set agrees completely;
otherwise keep the explicit remaining discrepancy. Neither outcome proves universal
KV/mask correctness or rewrites cross-path identity failures.

## Continuing-conversation fixture

One authored conversation per configuration/repeat begins with the frozen16K
`long-code-cache` prompt and a128-token response cap. Then append five turns from
`data/continuing-agent-turns.json`: code, extraction from a tool-result-shaped user
message, arithmetic, exact copying, and a topic change. Each has a64-token cap.
This is a client-driven conversation fixture, not autonomous tool execution or
a broad agent capability benchmark. Preserve all generated outputs, including errors.

Append prior emitted IDs directly, close the assistant message once (use its emitted
im_end if EOS, otherwise append im_end), and tokenize the authored next user/assistant
suffix with the pinned tokenizer. Never retokenize the existing prefix. Retain raw
suffix tokenization responses. Abort without trimming if capacity is exceeded.
Workloads and continuation construction are fixed here; exact later prompts depend
on preceding generated output. Persist each actual prompt before requesting it.

Two configurations, unchanged long placement and precision:

| Condition | Threshold | Max draft | Target GPU layers (including output) | Context / KV |
|---|---:|---:|---:|---|
| default |32|4|38|18432 / q8_0 both models|
| offload |8|16|38|18432 / q8_0 both models|

Target threads24/24, draft8/24 and fullyGPU. Two fresh-process repeats; repeat1
configuration order offload,default; repeat2 default,offload. For each configuration,
first run its retained six-turn conversation. Then a new process replays those six
**exact prompts** with `cache_prompt=false` on every request. The reset path's
responses never substitute into later prompts. Its source retained run must be complete
and match condition/repeat. This yields8 processes and48 scored requests. Each process
has the same short32-token warmup (8 additional requests). Retained always precedes
its paired reset; this ordering limitation is explicit, not randomized away.

Use native SSE, preserve raw bytes and event arrival timestamps, and reconstruct all
emitted IDs/content before computing metrics. Record time to first token-bearing
event, time to first nonempty text, full HTTP completion time, native prefill/decode,
accepted-prefix events, actual input/output lengths, stop reason, and sampled resources.
Headers/progress alone are not TTFT. Initial cold request is separate from the five
continuing turns. Report per-turn values and ratio-of-total-output to total-time,
never average instantaneous token rates. Compare outputs for each exact-input pair.
Different conditions may produce different histories; disclose this when comparing
them. Their paired reset controls isolate reuse on identical inputs.

Target reused/new counts come from native cache_n/prompt_n and must sum to the
supplied length. Reset must report zero reused. A retained miss is an outcome, not
a reason to drop a row. Report newly appended client tokens separately from any last
output token re-evaluated for cache synchronization.

Draft accounting has a stock-interface limit: `draft-simple::process` decodes the
same successfully executed target batch; server `decode` invokes it after every target
batch. The slot memory wrapper trims target and draft together. Thus native target
prompt_n determines the positions forwarded through this draft prompt-processing
path, **as a source-derived count**, not an independent draft performance counter.
Preserve pinned source excerpts/hashes in the report. Additional proposal work and
verification catch-up must not be called zero. Record attempted proposals and source
semantics; leave independent draft KV residency/reuse counters unavailable if the
stock interface does not expose them. No added instrumentation or invented counters.

## Validation, reporting and following work

Unit checks must detect missing SSE finals/tokens, EOS/limit inconsistencies,
teacher-forcing errors, incomplete trajectories and mismatched reset sources.
Independent analysis re-reads raw requests/responses/logs and resource samples,
checks exact expected coverage and source identities, reconstructs output/timing and
cache accounting, and counts discrepancies without trusting compact pass flags.
Keep failures and use fresh output directories for a corrected protocol. Native runs
are serial and have no concurrent tests, review, downloads or archive compression.
Archive compact/raw receipts and source snapshots; restore and reproduce analysis
before release. Full clean-candidate tests and independent code review precede commit
and publication; tag/release notes share the tracked release Markdown.

After this bounded delivery, separately preregister thresholds2/4/8 at fixedK16 and
the attention-resident/host-FFN allocation experiment. Verify eligible host buffers,
backend assignments and complete memory costs. K24, other drafts, learned control,
copy overlap and block-union analysis are optional follow-ups, not prerequisites.
