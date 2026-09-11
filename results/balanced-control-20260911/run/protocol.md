# Small balanced development control on two pinned checkpoints

Prospective issue #20 control. Commit and push source, corpus and scoring rules before
either checkpoint is evaluated. This separately authored development set contains
ten tasks: two each of code, data extraction, arithmetic, exact copying and topic
changes. This balanced mix was selected by the project owner. No Wikipedia article
or final held-out task is reused. Ten fixtures cannot establish general agent quality.
Topic-change fixtures switch instructions within a prompt. Each task starts a fresh
KV episode; this control does not test persistent execution state or residency across
tasks, long conversations, or an always-on agent.

Use the already pinned Qwen2.5-1.5B-Instruct and OPT-1.3B checkpoints and their archived
calibration-only popularity permutations from v0.7/v0.6. Do not calibrate or fit on
these tasks. The two models differ in size, activation, training, tokenizer and
instruction tuning; results compare each model with its own native dense reference,
not absolute cross-tokenizer PPL or an architecture-only causal effect.

Use the original FP32 CUDA environment, TF32 disabled, four CPU threads, seed 1729,
actual one-token KV execution. Keep the pretrained attention, normalization and
residual path. Verify the full checkpoint catalogs, source and parent-layout hashes.
Record parameter/restoration bytes; the diagnostic retains dense weights and backups.
Serialize GPU measurements with the other project experiments.

The prompt is exactly `Return only the requested answer on one line, without explanation.\n`
followed by the task string and `\nAnswer:`. Use no chat template and no added special
tokens for either model. Tokenize the prompt and the answer separately, then concatenate
their token IDs for teacher forcing. Record that boundary explicitly; do not retokenize
or truncate a task after inspecting output. Require nonempty prompt/answer and at most
256 combined tokens. If that bound fails, stop rather than changing the task.

Save native dense teacher-forced logits and 32-token greedy continuations for every
task. Generate exactly 32 tokens, retaining EOS and all later generated IDs in the
raw artifact. For scoring only, take tokens before the first occurrence of the
tokenizer's EOS ID, decode with special tokens omitted, strip surrounding whitespace,
and compare the entire remaining text exactly and case-sensitively with the frozen
one-line answer. Additional nonempty lines are wrong; the first nonempty line is
retained for inspection only. A missing answer is wrong. No execution of
generated code, semantic-equivalence substitution, retries or manual acceptance.
This is exact target/format compliance, not broad functional code accuracy.

Before masked conditions, require width-8 local reconstruction on every layer using
the first two dense FFN inputs from the first task, plus all-group popularity
reconstruction on every full teacher-forced task and exact dense generated token IDs.
Use the original <=0.01 relative-L2 and <=0.001 KL gates; failure stops that model's
masked probes and is retained. The second model still runs as its own control.
Retain both local reconstruction tensors. Non-finite grouped/layout outputs are
model-local numerical failures with an explicit error receipt and no invented finite
metric; finish the numerical-control grid and evaluate the other model. Unrelated
execution failures remain whole-run failures, preserving their exception receipt.
Compute local reconstruction metrics on the exact saved CPU tensors to avoid
CPU/GPU reduction-order mismatches. An infinite relative-L2 from a zero reference
norm is also an explicit failed metric. Observe finiteness at every native, layout
and masked generation visit, preserving all IDs and every non-finite logit tensor.
Non-finite native/layout generation blocks that model's masked conditions; continue
the numerical grid and the other model. Non-finite masked generation retains its
condition grid with a failed model status. Such decoded text is not valid accuracy
evidence even if its literal target-match field happens to be true.
If a shared prompt also makes native/masked teacher-forced logits non-finite, retain
the full logits, generated IDs and numerical-error receipts. Any affected quality
metric and its aggregate are null, never silently dropped or replaced with a finite
score. Native invalidity blocks that model's masked conditions; masked invalidity
retains all condition rows with a failed numerical status. The other model still runs.

Then evaluate static and hindsight selection at nominal 90% group retention, popularity
packing, width 8, on all ten tasks. Static keeps the first calibration-sorted groups.
Hindsight ranks current abs(z)*down-column-norm group sums with stable index ties and
is noncausal. Qwen keeps 1008/1120 groups (90%); OPT keeps 922/1024 (90.0390625%).
Charge OPT input bias with groups and output bias on every visit. Record applied masks,
selected FFN parameter volume and both complete teacher-forced and generated paths.

Report answer-only NLL/PPL relative to each model's dense output, full-text diagnostics,
top-1/KL, exact target compliance, generated-token divergence and all per-task results.
Answer-only positions start with the last prompt token predicting the first answer
token. Use explicit token-weighted aggregates; keep task accuracy denominators separate.
Preserve all dense and candidate generations even when dense answers are wrong.
Save full teacher-forced logits for the layout controls and masked conditions as well
as native dense references, so analysis can recompute every reported quality metric.

This is an exploratory transfer-of-observation control, not a new success screen or
runtime nomination. No model sweep, final-test evaluation, fitting, transfer timing,
achieved memory reduction or tokens-per-second claim. The broader held-out issue #8
and complete-policy/runtime gates remain open irrespective of these ten task scores.
