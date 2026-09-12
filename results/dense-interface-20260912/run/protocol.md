# Dense interface qualification (prospective)

This protocol, source, configuration and tasks must be committed and pushed before
pretrained execution. No scores from this experiment have informed this design.
v0.10 is preserved without modification or retrospective rescoring.

Use the same pinned Qwen2.5-1.5B-Instruct checkpoint, FP32, SDPA, Python 3.14.3,
PyTorch 2.10.0+cu130 and Transformers 5.13.1 as v0.10. Disable TF32, use four CPU
threads, seed 1729 and CUDA only. Verify the complete checkpoint catalog against
the v0.7 manifest and the popularity layout against the v0.10 configuration.

The documented interface uses the tokenizer's committed chat template with a
generation prompt. System content is the model card's helpful-assistant sentence;
the user content is v0.10's one-line instruction followed by the task. No examples,
answer hints, prompt search or model changes. Plain text uses the exact old prompt.

References: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[HF chat templates](https://huggingface.co/docs/transformers/chat_templating),
[OPT's pretrained role](https://huggingface.co/facebook/opt-1.3b).

Use greedy decoding with a new explicit GenerationConfig: no sampling, one beam,
no penalties or forced tokens, use_cache=True, maximum 64 new tokens, stopping at
the first of the checkpoint generation_config EOS IDs. Record EOS and padding IDs
and the full effective configuration. Never generate after EOS. Reset KV per task.
The 64-token bound and EOS behavior differ from v0.10; do not attribute a cross-release
score change to the template alone. The paired native plain/chat debug comparison
isolates formatting under the *same new* decoding settings.

Run all ten v0.10 tasks as disclosed debugging fixtures, both plain and chat. Run
20 newly authored tasks (four per domain: code, extraction, arithmetic, copying,
topic changes) with chat only. Both sets and the interface are frozen together;
debug results cannot alter the fresh evaluation. These are development fixtures,
not blinded held-out tasks. Code tasks test reasoning about code, not code synthesis.
Topic switches occur within prompts; they do not test persistence across requests.

For each of these 40 task/interface pairs:

1. Native uninstrumented model.generate uses a batched prompt prefill.
2. The project path feeds the exact same prompt one token at a time with KV cache,
   then greedily decodes with the same EOS and bound.
3. The incremental path uses the published popularity permutation and a down-input
   hook multiplying by an all-ones group mask. All groups execute and are counted.

Save generated IDs and every generation decision's raw logits. Also replay both
incremental paths on the native continuation, saving logits for every identical
prefix. Only these aligned logits establish reference fidelity; independently
generated continuations establish behavior. Check finite logits at every model
call, including prompt processing; retain any nonfinite tensors and fail the gate.
Hook cleanup and immutable-byte restoration are mandatory even on failure.

Scoring separates three outcomes, with no substring or expected-answer search:
answer correctness is exact equality of the first nonempty stripped line to the
frozen answer; format compliance requires exactly one nonempty line and the task's
declared scalar syntax (integer, A/B/C choice, or identifier). Copying requires
the *entire* stripped continuation to equal the target for correctness. Success
requires both. No markdown removal, prose parsing or numeric tolerance. Record
the entire continuation, first line, truncation/EOS and all three booleans.

Gate A requires native chat success >=7/10 debug and >=14/20 fresh, with >=2/4
successes in every fresh domain. Additionally every path must be finite, both
incremental paths must reproduce every native greedy continuation exactly, and
every aligned-logit comparison must satisfy relative L2 <=0.01 and mean KL <=0.001.
Failures are published, not grounds to change these criteria. A failure can lead
to a separately preregistered follow-up. All rows run even when an early task fails.

Save inputs, checkpoint and source identities, complete raw rows, generated text,
aligned and free-running logits, and an independently recomputable summary. The
analyzer verifies source against Git objects, corpus hashes, tensor hashes/shapes,
greedy IDs and EOS, scoring, comparison grids and the gate. Report domain counts
and individual failures. This experiment contains no sparse policy, acquired-byte
claim, general coding benchmark, physical paging or latency claim.
