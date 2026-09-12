# Dense interface qualification results

Qwen's documented chat template recovers useful behavior, but the frozen balanced
qualification gate fails. Native chat succeeds on **7/10 disclosed debug tasks**
and **12/20 fresh development tasks**. Matched native plain text succeeds on **0/10**.
All three execution paths produce identical continuations on all 40 task/interface
pairs. Numerical fidelity passes; utility qualification remains open.

## Frozen design and findings

The [prospective protocol](https://github.com/displague/dynamic-model-loading/blob/v0.11.0/docs/dense-interface-protocol.md) and source were
committed and pushed at `805ed348b271a34b20c2364a9127ceaa975f14af`
before any pretrained measurement. The same Qwen checkpoint, FP32 and SDPA were
used. Decoding is greedy, repetition penalty 1.0, maximum 64 new tokens, stopping at
either declared EOS. This differs from v0.10's fixed 32-token loop. Only the paired
plain/chat comparison within this experiment isolates formatting under matched
decoding. The old result is not rescored or replaced.

| Native condition | Tasks | Correct first-line answer* | Format compliant | Success |
|---|---:|---:|---:|---:|
| Old tasks, plain | 10 | 2 | 0 | 0 |
| Old tasks, chat | 10 | 7 | 9 | 7 |
| Fresh tasks, chat | 20 | 12 | 16 | 12 |

*Correctness is the prospectively declared exact first nonempty line, with whole
continuation matching for copying. This is a restricted scalar-answer endpoint;
it does not semantically grade prose or executable code. For example, `C: len(xs)`
does not satisfy the frozen single-letter rule even though it identifies the
right choice. Such output is shown verbatim, not silently reinterpreted as success.

| Fresh domain | Success | Frozen domain minimum |
|---|---:|---:|
| code | 2/4 | 2/4 |
| extraction | 4/4 | 2/4 |
| arithmetic | 3/4 | 2/4 |
| copying | 2/4 | 2/4 |
| topic_changes | 1/4 | 2/4 |

The 7/10 debug requirement passes. The 14/20 fresh requirement and 2/4 topic-change
requirement fail. Gate A is therefore **false**, without changing thresholds.
The twelve successes provide concrete regression examples, but this run does not
establish the declared balanced useful-agent baseline.

## Interface and execution isolation

Native `model.generate` performs a batched prompt prefill with original weights
and no FFN hooks. A passive model-output observer records finiteness without
modifying execution. The project path uses single-token incremental prefill and
decode. The third path adds the published popularity permutation and an all-ones
group hook; every FFN group executes. It tests that representation and
instrumentation, not a physical grouped kernel or partial repair.

Both incremental paths reproduce every native continuation exactly: **80/80 paired
comparisons**, with all model-call outputs finite. Replaying each native continuation
also provides identical-prefix logit comparisons. Maximum relative L2 is
**1.12160412e-05** and maximum mean KL is **3.835198681e-10**, below 0.01 and 0.001.
Generated agreement is assessed separately from logits and task success.

Review before scoring caught a Transformers 5.13 chat-token wrapper and inherited
generation defaults. The frozen implementation explicitly extracts `input_ids`
and fills neutral generation defaults, including the repetition penalty. It also
clones shared native/aligned tensors for serialization. These were premeasurement
apparatus fixes; no experimental result was used to choose them.

## Individual chat failures

| Split | Task | Expected | Raw continuation (JSON escaped) |
|---|---|---|---|
| debug | code-output-01 | `10` | `"11"` |
| debug | arithmetic-sequence-02 | `112` | `"85"` |
| debug | switch-extract-02 | `K8` | `"```python\n{\"region\":\"west\",\"code\":\"K8\"}\n```"` |
| fresh | fresh-code-01 | `10` | `"6"` |
| fresh | fresh-code-03 | `C` | `"C: len(xs)"` |
| fresh | fresh-arithmetic-02 | `76` | `"60"` |
| fresh | fresh-copying-02 | `D4\|e5\|F6` | `"F6"` |
| fresh | fresh-copying-04 | `Qz09_aB` | `"The requested answer is: Qz09_aB"` |
| fresh | fresh-topic_changes-02 | `M4` | `"The tag from the given JSON is \"M4\"."` |
| fresh | fresh-topic_changes-03 | `33` | `"Answer: 33"` |
| fresh | fresh-topic_changes-04 | `H7\|j8` | `"9"` |

Formatting is material, but it does not explain every failure. The model still
answers the old even-number sum as 11 and omits the final addition in the old
arithmetic sequence (85). On fresh tasks it answers a sum as 6, returns 60 before
the requested addition, copies only F6 from a delimited string, and answers the
previous topic's task as 9. Other failures are output-contract violations.

These authored development fixtures cover reasoning about code, scalar extraction,
arithmetic, copying and in-prompt topic changes. They are not a blinded held-out
agent benchmark, code synthesis, persistent multi-request state, or a model sweep.
OPT's separate ReLU sparsity evidence remains unchanged.

## Record and next dependency

All 120 raw rows, prompts, continuations, source/configuration identities, and 120
decision-logit payloads are archived. Independent analysis recomputes the complete
grid, greedy/EOS behavior, scoring, retention counts, same-prefix metrics and gate.
Restored release assets reproduce the analysis. The final clean release candidate
passes the complete 188-test suite in both declared PyTorch environments.

Gate A remains open under issue #8. Bounded issue #21 is complete as an experiment,
not as a satisfied hypothesis. Analytical causal repair #19 continues; it must not
claim preservation of a qualified balanced baseline from this result. Physical
paging #11 and refinement runtime #16 remain gated.
