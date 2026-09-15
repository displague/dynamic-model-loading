# v0.30: physical risk acquisition is correct but loses its economics gates

Sixth of six authorized research deliveries. The [frozen protocol](risk-screen-protocol.md)
finishes in **82.047 seconds**: worker 75.510s plus independent audit 6.536s.
**Hfaithfulness passes; Hacquisition and Hruntime fail. Stop this physical candidate.**
No long suite, native worktree/patch or deployment follows. Milestone 10 stays open.

## Generated, full-vocabulary result

Two documents outside this course's predictor-fit/diagnostic set, four-token
prefixes and eight generated tokens each. All 48 committed tokens across six
episodes match the separately retained FP32 target outputs; repeated scalar
references also match. These are real generated draft trajectories and full
readouts, not v0.29's teacher-forced binary-pair proxy.

| Final-FFN condition | Emitted accepts / proposals | H2D page bytes / accept | Charged total |
|---|---:|---:|---:|
| Fixed17 | 14/16 (87.5%) | 660.777 MiB | 13.690s |
| Risk17 | 14/20 (70.0%) | 740.813 MiB | 11.926s |
| All35 | 16/16 (100%) | 575.859 MiB | 10.526s |

Risk spends **12.112% more bytes per emitted accepted token than fixed** and
**28.645% more than all35**. Lower final-page traffic per call is outweighed by
extra proposal/fallback work, including earlier FFNs. Risk needs 29 draft calls
including prefill versus 25 fixed and 24 all35. The accepted/proposal denominator
charges complete K4 blocks even when a final block is output-capped.

Risk is 13.295% slower than all35 in aggregate. It appears faster than fixed in
aggregate, but the first fixed episode is 8.908s and the second 4.782s; there is
one counterbalanced pair, not enough repetition to separate order/warmup effects
from policy. Do not claim a fixed-baseline speedup. Per-document results are:

| Document / condition | Accepted / proposed | Charged seconds |
|---|---:|---:|
| diagnostic2 / fixed | 6/8 | 8.908 |
| diagnostic2 / risk | 7/8 | 5.062 |
| diagnostic2 / all35 | 8/8 | 5.271 |
| diagnostic3 / all35 | 8/8 | 5.255 |
| diagnostic3 / risk | 7/12 | 6.864 |
| diagnostic3 / fixed | 8/8 | 4.782 |

The better before/after resident-target reference totals **0.351s**. None of these
pager results is an end-to-end speedup over that resident 1.5B target. All35 is the
same-pager full-refinement control, not the best possible dense kernel. The stock
32B/small-draft deployment comparisons remain unchanged and are not directly
comparable evidence for this small HF apparatus.

## What was physically implemented and checked

The nominated policy actually buys three host precision planes for each chosen
256-neuron page, observes its projected correction, updates the current-direction
belief and reuses the loaded workspace for final computation. No unseen-page
oracle, second load of chosen pages or recursive belief is used. A new complete
readout owns each draft proposal; the separate dense target owns commitment.

All 28 high-eight FFN controls pass (maximum relative L2 **1.88245e-7**); parent
representation and integer-plane checks pass. Physical scalar/readout identities
pass, maximum absolute pair error **6.71622e-6**. Actual old-prefix and post-step
KV hashes remain unchanged through final-FFN refinement and link through five
rejection crops (two fixed, three risk). Cropped underlying storage is separately
charged. First-four-call high-precision prefill never re-enters after rejection.
Every full-readout proposal, K4 verification, fallback and committed ID reconciles.

Across the complete worker, page acquisition transfers **30,825,086,976 B**, plus
**660,602,880 B** construction H2D and separately charged mechanics/input/generator
copies. Condition page payloads include prefill and rejected work:
fixed: 9,700,245,504 B, risk: 10,875,174,912 B, all35: 9,661,317,120 B. All cache requests are actual
bypass loads; there are no hits or uncharged persistent page admissions.

Risk's 21 decode calls buy 357 pages and spend 1.148s in page observation and 0.497s
in non-probe acquisition work (controller/axis/selection bookkeeping). Fixed's 17
decode calls buy 289 pages; all35's 16 buy 560. These are arithmetic aggregates of
frozen subclock receipts, not new experiments. Total risk acquisition time, 1.645s,
is inside the 11.926s charged episode total. Runtime plus generator explicit D2H:
fixed: 35,789,028 B, risk: 46,986,492 B, all35: 33,037,440 B; KV hashing/readbacks are charged.
Counters cover explicit application operations, not every framework/driver DMA.

## Memory, provenance and validation

Extra CUDA peak **912.328 MiB**, under 1024 MiB. Sampled total GPU peak: 9,331,380,224 B;
peak RSS: 10,598,440,960 B; minimum available host memory: 5,040,648,192 B. All 281 samples pass.
Target/non-FFN-draft baseline: 7,725,494,272 B; original host FFNs: 4,624,220,160 B; GPU base/
scales: 639,959,040 B; dense workspace: 254,607,360 B; layer/page pools: 20,643,840 B / 589,824 B;
pinned staging: 10,321,920 B / 294,912 B. Original host increments: 578,027,520 B, final host
four-bit base: 20,643,840 B and new plane: 10,321,920 B remain charged. Predictor CPU arrays:
452,168 B; correction vectors, KV and transient tensors are included in CUDA/RSS peaks.
This establishes neither a reduced-VRAM frontier nor access to a larger model.

Reviewed source/protocol **f8796cef8b530aa0cc57dea2ca473d2743093e02** was committed
and pushed before inference, with 520 CPU tests passing. Review added generator/
verifier copy receipts, full reference timing containment and execution-setting
enforcement. Tiny-model rejection and receipt-tamper tests cover state/cost paths.
Exact model-free replay passes twice; second audit 6.166s. Its first launch addressed
the wrong module and stopped before analysis; that tooling receipt is preserved.
There was **one checkpoint run**, no changed tolerance or remeasured candidate.
CPU replay does not independently rerun every neural forward/all vocabulary logits;
the source, measured numerical checks, actual readout IDs and target references
jointly establish the stated screen contract.

Raw `risk-screen-v030-raw.zip`: 47,656,066 B, 118 members;
SHA256 `d44eb603a0f0f26c0357a61645b6118fbd2f76fd7c35927c512ca3826aac603d`.
Every member restored/checked; raw source rechecked. Includes source/protocol,
partial planes, frozen predictor, actual trajectory/state/cost ledgers, numerical
checks, licensing and receipts. Original HF/v0.23/v0.29 dependencies are explicit.

## Six-delivery conclusion

We implemented and tested a novel application path: decision-sensitive physical
page observations, a joint statistical field, changing-axis estimation, decision-
risk acquisition and a real verified draft runtime. It produced component results
but **not a winning inference configuration**. v0.29's binary development proxy
did not transfer into better full-readout/accepted-token economics here. New
documents, generated histories and full-vocabulary decisions changed together;
this screen does not isolate a unique reason for that failure.

Keep the algebraic observation shortcut and geometry result as evidence, preserve
the static/temporal/physical negatives, and require a new hypothesis/protocol before
further inference. A potential next question is full competing-token or accepted-
prefix decision risk; it is unmeasured, not an authorized native optimization.
Complete only issue #45 and the six-delivery authorization; do not close milestone 10
or declare novel loading impossible.

Reproduce research (not a recommended chat runtime):

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.risk_screen --output runs/risk-screen-<fresh-name>
.\.venv\Scripts\python.exe -m dynamic_model_loading.risk_screen --analyze --output runs/risk-screen-20260915-v1/worker
```
