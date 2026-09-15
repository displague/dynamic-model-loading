# v0.34: full-vocabulary acquisition risk loses actual-prefix economics

Fourth of six deliveries, [#49](https://github.com/displague/dynamic-model-loading/issues/49).
The [prospective protocol](vocabulary-risk-screen-protocol.md) completed in
84.876s: worker77.913s plus independent audit6.963s. Hfaithfulness passes;
Hacquisition and Hruntime fail. Stop this full-vocabulary ranker. No long matrix.

## Actual generated results

| Policy | Accepted / proposed | Charged H2D / accepted token | Episode wall |
|---|---:|---:|---:|
| Binary-pair risk,17 pages | 15/16 | 627,100,911 B | 9.557788s |
| Full-vocabulary risk,17 pages | 13/20 | 837,240,706 B | 35.847426s |
| Load all35 pages | 15/16 | 644,087,842 B | 9.532478s |

All48 committed tokens across six episodes match the independent target reference.
These are actual K4 proposed/verified prefixes, with rejected work and fallback
charged, not teacher-forced pair accuracy. Fullrisk needs29 draft calls versus24
for each control, including high-precision prefill. Its acquisition controller
alone consumes25.928s. Charged H2D totals are9,406,513,664 /10,884,129,172 /
9,661,317,632 B respectively. The new score uses more bytes per accepted token
and about3.76x the wall of all35. Better modeling of the vocabulary event did not
make this prior and action set an economical loader.

Document0 accepts: risk8/8, fullrisk7/8, all35 7/8. Document1: risk7/8,
fullrisk6/12, all35 8/8. Fullrisk performs three actual cache crops. Four scalar
target reference episodes bracket the comparisons; the best per-document sum is
0.322113s. No pager is faster than this fully resident1.5B target. The stock32B
deployment baseline remains unchanged, not directly comparable to these times.

## What changed, and what did not

The new score samples correlated page/remaining corrections, retaining shared
hidden-feature draws through every one of151936 vocabulary competitors. At each
of17 purchases,16 joint samples score every remaining page:459 full-readout action
evaluations per decode call. Gaussian conditioning uses purchased vectors only.
An independent dense verifier, never a sampled token, commits the output.

This is a plug-in uncertainty model of the fully refined draft, not a calibrated
target-acceptance posterior, omission certificate or optimization of budgets across
future tokens. Reusing the v0.29 matrix-normal prior also retains its limitations.
We stop this candidate and do not queue another order of these same pages.

Two newly authored prompts contribute only their first FOUR plain tokens; answers
are capped at eight tokens. Most authored text is unused. This is neither domain
task quality nor long-context qualification. One counterbalanced pair gives no
confidence interval or general performance guarantee. The prospective1% byte gate
was justified before inference by the roughly2.6% saturated-workload headroom;
neither the old5% verdict nor this new failed verdict is loosened after results.

## Correctness, memory and receipts

All28 eight-bit numerical controls pass (maximum relativeL2 1.88245e-7), maximum
physical pair/readout error7.91942e-6, and all target IDs, append/refinement KV
fingerprints and rejection crops reconcile. Target and draft KV remain separate
FP32 storage; maximum logical cache per model802,816 B. Fullrisk's extra controller
H2D is8,953,644 B. Explicit episode D2H is33,765,376 /52,933,019 /33,668,224 B.
The audit reconstructs recorded sampled-ID scores/ties and paid-vector updates;
it does NOT rerun every sampled full-vocabulary head independently. Source binding
and tiny numerical tests support that execution claim, not a mathematical proof.

Target parameters6,174,857,216 B; independent non-FFN draft CUDA parameters
1,550,637,056 B; original host draft parameters4,624,220,160 B. Their sum defines
the7,725,494,272 B CUDA-parameter baseline (no parameter sharing), not free draft
residency. Representation workspace254,607,360 B, resident representation639,959,040 B,
layer/page pinned staging10,321,920 /294,912 B, original host increments578,027,520 B,
and452,168 B prior arrays are retained and charged. Peak additional CUDA allocation
1,009,306,112 B stays below1GiB. Peak sampled totalGPU9,394,294,784 B, RSS9,443,315,712 B,
minimum free host9,017,225,216 B. No capacity frontier was tested.

Model load4.285s and representation construction5.180s are inside the worker but
outside conditional episode times. The construction counter660,602,880 B is
representation H2D, NOT the total model-initialization transfer: moving the target
and non-FFN draft additionally moves their parameter payload above (plus buffers).
Raw phases separately retain construction/equality/readback/mechanics costs.

589 source CPU tests pass. Mandatory source review independently checked the joint
sampling and original v0.30 exact replay; that older failed result is unchanged.
A second model-free audit reproduces every result field exactly in6.889s, without
inference. Final clean-tip full-suite receipts accompany the release separately.

Raw archive `vocabulary-v034-raw.zip`:56,826,528 B,130 members, SHA256
`b03b96f31538fa1c29adf1ff259f6bad96f42461813452733176bf91f913c5ac`.
Every member was decompressed and hash checked; source files were rechecked.
[Compact receipts](../results/vocabulary-risk-screen-20260915) preserve the raw
summary, supervision, archive inventory, review, source tests and repeat audit.

## Antecedents and next step

[Gaussian conditioning](https://gaussianprocess.org/gpml/chapters/RW2.pdf),
[RMSNorm](https://arxiv.org/abs/1910.07467), and
[speculative verification](https://proceedings.mlr.press/v202/leviathan23a.html)
motivate the calculation. No invention of these methods is claimed. This result
rejects this measured acquisition application, not Bayesian inference generally.
Complete only #49. Two deliveries remain: separately frozen packet capacity and
a stronger low-memory comparison/stress screen. Milestone10 remains open; no native
pivot or deployable runtime has been established.
