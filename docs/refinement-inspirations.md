# Progressive acquisition: inspiration and departure ledger

Recorded during implementation, before checkpoint measurement. Credit concerns
both the mechanism and the questions that motivate testing it. This ledger does
not establish publication novelty or transplant reported speedups to this runtime.

| Primary source | Relevant idea | What this experiment uses or leaves untested |
|---|---|---|
| [Equitz and Cover, Successive Refinement of Information (1991)](https://isl.stanford.edu/~cover/papers/paper94.pdf) | Successive descriptions can refine reconstructed information; optimal refinability is conditional. | Acquire small additional descriptions of weights. No rate-distortion optimality claim. |
| [Any-Precision LLM (2024)](https://arxiv.org/abs/2402.10517) | Multiple weight precisions and bit-plane representation. | Embedded bit planes are an antecedent, not our invention. Our simple affine quantizer is not a reproduction of their quantization algorithm or kernels. |
| [BitStack (2024)](https://arxiv.org/abs/2410.23918) | Incremental compressed model blocks and variable memory/performance trade-offs. | Supports asking a capacity-frontier question separately from transfer savings; no claim that incremental model storage is new. |
| [AnyBCQ (2025)](https://arxiv.org/abs/2510.10467) | Progressive precision expansion and reusable binary codes. | Further direct precedent for incremental precision; we do not use its specialized binary computation. |
| [PMPD (2025 revision)](https://arxiv.org/html/2410.13461v2) | Phase-aware precision, progressively lower decoding precision, static and prompt-conditioned scheduling; frequent switching has costs. | Owner supplied this source before freeze. Our current experiment refines upward locally instead of scheduling downward across a sequence. High-precision prefill is a distinct, untested alternative to revisit if KV contamination limits usefulness. All switching and repeated computation are charged. |
| [DecDEC (QDEC), v2](https://arxiv.org/html/2412.20185v2) | Low-bit weights with dynamically acquired quantized residual compensation. | Direct prior art for residual acquisition. We test observed inter-precision output changes and previous-token cache admission, not merely a renamed residual loader. |
| [Rump, Verification Methods (2010)](https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf) | Rigorous numerical enclosures distinguish known bounds from estimates. | Motivates explicitly labeling the geometric remainder proxy as heuristic. No certified argmax or target omission is implemented. |
| [Richardson (1911)](https://doi.org/10.1098/rsta.1911.0009) and [How Accurate is Richardson's Error Estimate? (2025)](https://doi.org/10.1002/cpe.70305) | Convergent approximations with a stable leading error term permit extrapolation and error estimation; the expansion assumptions matter. | Owner's formula-to-other-domains method led to H4 before measurement: yR=y6+(y6-y4)/3 for spacing ratio four and assumed order one. Test both predicted benefit and whether actual increment directions support the premise. No rounding-error expansion is assumed proven. |
| [Russell and Wefald, principles of metareasoning](https://people.eecs.berkeley.edu/~russell/papers/aij-cnt.pdf) | Spend computation according to expected decision value and cost. | The cache score uses observed local correction value as a cheap surrogate. It is not an optimal value-of-computation solution and may not predict final-token usefulness. |

The testable contribution is an implementation and controlled measurement of a
specific acquisition/reuse hypothesis, not a demonstrated globally novel algorithm.
Previous local work supplies the failed two-bit base, full-page correction and
stateless controls; those results are retained in v0.22. New sources and design
changes should be added prospectively rather than retroactively attributed.

## Formula -> antecedent -> consequence -> test

The current apparatus produces y4, y6 and y8 while reducing grid spacing by four
at each step. This resembles mesh-refinement sequences in numerical analysis.
The antecedent for Richardson's consequence is stronger than "more bits improve
average accuracy": errors need a stable asymptotic leading term and controlled
arithmetic error. With first-order error, successive differences should point
similarly and shrink roughly fourfold; extrapolation can cancel that term.

H4 explicitly tests the consequence on the stored dense-input FFN outputs. A
failure can localize a broken assumption (for example, alternating rounding-error
directions), rather than dismissing all cross-domain error prediction. Other
convergence orders, fitted coefficients, stochastic estimators or phase-aware
prefill are future hypotheses, not candidates selected after seeing these rows.
One concrete antecedent for that later branch is
[Probabilistic Richardson Extrapolation](https://arxiv.org/abs/2401.07562), which
connects extrapolation to multi-fidelity statistical modelling and uncertain
convergence orders. It is cited as a possible consequence to investigate, not
implemented or evidence that quantization satisfies its assumptions.

The same method also supplies a cache lower bound, not just a new algorithm:
each layer/stage is used once per consumed token, and all first increments are
mandatory. Eight cached first increments give the maximum eight hits per later
token. This observation caused addition of the fixed-admission control before
freeze. Reducing traffic versus streaming cannot establish value-ranking novelty
on this access pattern; the screen reports its gap to the fixed bound explicitly.
