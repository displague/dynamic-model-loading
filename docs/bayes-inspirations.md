# Bayesian acquisition: formula transfer and attribution

This is an experimental combination, not a claim that Bayesian prediction,
progressive precision, kriging or phase-aware inference was invented here.

- [Rasmussen and Williams, GPML, chapter 2 (2006)](https://gaussianprocess.org/gpml/chapters/RW2.pdf):
  Gaussian conditioning predicts an unobserved response from covariance with
  observed responses. We transfer that shape to log remaining FFN error. Nearby
  projected draft states are hypothesized to have related errors. Conditional
  variance motivates buying the missing precision when uncertain. A bad kernel,
  lost projection information, small sample or shifted trajectory can invalidate
  practical usefulness; compare point error and a constant-layer control.
- [Sacks et al., Design and Analysis of Computer Experiments (1989)](https://doi.org/10.1214/ss/1177012413):
  statistical emulation of expensive deterministic responses is the cross-domain
  motivation (publisher metadata checked; full text not relied on). We are not
  simulating fluids or claiming their conservation laws apply to transformer errors.
- [Angelopoulos and Bates, conformal prediction (2022 version)](https://arxiv.org/abs/2107.07511v6):
  calibrate residual ranks on a separate split. We use whole-document maxima to
  avoid counting four correlated token errors as four independent calibration
  examples. Exchangeability is an assumption, not established here. Our fixed
  development documents and adaptive generation support only empirical reporting.
- [PMPD, Chen et al. (2025 version)](https://arxiv.org/html/2410.13461v2):
  directly motivates the high-precision-prefill control requested by the owner.
  Here the draft initializes its own KV using eight-bit FFNs, then explicitly
  retains that prefix under six-bit/adaptive decoding and speculative rollback.
  We do not claim eight-bit FFNs equal a full-precision target.
- [v0.23 inspirations](refinement-inspirations.md): embedded precision and progressive
  physical increment acquisition remain unchanged. The failed deterministic
  correction-direction premise is preserved in [its results](refinement-screen-results.md).

Departure under test: causal statistical acquisition of packed precision increments
inside an independently verified draft, using magnitude uncertainty and explicit
mixed-precision KV lifetime. This tests a concrete version of the original
discussion's probabilistic side-index idea, not the entire space of Bayesian or
high-dimensional loading designs. No llama.cpp patching follows from a PyTorch
speed deficit alone. See the [prospective protocol](bayes-screen-protocol.md).
