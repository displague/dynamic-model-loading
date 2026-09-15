# v0.29: from changing-axis prediction to acquisition decisions

Prospective fifth delivery, issue44. Only v0.28's passing **static vector/current-
axis** premise is used. Recurrence failed and is not expanded. Fit/scoring follows
review, source/protocol commit and push; same immutable v0.26 development inputs.
No checkpoint inference, physical traffic claim, native change or long suite.

## Algebra and decision value

Under the separable model Cov(vec X)=C times D, observations of complete page
vectors use the same page-conditioning gain for every feature. Therefore projecting
the posterior onto a FIXED current direction a is equivalent to conditioning the
projected prior M a with covariance (a transpose D a) C on scalar page projections.
Test means and full page covariance against the full-vector calculation at all128
positions, using the v0.28 four-page schedule. Tolerance1e-10 absolute. This is a
conditional sufficient-observation result, not sufficiency for another direction
or all token logits. If the competing pair changes, retained scalar observations
cannot be assumed sufficient for the new axis.

Use the unchanged v0.28 fit_vector method: fit first96 positions, no diagnostic
updates to model parameters. Current base top-two axis and margin are known.
Within each position the controller may only call observe(page) for a purchased
page, once per page. Missing sites remain uncertain, not zero. All selections,
paid scalar observations and scores are archived. No cross-token state.

With loaded margin m and unobserved z ~ N(mu,C), full pair margin is
T=m+sum z. Loading s changes the ACTUAL partial margin to Y=m+z_s.
Risk(s)=E[1(sign(Y)!=sign(T)) | observations]. This is one-step plug-in decision
risk, **not** just posterior variance. Integrate the conditional Gaussian T|z_s
using32-point Gauss-Legendre quadrature over standard normal[-9,9], split at both
sign-change breakpoints. A diagnostic128-point audit of every available action
along both risk-policy paths must differ by <=1e-4. Tail mass outside[-9,9] is
negligible relative to this tolerance.
Near-zero variances floor1e-14 for numerical integration. The model's Gaussian
bands already under-cover; these are ranking scores, not certified probabilities.

The decision-value framing draws on
[Russell's rational metareasoning](https://people.eecs.berkeley.edu/~russell/papers/aij-cnt.pdf)
and [Gaussian conditioning](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
Our application compares acquisition actions by predicted decision error. It does
not invent Bayesian decision theory, provide optimal multi-step planning, or solve
the cost-calibrated stopping rule. A fixed17-page budget makes page payload costs
equal; no claim that controller costs are equal or free.

## Controls, frozen gates, next-step rule

All acquire17 pages: fixed first17; sequential largest absolute posterior mean
contribution; sequential lowest expected pair risk; risk with diagonal covariance
(same changing-axis means). Observations update only purchased sites and correlated
beliefs. Score actual additive partial margin against additive all35-page margin,
whose readout relation was checked in v0.26. This is a BINARY pair proxy, not
full-vocabulary argmax or accepted tokens. Report fit/diagnostic/per-document errors,
repairs, newly introduced errors. Posterior full-margin forecast errors are
descriptive only, never substituted for actual partial-computation results.

Hcommutation: all mean/covariance differences<=1e-10.
Hnumerical: all visited diagnostic risk-path quadrature differences<=1e-4.
Hrisk: at least TWO fewer diagnostic pair errors than fixed17; neither document
more than one worse than fixed; no more errors than contribution or independent-risk.
Hcontribution: at least TWO fewer diagnostic pair errors than fixed17; neither
document more than one worse than fixed. These are separate declared candidates,
not a post-result change of the risk hypothesis. No IID significance claim on two
reused documents. Nomination: risk if Hrisk, otherwise contribution if Hcontribution;
both require Hcommutation and Hnumerical. No nominee means stop these policies.
An algebraic shortcut alone does not rescue an acquisition-policy failure.

## Reproduction and costs

.venv CPU-only supervised300s worker; include parent loading, fitting, all policies,
numerical audit and raw writes. Record versions, peakRSS, tensor bytes and wall
time; separately timed analysis re-derives published inputs and exactly replays
all selections, predictions and aggregates. The supervisor bounds the worker, not
a new long inference suite. Corrections use new run directories.

17 page payloads would be15040512B; scalar observations68B versus104448B full-vector
readback, plus6144B current-axis readback, model/index storage and controller cost.
This is an analytical proposed reduction, not measured physical saving. v0.26's
original fully paid collection remains unchanged. No new GPU/H2D, KV mutation,
accepted-token, model-capacity, speedup or native-admission evidence here.

Run: `.\.venv\Scripts\python.exe -m dynamic_model_loading.acquisition_screen --output runs/acquisition-screen-20260915-v1`.
