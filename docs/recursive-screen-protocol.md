# v0.28: changing decision coordinates and recursive page estimation

Prospective fourth delivery, issue43. v0.27's static scalar field failed; its
result is unchanged. A correction vector can persist while its projection changes
as the competing token pair changes. Test that antecedent explicitly, separating
representation from recursion. No inference, native change or long suite.

## Model and antecedents

Apply [Kalman's state predict/update formulation](https://people.math.harvard.edu/archive/116_fall_03/handouts/Kalman1960.pdf)
to a35-by1536 matrix X of physical page correction vectors. Its mean is M;
Cov(vec X) is separable: page covariance C times a **diagonal** feature covariance D.
This matrix-normal simplification avoids a53760-by53760 covariance. It is not a
full high-dimensional posterior: off-diagonal feature uncertainty is omitted.
The observed readout direction a_t changes each token. Predict z_t=M_t a_t and
Cov(z_t)=(a_t transpose D a_t) P_t. This is a changing measurement operator,
not an assumed constant direction of correction.

Fit only v0.26's six development documents/96 positions. M is their vector mean.
D diagonal is pooled centered feature variance, floored at1e-12. Whiten by D,
estimate page sample covariance pooled over features with denominator95*1536,
and mix .5 covariance +.5 diagonal. Scalar controls use the v0.27 Gaussian fit.
Fit a single AR coefficient separately for scalar/vector representations using
adjacent pairs **within** each fit document, clipped[0,.95]. No tuning/sweep.

Predict Mminus=Mprior+rho*(Mprevious-Mprior),
Pminus=rho^2*Pprevious+(1-rho^2)*C. Observe four pages
[(4t+j) modulo35 for j=0..3], same deterministic schedule in every condition.
Condition on their actual vectors/scalars only. Noise is fixed1e-6*diag(C)
(times D for vectors), a numerical regularizer, not a measured sensor likelihood.
Missing pages are not zeros; they keep predictive uncertainty. Reset at each
document. Current direction is base-only and available before acquisition;
high/target decisions and unobserved corrections are scoring-only shadows.

Four conditions: static scalar, recursive scalar, static vector, recursive vector.
Static means rho=0 each step, not a change to the observation budget. Vector
observations contain more information: a physical sensor already returns1536
FP32 values, but the metadata/readback cost must be charged. This is not a claim
that rich vector observations cost the same as12 scalar bytes.

## Frozen screens

Score only31 unobserved pages per position, with fit, diagnostic and each
diagnostic document shown. Hgeometry: static-vector diagnostic MSE <=.90 times
static-scalar, neither document ratio>1.05. Htemporal: recursive-vector MSE <=.90
times static-vector, neither document ratio>1.05. Separate gates; a geometry win
does not rescue a temporal failure. Any pass permits a NEW bounded decision-
acquisition protocol for only the passing premise; both fail stops this filter.
Report all MSE/MAE and nominal Gaussian90 coverage, descriptive only. No IID
interval claim on992 correlated site-position observations from two documents.

Reuse published v0.26 development data, not a fresh holdout; same immutable parent
inventory as v0.27. The first96 fit and last32 diagnostic positions are never mixed
for fitting. Freeze source/config/protocol and push before calculation. CPU toys
test Kronecker conditioning, missing observations, reset and no shadow/future leak.
Raw posterior predictions/model parameters and inputs are archived and exactly
replayed with parent/Git/hash/resource validation. Corrected run uses a fresh path.

## Accounting and contract

.venv, CPU-only supervised300s worker including loading/fitting/scoring and raw
writes; separately timed analysis. Record actual versions, peakRSS/tensor bytes.
Float64 vector state430080B, page covariance9800B, diagonal feature covariance12288B
per filter, plus priors, scratch and observations in process RSS. Four hypothetical
page observations require3538944 H2D payload bytes and24576 FP32-vector D2H bytes.
These are analytically charged proposed costs, NOT new physical traffic savings.
v0.26's original full collection remains fully paid. No model execution or KV
mutation here. In a future speculative runtime, rejected-state beliefs must be
rolled back/reconditioned separately from retained physical weight cache bytes.

No full-argmax certificate, accepted-token, speedup, capacity or native-admission
claim. The separable covariance is an assumption to test, not evidence that token
decisions satisfy Gaussian/Markov assumptions.

Run after freeze: `.\.venv\Scripts\python.exe -m dynamic_model_loading.recursive_screen --output runs/recursive-screen-20260915-v1`.
