# v0.28: decision coordinates help; recursion does not clear its gate

Fourth authorized delivery. The [frozen protocol](recursive-screen-protocol.md)
finishes in12.177s (worker6.522s, audit5.655s). **Hgeometry passes; Htemporal fails.**
Only the geometry premise is eligible for a separately frozen decision-acquisition
screen. This is not a successful recursive loader or an accepted-token result.

| Four observed pages; score31 unseen pages | Diagnostic MSE | Nominal90 coverage |
|---|---:|---:|
| Static scalar | .09180881 | 78.02% |
| Recursive scalar | .09181015 | 78.02% |
| Static vector/current direction | .08232945 | 79.33% |
| Recursive vector/current direction | .08119994 | 79.54% |

Geometry reduces diagnostic MSE **10.325%**, against the frozen10% screen bar.
Document ratios .755410 and .972204 both clear the1.05 limit. This narrow pass
on two reused diagnostic documents is not statistical confirmation or a coverage
certificate. It is materially more signal than v0.27's0.049% static scalar gain.

Recursion improves only **1.372%** over static vectors, missing its separate10%
bar. Document ratios1.012384 and .975453. Fitted scalar rho .002427 and vector
rho .189807: scalar projection is nearly uncorrelated at lag one in the fit data;
vector persistence is larger but not useful enough under the frozen observation
schedule. Do not credit recurrence for the representation change.

The estimator uses35-by1536 correction-vector means,35-by35 page covariance and
diagonal feature variance. It projects onto the current base top-two readout axis.
The feature covariance is separable/diagonal, not a full53760-dimensional covariance;
nominal bands under-cover (particularly on diagnostic document1). Full-vocabulary
argmax is not certified. Only four currently observed pages and their causal past
enter updates; shadow pages never do, and each document resets state.

## Evidence and boundaries

Source `cc969abee45dad9ac29225bd0545b613f66e93cf` and protocol were reviewed,
validated with483 CPU tests, committed and pushed before calculation. Nine new
tests cover dense-Kronecker equivalence, missing data, document/future isolation,
current-axis behavior and invalid observations. Raw posterior/model arrays and
parent-bound inputs exactly replay; a second audit also matches.

Reuse96 fit positions/six documents and32 diagnostic positions/two documents from
v0.26. Not a new holdout. No GPU inference or physical traffic is newly measured.
PeakRSS220917760B; input tensors56665088B, prediction/model tensors753080B.
Float64 vector state430080B plus9800B page covariance and12288B diagonal feature
variance; priors, observations and scratch are included in process RSS.
Four hypothetical probes require3538944 page payload bytes and24576 FP32-vector
readback bytes per position. These analytical charges are not traffic savings.
Original full v0.26 observation costs remain. No larger-model or native claim.

Raw ZIP39895286B,77 members, each restored/checked;
SHA256 `96632e297c4dbefb19eb56fe4ff629186347f0c22759fcd928e7e75680fb5999`.
Full original measurement remains the v0.26 dependency; source/protocol, derived
inputs, models, predictions and receipts are included here.

## Next

Use the **static vector/current-direction** premise only. Test whether its
predictions and uncertainty change acquisition decisions, versus fixed pages and
contribution ranking. Prediction MSE is not decision value. A useful algebraic
question is whether projection and conditioning commute under the separable
model, allowing a scalar page observation without shipping its full vector.
Freeze that test and costs before measuring; do not expand the failed recurrence.
Complete issue43, keep milestone10 open and stock/native baselines unchanged.
