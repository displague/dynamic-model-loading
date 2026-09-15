# v0.27: a covariance field over physical page effects

Prospective freeze, before fitting or scoring. Third of six authorized deliveries.
ADRs 0004--0006 apply. No inference, native change or long performance matrix.

## Question and inspiration

Can a paid page observation predict effects on **unobserved** pages? Use Gaussian
conditioning ([Rasmussen and Williams, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf))
and the structured-plus-independent idea from
[Moraga's spatial models](https://www.paulamoraga.com/book-spatial/bayesian-spatial-models.html).
The geometry is a fit-only co-effect covariance, not physical adjacency or a
claim that SwiGLU pages are counties. This is empirical-Bayes covariance estimation,
not MCMC, a fitted CAR model, or a posterior over model weights.

For page s, z_s = (W_u-W_v) gain dot delta_s / RMS(h), with u,v the BASE top two.
This is a decision-sensitive signed quantity, not local FFN L2. It cannot certify
third-token exclusion. v0.26 already found three diagnostic high winners outside
that pair. A better predictor is not automatically a better loader.

## Fixed data and model

Reuse v0.26's published development data, not a new holdout. Six fit documents
(96 positions), two diagnostic documents (32 positions); never pool them for fit.
The published parent files.json SHA256 is
`ea879f8fb8389be7d2df2a299c608acbf8c7285252b0b345c6701812d430f8c7`.
Bind every consumed frame/tensor to that inventory; archive extracted inputs and
all predictions. Diagnostic observations have already been measured and inspected
for other questions. This is hypothesis screening, not independent confirmation.

Fit 35-dimensional mean and sample covariance. C = .5 sample + .5 diag(variance),
variance floored at max(median variance * 1e-8, 1e-12). The independent component
prevents low-rank overconfidence. Numerical observation noise is 1e-6 times each
site's prior variance; it is a fixed regularizer, not a learned sensor likelihood.
Controls: independent diagonal; field; shuffled correlation labels (seed2701)
preserving each site's variance and mean. No hyperparameter search.

Fit-only greedy probe order maximizes reduction of OTHER unobserved sites'
variance. The same order and budgets (one primary, four secondary) apply to all
models. This borrows the observation-placement question from
[Krause et al.](https://jmlr.csail.mit.edu/papers/volume9/krause08a/krause08a.pdf);
variance reduction is not decision EVSI and no submodular performance guarantee
is imported. Conditioning sees only the selected measurements, never shadow sites.

## Frozen verdict and accounting

Primary Hspatial: one-probe diagnostic MSE <= .90 times independent; neither
diagnostic document exceeds 1.05 times independent; field beats shuffled MSE.
Report fit/diagnostic/document MSE, MAE and Gaussian nominal90 coverage, always
excluding observed sites. Correlated pages/tokens and only two diagnostic documents
do not support an IID confidence claim. Coverage is descriptive, not a gate or
conformal guarantee. Four probes are descriptive, not a replacement primary.
Failure stops this static-field candidate; it does not close dynamic loading.

Run .venv, CPU only, one supervised <=300s worker; record actual environment,
wall time, tensor bytes and process peak RSS. Analysis independently reconstructs
inputs, checks Git/source/config and raw hashes, and exactly replays predictions
and aggregates. No new GPU execution/H2D or accepted tokens. Original acquisition
cost remains v0.26's fully paid collection; one page would require 884736 bytes
and its observed calculation cost, not a free sensor. No simulated bytes become
traffic savings, VRAM savings, speedup or model-capacity claims.

Run: `.\.venv\Scripts\python.exe -m dynamic_model_loading.spatial_screen --output runs/spatial-screen-20260915-v1`.
Only after review and protocol/source commit pushed. Corrected runs use fresh paths.
