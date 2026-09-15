# v0.27: the static page field does not transfer useful information

The frozen [protocol](spatial-screen-protocol.md) returns
**stop_this_static_field**. Worker6.176s + analysis5.290s =11.465s, CPU only.
Third of six authorized deliveries; complete bounded issue42, not milestone10.

| Unobserved diagnostic page effects | MSE | Nominal90 coverage |
|---|---:|---:|
| Independent, one observation | .09131851 | 79.69% |
| Field, one observation | .09127416 | 80.06% |
| Shuffled, one observation | .09095631 | 79.69% |
| Independent, four observations | .09301979 | 79.84% |
| Field, four observations | .09391837 | 80.34% |
| Shuffled, four observations | .09422627 | 79.33% |

Primary field/independent MSE ratio **.999514**: a0.049% reduction, not the frozen
10% improvement. Field loses to shuffled geometry. Document ratios .994817 and
1.002057. **Hspatial fails.** Four probes are descriptive, not a replacement
primary, and worsen field MSE by0.966%. Observed sites are excluded. The one-probe
diagnostic denominator is1088 site-position predictions but only **two documents**,
not1088 IID samples. Nominal Gaussian bands are not calibrated safety bounds.

Fit-only probe order [4,1,24,34], zero-indexed. Fit primary MSE improves2.715%.
The model genuinely conditions jointly across pages, unlike v0.24's per-layer
GPs, but this architectural distinction did not yield useful diagnostic transfer.
No parameter or gate changed after scoring.

## Provenance and costs

Reviewed source/protocol `28b55a46c945e70fea21548bd8676f58160128d2` was pushed
before scoring, with474 CPU tests passing. Review added seven completion/resource
tamper tests. All128 inputs are re-derived from SHA-bound v0.26 observations;
all predictions, covariance, probe order and aggregates exactly replay, including
a second audit. This is deterministic replay, not a separately implemented
statistical estimator or a new inference run.

Worker peakRSS162344960B; input tensors56665088B and predictions440192B. No GPU
or new physical H2D measurement. A hypothetical probe requires884736 page bytes;
the original observations were already fully paid in v0.26. CPU replay does not
make a sensor free. No online policy, accepted-token, speedup, VRAM-frontier or
larger-model claim follows. Reused96 fit positions/six documents and32 diagnostic
positions/two documents are development data, not an independent new holdout.
The base top-two projection cannot certify full-vocabulary argmax.

Raw ZIP39451356B,75 members;
SHA256 `0196a315d63f70d92818aa98e46bab1a05b2f14ae3202f309e0b3c51b20673c6`.
Every member restored and checked; raw sources rechecked. Includes extracted
vectors/effects, posteriors, frozen source/config/protocol and receipts. The
unmodified complete v0.26 observations remain an explicit parent dependency.

## Next hypothesis, not a rescued verdict

The decision axis changes with each token's competing candidates. Persistence of
a correction vector need not mean persistence of its signed scalar projection.
Test that distinction using only causally observed histories, missing-data rules
and explicit uncertainty under a new protocol. Do not retune this static covariance
to clear its gate. Stock/native baselines unchanged; no long matrix follows.
