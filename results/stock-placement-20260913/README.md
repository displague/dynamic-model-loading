# Stock threshold and attention-resident receipts

The combined analysis keeps three source/run boundaries distinct: the original
threshold/allocation study, the first fixed follow-up stopped by a byte-hash
failure, and the corrected fixed comparison. Native allocation failures are
unscored. No original allocation decision is manufactured.

`analysis.json` is the complete compact result; `summary.json` provides the main
comparison. `archive-inventory.json` binds every raw member, and `restoration.json`
records actual restoration and byte-identical analysis reproduction. Environment
versions and the GPU driver are recorded in `environment.json`. Model weights are
not distributed in the archive.

Download the [raw archive](https://github.com/displague/dynamic-model-loading/releases/download/v0.17.0/placement-v0170-raw.zip).
The archive includes all three raw roots, their measured source snapshots, the
pinned native source excerpts, protocol reviews and the standalone analyzer.
Run `analysis-code/analyze_attention_delivery.py` with the archived original,
initial and corrected source/run directories as shown in `restoration.json`.
No model inference or weight download is needed to reproduce this analysis.
