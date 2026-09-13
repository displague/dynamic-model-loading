# Continuing-agent evidence

`analysis.json` is independently reconstructed from all ten native runs.
`summary.json` pools continuing-turn totals and explicitly compares prompt/output
IDs across retained configurations and repeats. `environment.json` records the
orchestrator and installed packages; model execution is native CUDA b10919.

The release asset `continuing-v0160-raw.zip` contains all raw requests, SSE bytes,
arrival timestamps, native logs, sampled resources, replay fixtures, selected
historical source receipts, executable analysis snapshots and pinned native source.
`archive-inventory.json` records every member's size and SHA256.
`restoration.json` records an actual extraction with all member hashes checked and
byte-identical reanalysis. Original and restored copies are retained locally.

Archive SHA256:
`82b08500fb57afaa6b003838dae659b20402e6bd138add28f21a31bef552412f`.
Analysis SHA256:
`ee77016125b13506ea3d20e06dad56c892a1cef7ee4dfacac120a3b57ef56c8e`.

Reanalyze the restored evidence on the recorded Windows/Python environment:

```powershell
python source/scripts/analyze_continuing.py --runs runs/continuing-agent-20260913 --output new-analysis.json
```

No model weights or model execution are required for this analysis. Use a fresh
output path. The recorded command paths retain the measured Windows identities.
Top-two replay entries cannot reconstruct full-vocabulary KL or the old verifier's
distribution. Source-derived draft forwarding counts are not independent cache
counters. See the protocol and detailed findings for those limits.
