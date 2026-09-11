# Repair diagnostic archive

The Git files preserve all 260 raw ledger rows, source/configuration/protocol snapshots,
token IDs, independently recomputed analysis and complete curves. The two v0.8.0 release
assets contain 10 reference-logit and 144 applied-mask/audit payloads. `archive.json`
indexes every Git file and asset payload by size and SHA-256.

Five privileged conditions pass the aggregate diagnostic screen; only one also passes
both article quality lines. All runtime eligibility fields remain false. See the
[complete report](../../docs/refinement-feasibility-results.md) for scope and limits.

Restore the v0.7 parent archive first using its [instructions](../causal-control-20260911/README.md).
Then download and restore v0.8.0 without running another model experiment:

```powershell
gh release download v0.8.0 --repo displague/dynamic-model-loading --pattern 'repair-*.zip' --dir runs/repair-assets
.venv/Scripts/python.exe results/repair-control-20260911/restore.py --assets runs/repair-assets --output runs/repair-restored
.venv/Scripts/python.exe -m dynamic_model_loading.repair_analysis --run runs/repair-restored --parent runs/restored-causal --archive results/causal-control-20260911 --output runs/repair-reanalyzed
.venv/Scripts/python.exe -m dynamic_model_loading.repair_plot --run runs/repair-restored --analysis runs/repair-reanalyzed --output runs/repair-replotted
```

Use the recorded baseline environment and a repository clone containing the source
commit in Git history. The analyzer validates actual committed Git blobs, not only
self-reported source hashes. Parent receipts/layouts and selected original causal
masks are required. Large candidate logits are not retained for this repair run;
quality values are the immutable recorded measurements, with reference-side NLL,
ranking receipts, mask unions and traffic independently checked. No timing or causal
runtime claim is inferred from the archive.
