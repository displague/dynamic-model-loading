# Hardware-cost evidence

Every raw measurement and source/input receipt is tracked in this directory; no large
release assets are required. `archive.json` checks sizes and SHA-256 values for the
complete Git inventory. `analysis/summary.json` independently validates all 569 rows,
36 workload medians and two ranking medians. No pretrained model or pager is involved.

From a repository containing the recorded source commit in Git history:

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -m dynamic_model_loading.hardware_analysis --run results/hardware-cost-20260911/run --output runs/hardware-reanalyzed
```

Use the recorded baseline environment. This CPU analysis regenerates the seeded
synthetic pool receipt and recomputes all medians without CUDA timing. For a new
hardware measurement, follow the frozen protocol with a fresh output directory;
new hardware/timing does not replace the original results.
