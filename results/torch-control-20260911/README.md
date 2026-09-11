# PyTorch environment control archive

`torch210/` and `torch212/` preserve the two raw runs, source snapshots, full package
inventories, exact tokens, and manifests. `analysis/` preserves the cross-environment
comparison and exact historical-repeat check. `preparation/` contains the inventory
alignment and environment test receipts, including an explicitly labeled operator
receipt for the initial torchvision incompatibility. It is not a complete original
failure transcript.

Verify `archive.json`, then copy each run into a fresh working directory and decompress
its `layouts.json.gz`. Download the `torch-logits-v0.5.0.zip` release asset and verify
its SHA256 and each compressed payload against `archive.json`. Extract and decompress
the corresponding `ordinary-logits.safetensors.gz` into each restored run directory.
Embedded logits metadata must match that run's manifest. Use the v0.5.0 code:

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.environment_analysis --baseline <restored-210> --candidate <restored-212> --parent results/packing-pilot-20260911/precision-receipted --output <fresh-analysis>
```

This comparison does not need to load model weights or run CUDA. Repeating the actual
model runs requires the separately pinned environments and the exact inventories in
their control configurations. Existing run directories are never overwritten.
