# Restore and analyze the causal evidence experiment

The Git archive contains the raw ledger, source/configuration/input snapshots,
primitive timing grids, analysis and console receipts. Indexed release ZIPs hold
the fitted models, calibration examples, state/ranking/mask traces, teacher-forced
logits and generated logits. `archive.json` lists every container and payload size
and SHA256. The archive contains no pretrained model weights.

Use Python 3.14.3 with PyTorch 2.10.0+cu130, Transformers 5.13.1, safetensors 0.8.0,
huggingface-hub 1.19.0 and NumPy 2.3.5. Analysis runs on CPU and does not execute
pretrained weights, but verifies the full pinned checkpoint catalog. A fresh cache
containing exactly the listed files avoids extra-file catalog mismatches.

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:HF_HUB_CACHE=(Join-Path (Get-Location) 'runs/causal-evidence-hf-cache')
hf download Qwen/Qwen2.5-1.5B-Instruct config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
gh release download v0.12.0 --repo displague/dynamic-model-loading --pattern 'causal-evidence-part*-v0.12.0.zip' --dir runs/causal-evidence-assets
.venv/Scripts/python.exe results/causal-evidence-20260912/restore.py --assets runs/causal-evidence-assets --output runs/restored-causal-evidence
.venv/Scripts/python.exe -m dynamic_model_loading.causal_evidence_analysis --run runs/restored-causal-evidence --cost results/causal-evidence-20260912/cost --output runs/restored-causal-evidence-analysis
```

The restorer verifies all indexed Git files, ZIPs and payloads before writing a
fresh restored directory. Timing files stay in the indexed Git `cost` directory.
The analyzer checks archived source against the original Git commit, checkpoint
and input identities, complete calibration/evaluation grids, fitted coefficients,
permitted observation/state replay, exact ranking from retained GPU scores,
fallback decisions, numerical controls, generation, scoring, acquired/dispatch
bytes and the measured cost calculations.

Quality measurements retain their original frozen source snapshot. The corrected
timing has its own source/configuration/protocol snapshot and identifies the original
quality commit. The first cost run completed nine fixtures, then failed exact-mask
replay on full completion because a contiguous reload changed the fitted matrices'
layout and moved a near-tied selection. Its 468 timing rows and console failure are
retained under failed-cost and in the original cost log. Diagnosis scripts/logs
are retained too. They are not mixed with the complete corrected timing grid.

The timing correction reconstructs the original solve layout from the saved
calibration examples and requires every coefficient to match bit-for-bit. It then
reruns the unchanged timing workload in a fresh directory. See the committed
docs/causal-evidence-timing-correction.md; fit strides and identities are verified
independently. The released analyzer also derives decoder vocabulary
width from the pinned checkpoint config (151936) instead of the original analyzer's
incorrect hardcoded 152064. That correction changes no model execution, inputs,
policy, threshold or raw quality measurement. Its source SHA256 is recorded in the analysis
verification receipt. Do not execute the archived analyzer as a replacement for the
corrected released entrypoint above.

The 2 GiB FFN working allowance now includes 256 MiB of acquisition workspace for
both dense and sparse paths, plus a 32 MiB controller reservation for sparse paths.
Earlier 2 GiB weight-cache comparisons are preserved and are not interchangeable
with this new budget. Synthetic serialized cost sums are not exposed inference
latency, and restoration is not an independent experimental sample.
