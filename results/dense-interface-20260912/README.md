# Restore and analyze the dense interface experiment

The Git archive contains the raw 120-row ledger, source/configuration/corpus
snapshots, model catalog, summaries and console receipt. The three release ZIPs
contain 120 safetensors payloads with own-trajectory and aligned-prefix logits.
`archive.json` lists exact container and payload sizes and SHA256 identities.

Run from the repository root with Python 3.14.3 and the declared analysis packages
(PyTorch 2.10.0+cu130, Transformers 5.13.1, safetensors 0.8.0, huggingface-hub 1.19.0,
NumPy 2.3.5). The analysis loads tensors/tokenizers on CPU; it does not execute
pretrained weights. The strict catalog includes the checkpoint weights so that
the exact pinned snapshot identity is verified. Use a fresh cache for these exact
files if an existing cache includes additional files.

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:HF_HUB_CACHE=(Join-Path (Get-Location) 'runs/interface-hf-cache')
hf download Qwen/Qwen2.5-1.5B-Instruct config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
gh release download v0.11.0 --repo displague/dynamic-model-loading --pattern 'dense-interface-part*-v0.11.0.zip' --dir runs/interface-assets
.venv/Scripts/python.exe results/dense-interface-20260912/restore.py --assets runs/interface-assets --output runs/restored-interface
.venv/Scripts/python.exe -m dynamic_model_loading.dense_interface --run runs/restored-interface --output runs/restored-interface-analysis
```

The restorer verifies every indexed Git file, ZIP and payload before writing to a
fresh directory. The analyzer verifies archived source against the recorded Git
commit, frozen corpora and checkpoint, full comparison grids, greedy/EOS behavior,
scoring, finite observations, full retention and aligned-prefix metrics.

Expected outcome: 7/10 debug and 12/20 fresh native chat successes, 0/10 matched
plain successes, reference fidelity true, utility qualification false, Gate A
false and no runtime nomination. See `analysis/summary.json` for all nine path
conditions. This archive preserves a failed qualification experiment; restoring
it does not make a new independent experimental sample.
