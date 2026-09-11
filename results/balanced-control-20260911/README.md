# Balanced development archive

This archive retains all 138 raw rows, all 80 teacher/generated applied-mask traces,
the frozen ten-task corpus, exact token IDs, source/configuration/protocol snapshots
and independent analysis. Two v0.10.0 release ZIPs hold 132 tensor payloads: 20 native,
20 layout and 40 masked teacher-forced logit files, plus 52 local reconstruction pairs.
Every file is indexed by size and SHA-256 in `archive.json`.

Both numerical model runs complete. Dense exact target/format compliance is 0/10 for
both models; the task endpoint cannot establish preserved useful agent behavior.
No criteria, prompts or accepted outputs were changed after scoring. See the
[findings](../../docs/balanced-development-results.md) and
[all scored continuations](../../docs/balanced-development-generations.md).

Use the recorded baseline Python/PyTorch/Transformers environment and a clone containing
the source commit in Git history. Analysis requires the pinned checkpoints locally
for catalog hashes and tokenization; it does not execute either pretrained model.
The earlier Qwen/OPT parent manifests and layouts are already tracked in Git.

```powershell
$env:HF_HUB_CACHE=(Join-Path (Get-Location) 'runs/balanced-hf-cache')
hf download Qwen/Qwen2.5-1.5B-Instruct config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
hf download facebook/opt-1.3b config.json merges.txt pytorch_model.bin README.md special_tokens_map.json tokenizer_config.json vocab.json --revision 3f5c25d0bc631cb57ac65913f76e22c2dfb61d62
gh release download v0.10.0 --repo displague/dynamic-model-loading --pattern 'balanced-logits-*.zip' --dir runs/balanced-assets
.venv/Scripts/python.exe results/balanced-control-20260911/restore.py --assets runs/balanced-assets --output runs/balanced-restored
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -m dynamic_model_loading.balanced_analysis --run runs/balanced-restored --output runs/balanced-reanalyzed
```

Catalog validation requires exactly the files recorded in each parent manifest;
the fresh task-specific cache and explicit filename lists avoid unrelated formats.
The published run used the pre-existing cached snapshots. Artifact restoration validates
all Git files and ZIP payloads before writing a fresh run directory.
