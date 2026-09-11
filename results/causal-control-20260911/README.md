# Causal selector development archive

The protocol and apparatus were pushed as `85efbc7` before pretrained fitting or
evaluation. `run/` preserves the complete ledger, exact inputs, source snapshot,
configuration, protocol and summary. `analysis/` independently reconstructs the
quality aggregates, charged cache replay, timing medians and frozen decision.
`figures/` contains the standalone plot, its source and hashed input summary.

Large tensor/mask payloads are release assets. Two logit bundles preserve all 17
native dense reference episodes; the third bundle contains the 85 document traces,
20 generated-path traces, calibration sufficient statistics and fitted predictor.
Every asset and every uncompressed payload has a size and SHA256 in `archive.json`.
Git hashes establish the published archive identity; internal hashes alone are not
an independent attestation of model execution.

Download all three assets from [v0.7.0](https://github.com/displague/dynamic-model-loading/releases/tag/v0.7.0)
into `runs/`, then restore into a fresh directory and verify their contents:

```python
from pathlib import Path
import hashlib, json, shutil, zipfile

def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

archive = Path("results/causal-control-20260911")
index = json.loads((archive / "archive.json").read_text(encoding="utf-8"))
target = Path("runs/restored-causal")
shutil.copytree(archive / "run", target)
for asset in index["assets"]:
    path = Path("runs") / asset["name"]
    assert path.stat().st_size == asset["bytes"]
    assert sha256(path) == asset["sha256"]
    with zipfile.ZipFile(path) as bundle:
        assert set(bundle.namelist()) == {p["file"] for p in asset["payloads"]}
        for payload in asset["payloads"]:
            relative = Path(payload["file"])
            assert not relative.is_absolute() and ".." not in relative.parts
            value = bundle.read(payload["file"])
            assert len(value) == payload["bytes"]
            assert hashlib.sha256(value).hexdigest() == payload["sha256"]
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
```

Recompute the analysis without executing the pretrained model:

```powershell
.venv/Scripts/python.exe -m dynamic_model_loading.causal_analysis --run runs/restored-causal --parent results/cache-trace-20260911/run --output runs/restored-causal-analysis
```

The pinned v0.4 parent archive remains a required input. Repeating model execution
also requires the pinned Qwen checkpoint and measured CUDA environment. Use the
archived configuration and protocol with the causal-study CLI in a new output
directory. Normal Git source files use LF, while copied run receipts preserve their
original bytes, including line endings.

Cold/warm figures describe simulated FFN-cache traffic with selector tensors already
initialized. Initialization traffic, actual transfers, allocator peaks, KV and other
runtime workspaces are not measured by this cache replay. Timing covers query and
necessary selected-observation update relative to resident dense FFNs. The complete
model and restoration copies remain resident in this diagnostic. Generated paths
compare token sequences, not external task accuracy; local omission norms do not
define a validated unsafe-omission detector.
