# OPT ReLU positive-control archive

The unchanged apparatus and protocol were pushed at `cd33f1f` before both attempts.
The first attempt stopped before model loading because the local checkpoint lacked
its JSON/vocabulary files; preserve its failure and source in `preparation/`.
Fetching those files from the same pinned revision enabled the complete retry.
Neither dependencies nor thresholds changed.

`run/` preserves the 545-row ledger, inputs, exact OPT token IDs, manifest, source,
protocol and summary. `analysis-final/` independently validates the completed grid, original
numerical gates, gate ordering, hashes, denominators, bias accounting and aggregates.
`archive.json` indexes preserved bytes and the lossless layout compression. There is
no claim that archive hashes provide an independent signature of model execution.
`analysis/` and `analysis-v2/` preserve earlier validator outputs. The final version
adds dense-logit verification, reference-derived dense NLL, complete inventories,
and coupled active-neuron/short-tail feasibility checks. All versions reproduce the
same original measurement aggregates.

Restore into a fresh directory before using `python -m
dynamic_model_loading.relu_analysis --run RESTORED --output NEW_ANALYSIS`:

```python
from pathlib import Path
import gzip, shutil, zipfile
source = Path("results/relu-control-20260911/run")
target = Path("runs/restored-relu")
shutil.copytree(source, target)
target.joinpath("layouts.json").write_bytes(
    gzip.decompress(target.joinpath("layouts.json.gz").read_bytes()))
with zipfile.ZipFile("runs/relu-logits-v0.6.0.zip") as asset:
    asset.extractall(target)
```

Download the [dense logit asset](https://github.com/displague/dynamic-model-loading/releases/download/v0.6.0/relu-logits-v0.6.0.zip)
to the example path above. Verify its SHA256 against `archive.json` before extracting.
All 16 dense reference logit files are listed there with hashes and sizes. Validation
requires these files and checks their digests; it does not rescore the model.
Recomputing model quality requires the pinned OPT checkpoint and measured CUDA
environment. The adapter computes dense projections and then
masks, retaining full weights and extra restoration copies. Byte figures are
hypothetical selected FFN volume, including biases; no cache replay, transfers,
bounded residency or speed measurement is established by this control.
