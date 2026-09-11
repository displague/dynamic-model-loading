# Cache trace archive

`run/` preserves the exact applied-mask experiment. `analysis/` preserves complete
cache replay, aggregates, and its source snapshot. `figures/` contains the full
frontier, labeled detail, and the exact figure-v2 source. `archive.json` indexes
the original copied bytes, compression, external trace asset, and local dense-logit
hashes. Model checkpoint weights are external pinned inputs.

Download `cache-traces-v0.4.0.zip` from the GitHub v0.4.0 release and check its SHA256
against `archive.json`. Copy `run/` into a new working directory. Decompress
`layouts.json.gz` and `calibration-importance.safetensors.gz` there, preserving the
decoded bytes. Extract the trace asset there so that `traces/*.npz` sits beside
`manifest.json`. Use the repository's v0.4.0 code and baseline environment:

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.cache_analysis --run <restored-run> --output <fresh-analysis>
```

`analysis/replay.jsonl.gz` is lossless gzip of the full 36,864-row ledger. Decompress
it to inspect document/cache/state records. Do not replace completed run directories.
Run metadata and SHA256 checks reject mismatched inputs, missing gates or conditions,
and altered trace payloads. The large source and output tensors remain release
assets or explicitly indexed local files rather than Git objects.

`run/report.md` retains legacy apparatus boilerplate saying no cache simulator is
implemented. The separately preserved `analysis` directory supplies this release's
simulation; that boilerplate is obsolete. Actual model measurements still use dense
operations and full residency, and the replay performs no physical transfers.
