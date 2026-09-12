# Run the stock comparison

Use the [prospective protocol](stock-speculation-protocol.md) and pinned catalogue.
The Python scripts orchestrate the unmodified stock server. They do not implement
model kernels, sparse verification or a custom shared-state runtime.

Install the project's `native` extra when needed; this machine already has psutil
7.2.2 and NumPy available. The baseline PyTorch environment is unchanged. Obtain the
exact model files and Windows CUDA 13.3 release archives listed in
`configs/stock-speculation-artifacts.json`; retain one explicit model directory with
subdirectories `target`, `draft05`, `draft15` and `draft32`. Do not merge target shards.

Before inference, verify the files and tokenizer metadata:

```powershell
.venv/Scripts/python.exe scripts/stock_artifacts.py --models runs/stock-models --binary runs/llama-b10919 --llama-source runs/llama-b10919-source --output runs/stock-artifacts.json
```

The measurement source, protocol, inputs and scripts must be reviewed, committed
and pushed. Keep the execution worktree clean and on the same source revision for
calibration and evaluation; analysis can use a separate checkout afterward. The
runner checks ancestry against `origin/experiment/stock-speculation` and rejects
uncommitted code. All output directories must be new. Raw model responses and HTTP
errors are retained before interpretation. A runner error stops the driver; only
an observed resource failure can make a placement infeasible.

```powershell
.venv/Scripts/python.exe scripts/stock_study.py calibration --models runs/stock-models --binary runs/llama-b10919 --output runs/stock-calibration
.venv/Scripts/python.exe scripts/stock_study.py evaluation --models runs/stock-models --binary runs/llama-b10919 --selection runs/stock-calibration/selection.json --output runs/stock-evaluation
```

Use absolute interpreter/model/runtime paths when running from an isolated source
worktree. The native server's working directory is the verified runtime directory;
unexpected DLLs/executables there fail validation. Explicit CUDA devices and actual
offload logs establish layer placement. The stock input embedding remains on the
CPU even when every transformer/output layer is on the GPU. Both processes use
Windows Job Objects so cancellation cleans up the whole child tree.

The driver records its finite placement/thread search, frozen choice, evaluation
ordering and every attempted run. It preserves a target-only reference before the
three sustained repetitions and separate scalar smoke comparisons. Input manifests
must cover every expected case. Calibration source, ledger and selected runs are
checked before evaluation. Per-request files include raw HTTP, actual generated IDs,
native timings, stock acceptance events, metrics and sampled host/GPU resource use.

`completion.json` means that the runner completed its cases, not that outputs match
the reference or that speculation is faster. Acceptance telemetry reports no
survival curve when counters disagree or checkpoint replay prevents reconciliation.
Output fidelity, first-divergence investigation and performance comparison remain
analysis obligations. The first token comes from prefill: keep native decode-step
throughput, emitted-token/decode-time ratio and full-request throughput distinct.
