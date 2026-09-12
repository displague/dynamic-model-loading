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
and pushed. Keep the execution worktree clean and on the final calibration source
revision for evaluation; analysis can use a separate checkout afterward. The
runner checks ancestry against `origin/experiment/stock-speculation` and rejects
uncommitted code. All output directories must be new. Raw model responses and HTTP
errors are retained before interpretation. Runner errors stop the driver. Resource
failures and the protocol's separately classified seven-layer native startup abort
exclude a configuration without conflating their causes.

```powershell
.venv/Scripts/python.exe scripts/stock_study.py calibration --models runs/stock-models --binary runs/llama-b10919 --output runs/stock-calibration
.venv/Scripts/python.exe scripts/stock_study.py evaluation --models runs/stock-models --binary runs/llama-b10919 --selection runs/stock-calibration/selection.json --output runs/stock-evaluation
```

For the documented startup-abort continuation, use `calibration --resume-from`
with the original unfinished calibration directory and a fresh `--output` directory.
The driver verifies and copies each prior case, retains the original ledger/log,
and runs only unvisited grid points. Benchmark and input hashes must match; source
ancestry records orchestration changes. The original failed run is never overwritten.

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

After the full evaluation driver completes, analyze its immutable receipts from a
separate checkout. The analyzer rejects incomplete matrices and different input or
source provenance. It preserves per-request/per-repeat rates, stop conditions,
token differences, resource exclusions and reconciled cycle distributions. It
withholds an aggregate rate comparison when sustained coverage is incomplete.
An observed rate ratio on divergent continuations is not an exact-output speedup.

```powershell
.venv/Scripts/python.exe scripts/analyze_stock.py --evaluation runs/stock-evaluation --output runs/stock-analysis.json
.venv/Scripts/python.exe scripts/stock_replay.py --analysis runs/stock-analysis.json --evaluation runs/stock-evaluation --models runs/stock-models --binary runs/llama-b10919 --output runs/stock-replay
```

Commit and push the replay source before running it. Each distinct first-divergence
prefix is rebuilt under target-only at the baseline and candidate placements and
under the candidate configuration. Identical prefixes/configurations are deduplicated
while retaining every originating observation. These separate untimed one-token
requests expose the top ten pre-sampling log probabilities; they do not reproduce
the original speculative verification batch shape or certify correct historical KV
state. Preserve unresolved discrepancies. No full-vocabulary KL follows from these
top probabilities. Diagnostic responses, resource samples and failures are retained.

The request wall interval covers the non-streaming `/completion` request with
already tokenized input. Template rendering, separate tokenization/metrics calls,
artifact hash verification and process startup are outside it. Native prefill time
is retained; this harness does not measure client-observed streaming time to first
token. GPU offload logs and sampled device use do not establish that Windows never
evicts an allocation, and logical process I/O counters do not identify physical
storage reads. These limits also apply when the sampled memory bounds pass.
