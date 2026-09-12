# Dynamic model loading

Research into executing models under GPU memory pressure, with preserved FFN
sparsity experiments and a new target-scale stock speculative-decoding comparison.

The starting point is the final critical review in the
[shared research conversation](https://chatgpt.com/share/6aa40208-727c-83e9-a91b-b2ce031c96eb).
The executable first milestone checks packing correctness and produces optional
hindsight sparsity diagnostics. It is not yet a weight pager.

See the [research stages and delivery history](docs/plan.md),
[release notes](docs/releases/), and
[GitHub milestones](https://github.com/displague/dynamic-model-loading/milestones)
for the original plan, completed experiments, and remaining gates.

**Current direction:** [ADR 0003](docs/adr/0003-verified-speculation-boundary.md)
retires the tested per-token selective-execution path as the primary implementation.
The [completed stock comparison](docs/stock-speculation-results.md) measures an
observed 18.10 emitted tokens/s with a 0.5B draft at length four versus 8.52 for
target-only Qwen2.5-32B-Instruct Q4_K_M. Every nonbaseline configuration differs
from the fixed reference on sustained generation, including placement/thread
controls; all 260 scalar-smoke comparisons match. This does not yet establish
identical-output acceleration. [Execution-path fidelity #27](https://github.com/displague/dynamic-model-loading/issues/27)
is the next bounded investigation; a custom runtime remains deferred. See
[v0.14.0](docs/releases/v0.14.0.md) and the [frozen protocol](docs/stock-speculation-protocol.md).

**Preserved v0.12 result:** the [complete causal-repair experiment](docs/causal-evidence-results.md)
finds a mixed quality improvement from partial evidence at matched bytes: relative
PPL 1.089487, versus 1.112546 for a larger one-shot
decision and 1.107101 for predetermined repair. Individual-prefix
advantage and the combined quality/economics gate fail. The old pager is not pursued under ADR 0003.
See [v0.12.0](docs/releases/v0.12.0.md) for complete costs and retained failures.

The [dense chat-interface study](docs/dense-interface-results.md) recovers 7/10 debug
and 12/20 fresh balanced successes, with all native/incremental/repacked comparisons
passing. Its frozen balanced Gate A still fails. This historical qualification does not gate the new target-scale runtime study.

**Earlier privileged repair:** the [fixed-mask repair diagnostic](docs/refinement-feasibility-results.md)
finds five privileged repair settings that meet the aggregate quality/traffic lines
on two reused article prefixes. One also passes the quality line on both individual
prefixes; none of the 20 larger one-shot settings passes both aggregate lines. This
motivated the completed v0.12 causal test, with no runtime nominee or achieved memory gain.

The [hardware-cost characterization](docs/hardware-cost-results.md) separates transfer,
staging, gathering and resident computation. At a full-layer payload, its serialized
acquisition/packing/FFN path costs 15.924 ms wall time versus 0.566 ms for resident FFN
execution; these synthetic primitives do not establish end-to-end speed.

The [balanced development control](docs/balanced-development-results.md) records ten
fixed code, extraction, arithmetic, copying and topic-change tasks on Qwen and OPT,
with their dense baselines, exact target scores and every generated continuation.
This small control does not establish held-out agent quality or persistent state.

All four initial causal selectors fail the frozen development
quality screen at the nominated popularity/width-8/90% condition. Relative PPL ranges
from 1.089204 to 1.158203 against the 1.01 limit. All meet the simulated traffic
screen, but recency, EMA and the learned selector also exceed the cost screen.
Incremental hindsight reaches 1.009318 and remains ineligible because it uses current
dense activations. The [causal findings](docs/causal-control-results.md) retain every
article, transition, timing and generated path. The [historical dependency graph](docs/plan.md#revised-dependencies-after-the-v07-design-review)
permitted analytical refinement from failed first passes. ADR 0003 supersedes that
implementation priority; no physical transfer or runtime gain was established.

The earlier [cache findings](docs/cache-trace-results.md) nominated one of 72
conditions: relative PPL 1.008978 and 15.54% less simulated warm traffic at 2 GiB,
with 9/16 articles individually above 1% PPL increase. See also the
[layer study](docs/layer-study-results.md), and [packing pilot](docs/packing-pilot-results.md)
for the full curves and preserved negative results. The
[initial BF16 numerical failure](docs/initial-results.md) remains unresolved by a
practical execution policy. The [same-interpreter PyTorch 2.12 control](docs/torch212-results.md)
also preserves the failure; the default BF16 command still reproduces it. The
[OPT ReLU positive control](docs/relu-control-results.md) passes, with 96.04% exactly
zero neuron activations; wider physical groups consume more selected volume. This
architecture control does not establish equivalent sparsity in SwiGLU models.

## Run locally

The current Windows `.venv` uses Python 3.14.3 and inherits the existing installation's
PyTorch 2.10.0+cu130 and Transformers 5.13.1. No global packages were changed.
Every run records the actual dependency versions and CUDA environment.

```powershell
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m dynamic_model_loading.experiment --output runs/stage0
.\.venv\Scripts\python.exe -m dynamic_model_loading.experiment --diagnostics --output runs/diagnostics
.\.venv\Scripts\python.exe -m dynamic_model_loading.experiment --config configs/float32-control.json --diagnostics --output runs/fp32-control
```

The checkpoint is cached locally. On another machine add `--allow-download`, or use:

```powershell
hf download Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --include "*.json" "*.safetensors" "*.txt" "*.jinja"
```

For a fresh isolated environment, install an appropriate CUDA build of PyTorch using
the [official installation guide](https://pytorch.org/get-started/locally/), then install
this project with `pip install -e ".[dev]"`. Native Windows works for this milestone.
`--device cpu` is available explicitly. The default never silently falls back from CUDA.

Use a new output directory each time. A failed correctness gate exits with code 2 and
blocks sparsity diagnostics. See [the protocol](docs/protocol.md) for frozen numerical
tolerances and exact metric definitions. The small bundled corpus is a smoke fixture,
not a quality benchmark.

Each run contains:

- `started.json`, `manifest.json`: environment and source/checkpoint hashes.
- `config.json`, `corpus.jsonl`, `protocol.md`, `layouts.json`: the experiment inputs.
- `results.jsonl`: flushed raw measurements, including failures.
- `summary.json`, `report.md`: compact results written after raw measurements.

## Models and scope

The first actual checkpoint run uses
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), a small
SwiGLU model. The code also has FFN adapters for Granite and LFM2, tested with tiny
random instances; that is not validation of their full pretrained checkpoints.

Your local inventory includes LM Studio's Granite-4.1-3B (2.10 GB), LFM2.5-2.6B
(1.94 GB), and Llama-3.2-1B (1.32 GB), plus Ollama's Qwen3-4B-Instruct (2.5 GB), as
reported by their CLIs on 2026-09-11. These disk sizes describe the installed formats,
not BF16 parameter storage or total runtime memory. Existing GGUF models remain useful
deployment comparisons; this instrumentation uses native Hugging Face tensors.

[Granite-4.1-3B](https://huggingface.co/ibm-granite/granite-4.1-3b) uses conventional
SwiGLU FFNs. LFM2.5-2.6B also has SwiGLU FFNs: `w1` is the gate, `w3` the up projection,
and `w2` the down projection. Its hybrid convolution/attention mixing does not remove
those FFNs. See its [configuration](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/config.json)
and [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py).

The adapters leave the architecture's normalization, scaling, attention, and recurrent
state in place. Unsupported models or projection biases fail explicitly.

## What the numbers mean

The diagnostic hook computes dense gate/up activations, scores neurons using
`abs(z) * norm(W_down_column)`, and masks the dense down projection. It compares native,
random, and calibration-popularity layouts. It does not perform sparse GPU execution.
Hypothetical selected weight bytes are not measured PCIe transfers or memory savings.

The packing pilot additionally supports calibration-only co-activation signatures and
capacity-constrained grouping. It still has no bounded-cache or transfer runtime.
The first causal selector experiment is complete with no qualifying candidate.
The [repair feasibility protocol](docs/refinement-feasibility-protocol.md) tests
whether complete corrected outputs can recover within the traffic allowance; its
privileged diagnostics cannot qualify a runtime. CUDA timing and memory reporting follow the distinctions in
[PyTorch's CUDA notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html).

## Continue the experiment

[The pilot protocol](docs/packing-pilot-protocol.md) fixes the development data recipe,
numerical controls, grouping method, and exploratory comparison before measurements.
Install the optional analysis dependencies with `pip install -e ".[analysis]"` in the
project environment. Download only WikiText training and validation data:

```powershell
hf download Salesforce/wikitext --repo-type dataset --revision b08601e04326c79dfdd32d625aee71d232d685c3 --include "wikitext-2-raw-v1/train-*.parquet" "wikitext-2-raw-v1/validation-*.parquet"
.\.venv\Scripts\python.exe -m dynamic_model_loading.precision --output runs/precision
.\.venv\Scripts\python.exe -m dynamic_model_loading.corpus --output runs/pilot-corpus
.\.venv\Scripts\python.exe -m dynamic_model_loading.experiment --config configs/packing-pilot.json --corpus runs/pilot-corpus/corpus.jsonl --diagnostics --output runs/packing-pilot
.\.venv\Scripts\python.exe -m dynamic_model_loading.analysis --run runs/packing-pilot --output runs/packing-figures
```

The corpus builder selects distinct articles and records source hashes, titles, and
licensing. The test split is unused. The larger run spills reference logits to disk
(about 2.5 GB for the default slice); these are analytical reference files, not weight
offload traffic. New runs snapshot their source and protocol as well as their inputs.

The [per-layer study](docs/layer-study-protocol.md) reuses the archived pilot inputs
and layout hashes. Its 66 conditions mask individual layers, all layers, and four
fixed blocks while charging unmasked FFNs at full hypothetical weight volume:

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.layer_study --output runs/layer-study
.\.venv\Scripts\python.exe -m dynamic_model_loading.layer_plot --run runs/layer-study --output runs/layer-figures
```
