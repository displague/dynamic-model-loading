# Dynamic model loading

An experimental apparatus for testing whether dense FFN computation can be grouped
and selectively executed at useful quality and memory costs.

The starting point is the final critical review in the
[shared research conversation](https://chatgpt.com/share/6aa40208-727c-83e9-a91b-b2ce031c96eb).
The executable first milestone checks packing correctness and produces optional
hindsight sparsity diagnostics. It is not yet a weight pager.

**Current result:** the BF16 reordered-layout numerical gate failed. A separate FP32
control passed and completed 18 diagnostic configurations. See
[initial findings](docs/initial-results.md) for the failure, controls, and grouping
results. The default BF16 command currently reproduces the gated experiment; it must
not be treated as a passing reference.

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

The next substantive milestone is larger, separated data plus packing and cache traces.
A learned selector and corrective refinement should follow only if those measurements
show an opportunity. CUDA timing and memory reporting follow the distinctions in
[PyTorch's CUDA notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
