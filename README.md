# Dynamic model loading

Latest research: [v0.31 output-specialist prerequisite](docs/specialist-screen-results.md).
The next six-release course begins with a negative: learned general and domain
heads each match the target on 34/48 diagnostic positions versus the untouched
draft's 37/48. No specialist pager follows; sparse architecture is the next distinct
question. [Course and cited inspirations](docs/residency-research-course.md).

Previous course: [v0.30 physical acquisition result](docs/risk-screen-results.md).
The six-release course is delivered. A real risk-directed draft runtime preserves
every committed target output but accepts 14/20 proposals versus fixed's 14/16 and
all35's 16/16. It spends more bytes per accepted token and loses to all35 on time.
Stop this candidate; no larger-model speedup, long suite or native port follows.

Component evidence remains: [v0.29's binary-risk win](docs/acquisition-screen-results.md),
[v0.28's geometry pass/temporal failure](docs/recursive-screen-results.md), and
[v0.27's static-field negative](docs/spatial-screen-results.md). The
[course summary](docs/information-research-course.md) keeps these distinct from
physical acquisition and deployable performance.

Research into novel weight-acquisition mechanisms for inference under smaller
memory budgets. Stock llama.cpp configurations are practical baselines, not the
project’s research objective.

The starting point is the final critical review in the
[shared research conversation](https://chatgpt.com/share/6aa40208-727c-83e9-a91b-b2ce031c96eb).
The first restored experiment implements a physical, side-indexed FFN draft pager in
PyTorch, with dense target verification. Its [prospective protocol](docs/fault-pager-protocol.md)
and [implementation amendment](docs/fault-pager-amendment-1.md) freeze the inputs,
resource accounting and comparisons before measurement. This 1.5B FP32 study does
not by itself establish a large-model speedup or justify a native implementation.

**Preserved result:** [v0.26's output-margin sensor screen](docs/output-sensors-results.md)
finds selective-acquisition headroom in a controlled two-bit final FFN. A paid
17-page oracle repairs all nine diagnostic base/high-precision disagreements;
fixed17 repairs seven. Both introduce two new disagreements. All projected-margin
and numerical/KV checks pass in **2 minutes 25 seconds**. This admits a short
spatial-estimation test, not an online loading win or capacity/speedup claim.

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.output_sensors --output runs/output-sensors-<fresh-name>
```

**Preserved result:** [v0.25's downstream intervention screen](docs/decision-field-results.md)
finishes in **1 minute 58 seconds**. Additive predictions match all 48 paired
diagnostic argmaxes, but none of the tested singleton/pair loads repairs the one
base disagreement. This four-region action set stops; no predictor is fitted or
runtime gain claimed. The [six-release course](docs/information-research-course.md)
records the subsequent finer observation, estimation and physical-policy tests.

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.decision_field --output runs/decision-field-<fresh-name>
```

**Preserved result:** [v0.24's Bayesian acquisition screen](docs/bayes-screen-results.md)
predicts remaining local FFN log-error **30.59% more accurately** than a layer
mean, but does not establish better acquisition. In **3 minutes 9 seconds**, the
two-prompt screen finds 8/8 accepted proposals with uniform six bits versus 7/8
with uncertainty-directed precision; calibrated coverage is 50/56 blocks, below
90%. Explicit high-precision prefill/KV controls pass their state checks but have
no acceptance headroom on this subset. [All results and limitations](docs/releases/v0.24.0.md)
are preserved; no long matrix or native port follows.

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.bayes_screen --output runs/bayes-screen-<fresh-name>
```

**Preserved candidate:** [resident-base correction acquisition](docs/debt-screen-results.md)
computes a packed two-bit FFN base and physically fetches correction pages selected
by an error estimate updated after each actual correction. Its corrected screen
finishes in **2 minutes 14 seconds** including analysis. Feedback, fixed selection
and the base alone each accept **0/64 proposals**; feedback takes 24.31 seconds
versus 12.57 for the base and 23.51 for fixed selection. Exact final outputs come
from dense verification/fallback, not a faithful draft. The policy is rejected;
no long matrix or native port follows. [v0.22.0](docs/releases/v0.22.0.md) preserves
both the initial accounting failure and the prospectively corrected negative run.
This is new acquisition code and a measured hypothesis, not a global novelty claim.

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.debt_screen --output runs/debt-screen-<fresh-name>
```

**Preserved physical-pager result:** the [physical pager study](docs/fault-pager-results.md)
preserves all 72 scored outputs, but the tested policy loses: 30.2% draft acceptance,
zero LRU demand hits, and 5.3% more transferred bytes for the nominated prefetch
policy. [v0.20.0](docs/releases/v0.20.0.md) publishes the implementation and raw
receipts as a negative research result. No llama.cpp patch follows; novel loading
research remains the primary program.

**Screen first:** [v0.21's bounded subset](docs/fault-screen-results.md) detects the
same candidate's failure in **3 minutes 22 seconds**, including setup and analysis,
instead of another hours-scale matrix. Two prompts / eight output tokens retain
exact outputs but accept only 6/40 proposals and increase prefetch traffic by
5.66%. This validates a screening workflow, not a new policy or a speedup.
[ADR 0006](docs/adr/0006-screen-before-performance-matrix.md) requires short screens
before new long performance studies:

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.fault_screen --output runs/fault-screen-<fresh-name>
```

The measured worker has a fixed five-minute limit. A failed or timed-out screen
does not launch the full suite; a pass requires a separately frozen expanded study.

See the [research stages and delivery history](docs/plan.md),
[release notes](docs/releases/), and
[GitHub milestones](https://github.com/displague/dynamic-model-loading/milestones)
for the original plan, completed experiments, and remaining gates.

**Preserved practical baseline:** stock b10919, Qwen2.5-32B-Instruct Q4_K_M
with a 0.5B Q8_0 draft, resident attention/KV and 32 host-backed FFNs. Use
threshold 2, K16/p_min0 and q8_0 KV under the recorded 15,000 MiB GPU allowance.
The [continuing-turn study](docs/attention-agent-results.md) measures **23.98 seconds
for five retained turns and 0.623-second mean token-bearing TTFT**, versus 33.11
seconds and 0.776 seconds for whole-layer offload. Exact-prompt attention resets
take 172.97 seconds. Four of five answers hit the 64-token cap, and histories differ
after extraction; this is a bounded latency result, not task completion or universal
output equivalence. [v0.18.0](docs/releases/v0.18.0.md) completes the agreed queue.

**Try it locally:** run `.\local.ps1 serve`, then
`.\local.ps1 claude -Workspace C:\path\to\project` in another terminal.
The [interactive guide](docs/local-interactive.md) provides the local binary,
complete arguments and four stock comparison profiles. Claude's limited Read/Edit
loop passed the [integration check](docs/local-session-results.md); Codex's
Responses connection works but its coding check remains blocked by client policy.

The separate [v0.17 cold-request study](docs/stock-placement-results.md) measures
20.57 native decode steps/s after a populated 16K prefix, versus 15.29 for its fresh
whole-layer control. See the [configuration guide](docs/stock-long-context-configuration.md)
for complete settings and distinctions between these workloads. In-process prefix
retention is measured; complete target/draft disk restart and 32K are unqualified.

[ADR 0004](docs/adr/0004-fault-pager-research-track.md) restores novel physical
loading research as the primary program. [ADR 0005](docs/adr/0005-native-pivot-evidence-boundary.md)
requires measured policy evidence and a native-only research question before any
llama.cpp patch. [ADR 0003](docs/adr/0003-verified-speculation-boundary.md) preserves
the earlier selective-execution failures and the historical stock-program pivot.
[#27](https://github.com/displague/dynamic-model-loading/issues/27) remains open:
the earlier independent replay agrees on 256/256 long IDs and 1,527/1,536 short IDs.
No margin waiver or new numerical campaign changes those results. The new physical
pager study has its own contract; it does not rewrite those historical gates.

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
