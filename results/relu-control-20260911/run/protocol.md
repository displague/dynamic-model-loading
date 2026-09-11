# OPT-1.3B ReLU architecture control

Prospective Stage 0 positive control for issue #6, planned delivery v0.6.0. This
addresses the original plan's architecture sanity check; it does not generalize
ReLU sparsity to modern gated models or establish a weight-paging runtime.

Use `facebook/opt-1.3b` revision `3f5c25d0bc631cb57ac65913f76e22c2dfb61d62`, FP32,
the existing Python 3.14.3 / torch 2.10.0+cu130 / Transformers 5.13.1 baseline, SDPA,
four CPU threads, TF32 disabled, and seed 1729. Load the official checkpoint with
`trust_remote_code=False` and `weights_only=True`; preserve file hashes. The pinned
configuration declares ReLU, 24 layers, hidden dimension 2048 and FFN width 8192.
Retain OPT's attention, normalization, residual connections and other architecture.

Reuse the packing pilot's 80 calibration and 16 development article texts, pinned
by corpus SHA256. Retokenize for OPT without a chat template or added special tokens,
at most 256 tokens per text, and save exact token IDs before measurement. These
texts were already truncated for the earlier Qwen corpus; different tokenization
means this is a within-model relative-quality comparison, not a comparison of
absolute perplexities or a new held-out evaluation.

OPT uses two biased projections. A physical permutation must reorder `fc1` weight
rows and bias entries, and matching `fc2` weight columns. Keep `fc2`'s output bias
fixed and add it once in grouped reconstruction. Charge each selected neuron for
both projection vectors and its `fc1` bias; charge the entire output bias once per
FFN input position, independently of selection. Record fixed non-groupable parameters
separately. Do not reinterpret OPT as a three-matrix SwiGLU module.

Fit popularity order using calibration-only mean `abs(z) * norm(fc2_column)` after
ReLU. Compare native, seed-fixed random, and popularity layouts. Require every layer's
all-group reconstruction at width 128 and every development document's full-model
permutation to pass the original relative-L2 <=0.01 / KL <=0.001 numerical limits.
Any failure blocks every sparsity probe, and incomplete or interrupted runs retain
failure receipts. Restore physical permutations and remove hooks on errors.
This correctness apparatus retains original groupable FFN tensors during a layout
scope and uses temporary per-layer transaction copies; record these extra payload
bytes. Restore every layer before propagating a recoverable cleanup interruption.
After a persistent device/copy failure, attempt the remaining restorations and fail
the run explicitly. These diagnostic copies are not a bounded-memory implementation.

After those gates, evaluate widths 1/8/32/128 under two predeclared probes, for
24 layout/width/probe conditions across all 16 development articles:

1. **Exact-zero selection:** select every group containing any nonzero ReLU output.
   Whole groups whose activations are all zero can be omitted algebraically. The
   applied dense mask must pass the original numerical gates on every document.
   Record actual zero-neuron and all-zero-group rates, per layer and in aggregate.
2. **Approximate 75% retention:** select ceil(0.75 * group count) groups by summed
   contribution importance with stable group-index ties, using the same masking
   definition as the Qwen apparatus. Preserve every KL, NLL, relative PPL, and top-1
   result, including any failures or improvements. No new quality-success threshold
   is adopted from this control.

Run every exact-zero condition before approximate omission. If any exact-zero
document fails the numerical limits, preserve its controls and block all approximate
conditions rather than treating an invalid positive control as sparsity evidence.

Both probes are hindsight diagnostics: `fc1` runs densely before selection, and the
masked `fc2` still runs densely. Selected bytes are hypothetical volume, not fetched
traffic or reduced residency. Exact-zero skipping is a proposed positive control,
not an assumed rate of usable physical-group sparsity. Save all raw document and
layer accounting before summaries; do not drop unfavourable groups or layers.

Compare the approximate 75% curves descriptively with the earlier Qwen results while
retaining differences in architecture, tokenization, parameter counts, and corpus
use. The exit gate is trustworthy architecture-specific reconstruction and diagnostic
evidence. A useful modern-model operating point, causal prediction, held-out quality,
and actual bounded execution remain separate experiments.
