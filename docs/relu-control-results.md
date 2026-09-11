# OPT ReLU positive-control findings

The pinned FP32 OPT-1.3B control shows substantial **exact activation sparsity**:
96.0373% of neuron observations are zero on these 16 development articles. Selecting
every group containing a nonzero activation preserves outputs within the original
numerical limits. Physical grouping substantially reduces that opportunity, while
calibration-only popularity packing preserves more of it than native or random order.
This is the original plan's architecture control, not evidence of equivalent sparsity
in Qwen's SwiGLU FFNs and not a sparse runtime.

Protocol and apparatus commit `cd33f1f` preceded measurement. The checkpoint is
`facebook/opt-1.3b` at `3f5c25d0bc631cb57ac65913f76e22c2dfb61d62`, with 24 biased
2048-to-8192-to-2048 ReLU FFNs. The official PyTorch weight file has SHA256
`cf7d5c970d6ddbd3b03009b397c0422e147edd5c8020d47a8d2fac0b11a3b08d`.
Python 3.14.3 / torch 2.10.0+cu130 / Transformers 5.13.1, FP32 and SDPA on the RTX 5080
Laptop GPU remain the measured environment. The first attempt failed before loading
the model because pinned vocabulary/JSON files were missing from the local cache.
Fetching those files enabled the unchanged runner; the original failure is archived.

The input text is the existing 80-calibration / 16-development Wikipedia corpus,
already truncated using Qwen tokenization. OPT tokenization without added special
tokens yields 19,054 calibration tokens, 3,844 development input tokens and 3,828
next-token predictions. These are within-OPT comparisons; absolute perplexity across
the two tokenizers is not comparable. No final held-out documents were scored.

All 72 local reconstruction and 48 full-layout checks pass. Maximum local relative
L2 is 3.3455e-7; maximum full-layout logit L2 is 3.4171e-6 and KL 5.7206e-8. All 192
exact-zero document checks pass before the 192 fixed-75% probes begin. Native exact
zero masking is bitwise equal to its dense reference; random and popularity results
retain the tiny arithmetic differences introduced by permutation. All exact-zero
conditions have 100% top-1 agreement. Published PPL values rounded to 1 are not a
claim of bitwise equality for the permuted layouts.

## Complete condition grid

Fraction includes both projections, input bias for selected neurons, and the output
bias charged once per layer and token. Exact-zero selects every group with any
nonzero ReLU activation. Fixed-75% selects ceil(0.75 times group count), including
zero-importance groups when fewer groups are active. It does not mean 75% of active
groups. Both modes first compute the full dense activation.

| Probe | Layout | Width | Selected FFN bytes (%) | KL | Relative PPL | Top-1 agreement (%) |
|---|---|---:|---:|---:|---:|---:|
| exact_zero | native | 1 | 3.9685 | 0.00000000 | 1.00000000 | 100.0000 |
| exact_zero | native | 8 | 25.6226 | 0.00000000 | 1.00000000 | 100.0000 |
| exact_zero | native | 32 | 59.1472 | 0.00000000 | 1.00000000 | 100.0000 |
| exact_zero | native | 128 | 85.0444 | 0.00000000 | 1.00000000 | 100.0000 |
| exact_zero | random | 1 | 3.9685 | 0.00000005 | 0.99999994 | 100.0000 |
| exact_zero | random | 8 | 25.6475 | 0.00000005 | 0.99999994 | 100.0000 |
| exact_zero | random | 32 | 59.1513 | 0.00000005 | 0.99999994 | 100.0000 |
| exact_zero | random | 128 | 85.1267 | 0.00000005 | 0.99999994 | 100.0000 |
| exact_zero | popularity | 1 | 3.9685 | 0.00000005 | 0.99999986 | 100.0000 |
| exact_zero | popularity | 8 | 19.8017 | 0.00000005 | 0.99999986 | 100.0000 |
| exact_zero | popularity | 32 | 40.0992 | 0.00000005 | 0.99999986 | 100.0000 |
| exact_zero | popularity | 128 | 60.3959 | 0.00000005 | 0.99999986 | 100.0000 |
| approximate_75 | native | 1 | 75.0015 | 0.00000000 | 1.00000000 | 100.0000 |
| approximate_75 | native | 8 | 75.0015 | 0.00009121 | 1.00016532 | 99.8433 |
| approximate_75 | native | 32 | 75.0015 | 0.00524767 | 1.00603536 | 96.6301 |
| approximate_75 | native | 128 | 75.0015 | 0.06660990 | 1.07478095 | 85.0836 |
| approximate_75 | random | 1 | 75.0015 | 0.00000005 | 0.99999994 | 100.0000 |
| approximate_75 | random | 8 | 75.0015 | 0.00009960 | 0.99982789 | 99.9739 |
| approximate_75 | random | 32 | 75.0015 | 0.00548185 | 1.00684658 | 96.5778 |
| approximate_75 | random | 128 | 75.0015 | 0.06717838 | 1.06906311 | 85.0575 |
| approximate_75 | popularity | 1 | 75.0015 | 0.00000005 | 0.99999986 | 100.0000 |
| approximate_75 | popularity | 8 | 75.0015 | 0.00007537 | 0.99981062 | 99.9216 |
| approximate_75 | popularity | 32 | 75.0015 | 0.00143093 | 1.00188220 | 98.6416 |
| approximate_75 | popularity | 128 | 75.0015 | 0.01690355 | 1.01411177 | 93.8871 |

At width 128, exact-zero popularity packing requires 60.40% of FFN bytes; native
requires 85.04%. A fixed 75% selection can still omit active groups on particular
layer/token visits even when mean exact-zero volume is below 75%. Consequently,
width-128/popularity fixed-75% changes outputs: relative PPL 1.014112, KL 0.016904,
and top-1 agreement 93.887%. Do not infer safety from average selected volume.

## Descriptive comparison with the earlier Qwen curves

At the same nominal 75% group-retention target, these OPT controls change relative PPL less
than the earlier Qwen probes. This comparison uses each model's own dense reference;
it does not compare absolute perplexity or establish an architecture-only cause.

| Layout | Width | Qwen relative PPL | OPT relative PPL | Qwen KL | OPT KL |
|---|---:|---:|---:|---:|---:|
| native | 1 | 1.000341 | 1.000000 | 0.002150 | 0.000000 |
| native | 8 | 1.069628 | 1.000165 | 0.085004 | 0.000091 |
| native | 32 | 1.184640 | 1.006035 | 0.204588 | 0.005248 |
| native | 128 | 1.367578 | 1.074781 | 0.348355 | 0.066610 |
| random | 1 | 0.999971 | 1.000000 | 0.002153 | 0.000000 |
| random | 8 | 1.072287 | 0.999828 | 0.087731 | 0.000100 |
| random | 32 | 1.166276 | 1.006847 | 0.204791 | 0.005482 |
| random | 128 | 1.358706 | 1.069063 | 0.346282 | 0.067178 |
| popularity | 1 | 1.000113 | 1.000000 | 0.002173 | 0.000000 |
| popularity | 8 | 1.050740 | 0.999811 | 0.073388 | 0.000075 |
| popularity | 32 | 1.128710 | 1.001882 | 0.154117 | 0.001431 |
| popularity | 128 | 1.225246 | 1.014112 | 0.244443 | 0.016904 |

Rounding matters at width 128: Qwen retains ceil(0.75 * 70) = 53/70 groups
(75.714286%), whereas OPT retains 48/64 (75%). Widths 1/8/32 retain exactly 75%
of groups in both models. OPT's bias-inclusive selected byte fraction is separately
75.001525%; group retention and total FFN-byte fraction are distinct quantities.

Qwen is a 28-layer gated SwiGLU model with 1,543,714,304 parameters; OPT has
24 two-projection biased ReLU FFNs and 1,315,758,080 parameters. Their checkpoints,
training, attention and tokenizers differ. Both use the same 80/16 article text
split, already truncated using Qwen tokenization, but Qwen has 20,480 calibration
and 4,080 predicted development tokens versus OPT's 19,054 and 3,828. Layouts are
fitted separately from calibration in each model. Both are FP32 development probes;
neither establishes held-out task utility or a runtime saving. Width-1 Qwen values
come from the [packing pilot](../results/packing-pilot-20260911/pilot/summary.json);
widths 8/32/128 come from the [cache study](../results/cache-trace-20260911/run/summary.json).
Overlapping Qwen width-32/128 rows are identical between those archives. All OPT
values come from the complete control grid above.

Model parameter storage is 5,263,032,320 bytes, of which 3,222,011,904 are groupable
FFN weights/input biases. Fixed FFN output biases total 196,608 bytes. The adapter
also retains 3,222,011,904 bytes of pre-layout restoration copies and up to
268,500,992 bytes of per-layer transaction payload. These figures describe tensor
payloads, not allocator peaks. Fixed-75% FFN volume is 75.001525%, slightly above 75%
because output biases are always included. No weights were actually paged or skipped.

The independent validator reproduces all 24 aggregates from 384 document rows,
checks all 120 numerical gates, source/input/layout hashes, gate ordering, and
bias/short-tail accounting, complete environment/source inventories and reference-derived dense NLL. The archive contains 545 raw rows; dense reference
logits are a release asset with every payload hash listed in the archive index. The 89-test suite
includes tiny real OPT loading, bias/tail reconstruction, interruption restoration,
blocked probes, and rejection of altered receipts. Three pre-measurement reviews
included 153 additional CPU fault-injection cases. Publication review and a clean
candidate full suite remain release checks, not new model evaluations.

Evidence: [protocol](relu-control-protocol.md), [archive](../results/relu-control-20260911),
[release note](releases/v0.6.0.md). This closes the bounded ReLU control (#6). Practical
BF16 execution (#4), held-out multi-domain quality (#8), causal selection (#10),
physical paging (#11) and corrective refinement (#12) retain their own gates. The
next causal experiment uses the already nominated Qwen condition; this ReLU result
does not replace that model or tune that operating point.
