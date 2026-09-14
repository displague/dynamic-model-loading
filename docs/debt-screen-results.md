# Resident-base correction acquisition: implemented, measured, negative

The feedback-directed acquisition candidate **fails its short screen**. It is an
implemented PyTorch loading policy, not a stock llama.cpp configuration study.
All numerical, output, physical-accounting and resource checks pass in the
corrected run, but no proposed token from any approximate control is accepted.
There is no long matrix, native port, target-scale claim or deployed speedup.

## What was built

A packed two-bit base executes all FFN neurons. Small calibration-only sketches
estimate the output correction for each paired gate/up/down page. The new
controller selects a page by predicted reduction of total projected output error,
fetches its full-precision weights from CPU into one GPU page slot, and updates
the outstanding error estimate using the actual correction before selecting again.
The maximum is four pages per FFN. Independent dense verification owns commitment.

The [protocol](debt-screen-protocol-v2.md) defines the nonlinear correction and
causal information boundary. Resident low-bit weights with residual acquisition
already have antecedents, including [DecDEC](https://arxiv.org/html/2412.20185v2).
The tested departure is feedback-directed, paired nonlinear page selection. This
is a new research candidate in this repository, not a proven global invention.

The zero-fetch base and fixed top-four sketch-ranking ablation separate policy
value from compression. Dense streaming executes all 35 full pages. All four
conditions retain and pay for the same constructed resident apparatus; unused
base/sketch storage is not silently removed from the controls.

## Frozen subset and result

Root `.venv`: Python 3.14.3, torch 2.10.0+cu130, transformers 5.13.1, FP32 CUDA
SDPA, four CPU threads, TF32 disabled. HF Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; checkpoint hashes are retained.
Two previously seen diagnostic documents, eight prefix IDs and eight committed
output tokens each, one repeat. Order reverses for the second document. All four
proposals are charged even when rejected or beyond the terminal output cap.

The corrected source and protocol were pushed at
[`bb34a8f`](https://github.com/displague/dynamic-model-loading/commit/bb34a8f91dbf5fc2b7f4cc210affab7b98480780)
before `runs/debt-screen-20260914-v2` began. Its worker finishes in **125.438 s**;
independent analysis takes **8.703 s**: **2 minutes 14 seconds** total. Construction
and serialization take 2.235 s; all warmups total 19.367 s, already included in
worker time. Compilation, implementation and CPU review time are not benchmark time.

All rows below aggregate the same 16 committed tokens across two scored episodes.
Wall time includes prefill, draft, dense verifier, synchronization, policy/page
records and ledger flushing. H2D includes prefix and discarded proposal work.

| Condition | Accepted / proposed | Charged wall seconds | Committed tokens/s | Page H2D GiB |
|---|---:|---:|---:|---:|
| Resident dense target only | n/a | 0.416 | 38.494 | 0 |
| Full dense streaming draft | 16/16 | 25.093 | 0.638 | 137.8125 |
| Two-bit base, no corrections | 0/64 | 12.571 | 1.273 | 0 |
| Fixed sketch ranking, four corrections | 0/64 | 23.515 | 0.680 | 46.265625 |
| Feedback-directed corrections | 0/64 | 24.313 | 0.658 | 46.265625 |

The candidate is **93.40% slower than the base** and **3.39% slower than fixed
selection** by total wall time. Both correction controls load 10,528 pages and
transfer 49,677,336,576 bytes plus 6,569,472 controller D2H bytes. Feedback uses
all four corrections on every one of its 2,632 scored FFN executions; its stopping
rule does not reduce acquisition on this subset.

Feedback saves 66.43% page traffic versus dense streaming, but that is not a
policy win: fixed selection saves exactly as much and the zero-fetch base is
faster. The required acceptance >=50%, acceptance gain >=10 percentage points,
and >=5% wall improvement over each of base/fixed all fail. The two transfer
criteria pass. The frozen conjunction therefore returns **stop**.

All eight scored draft outputs and all four draft warmups match their scalar
dense references, including stop reasons. This does not imply a faithful draft:
for each approximate condition, all scored output is supplied by fallback after
rejection. The dense target is GPU-resident in this feasibility apparatus, so its
0.416-second reference is not a memory-constrained 32B deployment comparison.

## Numerical and memory evidence

All-page correction reconstructs all 56 fixed FFN inputs: maximum relative L2
1.54899e-6. Both full-model logit positions pass, relative L2 1.52643e-5 and mean
KL 9.51202e-8, below the unchanged .01 / .001 thresholds.

The predeclared local FFN diagnostic shows why local improvement cannot stand in
for acceptance. Mean relative L2 is **1.216407** for the two-bit base, **0.945419**
for fixed correction and **0.980107** for feedback. Both correction methods reduce
that local error, but feedback is worse than fixed and neither yields an accepted
proposal. These are dense-input local errors, not an independent quality benchmark
or proof that one specific source of approximation error caused every rejection.

Physical charges include independent target CUDA parameters (6,174,857,216 B),
draft CUDA non-FFN parameters (1,550,637,056 B), and CPU draft FFNs
(4,624,220,160 B) aliased by the page catalogue. There is no shared model storage.
Packed bases and sketches use **428,343,296 B**; reusable workspace and unpack
scratch use **182,353,920 B**. One GPU page and one pinned host staging page each
use 4,718,592 B. Construction transfers 4,627,070,976 H2D bytes and exports
428,343,296 D2H bytes of constructed tensors. These setup costs are separate from
the scored page-traffic table, not hidden.

Peak extra inference CUDA allocation is **619.125 MiB**, below the unchanged
768 MiB cap after charging all baseline overhead. Target/draft logical KV peaks
are checked exactly against replayed positions. All 734 joint resource samples
pass: peak total GPU used **8,873.098 MiB**, minimum available host
**10,268.809 MiB**, peak process RSS **9,750.285 MiB**. Setup allocator peaks,
metadata, source/sketch storage and physical transfer events are retained.

## Preserved first attempt and accounting correction

The [original protocol](debt-screen-protocol.md) and source
[`5c0ca04`](https://github.com/displague/dynamic-model-loading/commit/5c0ca048b8ea4dfc663a5d39cc0346b5779cb401)
preceded the first 138.939-second worker. That run also records 0/64 acceptance for
all approximate controls, but its **overall decision remains inconclusive**:
analysis rejects a measured baseline of 7,749,224,960 B against known parameter
storage of 7,725,494,272 B plus an assumed 16 MiB overhead margin. The observed
difference is 22.631 MiB; its exact allocation-source breakdown was not diagnosed.

The separate corrected protocol charges *all* bytes above the known parameter
storage inside the same 768 MiB allowance. It does not enlarge that allowance,
alter selection, change thresholds or repair the first receipts. The corrected
run has a new directory and a newly pushed source freeze. All **29 constructed
artifact hashes are identical between attempts**, confirming that the candidate
representation was unchanged. Do not pool the two attempts as independent repeats.

## Validation, raw artifacts and reproduction

Independent core/integration reviews approved before each inference freeze.
Review led to actual-feedback and forced-rejection tests, reconciled charged
timing, exact KV replay and conservative allocator lower bounds. The corrected
pre-inference CPU suite passes **384 tests**. A second model-free analysis takes
8.380 s and reproduces the full corrected summary exactly. Review and test
receipts are in [the compact evidence directory](../results/debt-screen-20260914/).

The [v0.22 prerelease](https://github.com/displague/dynamic-model-loading/releases/tag/v0.22.0)
attaches both attempts with source snapshots, constructed tensors, numerical
inputs/outputs, every page/policy/verification ledger, supervisor and resources:

- `debt-screen-v022-first-with-license.zip`: 485,103,609 B, 145 members,
  SHA-256 `435e43e4937690c1b3f6785dd1917ebbdf1702f44e0f55dd36b5d7e2b1bcf084`.
- `debt-screen-v022-corrected-raw.zip`: 485,110,809 B, 146 members,
  SHA-256 `22805ae6ef3d7209994721ea517e7c47178b069161e7dddf4f7b3720d5b74e42`.

Every member is decompressed and SHA-checked; source files are rechecked after
compression. The first pre-packaging archive receipt also remains preserved.
Distribution copies add Qwen's Apache-2.0 license and a modification notice
outside the unchanged experimental worker trees; those are packaging metadata,
not new measurement records. Corpus provenance remains separate.

From the prepared environment with pinned HF weights and parent artifacts:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_debt_screen.py tests/test_residual_debt.py
.\.venv\Scripts\python.exe -m dynamic_model_loading.debt_screen --output runs/debt-screen-<fresh-name>
```

To audit the corrected release archive without inference, extract it under
`runs`, retain the supervisor alongside `worker`, then run:

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.debt_screen --analyze --output runs/debt-screen-20260914-v2/worker
```

This release completes [issue #37](https://github.com/displague/dynamic-model-loading/issues/37),
not [the novel-loading milestone](https://github.com/displague/dynamic-model-loading/milestone/10).
The next representation/acquisition hypothesis needs a cheap causal-faithfulness
check and a new short protocol; this two-bit/four-page policy is not expanded or
ported. The result does not disprove all residual-loading approaches. Stock 32B
baselines and historical ADR 0005 remain unchanged.
