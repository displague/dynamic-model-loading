# v0.31: output-only specialists fail the prerequisite

First of six authorized new deliveries. The frozen
[protocol](specialist-screen-protocol.md) completes in **23.417 seconds**:
18.932s worker plus 4.485s independent model-free audit. Both Hgeneral and
Hspecialization fail. Stop this output-only ridge specialist candidate before
physical paging. No long matrix or native work follows.

## Full-vocabulary target agreement

| Condition | Fit (96 positions) | Diagnostic (48 positions) |
|---|---:|---:|
| Untouched 0.5B draft | 75 | 37 |
| General correction head | 79 | 34 |
| Matching-domain specialist | 78 | 34 |
| Wrong-domain specialist | 75 | 36 |

The learned corrections improve their training counts but each loses three
diagnostic matches to the untouched draft. Matching specialists do not outperform
the general head and lose to the wrong-domain control. Diagnostic code/prose/math
counts respectively are base 13/14/10, general 13/11/10, specialist 12/12/10 and
wrong-domain 13/11/12, out of 16 each. No domain supplies the required specialist
advantage over the general control. We do not select a favorable domain afterward.

This is **teacher-forced agreement on authored prefixes**, not speculative
acceptance, generated answers or task quality. Domain labels are supplied, not
predicted. The test uses 12 fit and six diagnostic synthetic documents, not an
independent natural benchmark. Small sample size, only 16 projected features,
the squared-logit loss and limited training are possible limitations; this screen
does not identify a unique failure cause or rule out domain draft training.
No alternative rank, regularization, loss or document selection was tried.

## What was implemented

A frozen Qwen2.5-0.5B-Instruct HF core, with target-aligned 17-by-151936 linear
output corrections fitted against the pinned 1.5B target. One general head and
three code/prose/math heads use fit rows only. All vocabulary entries remain
eligible; the wrong-domain control rotates heads. No backbone weights, hidden
states or KV are rewritten. This collection runs with `use_cache=False`, so it
does not qualify a switching or rollback runtime. It is not another FFN pager.

Source **b9be9e0eca90094b94ffe4697a82e9310e545329** and all protocols/inputs were
reviewed, committed and pushed before inference. There was one checkpoint run.
The full readout reconstruction and repeated target/draft/hidden controls are
exact on their checked arrays (relative L2 zero, identical argmax). The auditor
retokenizes the frozen texts, refits only on training rows and reproduces all
heads, predicted IDs and gates exactly. A second model-free audit matches the
complete summary in 4.329s without another checkpoint forward.

## Costs and limits

Draft artifact download and hashing took 14.155s before the experiment, separately
reported. Rehashing and loading both models, imports, collection, fitting, scoring
and raw writes are charged inside the worker. Fitting takes 0.418s and scoring
0.481s; these are component CPU clocks, not an online routing benchmark. PyTorch
and NumPy/OpenBLAS both use four threads; the actual BLAS DLL hash is archived.

Target and draft parameter payloads are 6,174,857,216 B and 1,976,131,072 B.
Registered rotary buffers add 768 B; construction H2D totals 8,150,989,056 B.
Explicit data copies add 40,896 B H2D and 190,160,896 B D2H. These counters do not
claim to trace every framework/driver DMA operation. Raw observation tensors are
175,546,368 B. The four FP64 heads occupy 82,653,184 B; their hypothetical FP32
equivalent is 41,326,592 B. This small bank fits without paging; no artificial
eviction experiment is presented as a capacity win.

All 64 resource samples pass: sampled total GPU peak 8,955,990,016 B, sampled
process RSS peak 9,446,006,784 B, minimum host available 9,557,946,368 B. CUDA
inference allocation peak is 8,167,445,504 B. These measurements establish neither
lower required VRAM nor access to a larger model. Stock 32B baselines remain
unchanged and are not directly comparable with these HF component timings.

Review caught successful exit statuses after worker failure, omitted construction
buffers and unrecorded NumPy threading. They were fixed before source freeze;
allocation-free model tests caught the two registered frequency buffers. The
corrected source passes 543 CPU tests. Failed review receipts remain alongside
the approval; they are not failed or rerun checkpoint experiments.

Raw archive `specialist-screen-v031-raw.zip`: **293,695,835 B**, 96 members;
SHA256 `7b9d9774438d9ea14471ee041cb0b22858cd161b274c03aa491c9c97f1334333`.
Every member was decompressed and SHA-checked and every raw source rechecked.
The archive includes full observations, heads, predictions, source/protocol,
documents/tokens, numerical/cost receipts and licensing. Pinned full HF weight
files remain external dependencies rather than being duplicated in the release.

## Next boundary

Complete only issue #46, the first delivery. Milestone 10 remains open. Next move
to **sparse architecture**, testing physical acquisition after exact ReLU-zero
discovery with the first projection fully charged. Do not build or benchmark a
pager around these failed output heads. Other specialist representations remain
open questions requiring materially distinct, prospectively frozen protocols.

See [course and cited antecedents](residency-research-course.md) and
[compact receipts](../results/specialist-screen-20260915/).
