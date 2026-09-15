# v0.25 decision-field intervention screen

Freeze this protocol, config and source in a clean reviewed pushed commit before
inference. ADRs 0004--0006 and the [research course](information-research-course.md)
apply. The five-minute supervisor includes imports, model verification/loading,
construction, mechanical checks, all inference, readbacks and raw hashes. Analysis
is separately timed. Timeout/error/incomplete receipts are inconclusive; corrections
use a new directory. No native pivot, long matrix or deployable result follows.

## Question

Does buying precision at four separated FFN regions change the downstream token
decision? Can a sum of individually measured logit changes predict a joint change?
This is a privileged opportunity/interaction screen, not a fitted causal policy.
It supplies observations for separately frozen spatial/temporal tests, not another
GP fit to local error. Borrowed methods and intended departures are cited in the
course. No claim that these four regions represent all model pages.

## Frozen apparatus and workload

Use the same local Qwen2.5-1.5B-Instruct HF revision
989aa7980e4cf806f80c7fef2b1adb7bc71aa306 and checkpoint inventory as v0.24,
Python 3.14.3 / torch 2.10.0+cu130 / transformers 5.13.1, FP32, SDPA, CUDA,
four CPU threads, no TF32. Retain separate FP32 target and draft non-FFN weights;
original draft FFNs stay on CPU. Reconstruct v0.23's group-128 embedded 4/6/8-bit
format and compare every packed tensor exactly to that frozen parent. The parent
release archive is an explicit representation dependency; do not duplicate it.
Check all 28 full-eight FFNs against saved same-input parent outputs, rel-L2<=.01.

First six calibration and first two diagnostic documents of the pinned parent
corpus/token inventory. Both are development data, not new held-out qualification.
For each document consume four corpus tokens with eight-bit draft FFNs. Then
teacher-force the next four tokens. At each position use identical already-computed
KV for twelve counterfactual branches: all-eight, four singleton promotions,
six pairs, and four-bit base last. Sites are zero-based layers [1,8,15,22]. Each
promotion fetches both physical two-bit increment slabs (20,643,840 bytes); all
other layers remain four-bit. One persistent slot plus bypass is allocated but
no admission occurs, so every requested slab is a physical load. No hot-cache
discount or inferred traffic saving. All-eight costs 56 slab loads per position.

The all-eight branch is a *same-history* draft reference, not a high-eight KV
trajectory and not the FP32 target. A separate FP32 target cache teacher-forces
the same corpus independently, supplying audit logits only. It supplies no
acquisition feature or selection label. No generated acceptance or speedup claim.

After every non-base branch crop draft KV to its pre-step length and require all
old K/V bytes to have the same SHA256; last/base branch is retained for the next
teacher-forced position. Precision never rewrites old KV. Reset document caches,
page counters and state. Charge fingerprints, D2H logits, ledger flushes and all
replayed full-model computation. Save every full-vocabulary logit vector, every
page event, branch timing, logical K/V size and unique underlying K/V storage,
allocation and resources. Cropped views can retain the just-computed suffix in
storage; do not report their shorter logical length as freed memory. The inherited
precision routine also reads back two FP32 correction norms per promoted layer;
charge those eight bytes separately from the full logits and KV fingerprints.

## Frozen analysis and interpretation

For each diagnostic pair (i,j), additive prediction is L_i+L_j-L_base. Measure
full-logit relative-L2 against L_ij, and argmax identity. Also measure interaction
energy ||L_ij-L_i-L_j+L_base|| / max(||L_ij-L_base||,1e-12).
Report all document/position rows, including negative precision effects. Define
an opportunity when base argmax differs from same-history all-eight, but at least
one singleton or pair agrees. This oracle uses outcomes unavailable to a loader.

- Hdecision: at least two of eight diagnostic positions are opportunities.
- Hjoint: additive argmax agrees on at least 90% of 48 diagnostic pairs AND
  aggregate interaction norm / aggregate joint-change norm <= .25.

These are small development-screen bars, not significance tests or correctness
certificates. Report fit and diagnostic counts separately. A pass only permits
the next short estimation hypothesis; failure of Hjoint requires directly
modeled interactions or abstention, not an additive-policy claim. Hdecision
failure means this action/workload does not establish a useful decision frontier.
Preserve the outcome and change the hypothesis prospectively. Neither gate is
native admission or larger-model feasibility.

Extra CUDA <=1024 MiB above actual target+draft non-FFN parameter storage, total
GPU <=15000 MiB, available host >=2048 MiB. Charge original host weights, packed
increments, base/scales, workspace, two GPU slabs, pinned staging, KV and Python
metadata/RSS separately. Traffic is not capacity. CPU independent analysis checks
Git-bound source, complete hashes, exact branch and transfer sequences, fixed
workload, crop fingerprints, all logits and recomputed gates. Full CPU tests and
review precede publication; publish raw receipts even for a failed screen.
