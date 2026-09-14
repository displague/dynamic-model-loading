# Physical side-indexed draft pager

Completed on 2026-09-14. The physical loader and verification boundary pass, but
the tested acquisition policy is a **negative result**. All 72 scored outputs match
the independent dense reference. Both LRU budgets produce zero demand hits;
prefetch increases traffic and latency. Neither physical-improvement decision nor
the frozen native-admission predicate passes.

## Measured result

Each row contains twelve episodes: four fixed prompts by three repetitions, 768
committed tokens. Wall time includes the 32-token prefix, drafting, verification,
rejected work, page gathering/copying, control and page/round logging. The target
reference has no draft-page traffic; its zero below does not mean zero total PCIe
traffic or zero model-loading cost. Model startup is outside episode timing.

| Policy / cache MiB | Median episode s | Median s/token | Committed tokens/s | Draft H2D GiB/committed token | Demand misses/committed token |
|---|---:|---:|---:|---:|---:|
| Dense target only |1.416|0.0221|45.116|0|0|
| Eager / 128 |115.800|1.8094|0.5654|9.5385|2170.547|
| Eager / 512 |115.743|1.8085|0.5651|9.5385|2170.547|
| LRU / 128 |117.737|1.8396|0.5556|9.5385|2170.547|
| LRU / 512 |119.762|1.8713|0.5454|9.5385|2170.547|
| Prefetch / 128 |123.385|1.9279|0.5286|10.1422|2170.547|
| Prefetch / 512 |124.454|1.9446|0.5258|10.0444|2160.473|

Every candidate condition accepts **456/1512 proposed tokens, 30.1587%**. Those
are emitted accepted tokens divided by all charged proposals, not accepted tokens
divided by the 768 committed tokens. Output IDs, stop reasons and per-prompt
acceptance counts reproduce across all three repetitions. All scored episodes reach
the 64-token cap; this is not a complete-answer or agent-utility test.

| Policy / cache MiB | Total wall s | Target computation s | Demand H2D TB | Prefetch H2D TB | Total H2D TB |
|---|---:|---:|---:|---:|---:|
| Dense target only |17.023|16.967|0|0|0|
| Eager / 128 |1358.396|15.545|7.865798|0|7.865798|
| Eager / 512 |1359.106|15.708|7.865798|0|7.865798|
| LRU / 128 |1382.248|15.476|7.865798|0|7.865798|
| LRU / 512 |1408.076|15.850|7.865798|0|7.865798|
| Prefetch / 128 |1452.770|15.861|7.865798|0.497830|8.363629|
| Prefetch / 512 |1460.595|15.803|7.829291|0.453650|8.282941|

TB here is decimal; GiB in the first table is binary. Target computation includes
target prefill and fallback consumption as well as block verification. Throughput
is total committed tokens divided by total wall time, not an average of rates.

At 128 MiB, prefetch saves no demands and cancels **105,504** unused fetched pages.
At 512 MiB it serves **7,737 demand hits**, but cancels **88,404** prefetched pages.
Its extra acquisitions exceed the demand transfers saved. Relative to matched LRU,
traffic increases **6.3291% / 5.3032%** at 128 / 512 MiB. Prefetch-512's median
episode is **3.9177% slower** than LRU-512. Both original physical-improvement
decisions therefore fail. Relative to eager-512, the nominated policy also fails
ADR 0005's 50% acceptance and 10% byte-saving requirements. No candidate is swapped
in and no threshold is changed.

The dominant cost is acquisition, not target verification: prefetch-512 spends
916.728 seconds in synchronized gathering/copy intervals, versus 15.803 seconds
in target computation. Its CUDA copy events total 309.051 seconds; these component
intervals are not an independent end-to-end speed measurement or an overlap claim.

## Correctness and memory

All 28 two-input FFN checks and all 16 full-model checks pass. Worst per-input FFN
relative L2 is `2.709208e-7`; worst full-model relative-logit L2 is `1.593579e-6`;
worst mean KL is `5.265001e-8`, below the unchanged `0.01 / 0.001` tolerances.
The analyzer independently reconstructs these metrics, the calibration index,
page-demand/LRU ledgers, committed prefixes, KV rollback boundaries, exact seeded
matrix, source snapshots and resource receipts. No replacement run was needed.

There are **45,198 resource samples**. Peak device-used memory is **8664.418 MiB**
against the 15,000 MiB ceiling; minimum available host memory is **8537.672 MiB**
against the 2048 MiB floor. Peak process RSS is **10486.941 MiB**. These are
whole-run values, not claims that every condition has the same phase-local peak.

The dense target holds 6,174,857,216 parameter bytes on GPU. The independent draft
holds 1,550,637,056 non-FFN parameter bytes on GPU and 4,624,220,160 FFN bytes on
host. The catalogues alias those host tensors; **no target/draft weight storage is
shared**. CPU and GPU controller copies each use 2,877,952 bytes; pinned staging
uses 4,718,592 bytes. Python catalogue/cache/queue object sizes are recorded at
layer boundaries, starting from 14,399 bytes before cache population. These object
sizes are not a substitute for RSS or allocator accounting.

Actual maximum page payload is 4.5 MiB for eager, 126 MiB for the 128 MiB caches,
and 508.5 MiB for the 512 MiB caches. Candidate target and draft logical KV peaks
are each 5,677,056 bytes. CUDA allocator peaks and reservations are preserved per
episode; reserved free blocks are not counted as usable resident pages, but remain
charged in total device-used memory. Cache/queue/KV reset between episodes; the
CUDA allocator is not emptied between conditions.

Warm-ups are retained but excluded from scored summaries. Their raw
`reference_match=false` field means the runner did not compare a warm-up to the
scored reference dictionary; it is not a reported warm-up output disagreement.

## What was built

This is a custom PyTorch loading path, not a llama.cpp configuration. Draft FFN
weights stay in host memory. Each selected page contains paired gate/up rows and
down columns; a missing page is packed into pinned staging, copied to CUDA and
executed there. A byte-bounded explicit residency map controls acquisition.

A calibration-only medoid index chooses 27 of each layer's 35 pages from the
current FFN input. It does not calculate omitted neurons or consult the target to
choose pages. The approximate draft proposes four tokens; an independently resident
dense target verifies them, commits the matching prefix and supplies a fallback on
rejection. Both KV caches are cropped at that boundary. All four proposals are paid
for, even beyond EOS or the remaining 64-token allowance.

The selection and arithmetic are identical across eager streaming, LRU and
synchronous related-token prefetch. Only acquisition/residency changes. Prefetch
uses the previous token's top two pages per layer. It makes no overlap claim.

## Frozen evidence

- [Original protocol](fault-pager-protocol.md) and
  [pre-inference amendment](fault-pager-amendment-1.md).
- [Executed source freeze](https://github.com/displague/dynamic-model-loading/commit/26bee33)
  was reviewed and pushed before calibration. Final pre-inference validation:
  325 tests passed. The independent follow-up review approved the candidate.
- Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
  FP32, root Python 3.14.3 / PyTorch 2.10.0+cu130 / Transformers 5.13.1;
  NVIDIA RTX 5080 Laptop GPU, four CPU threads, SDPA, TF32 disabled.
- Calibration: 80 documents, 128 positions each; mechanics: 28 layer checks and
  all 16 diagnostic documents. Generation: first 32 plain-text IDs of the first
  four diagnostic documents, up to 64 new tokens, three repetitions of each of
  six policy/budget conditions. Warm-ups are excluded from scored summaries.
- Cache payload budgets: 128 and 512 MiB. The independent dense target, draft
  non-FFN residency, CPU source, both indexes, staging, KV, Python metadata and
  allocator footprint are additional costs, not hidden inside that subtotal.
- Exclusive raw run: `runs/fault-pager-20260914-v1`. No corrected run or
  post-score selection is substituted for it.

The [compact receipts](../results/fault-pager-20260914/) include the complete raw
episode ledger, summary, frozen input identifiers and archive inventory. Four
partitioned raw ZIP assets on [v0.20.0](https://github.com/displague/dynamic-model-loading/releases/tag/v0.20.0)
preserve calibration tensors, numerical reference/candidate logits, every page and
verification event, resource samples, corpus, token IDs and source snapshots.
Every ZIP member was decompressed and SHA-256 checked against the analyzed run.
A second model-free audit reproduced the summary and all-file inventory
byte-for-byte; its hashes are retained in `analysis-replay.json`.

## Reproduction

The inference command requires the original cached checkpoint, corpus and parent
manifest, the frozen root environment, CUDA, and a clean pushed source tree.
Use a new output directory; completed or failed directories are never resumed.

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.fault_pager_run --output runs/fault-pager-reproduction-NEW
.\.venv\Scripts\python.exe -m dynamic_model_loading.fault_pager_analysis --run runs/fault-pager-reproduction-NEW --output runs/fault-pager-analysis-NEW
```

The analyzer reconstructs calibration medoids and numerical metrics, reconciles
page-demand transactions and KV boundaries, checks the exact document/repetition
order and raw resource ledger, and then applies the unchanged decision rules.

## Scope

This is a small-model physical-runtime experiment. The 1.5B dense target fits on
GPU; the draft cache is artificially constrained. It does not demonstrate faster
32B inference, deployment-grade memory reduction, long-context behavior or agent
utility. Historical stock 32B target-only and small-draft receipts remain practical
baselines, not directly comparable throughput controls for different FP32 weights.

The four short prompts and three repeats support this bounded measurement, not a
population-level generalization. Synchronous Python control, logging and packing
are charged. A slow Python runtime alone neither disproves a useful policy nor
authorizes a native pivot under [ADR 0005](adr/0005-native-pivot-evidence-boundary.md).

At approximately 14:03 local time during measurement, a new monitoring command
could not launch PowerShell 7 because `pwsh.dll` was unavailable. Monitoring switched
to Windows PowerShell; the already-running measured Python process was not
restarted or modified. The cause of the shell change was not established. This
shared-machine observation is retained; timing is not a claim of isolated-host
execution, and the raw resource series remains the authority for memory limits.

[Related-work context](fault-pager-related-work.md) distinguishes this specific
custom experiment from established mechanisms and avoids a global novelty claim.
The broad novel-loading research program remains open regardless of this policy's
outcome; the completed stock tuning program is not reinstated as its objective.

## Resulting next question

The measured request stream visits 756 selected pages, or 3402 MiB, per token
processed by the draft. That cyclic working set greatly exceeds either cache. The zero LRU hits
and cancelled prefetches identify a concrete acquisition-lifetime problem: fetching
pages ahead of time does not ensure they survive until use. Low acceptance then
amplifies the bytes paid per committed token.

A follow-up should test a different acquisition representation or lifetime policy
that reduces reacquisition **and** improves verified tokens per transferred byte.
For example, a compact resident contribution plus causally chosen corrections is
a research question to distinguish from the residual/low-rank antecedents, not an
implemented improvement. It requires a new protocol and an explicit departure
from prior work. Retuning this prefetch queue or porting this same policy to
llama.cpp is not the next delivery. The original larger-model question is still open.
