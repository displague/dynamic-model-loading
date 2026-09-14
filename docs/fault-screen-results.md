# A minutes-scale screen rejects the known physical pager

The frozen screen completes in **202.082 seconds (3 minutes 22 seconds)**:
195.308 seconds for the measured worker, including imports, model/artifact checks,
loading, numerical tests and warmups; 6.774 seconds for independent analysis.
This is a shorter *experiment*, not faster inference at an unchanged workload.

The screen returns **stop**. All four scored outputs and both paged warmups match
their dense references exactly. The unchanged prefetch candidate accepts only
**6/40 proposals (15%)** and transfers **5.6642% more bytes** than eager streaming.
Neither the >=50% acceptance nor >=10% transfer-saving screen passes. No long
suite, larger-model inference, llama.cpp patch, fork or native build was launched.

## Frozen scope

The [protocol](fault-screen-protocol.md), [configuration](../configs/fault-screen.json)
and complete source were reviewed and pushed at
[`1f5bd89`](https://github.com/displague/dynamic-model-loading/commit/1f5bd89ae32f31942facdd4022ce69c6a1ec64a8)
before any new inference. The one run is `runs/fault-screen-20260914-v1`; there
were no failed measurement attempts, candidate replacements or threshold changes.

This **retrospective workflow validation** uses the v0.20 pager, frozen calibration
index, and first two previously used diagnostic documents. Each has a 32-token
prefix and an eight-token output cap; one repeat, eager versus prefetch at 512 MiB,
counterbalanced order. It is not a novel new loading policy or held-out validation.
The old 72-episode experiment, failed gates and published history remain unchanged.

The root `.venv` records Python 3.14.3, torch 2.10.0+cu130 and transformers 5.13.1,
with FP32 SDPA on the RTX 5080 Laptop GPU, four CPU threads and TF32 off. Model
and parent artifacts match their frozen SHA-256 identities. The same independent
dense target, CPU-backed FFN draft, 27/35-page causal selector, four-token proposals,
KV rollback and terminal-waste charging are used. No stock/native configuration
is substituted for the custom pager.

## Measured subset

Each row below totals two scored episodes, 16 committed tokens. Wall time includes
prefix processing, draft/verifier work, rejected proposals and ledger cleanup;
it excludes the separately reported warmups and model startup.

| Condition | Total episode seconds | Committed tokens/s | Accepted/proposed | Total draft H2D GiB |
|---|---:|---:|---:|---:|
| Resident dense target | 0.4752 | 33.6710 | n/a | 0 |
| Eager, 512 MiB | 69.8840 | 0.2290 | 6/40 | 372.09375 |
| Prefetch, 512 MiB | 75.3707 | 0.2123 | 6/40 | 393.169921875 |

Prefetch's scored total wall time is 7.8513% higher. Its H2D bytes per proposed
token are 10,554,074,726.4 versus 9,988,315,545.6 for eager. Both policies accept
2/24 proposals on diagnostic 0 and 4/16 on diagnostic 1. All proposal and target
prediction paths match between the conditions. Warmup wall time totals 33.6954
seconds across the dense reference and the two paged controls; it is included
in the supervised worker duration, never hidden from the screening cost.

The three-minute cost must not be compared to a 64-output-token episode as a model
speedup. Small caps increase terminal proposal waste and the relative share of
prefix traffic. Two known prompts and one repeat do not estimate uncertainty,
establish long-context behavior, qualify unseen prompts or prove 32B economics.

## Correctness and physical accounting

All 28 FFNs pass at both tested positions; maximum relative L2 is
2.44454e-7. The full-model two-position smoke check, including its final logit
position, has relative L2 1.21293e-6 and mean KL 8.51221e-8. Its numerical page
ledger verifies all 56 full-page FFN executions. These are reduced smoke checks,
not satisfaction of the original larger numerical gate.

Independent analysis reconciles every scored/warmup commit, fallback, KV boundary,
selected page, eviction, canceled prefetch, physical transfer, memory counter and
source snapshot. A second model-free replay produces identical results. It also
requires successful supervisor evidence; invoking the internal worker directly
cannot qualify a screen without that receipt.

Joint resource monitoring records 957 samples: peak total GPU used 8,583.098 MiB,
minimum available host memory 10,635.844 MiB and peak process RSS 9,604.445 MiB.
All pass the unchanged 15,000 MiB GPU / 2,048 MiB available-host bounds, including
the final sample. Physical charges include:

- independent target CUDA parameters: 6,174,857,216 bytes;
- draft CUDA non-FFN parameters: 1,550,637,056 bytes;
- draft CPU FFNs: 4,624,220,160 bytes, aliased by their page catalogue;
- side index: 2,877,952 bytes on each of CPU and CUDA;
- pinned staging: 4,718,592 bytes; maximum cached page payload: 533,200,896 bytes;
- target and draft peak logical KV: 2,465,792 bytes each;
- peak measured Python catalogue/cache metadata: 94,989 bytes, excluding tensor
  storage. CUDA allocator snapshots are retained separately from these payloads.

## Validation and receipts

The pre-inference full CPU suite passes **350 tests in 31.88 seconds**. Independent
review approved after correcting the adapter hook and adding coverage for final
logit positions, supervisor evidence, allocation charges and full-page replay.
No GPU run was needed to find those implementation faults. Test receipts and both
review outcomes are retained in [the compact result directory](../results/fault-screen-20260914/).

The [v0.21 prerelease](https://github.com/displague/dynamic-model-loading/releases/tag/v0.21.0)
includes `fault-screen-v021-raw.zip`: 14,804,797 bytes, SHA-256
`830526ca089329be8e57456c234b4a55c9e243a92f9072fe9044bce54d44d425`.
Every one of its 81 members was decompressed and hash-checked against the original
16,864,217 raw bytes. It includes the full worker source/config/protocol, reused
index and tokens, numerical tensors, page/verification ledgers, resource samples,
supervisor receipt, log and decision. Nothing was substituted by an aggregate.

To reproduce (requires the pinned local HF weights and parent artifacts):

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_fault_screen.py tests/test_fault_generation.py tests/test_fault_pager_analysis.py
.\.venv\Scripts\python.exe -m dynamic_model_loading.fault_screen --output runs/fault-screen-<fresh-name>
```

To replay the archived result without inference, extract the archive beneath
`runs` and use:

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.fault_screen --analyze --output runs/fault-screen-20260914-v1/worker
```

## Decision and next step

[Issue #36](https://github.com/displague/dynamic-model-loading/issues/36) completes
the screening workflow delivery. [The novel-loading milestone](https://github.com/displague/dynamic-model-loading/milestone/10)
remains open. A materially different acquisition representation or lifetime needs
its own short, frozen screen before another full matrix. This failed candidate is
not ported to llama.cpp. ADR 0005 is neither satisfied nor rewritten by this subset.
