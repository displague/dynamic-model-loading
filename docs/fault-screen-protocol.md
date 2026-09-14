# Five-minute physical-pager screen, version 1

Prospective for these new measurements; frozen by the reviewed, pushed source
commit before inference. This is a retrospective **screen-workflow validation**
using the known v0.20 negative candidate, not a fresh candidate or held-out result.
See [ADR 0006](adr/0006-screen-before-performance-matrix.md). The old 72-episode
protocol and its results are immutable; this is not a corrected v0.20 run.

## Frozen subset and costs

- Use the same root `.venv`, HF Qwen2.5-1.5B-Instruct revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP32, CUDA SDPA, four CPU threads,
  TF32 off, independent dense target and CPU-backed FFN draft. Verify model bytes
  against the pinned earlier manifest and record actual versions/resources.
- Reuse the **unchanged** v0.20 calibration side index and token IDs, with hashes
  in `configs/fault-screen.json`; do not retrain/calibrate on diagnostics. Old raw
  files can be recovered from the v0.20 release assets. No new downloads in a run.
- First two diagnostic documents in original corpus order, first 32 prefix IDs,
  **8 output tokens**, one repetition. Both conditions have 512 MiB page capacity:
  eager selection-matched draft and related-token prefetch. Document 0 order:
  eager then prefetch; document 1: prefetch then eager. Exactly four scored paged
  episodes and two dense references; no 128 MiB, LRU, four-document or 64-token
  matrix. Fixed four-token proposal charging and rollback from v0.20 are unchanged.
- Before scoring: dense warmup on calibration document 0 (prefix 4, cap 4);
  check all 28 FFNs on the first two positions of diagnostic 0 against grouped
  dense execution, then one full-model two-position dense-completion check. This
  full-page numerical path uses the unchanged 512 MiB LRU mechanics cache. Keep
  per-position FFN relative L2 <= .01, logits relative L2 <= .01, KL <= .001.
  Both full-model logit positions are checked, including the final position;
  this is reconstruction, not next-token corpus NLL (which omits that position).
  These reduced numerical checks are a smoke test, not the original full gate.
- Warm each paged condition once on calibration document 0 (prefix 4, cap 4).
  Independently compare these warmups to the dense warmup too. Reset both KV
  states and the page cache each episode. Preserve target references, complete
  proposal/fallback/KV records and all physical page transactions, including
  warmups, canceled prefetch and episode cleanup. No early selection based on
  observed performance: run all four scored episodes unless a safety/mechanics
  failure or wall timeout intervenes.
- A separate standard-library supervisor gives the worker **300 seconds wall
  time**, starting at process launch, including its imports, provenance checks,
  model loads, numerical checks, warmups and scoring. On timeout kill only this
  owned worker, retain partial raw files and write an inconclusive timeout
  receipt; do not start a matrix or overwrite the directory. Analysis runs after
  the worker, separately timed. Process launch/reaping may add small supervisor
  overhead; replay requires a successful supervisor receipt with no more than one
  second of such overhead beyond the 300-second wait. A worker invoked directly
  without the supervisor cannot qualify. Each attempt requires a fresh directory;
  the normal CLI has no timeout override.
- Joint GPU-used ceiling 15,000 MiB, available host floor 2,048 MiB, 200 ms NVML
  sampling and final fail-closed check. Report target/draft/controller/pinned
  staging/cache/KV memory and actual CUDA execution. No uncharged shared weights.
  Report aggregate verifier-charged wall time, committed tokens/s and H2D bytes
  per proposal and committed token. All preprocessing/warmups count in worker
  wall time, but not scored generation totals. Never compare a short episode's
  wall time directly to an old 64-token episode as a speedup.

## Screen decision (not native admission)

Independent raw-ledger analysis must establish the complete specified inventory,
numerical checks, resource bounds and exact IDs/stop reasons for every episode.
For the two scored prefetch episodes combined, require both:

1. emitted accepted draft tokens / **all** proposed draft tokens >= 50%;
2. fully charged page H2D bytes per proposed token >= 10% lower than the paired
   eager control. Require selection-matched proposal and target-prediction paths
   across conditions. Preserve warmup costs separately.

Report each Boolean and the raw counts. If either economics threshold fails,
the candidate is **stop**, with no long suite launched. A correctness/resource
failure or timeout is **inconclusive/error**, never evidence of useful speed.
Passing is **eligible for expanded protocol**, not a research result proving
acceleration and never automatic permission to execute a larger matrix.

The acceptance/byte thresholds intentionally mirror the two necessary v0.20
numeric eligibility conditions for this unchanged policy. They do not evaluate
the complete ADR 0005 predicate or set universal gates for unrelated future
designs. Tiny caps amplify terminal proposal waste and prefix traffic. Two known
prompts and one repeat cannot establish generalization, confidence intervals,
long-context performance, or 32B gains. Prediction: this screen rejects the
known pager without spending hours; that tests the workflow, not a new hypothesis.

## Invocation

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_fault_screen.py tests/test_fault_generation.py tests/test_fault_pager_analysis.py
.\.venv\Scripts\python.exe -m dynamic_model_loading.fault_screen --output runs/fault-screen-<fresh-name>
```

The screen has no full-suite switch. Existing reproduction commands remain
available for historical work; they are not the default for new research.
