# Completed stock calibration

These byte-preserved receipts copy the final v4 continuation's `selection.json`,
`driver-ledger.jsonl` and `driver-complete.json`. The summary records all 41 attempted
configurations, 21 completions, 19 resource failures, one separately classified
native startup failure and 63 generation requests (21 warmups, 42 measured).
Source is `5b4a902b0ccff26c5313aeb5eef7a5a35a46e49a`.

The original 39-case ledger is preserved in the interrupted-calibration archive;
the continuation validates and imports those cases without new generation, then
tries only the missing 16/24-thread cases at draft32/target-ngl7. Both explicitly
report CUDA out of memory during startup. The earlier 8-thread failure retains its
unclassified native-abort cause. No calibration score was rerun or selected using
evaluation outcomes. Selection is frozen before the evaluation's first request.

The [raw archive inventory](../stock-speculation-20260912/archives.json) and
[restoration instructions](../stock-speculation-20260912/README.md) preserve the
initial verbosity/mmap failures, original non-mmap attempt and complete continuation.
The [report](../../docs/stock-speculation-results.md) explains the frozen tie rule,
finite placement/thread grid and timing boundaries. This calibration summary alone
contains no sustained evaluation outcomes.
