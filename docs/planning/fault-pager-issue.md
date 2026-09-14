# Implement and measure the frozen physical FFN draft pager

Restore novel dynamic-loading research under ADR 0004. Deliver the actual PyTorch
page runtime and the frozen 1.5B FP32 experiment: calibration-only medoid index,
27 of 35 paired pages, eager/LRU/prefetch controls at 128/512 MiB, dense target
verification, explicit KV rollback, actual H2D/cancelled-prefetch receipts, total
host/device accounting, numerical mechanics and all 72 generation episodes.

Preserve the stock 32B profiles as practical baselines. Keep source/protocol
snapshots, raw failures, independent receipt reconciliation and complete outputs.
The completed delivery is a bounded experiment; the broader novel-loading
milestone remains open. ADR 0005's frozen native-admission predicate governs any
future llama.cpp implementation.

- https://github.com/displague/dynamic-model-loading/blob/main/docs/fault-pager-protocol.md
- https://github.com/displague/dynamic-model-loading/blob/main/docs/fault-pager-amendment-1.md
- https://github.com/displague/dynamic-model-loading/blob/main/docs/adr/0004-fault-pager-research-track.md
- https://github.com/displague/dynamic-model-loading/blob/main/docs/adr/0005-native-pivot-evidence-boundary.md
