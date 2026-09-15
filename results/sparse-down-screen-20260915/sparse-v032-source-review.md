**BLOCKED — three actionable findings:**

1. **P1: Recorded CUDA peaks do not enforce the memory cap.** [sparse_analysis.py:112](C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/sparse_analysis.py:112) checks allocator consistency but permits a 16,000 MiB peak against the 15,000 MiB cap when resource samples remain below it. Reject over-cap allocator peaks for episodes and setup; add a regression test.

2. **P2: Replay accepts overlapping calls to the shared workspace.** [sparse_analysis.py:40](C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/sparse_analysis.py:40) validates each interval separately. All 24 layers can report simultaneous execution and pass. Require each layer’s start to follow the previous layer’s completion; correct the overlapping test fixture and test rejection.

3. **P2: Exactly 10% traffic saving fails the inclusive gate.** [sparse_analysis.py:64](C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/sparse_analysis.py:64) computes `0.09999999999999998` for an exact 10% reduction. Use integer comparison, such as `10 * sparse_bytes <= 9 * stream_bytes`, and test the boundary before freezing.

**Validation:** All 20 targeted tests passed. Tiny CPU OPT generation, populated-stale-row omission, restoration, and meta-model allocation checks passed. No files changed or real-checkpoint inference run.