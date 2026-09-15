**BLOCKED for source freeze — one remaining issue.**

[The allocation audit](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/specialist_analysis.py:124) undercounts registered buffers by half. Both `inv_freq` and `original_inv_freq` are registered:

- Target: **512 bytes**, not 256.
- Draft: **256 bytes**, not 128.
- Construction and baseline checks need **768 additional bytes**, not 384.

The worker now counts both correctly, so the current auditor would reject its receipt. Update the audit and [meta-device test expectations](/C:/Users/displ/Documents/dynamic-model-loading/tests/test_specialist_heads.py:167).

Validation: **21 passed, 2 failed**; both failures confirm these buffer totals. The CLI and BLAS fixes resolve the other blockers. No real checkpoints loaded, experiments scored, or files modified.