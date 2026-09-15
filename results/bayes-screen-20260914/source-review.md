**Blocked: one high-priority issue.**

### High — Successful analysis cannot be serialized

[bayes_analysis.py:144](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/bayes_analysis.py:144) stores NumPy booleans for upper/constant coverage. Summing them at [line 150](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/bayes_analysis.py:150) produces `numpy.int64` values in `covered_blocks`.

A small CPU reproduction confirmed:

```text
TypeError: Object of type int64 is not JSON serializable
```

Consequently, a completed screen fails when the supervisor writes its summary and becomes `analysis_error`/`inconclusive`. Convert report values—including gate booleans—to native Python types and add a test that serializes the complete analysis report.

**Validation and limitations:** All 13 tests in `test_calibrated_error.py` passed. No other high/critical blockers found in the reviewed statistics, accounting, rollback, or ADR boundaries. CUDA behavior and physical measurements remain unverified; no checkpoint inference or file edits were performed.