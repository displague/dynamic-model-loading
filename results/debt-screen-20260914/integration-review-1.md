**Block — two accounting-validation gaps remain.**

- **P1: Throughput timing is not independently checked.** [debt_analysis.py:235](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/debt_analysis.py:235) trusts `wall_seconds` without reconciling timestamps or timing components. [debt_screen.py:195](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/debt_screen.py:195) adds cleanup time without updating the finish timestamp or recording cleanup separately. Record a consistent charged interval, validate each episode, and add a test rejecting understated wall time before evaluating throughput gains.

- **P1: Impossible CUDA/KV charges pass validation.** [debt_analysis.py:210](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/debt_analysis.py:210) accepts positive values below loose upper bounds. A CPU probe of these checks accepted **1 byte of extra CUDA and 1 byte per KV cache**, despite **610,697,216 bytes** of resident representation plus workspace. Enforce physical lower bounds and allocator consistency; derive expected KV peaks from replayed cache positions.

No additional blockers found in source freeze, supervision, artifact hashes, numerical checks, choice/page replay, or advance-predicate formulas.

**Validation:** 61 selected CPU tests passed. No source edits, GPU inference, matrix, or release actions.