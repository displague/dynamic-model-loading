### P2 — Target-reference runs bypass the extra-CUDA limit

[refinement_screen.py:168](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/refinement_screen.py:168), [refinement_analysis.py:195](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/refinement_analysis.py:195)

The three target-reference episodes run after the mechanics memory check without saving or checking their allocator peaks. Line 185 then resets those peaks, and the analyzer skips CUDA validation for target rows. A reference episode exceeding the frozen 1024 MiB extra allowance could therefore pass if total GPU usage remains below 15000 MiB.

**Fix:** Record and enforce allocator peaks for each target-reference episode before resetting them, and validate those receipts in the analyzer.

Concurrent revisions addressed the earlier findings. **18 targeted CPU tests passed** on the updated files. I made no edits and ran no checkpoint inference, GPU work, or full suite.
