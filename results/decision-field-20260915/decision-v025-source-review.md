## BLOCKED — two high-priority findings

1. **The numerical reference is not independently bound to the frozen parent.**  
   [decision_field_analysis.py:111](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/decision_field_analysis.py:111) verifies the parent inventory’s hash, but line 133 loads `parent-mechanics.safetensors` without checking it against that inventory. Replacing the reference outputs and regenerating the run’s `files.json` can make the numerical gate pass against a changed reference.

   **Fix:** Verify the copied reference’s SHA256 against the frozen inventory’s `mechanics.safetensors` entry before comparison. Check matching shapes/dtypes and add a test that replaces the reference and regenerates raw hashes.

2. **The analyzer accepts incomplete phase-accounting receipts.**  
   [decision_field_analysis.py:123](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/decision_field_analysis.py:123) checks phase names, memory and overall time, but ignores construction H2D, snapshot D2H, initial KV fingerprints, target readbacks and target timing. Those fields can be missing or incorrect without preventing a verdict. Additionally, [decision_field.py:174](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/decision_field.py:174) transfers mechanical inputs and outputs without recording their bytes.

   **Fix:** Record those mechanical transfers and validate required fields, identities and expected byte counts for every phase. Add missing-field and incorrect-count tests. ADR 0006 requires incomplete receipts to fail closed.

**Validation and limitations:** Reviewed the seven requested files against accepted ADRs 0004–0006. Nine non-model test cases executed directly passed, as did a synthetic KV-storage check. Full pytest collection was blocked by temporary-cache initialization in this read-only session. The updated KV-storage accounting was included in this review. No checkpoint inference, writes or external actions were performed.