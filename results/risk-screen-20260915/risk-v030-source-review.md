**BLOCKED — three accounting/integrity issues.**

1. **Explicit D2H accounting omits generator/verifier copies.** [risk_analysis.py:90](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/risk_analysis.py:90) totals runtime receipts only. Each K4 round additionally reads proposal argmax IDs, verifier inputs, committed IDs, and ledger predictions back to CPU—at least 128 bytes plus committed IDs. Account for these separately and reconcile the total; they are explicit application copies.

2. **Reference timing is not contained or fully reconciled.** [risk_analysis.py:166](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/risk_analysis.py:166) accepts any positive reference duration, even one exceeding the entire worker, and uses it for `beats_resident_target_clock`. Validate the complete phase sequence/document bindings, reference durations against their phases, and generator timing components/recorded intervals.

3. **Frozen execution settings are not enforced by analysis.** [risk_analysis.py:115](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/risk_analysis.py:115) checks package versions but accepts missing or contradictory device, thread, and TF32 receipts. Require `cuda:0`, four threads, and both TF32 flags false before declaring `Hfaithfulness`.

No additional blockers found in callback causality, refined full readouts, workspace reuse, or the draft KV journal/crop flow. No tests, inference, scoring, or edits performed.