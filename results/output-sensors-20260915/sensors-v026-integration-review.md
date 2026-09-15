**BLOCKED for the frozen short screen.**

1. **KV continuity is not validated between frames.** [validate_rows](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/output_sensors_analysis.py:124) checks hashes only within each frame. Disconnected cache histories can pass; the test fixture already uses unrelated starting/ending hashes throughout. Require each subsequent `prior_sha` to equal the preceding frame’s final post-step hash, with a rejection test.

2. **Fixed/oracle reload timings are missing.** [Partial controls](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/output_sensors.py:246) have separate page events but no separate elapsed-time receipts. Their costs are mixed with readouts, fingerprints, and serialization inside frame time, despite the protocol requiring separate costs. Record and validate synchronized timings for each control.

Static review only; no inference, tests, or writes performed. Results and final-tip validation remain separate.