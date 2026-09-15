**BLOCKED — completion/resource receipts are not validated.**

[field_replay.py:100](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/field_replay.py:100) returns `completion.json` unchanged. An empty receipt—or incorrect completion, timing, inference, or tensor-byte fields—can accompany an eligible verdict if its hash is updated in `files.json`.

Before approval, validate:

- `complete=True` and `new_inference=False`.
- Finite, valid timing/RSS values and timing containment within the supervisor receipt.
- Input/prediction byte counts against the loaded arrays.
- Tamper cases covering missing and incorrect fields.

No additional blockers found in conditioning, leakage boundaries, parent hash binding, frozen gates, claim limits, or the publisher UTF-8 fix.

No tests, scoring, inference, or edits performed.