**BLOCK — one remaining accounting regression.**

[debt_analysis.py:73](/C:/Users/displ/Documents/dynamic-model-loading/src/dynamic_model_loading/debt_analysis.py:73) changes `extra` to use the parameter-only baseline, but its lower bound omits measured baseline overhead. That overhead can therefore mask missing KV/page allocation.

A CPU probe with 24 MiB additional baseline overhead accepted a peak omitting **1,376,256 bytes of simultaneously live KV storage**.

Add `baseline - charged_baseline_bytes` to the lower bound and test that this understated peak is rejected.

All **13 scoped tests passed**. The original protocol/config remain unchanged; the v2 protocol changes and archive helper’s exclusive outputs and byte verification otherwise look sound. No edits, GPU inference, or publication.