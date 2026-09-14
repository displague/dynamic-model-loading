# Qwen-derived research tensors

The v0.22 research archives contain modified tensors derived from
Qwen/Qwen2.5-1.5B-Instruct, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, provided by the Qwen team under
Apache License 2.0. The accompanying LICENSE is copied from that revision:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE

Modifications by this research project, 2026-09-14: the archived
`worker/constructed/layer-*.safetensors` contain affine two-bit packed FFN
gate/up/transposed-down weights, minima/scales and calibration-basis/error
projections. These are modified research representations, not the original
checkpoint or a standalone deployable model. `projection.safetensors` is the
separately seeded random projection. Source code and precise transformation
rules are preserved in each worker snapshot and protocol. No endorsement by
the original model authors is claimed.

The corpus, token IDs and calibration records retain their separate historical
source provenance; this model license notice does not relicense that data.
