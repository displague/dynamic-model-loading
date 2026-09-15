# Qwen-derived Bayesian acquisition research artifacts

The v0.24 research archive contains the same modified embedded-precision FFN
tensors as v0.23, derived from Qwen/Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, by the Qwen team under Apache 2.0:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE

Project modifications, 2026-09-14: group-128 affine eight-bit gate/up/transposed
down weights stored as four-bit bases and two two-bit increments with shared
FP32 minima/scales. The new archive also includes statistical error surrogates,
features/observations, local numerical checks and execution receipts. These are
research representations, not an original or standalone Qwen checkpoint. Source
transformations and protocol are archived. No endorsement is claimed.

The corpus/token IDs retain their historical source provenance and separate
licensing. This notice and the upstream LICENSE are packaging outside the
immutable experimental worker tree.
