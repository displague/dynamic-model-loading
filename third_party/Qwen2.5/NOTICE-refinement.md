# Qwen-derived progressive-precision research tensors

The v0.23 research archive contains modified tensors derived from
Qwen/Qwen2.5-1.5B-Instruct, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, provided by the Qwen team under
Apache License 2.0. The accompanying LICENSE is copied from that revision:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE

Modifications by this research project, 2026-09-14: the archived
`worker/constructed/layer-*.safetensors` contain affine eight-bit quantized
FFN gate/up/transposed-down weights split into four-bit base codes and two
two-bit increment slabs, with shared FP32 minima/scales. These are modified
research representations, not the original checkpoint or a standalone model.
The transformation code and protocol are preserved in the worker snapshot.
No endorsement by the original model authors is claimed.

The corpus and token IDs retain their separate historical source provenance;
this model license notice does not relicense that data. This notice is release
packaging outside the unchanged experimental worker tree.
