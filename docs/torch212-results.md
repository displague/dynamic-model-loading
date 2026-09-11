# Same-interpreter PyTorch 2.12 arithmetic control

PyTorch 2.12.0+cu130 does **not** resolve the BF16 permutation failure. Four of the
five arithmetic policies still fail the original 0.01 relative-logit L2 / 0.001 KL
limits. Canonical down-projection order remains exactly equal to its own environment's
ordinary BF16 reference. The torch 2.10 repeat reproduces all 60 archived rows exactly.

| Policy | 2.10 maximum L2 | 2.10 maximum KL | 2.12 maximum L2 | 2.12 maximum KL | All documents pass, both environments |
|---|---:|---:|---:|---:|---|
| Ordinary BF16 | 0.029594 | 0.001459 | 0.027114 | 0.001511 | No |
| Reduced-precision reduction off | 0.020588 | 0.001264 | 0.021761 | 0.001368 | No |
| FP32 down projection | 0.019678 | 0.001226 | 0.025195 | 0.001320 | No |
| FP32 FFN | 0.020914 | 0.001292 | 0.017395 | 0.001273 | No |
| Canonical down order | 0 | 0 | 0 | 0 | Yes |

Each comparison is a repacked output against its **own policy's native output**.
The complete raw ledger also preserves each policy's departure from ordinary native
BF16. Canonical inverse gathers remain diagnostic; their exactness does not establish
a practical group-local implementation or its cost.

Ordinary native BF16 outputs also change across the environments, with identical
checkpoint and token IDs. Relative logit L2 ranges 0.015831–0.024369; KL ranges
0.001018–0.001567; top-1 agreement ranges 95.51–98.78%. All six cross-environment
documents exceed the original numerical limits. Relative PPL ranges 0.992949–1.008763.
This comparison measures environmental drift, not which implementation is closer to
external truth, and not sparse-task equivalence. It supports retaining the baseline
and recording future environment changes explicitly.

The protocol and implementation were pushed as `fe1dd44` before both runs. Both use
Python 3.14.3, Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, the six original smoke documents (469 input
tokens / 463 predictions), Transformers 5.13.1, SDPA, four CPU threads, unchanged
TF32 settings, and identical random permutations. Full distribution inventories
confirm only torch and torchvision differ: 2.10.0/0.25.0 versus 2.12.0/0.27.0, all
CUDA 13.0 builds. A torch-only installation initially failed test collection because
the shared torchvision binary was incompatible. The matching companion resolved it;
all other effective package versions were aligned with the baseline. The initial
failure has an operator receipt, not a complete original terminal transcript.
The shared optional torchaudio package remains 2.10.0 and declares a torch-2.10 pin;
audio is outside the evaluated text-model path. Direct requirements for this project,
torch, torchvision, Transformers, and safetensors are satisfied. This control validates
the measured text path, not every installed optional package in the shared environment.

Both environments pass all 59 tests. Input guards verify the full inventory, Python,
checkpoint files, corpus, token IDs, and permutations before scoring. Review found
and corrected missing invariant-settings checks, misattributable logits bundles,
and lost early failure receipts. The analyzer requires the original gates and
anchored historical configuration, checks all 60 comparisons in each environment,
recomputes every policy aggregate, and validates logits bound to their producing
manifest. Follow-up review found no actionable regressions.

[Raw receipts and inventories](../results/torch-control-20260911/) are indexed by
`archive.json`. Dense logits are a separately hashed `torch-logits-v0.5.0.zip` release
asset, with lossless gzip payloads; the archive README describes restoration. No
historical raw data or baseline environment was replaced. This completes issue #5;
practical BF16 grouping remains issue #4. OPT's architecture-specific ReLU positive
control and the nominated causal-predictor study remain independent next experiments.
