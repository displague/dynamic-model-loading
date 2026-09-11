# Same-interpreter PyTorch 2.12 environment control

Prospective Stage 0 control for issue #5; planned delivery v0.5.0. This is independent
of the cache-trace development screen. Do not replace the measured `.venv` baseline.

Create an isolated environment using the same `C:\Python314\python.exe` interpreter
(Python 3.14.3), with torch 2.12.0+cu130 from the official PyTorch CUDA 13.0 wheel index.
Share the existing system packages and install torch and its matching torchvision
locally. Match every other effective distribution version to the baseline. Record the effective full distribution inventory in both environments
and every version difference. The prepared pair changes torch 2.10.0 to 2.12.0
and torchvision 0.25.0 to 0.27.0, all CUDA 13.0 builds. The initial torch-only
environment failed test collection because the shared torchvision binary was
incompatible; preserve that preparation failure separately from model measurements. An unresolved required dependency, unavailable CUDA,
or failed applicable test blocks model measurements; preserve the failure evidence.
Keep Transformers, tokenizer, checkpoint, attention implementation, CPU threads,
random seeds, TF32 settings, and original numerical gates unchanged.

Repeat the original six-document BF16 arithmetic diagnosis using the exact archived
smoke corpus and config values. Freeze the corpus and expected token/layout hashes
from `results/packing-pilot-20260911/precision-receipted` in the control config before
measurement. For each environment, evaluate native and random layouts under the five
declared policies: ordinary BF16, disabled reduced-precision reduction, FP32 down,
FP32 FFN, and canonical down order. Compare a repacked output with its own policy's
native output. Preserve each policy's departure from ordinary native BF16 separately.
Apply the unchanged relative-logit L2 <= 0.01 and mean KL <= 0.001 correctness limits
to every document; never reinterpret a numerical failure as sparse-quality evidence.

Run both environments on the same committed implementation. Require exact token IDs,
permutations, checkpoint file hashes, and unchanged dependency versions except torch and torchvision.
Archive dense ordinary BF16 logits for direct cross-environment quality comparison;
these comparisons describe environmental drift, not equivalence to external truth.
Report all five policies and every document. Compare the 2.10 repeat with the prior
receipted study before interpreting 2.12 differences.

Complete the full test suite in each environment before measurement and on the final
release candidate. Preserve source snapshots, inventories, failures, and raw receipts.
No performance improvement or practical BF16 sparse execution is established here;
even a passing arithmetic policy needs its own grouped execution and runtime gate.
