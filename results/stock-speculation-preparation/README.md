# Stock runtime preparation, not a measured inference experiment

This directory records binary help, version and hardware/storage observations for
the prospective stock speculation comparison. `configs/stock-speculation-artifacts.json`
pins every selected model file and stock executable/DLL hash. Model hashes originate
from pinned Hugging Face LFS metadata; they are not local-download receipts.
The runtime archives were downloaded and verified against GitHub release digests.

No target inference, placement calibration or acceptance measurement has run.
The 32.38 GB model catalogue exceeds present free disk space. A storage clarification
is pending. The measurement harness must receive its own review and clean pushed
preregistration before model evaluation. The authored workload and prospective
protocol are part of this delivery; they are not scored results.

The stock source is commit `d3146f2b56c2db4711ac8391871c9e529d1946d7`:
`git clone --depth 1 --branch b10919 https://github.com/ggml-org/llama.cpp.git`.
The two Windows x64 CUDA 13.3 ZIPs are obtainable from its official b10919 release.
Verify catalogue hashes before executing them. CLI inspection identified
`LLAMA_TRACE=1` acceptance logs; source inspection alone cannot establish runtime
telemetry integrity or performance.
