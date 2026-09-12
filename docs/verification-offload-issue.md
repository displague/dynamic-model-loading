Test the existing b10919 GGML_OP_OFFLOAD_MIN_BATCH setting under the prospective verification-offload protocol. Compare thresholds 32 and 8 with fixed 0.5B-draft placement/threads, one disabled-offload mechanism control, isolated backend-assignment diagnostics and repeated sustained-generation measurements. Then run the predeclared populated 16K/q8_0-KV control with its own target-only reference and separate placement. Preserve failures, actual shapes, acceptance events, effective process environment, resource traces and all output IDs. Distinguish graph assignment evidence from physical copy bytes and counter reconciliation from independent KV/argmax verification. The reviewer forecast of 25 tokens/s remains a forecast. No runtime patch or controller is required.

Protocol: https://github.com/displague/dynamic-model-loading/blob/main/docs/verification-offload-protocol.md

Fidelity remains separately tracked in #27. #25 is deferred; #26 follows measured draft-length/backend economics.
