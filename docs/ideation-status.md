# Original ideas and what has actually been tested

This inventory is retrospective. It changes no historical protocol or result.

| Idea | Status and evidence | Remaining scope |
|---|---|---|
| Activation-aligned pages |Partially tested: popularity/co-activation packing, widths, quality curves and cache traces, v0.2-v0.4.|Successful physical selective transfers/kernels on32B remain untested.|
| Activation/history side index |Tested analytically: static, recency, EMA, learned causal prediction, current-input/history/residency and partial-evidence controllers, v0.7/v0.12.|Stronger predictors and useful target-scale physical execution remain deferred.|
| Miss, acquire, correct before commitment |Partially tested: privileged additions v0.8 and causal additive repair v0.12.|Economical physical recovery and target-relative omission certification were not demonstrated.|
| Literal zero sentinel / demand fault |Requires a distinct contract.|Residency metadata can detect an explicitly requested absent block; it cannot detect an important block never requested. Zero-valued weights do not supply that missing signal.|
| Related-token lookup as a weight fetch key |Deferred.|Separate draft models propose tokens; they do not constitute that lookup experiment.|
| Dense model with slice routing |Partially tested by causal FFN selection, not trained into a new routed model.|Target-scale router training or physical sparse execution remains deferred.|
| Self-derived approximation |Non-sharing low-bit32B draft measured in v0.14.|A shared-component resident draft, pruned/low-bit/layer-skipped frontier remains deferred (#25).|
| Persistent execution |New bounded continuing-conversation fixture (#29).|Autonomous agent service, context consolidation and always-on memory policies remain untested.|
| Hierarchical chunks, approximation plus residual, uncertainty-aware acquisition, multi-layer lookahead, prompt-conditioned initialization |Deferred hypotheses.|No result disproves these broader representations or controllers.|
| Block-wise dense acquisition |Measured physical execution in v0.15.|Speculation supplies weight reuse; target verification controls committed output. Practical long conversations and placement tuning follow.|

A block-union study could measure union and cold-union bytes for retained1.5B masks
over1/5/9/17-token windows. A dense union would constrain those masks' transfer savings;
per-position sparse arithmetic could still remain. Hindsight approximation masks
are not exact-target certificates, and1.5B results do not settle quantized32B behavior.
This optional offline analysis does not block the practical stock configuration.
