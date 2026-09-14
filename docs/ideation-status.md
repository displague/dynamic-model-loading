# Original ideas and what has actually been tested

This inventory is retrospective. It changes no historical protocol or result.

| Idea | Status and evidence | Remaining scope |
|---|---|---|
| Activation-aligned pages |Popularity/co-activation packing and analytical traces were tested in v0.2-v0.4. v0.20 physically transfers contiguous 256-neuron paired pages and passes dense-completion checks; it does not establish a benefit from activation-aligned packing.|Successful physical selective transfers/kernels on32B remain untested.|
| Activation/history side index |v0.7/v0.12 tested causal controllers analytically. [v0.20](fault-pager-results.md) physically executes a current-FFN-input medoid-index draft, with 30.2% verified proposal acceptance.|Useful acquisition and target-scale behavior remain open research questions.|
| Miss, acquire, correct before commitment |v0.8/v0.12 tested additive repair. v0.20 implements real residency misses and dense target verification of a selected draft; all72 committed outputs match.|The tested acquisition economics fail. This is not target-relative omission certification or a32B gain.|
| Literal zero sentinel / demand fault |v0.20 uses explicit residency metadata and recorded physical demand completion, not numerical-zero inference.|Residency metadata detects a requested absent block, not an important block never requested. A zero-sentinel kernel remains unbuilt.|
| Related-token lookup as a weight fetch key |v0.20 tests top-two page prefetch from the prior token's side-index lookup. It increases traffic at both budgets.|This is not a semantic token-to-weight map; that broader idea remains untested.|
| Dense model with slice routing |Partially tested by causal FFN selection, not trained into a new routed model.|Target-scale router training or physical sparse execution remains deferred.|
| Self-derived approximation |Non-sharing low-bit32B draft measured in v0.14.|A shared-component resident draft, pruned/low-bit/layer-skipped frontier remains deferred (#25).|
| Persistent execution |Stock continuing-conversation fixture (#29); v0.23 physically retains precision increments across consumed draft tokens. Fixed retention saves 12.5% traffic; value-ranked retention saves 7.25% and is worse than fixed.|This tiny per-document test does not qualify long sessions, an autonomous service, or larger-model access.|
| Approximation plus residual / progressive precision |v0.22 tests a two-bit base plus full FP32 pages (0/64 accepted). [v0.23](refinement-screen-results.md) implements embedded 4/6/8-bit weights and real two-bit increment loading; eight-bit proposals match 8/8 on the tiny screen.|General faithful drafting and a competitive memory/quality/time frontier remain open; the v0.23 adaptive controller always reaches eight bits.|
| Uncertainty-aware acquisition / formula transfer |v0.23 tests an inter-precision norm proxy and a prospectively registered Richardson extrapolation diagnostic. Extrapolation worsens 28/28 local layers despite fourfold norm contraction.|The directional leading-error assumption fails on those inputs. Statistical error prediction and PMPD-inspired prefill controls remain untested, not disproven.|
| Hierarchical chunks, multi-layer lookahead, prompt-conditioned initialization |Still deferred hypotheses.|No result disproves these broader representations or controllers.|
| Block-wise dense acquisition |Measured physical execution in v0.15.|Speculation supplies weight reuse; target verification controls committed output. Practical long conversations and placement tuning follow.|

A block-union study could measure union and cold-union bytes for retained1.5B masks
over1/5/9/17-token windows. A dense union would constrain those masks' transfer savings;
per-position sparse arithmetic could still remain. Hindsight approximation masks
are not exact-target certificates, and1.5B results do not settle quantized32B behavior.
This optional offline analysis does not block the practical stock configuration.
