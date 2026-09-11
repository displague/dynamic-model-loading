# Packing pilot and precision diagnosis

Declared before these measurements, following initial commit 119cbbc. This is an
exploratory development experiment. The original BF16 failure remains a failure.

## Precision diagnosis

Use the same pinned Qwen checkpoint and all six smoke diagnostic documents. Compare
native and the original seeded random order under five policies: ordinary BF16,
BF16 with reduced-precision reduction disabled, FP32 down-projection computation,
FP32 computation of the entire FFN, and down-projection computation restored to the
original neuron order. FP32 policies keep BF16 parameters but materialize FP32
temporaries and cast the result back to BF16. They have no performance claim.

Report two independent comparisons: permutation under the same arithmetic policy,
and the policy's native-order result against the original BF16 reference. Retain the
same 0.01 relative-logit-L2 and 0.001 mean-KL limits as descriptive numerical gates.
Restoring canonical summation order requires inverse gathers of activations and
weights; it is a diagnostic, not an efficient implementation of paging.

## Larger packing sample

Dataset: Salesforce/wikitext, wikitext-2-raw-v1, revision
b08601e04326c79dfdd32d625aee71d232d685c3. Read train and validation only; do not fetch or
score test. Reconstruct articles using top-level title boundaries. Exclude title
overlap across splits, hash normalized text to detect duplicate documents, then rank
articles by SHA256(seed, split, title). Use the first 80 eligible training articles
and first 16 eligible validation articles, each with a prefix of 256 tokens. Require
at least 256 tokens/article. Calibration and diagnostics use distinct articles.
This is an English Wikipedia development slice, not a multi-domain held-out result.

Keep the initial Qwen checkpoint and use FP32 to isolate packing from the unresolved
BF16 numerical contract. Preserve the original numerical gates. Do not retune them.
Run all FFNs together at retained group fractions 0.5 and 0.75, group widths 1, 32, and
128, with native, random, popularity, and coactivation layouts. All selectors are
hindsight heuristics; no selection uses future diagnostic tokens to train layouts.

Coactivation layout: retain a uniform priority reservoir of 512 calibration token
importance vectors per layer. Normalize each neuron's observed vector to unit L2,
project onto 64 seeded random sign directions, and recursively bisect neurons using
five power iterations of centered covariance, with capacity-constrained splits into
32-neuron leaf groups. A zero-variance node falls back to current neuron order.
This is a deterministic packing heuristic, not an optimum or a claim about semantic
regions. One learned physical order is reused for all probe group widths.

Document the resulting quality/volume table and paired, whole-document bootstrap
intervals (2,000 replicates, seed 1729) for coactivation minus popularity mean KL at
width 32 and 75% retention. This is an exploratory comparison with no multiple-test
or deployment-equivalence claim. Save the reservoir token ordinals and layout hashes.

## Additional controls and accounting

Verify checkpoint provenance and freeze source/config/data into each new run. The
corpus builder records raw parquet hashes, selected article titles, and token counts.
Raw data are licensed CC BY-SA 3.0/GFDL through the dataset; retain attribution when
archiving excerpts. Do not silently overwrite existing corpus or run paths.

The larger runner may spill reference logits to disk rather than retain the entire
corpus in RAM. Reference I/O is outside baseline timing. The experiment still holds
all weights on the GPU and measures no actual offload traffic. The reported weight
fractions are hypothetical and cannot be interpreted as achieved memory savings.

Any newly found flaw gets its own correction and fresh run; previous receipts remain.
Evidence from this stage can motivate later per-layer omission and bounded-cache
trace studies, but does not replace those measurements.

Sources: [WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html).
