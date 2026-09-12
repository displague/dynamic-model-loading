# Causal evidence: complete result tables

These tables describe the frozen v0.12 development experiment. Every scored row is retained. Cost estimates sum serialized synthetic primitives; they are not measured inference latency.

## All ten quality and acquisition points

| Policy (extra groups) | Relative PPL | Prefixes <=1.01 | Warm GiB/input | Saving | Retained | Repair visits | Fallback visits | Gate B | Gate C |
|---|---|---|---|---|---|---|---|---|---|
| Initial only (0) | 1.22908664 | 0/4 | 2.157557 | 15.615% | 90.000% | 0.000% | 0.000% | Fail | Pass |
| Larger one-shot (28) | 1.11254611 | 0/4 | 2.265229 | 11.404% | 92.500% | 0.000% | 0.000% | Fail | Pass |
| Larger one-shot (56) | 1.05399466 | 2/4 | 2.372901 | 7.192% | 95.000% | 0.000% | 0.000% | Fail | Fail |
| Predetermined repair (28) | 1.10710081 | 0/4 | 2.265229 | 11.404% | 92.500% | 100.000% | 0.000% | Fail | Pass |
| Predetermined repair (56) | 1.06644427 | 1/4 | 2.372901 | 7.192% | 95.000% | 100.000% | 0.000% | Fail | Fail |
| Partial-evidence repair (28) | 1.08948730 | 0/4 | 2.265229 | 11.404% | 92.500% | 100.000% | 0.000% | Fail | Pass |
| Partial-evidence repair (56) | 1.02155126 | 2/4 | 2.372901 | 7.192% | 95.000% | 100.000% | 0.000% | Fail | Fail |
| Current-input only (28) | 1.12954718 | 0/4 | 2.265229 | 11.404% | 92.500% | 0.000% | 0.000% | Fail | Pass |
| Detector / fallback (28) | 1.01161344 | 1/4 | 2.577294 | -0.802% | 99.746% | 100.000% | 96.610% | Fail | Fail |
| Full completion (112) | 0.99999991 | 4/4 | 2.588245 | -1.230% | 100.000% | 100.000% | 0.000% | Pass | Fail |


## Individual Wiki prefixes

- P1: wikitext-validation-59a50d9ab4d0c84c
- P2: wikitext-validation-2c7c602b5ac557ff
- P3: wikitext-validation-9fb0c61a50d8cdb9
- P4: wikitext-validation-b3e260a8369ed0d7

| Policy | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| Initial only (0) | 1.13470094 | 1.16586514 | 1.16827264 | 1.47657683 |
| Larger one-shot (28) | 1.03253688 | 1.09817811 | 1.04665274 | 1.29089557 |
| Larger one-shot (56) | 1.00480884 | 1.11197997 | 0.99920602 | 1.10539674 |
| Predetermined repair (28) | 1.11690177 | 1.10955436 | 1.02683484 | 1.18055034 |
| Predetermined repair (56) | 1.07269150 | 1.10730978 | 0.98442119 | 1.10618491 |
| Partial-evidence repair (28) | 1.07081456 | 1.07252258 | 1.04683515 | 1.17189765 |
| Partial-evidence repair (56) | 1.00036436 | 1.05950977 | 1.00009196 | 1.02739521 |
| Current-input only (28) | 1.04341129 | 1.19749008 | 1.06752344 | 1.22042931 |
| Detector / fallback (28) | 1.01094801 | 1.02693837 | 1.01565553 | 0.99320461 |
| Full completion (112) | 0.99999981 | 0.99999979 | 0.99999970 | 1.00000034 |


## Serialized action accounting

Dense static comparator: **578.222378 ms/input token**. This uses the same 256 MiB workspace allowance, exact-size gathering/packing primitives, and the fastest measured one-addition fixture. The sparse controller also pays its 32 MiB reservation.

| Policy | batch_ms / input | resident_ms / input | controller_ms / input | output_addition_ms / input | total_ms / input | Headroom ms/input |
|---|---|---|---|---|---|---|
| Initial only (0) | 246.144400 | 4.551900 | 35.233841 | 0.568394 | 286.498534 | 291.723843 |
| Larger one-shot (28) | 254.750400 | 4.551900 | 34.544695 | 0.688633 | 294.535628 | 283.686750 |
| Larger one-shot (56) | 264.183300 | 4.551900 | 33.886972 | 1.349791 | 303.971963 | 274.250415 |
| Predetermined repair (28) | 274.763200 | 4.551900 | 42.634258 | 1.053483 | 323.002841 | 255.219537 |
| Predetermined repair (56) | 267.682000 | 4.551900 | 49.004480 | 1.210514 | 322.448894 | 255.773484 |
| Partial-evidence repair (28) | 274.763200 | 4.551900 | 71.876034 | 1.318250 | 352.509384 | 225.712994 |
| Partial-evidence repair (56) | 267.682000 | 4.551900 | 71.698814 | 1.036673 | 344.969387 | 233.252991 |
| Current-input only (28) | 254.750400 | 4.551900 | 45.169556 | 0.515478 | 304.987334 | 273.235044 |
| Detector / fallback (28) | 282.919011 | 4.551900 | 69.389020 | 1.515908 | 358.375839 | 219.846539 |
| Full completion (112) | 283.205200 | 4.551900 | 54.285317 | 1.043441 | 343.085858 | 235.136520 |


All components are in milliseconds. Down-product timing is diagnostic and is not added again to FFN primitives.


The primitive medians are non-monotonic with batch size. Full completion itself
estimates **343.086 ms/input** despite retaining
100% of FFN volume. Its initial/corrective split avoids the expensive larger cold
batch used by the declared dense schedule. This means the apparent headroom does
not isolate an omission benefit. The dense comparator is not optimized over batch
partitions; a future runtime proposal must compare against that stronger scheduling
control. The frozen Gate C result is preserved as a conditional plausibility screen.

## Controller and execution timing

Each entry is a median of ten synchronized repetitions after three warmups, normalized by 896 layer visits. The fixture replays the first 32 inputs of the first Wiki prefix.

| Policy | Workload | Wall ms/visit | CUDA ms/visit |
|---|---|---|---|
| Initial only (0) | dense_ffn | 0.351258 | 0.351162 |
| Initial only (0) | down_products | 0.106840 | 0.106773 |
| Initial only (0) | controller | 1.258351 | 1.258241 |
| Initial only (0) | output_additions | 0.020300 | 0.020250 |
| Larger one-shot (28) | dense_ffn | 0.330062 | 0.329962 |
| Larger one-shot (28) | down_products | 0.120700 | 0.120638 |
| Larger one-shot (28) | controller | 1.233739 | 1.233669 |
| Larger one-shot (28) | output_additions | 0.024594 | 0.024554 |
| Larger one-shot (56) | dense_ffn | 0.362232 | 0.362135 |
| Larger one-shot (56) | down_products | 0.116832 | 0.116735 |
| Larger one-shot (56) | controller | 1.210249 | 1.210181 |
| Larger one-shot (56) | output_additions | 0.048207 | 0.048130 |
| Predetermined repair (28) | dense_ffn | 0.345037 | 0.344941 |
| Predetermined repair (28) | down_products | 0.237522 | 0.237436 |
| Predetermined repair (28) | controller | 1.522652 | 1.522524 |
| Predetermined repair (28) | output_additions | 0.037624 | 0.037579 |
| Predetermined repair (56) | dense_ffn | 0.341690 | 0.341587 |
| Predetermined repair (56) | down_products | 0.247733 | 0.247634 |
| Predetermined repair (56) | controller | 1.750160 | 1.750087 |
| Predetermined repair (56) | output_additions | 0.043233 | 0.043171 |
| Partial-evidence repair (28) | dense_ffn | 0.351200 | 0.351075 |
| Partial-evidence repair (28) | down_products | 0.242533 | 0.242400 |
| Partial-evidence repair (28) | controller | 2.567001 | 2.566916 |
| Partial-evidence repair (28) | output_additions | 0.047080 | 0.046964 |
| Partial-evidence repair (56) | dense_ffn | 0.366393 | 0.366299 |
| Partial-evidence repair (56) | down_products | 0.259589 | 0.259438 |
| Partial-evidence repair (56) | controller | 2.560672 | 2.560596 |
| Partial-evidence repair (56) | output_additions | 0.037024 | 0.036984 |
| Current-input only (28) | dense_ffn | 0.319386 | 0.319187 |
| Current-input only (28) | down_products | 0.123673 | 0.123591 |
| Current-input only (28) | controller | 1.613198 | 1.613129 |
| Current-input only (28) | output_additions | 0.018410 | 0.018368 |
| Detector / fallback (28) | dense_ffn | 0.321376 | 0.321252 |
| Detector / fallback (28) | down_products | 0.265576 | 0.265463 |
| Detector / fallback (28) | controller | 2.478179 | 2.478061 |
| Detector / fallback (28) | output_additions | 0.054140 | 0.054072 |
| Full completion (112) | dense_ffn | 0.350969 | 0.350830 |
| Full completion (112) | down_products | 0.248321 | 0.248186 |
| Full completion (112) | controller | 1.938761 | 1.938692 |
| Full completion (112) | output_additions | 0.037266 | 0.037219 |


## Exact acquisition batch primitives

Dynamic gathering, staging, transfer, packing and execution are inside the cold clock. Startup allocations are outside it. All thirty integrity conditions must pass.

| Groups | Workload | Wall ms | CUDA ms |
|---|---|---|---|
| 28 | resident_ffn | 0.502200 | 0.424000 |
| 28 | gather_transfer_pack_ffn | 1.022100 | 0.998480 |
| 56 | resident_ffn | 0.097150 | 0.084736 |
| 56 | gather_transfer_pack_ffn | 0.769200 | 0.745312 |
| 112 | resident_ffn | 0.094350 | 0.082752 |
| 112 | gather_transfer_pack_ffn | 1.323600 | 1.303248 |
| 446 | resident_ffn | 0.289550 | 0.240432 |
| 446 | gather_transfer_pack_ffn | 6.149950 | 6.115952 |
| 447 | resident_ffn | 0.152800 | 0.133488 |
| 447 | gather_transfer_pack_ffn | 6.357250 | 6.321808 |
| 455 | resident_ffn | 0.152800 | 0.133504 |
| 455 | gather_transfer_pack_ffn | 6.469950 | 6.437680 |
| 456 | resident_ffn | 0.141200 | 0.126496 |
| 456 | gather_transfer_pack_ffn | 6.347850 | 6.323728 |
| 561 | resident_ffn | 0.172900 | 0.157664 |
| 561 | gather_transfer_pack_ffn | 8.258000 | 8.195696 |
| 562 | resident_ffn | 1.264150 | 0.938736 |
| 562 | gather_transfer_pack_ffn | 15.718200 | 15.349553 |
| 589 | resident_ffn | 0.174850 | 0.160240 |
| 589 | gather_transfer_pack_ffn | 9.127550 | 9.098112 |
| 590 | resident_ffn | 0.188250 | 0.169024 |
| 590 | gather_transfer_pack_ffn | 8.717050 | 8.682784 |
| 617 | resident_ffn | 0.267500 | 0.219344 |
| 617 | gather_transfer_pack_ffn | 8.586250 | 8.552064 |
| 618 | resident_ffn | 0.182000 | 0.169696 |
| 618 | gather_transfer_pack_ffn | 20.470400 | 20.232240 |
| 664 | resident_ffn | 0.199300 | 0.185200 |
| 664 | gather_transfer_pack_ffn | 19.518000 | 19.290481 |
| 665 | resident_ffn | 0.304000 | 0.250064 |
| 665 | gather_transfer_pack_ffn | 20.554450 | 20.502416 |


## Matched-byte evidence ablation

| Additions | Aggregate better than both controls | No worse on every prefix | Quality advantage |
|---|---|---|---|
| 28 | True | False | False |
| 56 | True | False | False |


These columns establish only the declared quality comparison. The action-cost table separately shows whether overhead consumes the opportunity.

## Initial, corrective and resident work

Values are normalized over 512 processed Wiki input tokens. The fixed resident core costs 1,845,264,384 weight bytes to preload at cold start; it is retained in this warm simulation. Each executed group contains eight neurons. No probes are used.

| Policy | Initial cold GiB/input | Repair cold GiB/input | Dispatch bytes/input | Resident group computations/input |
|---|---|---|---|---|
| Initial only (0) | 2.157440 | 0.000000 | 125680.000 | 12514.000 |
| Larger one-shot (28) | 2.265106 | 0.000000 | 131952.000 | 12514.000 |
| Larger one-shot (56) | 2.372772 | 0.000000 | 138224.000 | 12514.000 |
| Predetermined repair (28) | 2.157440 | 0.107666 | 131952.000 | 12514.000 |
| Predetermined repair (56) | 2.157440 | 0.215332 | 138224.000 | 12514.000 |
| Partial-evidence repair (28) | 2.157440 | 0.107666 | 131952.000 | 12514.000 |
| Partial-evidence repair (56) | 2.157440 | 0.215332 | 138224.000 | 12514.000 |
| Current-input only (28) | 2.265106 | 0.000000 | 131952.000 | 12514.000 |
| Detector / fallback (28) | 2.157440 | 0.419714 | 150130.125 | 12514.000 |
| Full completion (112) | 2.157440 | 0.430664 | 150768.000 | 12514.000 |


## Detector audit

Post-evaluation diagnostic only: compare the frozen >0.05 detector decision with the isolated dense audit of initial relative FFN error >0.05. These audit values never enter evaluation-time decisions or history. Local FFN error is not a task or PPL failure label.

| True positives | False positives | True negatives | False negatives |
|---|---|---|---|
| 13793 | 57 | 383 | 103 |


## Reused balanced development tasks

The v0.11 chat interface and scalar-answer scoring are unchanged. Its Gate A remains failed. New successes cannot cancel regressions on previously successful dense tasks.

| Policy | Correct /20 | Format /20 | Success /20 | Lost dense successes | New format failures on dense successes | Exact dense token sequences /20 |
|---|---|---|---|---|---|---|
| Native dense | 12 | 16 | 12 | 0 | 0 | 20 |
| Larger one-shot | 8 | 15 | 8 | 5 | 1 | 10 |
| Predetermined repair | 6 | 13 | 6 | 6 | 2 | 7 |
| Partial-evidence repair | 9 | 15 | 9 | 4 | 2 | 10 |


| Domain | Dense successes | One-shot successes | Predetermined successes | Partial successes |
|---|---|---|---|---|
| code | 2 | 0 | 1 | 0 |
| extraction | 4 | 4 | 2 | 4 |
| arithmetic | 3 | 1 | 2 | 3 |
| copying | 2 | 3 | 1 | 2 |
| topic_changes | 1 | 0 | 0 | 0 |


- Larger one-shot lost dense successes: fresh-code-02, fresh-code-04, fresh-arithmetic-03, fresh-arithmetic-04, fresh-topic_changes-01.
- Predetermined repair lost dense successes: fresh-code-02, fresh-extraction-02, fresh-extraction-04, fresh-arithmetic-04, fresh-copying-01, fresh-topic_changes-01.
- Partial-evidence repair lost dense successes: fresh-code-02, fresh-code-04, fresh-copying-01, fresh-topic_changes-01.

## Teacher-forced task reference fidelity

These compare logits on the same frozen prompt/answer prefixes. Answer-only metrics start at the last prompt token; prompt-plus-answer metrics also score prompt tokens. They do not compare logits after independently generated contexts have diverged, and are separate from task success.

| Policy | Scope | Predicted tokens | Relative PPL | Mean KL | Top-1 agreement |
|---|---|---|---|---|---|
| Larger one-shot | Prompt + answer | 1414 | 1.018381 | 1.005849 | 66.973% |
| Larger one-shot | Answer only | 71 | 1.206931 | 0.468664 | 81.690% |
| Predetermined repair | Prompt + answer | 1414 | 0.870216 | 0.817677 | 66.478% |
| Predetermined repair | Answer only | 71 | 1.591323 | 0.571296 | 78.873% |
| Partial-evidence repair | Prompt + answer | 1414 | 0.833839 | 0.824324 | 68.670% |
| Partial-evidence repair | Answer only | 71 | 1.258800 | 0.445240 | 84.507% |



## Generation work and traffic

Every generated path includes its prompt processing. Bytes per processed input and total warm sequence bytes are both reported. A shorter or incorrect output does not establish lower cost at equivalent utility. Cold-start loading additionally acquires the fixed resident core.

| Policy | Processed input tokens | Generated tokens incl. EOS | Total warm GiB | Warm GiB/input | Correction rounds |
|---|---|---|---|---|---|
| Larger one-shot | 1458 | 115 | 3302.704015 | 2.265229 | 0 |
| Predetermined repair | 1491 | 148 | 3377.456575 | 2.265229 | 41748 |
| Partial-evidence repair | 1477 | 134 | 3345.743367 | 2.265229 | 41356 |
