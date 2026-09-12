# Stock target-scale speculation: measurement record

The target is Qwen2.5-32B-Instruct Q4_K_M, executed by the pinned stock Windows
CUDA llama.cpp b10919 runtime. This study jointly charges draft storage and the
target residency it displaces. It compares target-only execution, two small
Q8_0 drafts and a separate IQ2_XS version of the target. The compressed draft
shares neither model allocations nor KV storage with its verifier.

**The fastest observed emitted rate is 18.10 tokens/s with the 0.5B draft at
maximum length four, versus 8.52 for target-only execution (2.124x). Strict
target-relative identity fails on sustained generation for every nonbaseline
configuration, including the target-only placement/thread controls. This is an
observed throughput advantage on differing continuations, not demonstrated
identical-output acceleration.**

All 260 candidate scalar-smoke comparisons match their fixed target reference.
The 58-request common-prefix diagnostic also completes, but does not explain the
sustained differences: rebuilt decisions agree across configurations while four
of eleven rebuilt baseline decisions differ from their own original incremental
decisions. The next bounded investigation is [execution-path fidelity #27](https://github.com/displague/dynamic-model-loading/issues/27).
No custom runtime or further repair-controller tuning follows from this result.

## Substrate and storage

The project owner's storage cleanup resolved the v0.13 acquisition block.
Thirteen merged worktrees with no untracked artifacts or unreadable directories
were removed with ordinary `git worktree remove`, recovering about 530 MB.
Eight older checkouts with artifacts or incomplete readability checks were
preserved, as were restoration copies and published measurements.

Eight GGUF files totaling **32,379,130,304 bytes** were downloaded from the pinned
revisions and verified by full-file SHA-256. The five target shards total
**19,851,336,384 bytes**, exceeding this GPU's capacity before KV and workspaces.
The 0.5B and 1.5B drafts occupy 675,710,816 and 1,894,532,128 file bytes; the
32B IQ2_XS control occupies 9,957,550,976. File bytes are not runtime allocations.

All shared tokenizer IDs match. The target and IQ2 draft have 152,064 entries;
the two smaller drafts have 151,936. The native runtime accepts all three draft
representations. This metadata check does not itself establish correct
speculative state handling or output identity.

The machine has an Intel Core Ultra 9 275HX, 24 cores/threads, approximately
32 GiB host RAM and an RTX 5080 Laptop GPU reporting 17,094,934,528 bytes to
NVML. The common measured device-used ceiling is **15,000 MiB**, including
desktop allocations, and host available memory must remain at least **2 GiB**.
These are sampled constraints; loader declarations alone cannot prove absence
of WDDM eviction. The driver is 616.64. Runtime commit is
`d3146f2b56c2db4711ac8391871c9e529d1946d7`.

## Startup amendments and preserved failures

The first attempt stopped before generation because default verbosity omitted
the allocation records required by the harness. Commit `a4ff1a6` prospectively
enabled stock verbosity 4. The next attempt exhausted the host-memory floor
with mmap loading, including when GPU allocation had fallen within its limit.
Neither attempt generated evaluation tokens.

Pinned Windows source makes fragmented mapped-file unmapping a no-op. Mapped
file prefetch and retained host views are plausible contributors to the measured
host pressure; these observations do not fully isolate its cause. Commit
`4f115ae` prospectively changed to stock `--load-mode none`, keeping lazy loading
off and preserving both resource limits. A target loading observation at 48
offloaded layers then retained about 14.96 GB host available, compared with
roughly 235 MB in the earlier mmap attempt. The GPU limit still excluded that
placement. This is a loading observation, not an inference speed comparison.

Non-mmap calibration completed 21 configurations before an unclassified native
CUDA abort at `draft32-ngl7-t8`. It exited 3221226505 during target loading,
before draft loading or generation. A startup-only verbosity-5 diagnostic
reproduced the abort but did not reveal its cause. Both originals remain intact.

Commit `5b4a902` froze a continuation that imports exactly the original 39-case
ledger and executes only the two missing cases, `draft32-ngl7-t16` and
`draft32-ngl7-t24`. Both new attempts explicitly reported CUDA out of memory
during loading and generated no tokens. They are resource failures. The earlier
8-thread attempt retains its distinct native-failure classification because its
own log did not identify an allocation error.

The continuation checks byte identity of the measurement script, catalogue and
both workload files, ancestor source commits, every copied case receipt and
copied environment/driver metadata. It preserves the original ledger rather
than replacing prior attempts. No scored calibration case was rerun.

## Frozen calibration selection

The completed inventory contains **41 attempted configurations: 21 complete,
19 resource failures and one separately classified native startup failure**.
There are 63 generation requests: one warmup and two measured calibration
requests for each completed configuration. Calibration uses a separate authored
library-organization prompt, EOS enabled and a 128-token cap. Draft calibration
uses maximum length 16. It is not the sustained evaluation workload.

| Configuration | Target GPU layers, including output | Target CPU threads | Median native decode interval |
|---|---:|---:|---:|
| Target only | 45 | 16 | 14.897 s |
| 0.5B Q8_0 draft | 44 | 24 | 23.636 s |
| 1.5B Q8_0 draft | 41 | 8 | 26.298 s |
| Separate 32B IQ2_XS draft | 11 | 8 | 39.559 s |

These are two-request medians for 128 emitted tokens. The native decode interval
excludes the first emitted token; it is not full request time. The finite
placement/thread grid and predeclared 1% tie rule do not establish global
configuration optimality. Smaller draft lengths and output fidelity require the
separate evaluation. The K=16 calibration measurements alone favor target-only
execution on this prompt.

All draft transformer/output layers are assigned to the GPU. The stock input
embedding stays on the CPU. Target CPU/GPU layer counts describe native placement,
not a custom neuron selection or a weight-streaming implementation.

The selected calibration runs expose the actual memory competition:

| Configuration | Target CUDA model buffer (MiB) | Draft CUDA model buffer (MiB) | Sampled device-used peak (bytes) | Minimum host available (bytes) |
|---|---:|---:|---:|---:|
| Target only | 12,842.65 | 0 | 15,401,930,752 | 14,048,092,160 |
| 0.5B draft | 12,581.02 | 500.84 | 15,558,365,184 | 13,234,847,744 |
| 1.5B draft | 11,760.04 | 1,564.63 | 15,685,767,168 | 12,036,345,856 |
| 32B IQ2_XS draft | 3,550.23 | 9,246.93 | 15,654,309,888 | 3,844,747,264 |

Model-buffer numbers are rounded native log values. Device-used peaks include
other allocations and the desktop; host minima and device peaks span startup,
warmup and measured calibration requests and need not occur simultaneously.
The large compressed draft displaces roughly 9,292 MiB of target CUDA model
buffers relative to target-only. The separate model can be highly aligned and
still impose substantial verification and host-memory costs. Its measurement is
a non-sharing control, not a formal performance bound on a future shared draft.

## Evaluation and accounting contract

The frozen matrix has 13 configurations: target-only, each draft at maximum
lengths 4/8/16, and three distinct target-only placement/thread-matched controls.
Six sustained code/prose/data prompts run in three recorded orders. Twenty
balanced scalar smoke fixtures separately check target-relative continuation
identity and stopping. A 4,096-token context is capacity; actual prompt and output
lengths must be reported. EOS remains enabled. No 1.5B utility threshold transfers
to this target-scale runtime experiment.

The fixed target-only greedy reference remains authoritative. Compare generated
IDs and stopping state, then replay first divergences on identical prefixes.
One-token diagnostic replay retains ten pre-sampling probabilities but does not
recreate an original verification batch, establish historical KV correctness,
recover full-vocabulary KL or certify stochastic exactness.

Warm request time covers the non-streaming completion HTTP request. Template
application, tokenization, metrics reads, startup and artifact hashing are
recorded outside that interval. Native prefill is not streaming client time to
first token. Artifact hashing warms host file cache, so startup is not a
storage-cold benchmark. Logical I/O counters do not establish physical disk reads.

Accepted-prefix survival uses reconciled native cycle events. Aggregate acceptance
percentages cannot reconstruct that curve. Checkpoint replay, truncated draft
lengths and residual direct decode steps remain visible. Stock logs do not
separately time draft computation, target verification and coordination; dividing
request totals by cycle count does not isolate those components.

No reconstructed v0.12 timing is used as a calibration constant here. The stock
measurements must choose the next implementation task.

## Completed evaluation and fidelity

All **54 native run groups and 574 requests** complete: 54 warmups, 26 fixed
reference requests, and 494 candidate comparisons. The 494 comprise 234 sustained
observations and 260 smoke observations. Every configuration emits 4,608 sustained
tokens across six prompts and three repeats. Prompts contain **73–103 tokens**;
every sustained response reaches the **256-token cap**, with EOS enabled. Actual
prompt-plus-output length is **329–359**, so this is a short-context pilot, not a
4K-context performance study. The smoke tasks retain ordinary stopping behavior.
No timed request is replaced or rerun after inspection.

| Configuration | Emitted/request s | Emitted/native decode s | Observed request-rate ratio | Sustained identical /18 | Smoke identical /20 |
| --- | --- | --- | --- | --- | --- |
| target | 8.521 | 8.699 | 1.000 | 18 | 20 |
| draft05-k4 | 18.102 | 18.489 | 2.124 | 9 | 20 |
| draft05-k8 | 16.464 | 16.780 | 1.932 | 12 | 20 |
| draft05-k16 | 11.724 | 11.875 | 1.376 | 9 | 20 |
| matched-ngl44-t24 | 8.061 | 8.209 | 0.946 | 6 | 20 |
| draft15-k4 | 17.095 | 17.497 | 2.006 | 12 | 20 |
| draft15-k8 | 17.000 | 17.376 | 1.995 | 6 | 20 |
| draft15-k16 | 11.992 | 12.160 | 1.407 | 6 | 20 |
| matched-ngl41-t8 | 6.776 | 6.831 | 0.795 | 9 | 20 |
| draft32-k4 | 8.555 | 8.721 | 1.004 | 9 | 20 |
| draft32-k8 | 8.049 | 8.191 | 0.945 | 6 | 20 |
| draft32-k16 | 6.118 | 6.201 | 0.718 | 9 | 20 |
| matched-ngl11-t8 | 3.614 | 3.640 | 0.424 | 9 | 20 |

The target-only configuration reproduces all 18 sustained references. Every other
configuration differs on two to four of the six unique prompts; those mismatches
repeat in all three runs. There are **114 differing sustained observations**, or
**38 unique configuration/prompt situations**, and **380/494** complete candidate
comparisons match generated IDs and stop state. The 260/260 smoke agreement is a
target-relative correctness check, not a score of answer utility or proof of
sustained-generation equivalence.

The fixed-length comparisons favor short small-model drafts. The 0.5B K=4 rate
is higher than its K=8 and K=16 rates in this workload. The 1.5B K=4 and K=8
aggregates differ by only about 0.56%; repeat variation is larger than that gap.
The separate IQ2_XS 32B K=4 rate is only 0.39% above the plain target aggregate;
this pilot does not demonstrate a gain for it. Its longer settings are slower.
None of these rate comparisons qualifies as identical-output acceleration.

The matched target-only controls retain both the draft configuration's target
placement **and its CPU thread setting**. They measure that combined baseline;
their differences from target45/t16 do **not isolate residency alone**. The IQ2
control reduces target CUDA model storage by approximately 9,292 MiB and its
matched target11/t8 baseline emits 3.61 tokens/s. Strong alignment cannot by itself
pay for a large draft's compute and allocation cost.

## Actual timing and accepted-prefix evidence

Each row below aggregates 18 sustained requests. The three repeat rates are
descriptive; they are not independent-domain confidence intervals. Startup ranges
span the configuration's three sustained groups and smoke group, after artifact
verification has warmed the file cache. Startup is excluded from warm throughput.

| Configuration | Request total (s) | Native prefill total (s) | Native decode total (s) | Three repeat request rates (tokens/s) | Startup range (s) |
| --- | --- | --- | --- | --- | --- |
| target | 540.751 | 10.846 | 529.699 | 8.523, 8.586, 8.456 | 8.94-9.75 |
| draft05-k4 | 254.553 | 5.216 | 249.230 | 18.095, 18.012, 18.201 | 9.79-10.22 |
| draft05-k8 | 279.875 | 5.069 | 274.621 | 16.785, 16.326, 16.292 | 9.31-11.80 |
| draft05-k16 | 393.036 | 4.849 | 388.040 | 11.499, 11.839, 11.841 | 9.73-10.08 |
| matched-ngl44-t24 | 571.622 | 10.198 | 561.319 | 8.140, 7.987, 8.058 | 8.93-9.77 |
| draft15-k4 | 269.555 | 6.062 | 263.360 | 16.571, 17.254, 17.486 | 11.79-12.94 |
| draft15-k8 | 271.065 | 5.708 | 265.197 | 16.956, 16.863, 17.183 | 11.90-13.20 |
| draft15-k16 | 384.268 | 5.190 | 378.950 | 12.298, 11.883, 11.806 | 11.87-13.51 |
| matched-ngl41-t8 | 680.020 | 5.232 | 674.581 | 6.758, 6.810, 6.762 | 9.81-11.55 |
| draft32-k4 | 538.636 | 10.066 | 528.402 | 8.615, 8.561, 8.490 | 20.96-21.49 |
| draft32-k8 | 572.516 | 9.743 | 562.574 | 8.065, 8.116, 7.966 | 21.28-23.59 |
| draft32-k16 | 753.183 | 9.906 | 743.141 | 6.118, 6.092, 6.145 | 21.16-21.74 |
| matched-ngl11-t8 | 1275.158 | 8.885 | 1266.042 | 3.606, 3.613, 3.623 | 15.60-15.90 |

The protocol's emitted/native-decode ratio includes the first emitted token in
its numerator even though native `predicted_ms` excludes that first token. Native
decode-step rates use 4,590 steps rather than 4,608 emitted tokens; the target
value is **8.665** steps/s and the 0.5B K=4 value **18.417**. The table above keeps
both the declared emitted/decode metric and the complete completion-request metric
explicit. Native prefill and decode sums do not include every request overhead.
The sum of HTTP completion intervals includes discarded drafting, verification
and coordination, without isolating their individual times.

Every native draft/accepted aggregate reconciles with the final cycle records.
These are measured accepted-prefix survival probabilities over all recorded
cycles, including cycles that attempt fewer than the maximum near the output cap.
They are not reconstructed from teacher-forced agreement or an independence model.

| Draft / maximum | Cycles | Attempted tokens | Accepted tokens | Mean accepted/cycle | P(A >= 1) | P(A >= 4) | P(A >= 8) | P(A >= 16) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft05-k4 | 1389 | 5526 | 3198 | 2.302 | 0.767 | 0.430 | — | — |
| draft05-k8 | 1080 | 8574 | 3504 | 3.244 | 0.733 | 0.386 | 0.219 | — |
| draft05-k16 | 921 | 14466 | 3663 | 3.977 | 0.713 | 0.365 | 0.192 | 0.075 |
| draft15-k4 | 1269 | 5052 | 3315 | 2.612 | 0.804 | 0.501 | — | — |
| draft15-k8 | 882 | 6987 | 3702 | 4.197 | 0.830 | 0.514 | 0.323 | — |
| draft15-k16 | 720 | 11385 | 3864 | 5.367 | 0.804 | 0.467 | 0.296 | 0.125 |
| draft32-k4 | 1116 | 4434 | 3471 | 3.110 | 0.874 | 0.680 | — | — |
| draft32-k8 | 756 | 6003 | 3831 | 5.067 | 0.833 | 0.651 | 0.472 | — |
| draft32-k16 | 558 | 8769 | 4032 | 7.226 | 0.812 | 0.581 | 0.441 | 0.247 |

The full survival arrays, attempted-length histograms, per-request counters and
checkpoint-replay records are retained in `analysis.json`. Small residual direct
decode steps are accounted for separately from accepted drafts and recorded cycles;
the first emitted token comes from prefill. Rejected suffixes remain in attempted
counts and measured time. The IQ2 representation achieves longer accepted prefixes,
but this does not imply lower time per emitted token. Output-path differences also
mean these acceptance curves are observations of the tested stock paths, not proof
of acceptance against an implementation-independent exact target distribution.

## Paired prompt observations

Each value below is the candidate/target request-rate ratio for the same authored
prompt, using the three repeat totals. The underlying per-repeat pairs remain in
`summary.json`. A prompt with a differing continuation still contributes an observed
rate, not an identical-output speedup; no mismatch is filtered out of the timing
summary. These six episodes do not establish population uncertainty or behavior on
long contexts. The combined aggregate weights request time rather than averaging
these ratios.

| Configuration | code-cache | code-parser | code-sql | data-audit | prose-debug | topic-transition |
|---|---:|---:|---:|---:|---:|---:|
| target | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| draft05-k4 | 2.535 | 2.029 | 2.071 | 2.654 | 1.832 | 1.868 |
| draft05-k8 | 2.713 | 1.766 | 2.009 | 2.708 | 1.637 | 1.431 |
| draft05-k16 | 2.194 | 1.198 | 1.370 | 2.282 | 1.184 | 0.936 |
| matched-ngl44-t24 | 0.943 | 0.964 | 0.934 | 0.951 | 0.953 | 0.931 |
| draft15-k4 | 2.560 | 1.860 | 2.150 | 2.188 | 1.696 | 1.808 |
| draft15-k8 | 2.615 | 1.900 | 2.138 | 2.407 | 1.588 | 1.695 |
| draft15-k16 | 2.033 | 1.428 | 1.413 | 1.926 | 1.009 | 1.160 |
| matched-ngl41-t8 | 0.802 | 0.799 | 0.788 | 0.803 | 0.791 | 0.787 |
| draft32-k4 | 1.117 | 1.050 | 1.040 | 1.080 | 0.937 | 0.849 |
| draft32-k8 | 1.151 | 0.958 | 0.996 | 1.069 | 0.797 | 0.799 |
| draft32-k16 | 0.958 | 0.788 | 0.803 | 0.951 | 0.546 | 0.517 |
| matched-ngl11-t8 | 0.427 | 0.424 | 0.420 | 0.432 | 0.421 | 0.420 |

## Joint memory accounting

All full resource traces, including startup, warmup, measured requests and gaps,
remain within the frozen 15,000 MiB device-used and 2 GiB host-available limits.
The largest compressed-draft peak is only about 7 MiB below the device ceiling;
this is a narrow fit on the measured desktop state, not a deployment margin.

| Configuration | Sampled device-used peak (MiB) | Minimum host available (GiB) |
| --- | --- | --- |
| target | 14408.77 | 13.486 |
| draft05-k4 | 14706.77 | 13.208 |
| draft05-k8 | 14708.77 | 11.399 |
| draft05-k16 | 14708.77 | 13.109 |
| matched-ngl44-t24 | 14130.77 | 13.380 |
| draft15-k4 | 14986.77 | 12.264 |
| draft15-k8 | 14988.77 | 12.215 |
| draft15-k16 | 14988.77 | 11.013 |
| matched-ngl41-t8 | 13268.77 | 10.519 |
| draft32-k4 | 14992.77 | 2.956 |
| draft32-k8 | 14992.77 | 3.641 |
| draft32-k16 | 14992.77 | 3.454 |
| matched-ngl11-t8 | 4610.77 | 3.953 |

Representative first-repeat allocation records show what is charged to both models:

| Configuration | Target / draft CUDA model MiB | Target / draft CUDA KV MiB | Target / draft CUDA compute MiB | Target / draft host model MiB |
| --- | --- | --- | --- | --- |
| target | 12842.65 / 0.00 | 704.00 / 0.00 | 171.83 / 0.00 | 6083.36 / 0.00 |
| draft05-k4 | 12581.02 / 500.84 | 688.00 / 48.00 | 171.83 / 18.88 | 6344.99 / 137.94 |
| draft15-k4 | 11760.04 / 1564.63 | 640.00 / 112.00 | 171.83 / 32.76 | 7165.97 / 236.47 |
| draft32-k4 | 3550.23 / 9246.93 | 160.00 / 1024.00 | 171.83 / 98.01 | 15375.77 / 243.63 |

Target CPU KV is 320/336/384/864 MiB for those four respective configurations;
the draft KV is wholly CUDA-assigned. Each model also declares a 0.58 MiB host
output buffer. Target host compute is 12.01 MiB; draft host compute is
3.76/5.01/12.01 MiB. The complete rounded allocation log lines are included in
`summary.json`, alongside the sampled whole-device and whole-host records.
Buffers are separately allocated: no target/draft model or KV sharing is credited.

Native buffer declarations and 200 ms NVML/host samples do not prove physical
residency at every instant, lack of WDDM eviction, or absence of physical storage
reads. CPU logical I/O counters are retained without interpreting them as disk
traffic. Host and GPU peaks need not be simultaneous. Other desktop allocations
are included in the device-used budget; no unrelated services were stopped.

## Measured aggregate cost model

The directly supported quantity is:

$$t_{\mathrm{emitted}}=\frac{\sum_r T_{\mathrm{completion\ request},r}}{\sum_r N_{\mathrm{emitted},r}}.$$

For target-only it is **117.350 ms/emitted token**; the fastest observed small
draft gives **55.241 ms/emitted token**. Because fidelity fails, these are rates
of stock-emitted output, not certified identical target output. The practical
performance comparator for later proposals is the best complete stock draft
configuration as well as plain offload, with this fidelity limitation retained.

The following divides those same request totals by the number of recorded cycles.
It is an **accounting normalization**, not a direct measurement of cycle latency:
prefill and residual direct decode are inside the request numerator but outside
the cycle count. No per-component draft/verify/control timing is invented.

| Draft / maximum | Whole-request ms / recorded cycle | Emitted tokens / recorded cycle | Whole-request ms / emitted token |
| --- | --- | --- | --- |
| draft05-k4 | 183.263 | 3.317 | 55.241 |
| draft05-k8 | 259.144 | 4.267 | 60.737 |
| draft05-k16 | 426.749 | 5.003 | 85.294 |
| draft15-k4 | 212.415 | 3.631 | 58.497 |
| draft15-k8 | 307.330 | 5.224 | 58.825 |
| draft15-k16 | 533.706 | 6.400 | 83.392 |
| draft32-k4 | 482.649 | 4.129 | 116.892 |
| draft32-k8 | 757.296 | 6.095 | 124.244 |
| draft32-k16 | 1349.791 | 8.258 | 163.451 |

For a hypothetical candidate whose complete, fully charged cycle costs C seconds,
beating a measured rate R by factor S requires E[N] > S C R. This simple sensitivity
calculation holds the supplied cost fixed; it is neither a forecast nor evidence
that a representation attains the necessary accepted length. Actual target/draft
memory, both KV/workspaces, displaced target placement and all rejected work must
already be included in C. All factors 1/2/3 are retained in `summary.json`.

| Hypothetical fully charged cycle cost | Tokens needed to beat target-only | Tokens needed to beat fastest observed draft | Tokens needed for 2x fastest observed draft |
| --- | --- | --- | --- |
| 100 ms | > 0.852 | > 1.810 | > 3.620 |
| 200 ms | > 1.704 | > 3.620 | > 7.241 |
| 400 ms | > 3.409 | > 7.241 | > 14.482 |
| 800 ms | > 6.817 | > 14.482 | > 28.964 |

For example, a 400 ms fully charged cycle must emit more than 7.24 tokens on average
to beat the fastest observed stock draft, or more than 14.48 for a hypothetical
2x advantage over it. This introduces no mandatory multiplier gate. Increasing
draft length also changes verification and draft cost; a fixed-C sensitivity is
not an adaptive-length policy. No v0.12 reconstructed timings or 1.5B acceptance
figures enter this model. The stock observations support investigating fidelity
before investing in a new representation, scheduler or shared-resident runtime.

## Common-prefix diagnostic: reproducible ambiguity

The analyzer identifies each first differing token before contexts separate.
Its original replay plan covers 58 distinct prefix/configuration requests in
13 native groups. An initial diagnostic completes all 11 target-baseline prefixes,
then its exclusive reuse of local port 8099 fails with WinError 10048 before the
second native group starts. The completed 11 responses and failure traceback are
preserved. No surviving native listener was found afterward; the exact port-failure
cause is not established.

Commit `1cf1101` prospectively assigns distinct ports 18099–18111, records the
mapping, validates the range and preserves probe failures. A fresh diagnostic
directory completes the same **58 cases / 13 groups**. It repeats the first eleven
baseline cases explicitly as a replication check, without replacing either the
old diagnostic or any timed request.

- All **38 unique baseline/matched/candidate comparisons return the same next
  token** after rebuilding an identical full prefix. These correspond to the
  114 original differing observations across three repeats.
- The rebuilt baseline matches its original incremental reference decision on
  only **7/11 unique prefixes**.
- All **11/11 repeated baseline next tokens and top-ten probability records**
  match the first diagnostic attempt exactly.

These observations reproduce the ambiguity; they do not resolve it. The diagnostic
uses a rebuilt prefix, a one-token cap and probability reporting, rather than the
original incremental KV trajectory and verification batch. It does not prove harmless
numerical drift, a KV rollback bug, or identical-output correctness. Top-ten records
cannot recover full-vocabulary KL. The fixed target-only reference is not redefined.

## Provenance, reproduction and next action

Calibration and evaluation are bound to source
`5b4a902b0ccff26c5313aeb5eef7a5a35a46e49a`; the native measurement script itself
is unchanged from `4f115ae`. The corrected port diagnostic uses `1cf1101`.
The original calibration ledger digest is
`4af2c1c287491b0416bf52f3fca2970118b12055c7336f33476a31c7e03004c0`;
the completed continuation ledger digest is
`0010c43dcee60166aecb36eb7ae067a41630c1d11989660a7461339f30e6c466`.
Selection is frozen before evaluation, digest
`e3dbf3301d13e77ff4edb39ffa53bc2cf809671560eaddc84491784c8c55d7b3`.

The [committed receipts and reproduction instructions](../results/stock-speculation-20260912/README.md)
link five losslessly verified raw archives. They retain startup failures, all
calibration attempts, the continuation, complete evaluation, both diagnostics,
native commands/logs, compact generated-token records and resource traces.
The 39 imported calibration cases are copied receipts, not 39 additional runs.
Generated IDs, request/response bodies and full acceptance records precede compact
tables; no full-vocabulary logit spill is needed for the declared metrics.

Bounded issues [#23](https://github.com/displague/dynamic-model-loading/issues/23)
and [#24](https://github.com/displague/dynamic-model-loading/issues/24) are completed
as measurement and aggregate-analysis deliveries. Their completion does not qualify
the speedup hypothesis. [Issue #27](https://github.com/displague/dynamic-model-loading/issues/27)
will isolate prefill versus incremental state, batch/request shape, probability
reporting, placement and threads under a prospective fixed-reference protocol.
The stock milestone stays open. Representation #25 and adaptive-control #26 remain
optional and deferred. All historical sparse-execution failures retain their
original meanings under ADR 0003.
