# Bayesian error prediction and explicit prefill-KV screen

The covariance predictor improves **log-error prediction MAE by 30.5921%** over
a constant per-layer mean. That statistical component did not establish useful
acquisition: uncertainty-directed promotion accepts 7/8 proposals versus 8/8 for
uniform six bits and uniform eight bits, and saves only 0.3151% traffic relative
to the constant-bound controller. All four registered compound gates fail. Stop
this configuration's expansion; do not close Bayesian or novel-loading research.

## Scope and provenance

- [Prospective protocol](bayes-screen-protocol.md), [configuration](../configs/bayes-screen.json),
  [inspiration ledger](bayes-inspirations.md), [bounded issue #39](https://github.com/displague/dynamic-model-loading/issues/39).
- Reviewed source freeze, pushed before checkpoint inference:
  [`4cb6f5f78bf224ba3293b92756993b892680ae7d`](https://github.com/displague/dynamic-model-loading/commit/4cb6f5f78bf224ba3293b92756993b892680ae7d).
- Fresh run `runs/bayes-screen-20260914-v1`; **179.970049 seconds** supervised worker
  plus **8.984892 seconds** independent CPU analysis = **188.954940 seconds**.
  Five-minute worker includes imports, checkpoint hashing/loading, construction,
  calibration, fitting, diagnostic shadow work, warmups, scored inference and raw
  hashing. A second model-free audit reproduces the complete summary exactly.
- Unchanged pinned Qwen2.5-1.5B-Instruct artifact, `.venv`, FP32 target arithmetic,
  torch 2.10.0+cu130 / transformers 5.13.1 / Python 3.14.3. Actual CUDA execution.
  No llama.cpp changes, larger-model inference, long suite or native admission.
- First eight calibration documents fit the model; next nine calibrate block
  maxima. Two reused diagnostic documents are development evidence, not unseen
  qualification. Each supplies eight prefill tokens and four teacher-forced shadow
  decode positions. Scored generation is separately own-trajectory, K4, four
  committed output tokens per document, one repetition, reversed condition order.

## What was implemented and what the statistical result means

One Gaussian process per FFN layer uses 18 features: a seeded 16D projection of
the current normalized 1536D draft input, log 4-to-6 change and log output norm.
It predicts the log normalized magnitude of the unfetched 6-to-8 correction.
Shrinkage covariance defines feature geometry; an RBF kernel supplies posterior
mean and predictive uncertainty. This is Bayesian function prediction, not vector
Richardson extrapolation, a full residual-field posterior, or online assimilation.

Fit/calibration uses current **draft-state teacher-forced** inputs. Shadow y8 is
observed only to label the missing correction, then y6 is returned so subsequent
states remain six-bit. No target activations/logits or acceptance outcomes enter
the predictor. Parameters are frozen before diagnostics and generation; skipped
increments are not secretly fetched during scored generation.

| Diagnostic, 224 rows / 56 layer-document blocks | Result |
| --- | ---: |
| Constant layer-mean log-error MAE | 0.0880599480 |
| GP mean log-error MAE | 0.0611206045 |
| Relative MAE improvement | 30.5921% |
| GP calibrated upper block coverage | 50/56 = 89.2857% |
| Constant calibrated upper block coverage | 52/56 = 92.8571% |
| GP mean threshold-safe / false-safe rows | 224 / 0 |
| GP upper threshold-safe / false-safe rows | 220 / 0 |
| Constant upper threshold-safe / false-safe rows | 216 / 0 |

The GP's point-error component passes its 10% improvement threshold, but Hcal's
coverage component falls below the unchanged 90% requirement. Do not round
89.2857% into a pass. The constant bound covers more blocks. Nominal calibration
requires exchangeability not demonstrated by this fixed corpus; these are
empirical counts, not guaranteed generation coverage or target correctness.

The logged shadow errors have maximum **0.0575225**, below the fixed **0.06**
acquisition threshold. Consequently both mean and upper predictions have zero
false-safe rows. Hunc requires a strict reduction and fails; this subset does
not exercise above-threshold hazards. That is a test limitation, not evidence
that uncertainty is useless. The threshold was not moved after seeing results.
GP upper misses occur in layer-document blocks (doc0, layer5) and (doc1,
layers0/4/5/7/19). These are descriptive readings of saved receipts, not a second
selection or a new measurement.

## Physical generation results

Every row below commits the same eight target-reference tokens across two prompts.
Every candidate consumes 24 draft tokens: 16 total prefill and eight proposals.
Times include charged acquisition, synchronization, feature readback, prediction,
prefix-KV hashing, verification and ledger flush; setup/calibration is separate
but included in supervised worker time. H2D includes prefill and rejected work.

| Condition | Accepted / proposed | Seconds | Increment H2D GiB | Decode FFNs at 8 bits |
| --- | ---: | ---: | ---: | ---: |
| Six-bit prefill and decode | 8/8 | 6.1880 | 4.7681 | 0/224 |
| Eight-bit prefill, six-bit decode | 8/8 | 8.2849 | 9.0747 | 0/224 |
| Eight-bit throughout | 8/8 | 9.3794 | 11.2280 | 224/224 |
| High prefill + GP mean | 8/8 | 8.5132 | 9.0747 | 0/224 |
| High prefill + calibrated GP | 7/8 | 8.4930 | 9.1228 | 5/224 |
| High prefill + constant bound | 7/8 | 8.5872 | 9.1516 | 8/224 |

The scalar resident target alone takes **0.5029 seconds**. That is an in-apparatus
control, not a target-scale or stock llama.cpp benchmark. The first scored target
reference takes 0.4050 seconds versus 0.0980 for the second despite a separate
short warmup; one-repeat timing variation remains visible in raw episodes.

Higher-precision prefill adds cost without an acceptance benefit on this subset:
both q6 and p8d6 already accept 8/8. Its 19.1781% traffic saving versus all-eight
comes from six-bit decode, not a measured improvement over all-six. Hprefill fails
its acceptance requirement; a ceiling-limited subset cannot settle harder prompts.

GP mean requests no decode eighth-bit increments and reproduces p8d6's policy.
Its prediction overhead adds work rather than demonstrating useful selection.
Calibrated GP saves **18.75%** traffic versus all-eight, but loses **12.5 percentage
points** of acceptance (the allowed loss was ten), and beats the constant bound
by only **0.3151%** traffic (required five). Hpolicy fails both those components.
The GP's 9.45% observed time reduction versus all-eight is not a reliable speedup
claim or superiority to the simpler six-bit control.

All five GP promotions and all eight constant-control promotions are at layer1.
The two calibrated policies each reject one proposal on the second document;
all-six, all-eight and the GP-mean policy do not. Selectively increasing local
precision is therefore not monotonic in draft agreement in this sample. These
receipts do not isolate the causal contribution of each promotion or establish
that local FFN norm error predicts downstream logit/argmax sensitivity.

## Explicit KV contract, correctness and resources

Only the first N consumed draft inputs use prefill precision. The last prefix
logit predicts the first proposal; no decode/fallback call re-enters prefill.
Both caches independently crop to base+accepted on rejection; accepted mixed
states remain and rejected suffix states are discarded. Changing FFN precision
does not rebuild old KV, and no target KV is copied into the draft. K/V storage
is FP32 throughout; eight-bit refers to FFN weights only.

Actual prefix K/V SHA256 is unchanged at every later step and crop. q6/p8d6/p8d8/
p8mean each pay **4,587,520 bytes** of prefix readback across scored episodes;
GP/constant upper policies pay **5,046,272 bytes**, including the rejection crop.
These instrumented costs are included in time. All 12 scored outputs and six
warmups match the dense target IDs and stop reason. Independent replay verifies
four-proposal blocks, fallback, physical slots/copies and both cache lengths.
CPU tests separately force rejection at positions0/1/2/3 and verify continued
fallback consumption; final-position rejection in the measured sample ends at cap.

| Physical charge | Bytes / observation |
| --- | ---: |
| Target CUDA parameters | 6,174,857,216 B |
| Separate draft non-FFN CUDA parameters | 1,550,637,056 B |
| Draft original CPU FFNs (not discarded) | 4,624,220,160 B |
| GPU resident base/minima/scales | 650,280,960 B |
| GPU reconstruction workspace | 254,607,360 B |
| Eight persistent slabs + bypass | 92,897,280 B |
| GPU feature projection | 98,304 B |
| Host packed increments | 578,027,520 B |
| Pinned staging | 10,321,920 B |
| CPU GP tensors / reported Python metadata | 447,328 / 24,483 B |
| Fit/calibration observation Python metadata | 1,788,312 B |
| Peak extra CUDA, all phases | 993.107910 MiB <=1024 MiB |
| Peak sampled total GPU | 8987.097656 MiB |
| Minimum sampled available host RAM | 5266.582031 MiB |
| Peak sampled process RSS | 9411.113281 MiB |

All original host backing, actual target/draft residency, allocator overhead,
construction, scratch, cache, projection and joint KV remain charged. No weights
are shared. Traffic savings are not VRAM savings: the same allocated apparatus
is used across modes. The 19 shadow documents take **92.5004 seconds** and transfer
**114,532,024,320 bytes**; GP fitting itself takes about **0.126 seconds**. Calibration
acquisition is real setup cost, not free training, and its long-session amortization
is untested. This 1.5B reference already fits; no larger-model capacity frontier,
longer-context access, retained-session benefit or deployment speedup is established.

## Validation, archive and next question

Pre-inference review resolved a NumPy summary-serialization bug before any checkpoint
run. **415 CPU tests passed** on the frozen source candidate. Independent replay
reconstructs GP fitting/calibration with NumPy rather than runtime GP methods;
every saved packed tensor is byte-identical to v0.23, and all 28 numerical outputs
match the parent's eight-bit outputs exactly. The unchanged rel-L2 tolerance .01
is a mechanics check, not eight-bit equality to FP32.

[Compact receipts](../results/bayes-screen-20260914/) contain summary, episodes,
phases, allocation, fit provenance, reviews and validation. The release attaches
the final clean-tip test receipt and commit-bound validation separately. Raw
archive `bayes-screen-v024-raw.zip`: **1,261,549,089 bytes**, **252 members**,
SHA256 `7bdbf9ef97c536daaa3f089523bcefbacc216cb25616b1285eee576c3b386cd6`.
Every member was decompressed/SHA-checked and the raw source files rechecked.
It includes self-contained constructed representations, raw feature/label and
physical/KV ledgers, frozen source, environment, model/corpus provenance and notices.

Reproduce (fresh directory, reviewed clean pushed source, pinned local artifacts):

```powershell
.\.venv\Scripts\python.exe -m dynamic_model_loading.bayes_screen --output runs/bayes-screen-<fresh-name>
.\.venv\Scripts\python.exe -m dynamic_model_loading.bayes_screen --analyze --output runs/bayes-screen-20260914-v1/worker
```

Preserve the positive component: state covariance predicts remaining local error
better than a layer mean. The unsatisfied link is from that local prediction to
the value of acquiring precision for a draft token. A future short protocol could
test downstream/logit-sensitive uncertainty and a prospective hazard-containing
subset, keeping ordinary examples and simple controls. Neither that next design
nor a new threshold has been measured here. No long suite or native pivot follows
from this run; [milestone10](https://github.com/displague/dynamic-model-loading/milestone/10)
and the broader research question remain open.
