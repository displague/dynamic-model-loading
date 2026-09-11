# Causal selection at the nominated Qwen operating point

Prospective Stage 3 development experiment, issue #10, intended delivery v0.7.0.
This protocol must be committed and pushed with the apparatus before calibration
fitting or evaluation. It follows the frozen v0.4 choice: Qwen2.5-1.5B-Instruct,
popularity packing, width 8, 90% group retention, 2 GiB equal-layer static cache.
The ReLU positive control does not alter this choice.

## Inputs and gates

Use the pinned Qwen revision, corpus, saved layouts and calibration mean importance
from v0.4 under baseline Python 3.14.3 / torch 2.10.0+cu130, FP32, SDPA, four CPU
threads, TF32 disabled, seed 1729. Require parent manifest, raw ledger, summary,
corpus, layout and exact-token hashes. The 80 calibration articles and 16 reused
development articles remain distinct; do not access final held-out inputs.

Use actual batch-one, one-token-at-a-time teacher forcing with KV caching. Record
native dense incremental logits and require the popularity all-group incremental
path to pass the original 0.01 relative-L2 / 0.001 mean-KL limits on all 16 articles.
Also require width-8 group reconstruction on every layer with L2 <=0.01. Block all
selector probes if either gate fails. Record fresh dense prefill using the same
parent inputs versus incremental dense differences separately; do not assume prefill and decode arithmetic or masks
are identical. Save dense logit hashes and exact token IDs.

Tiny CPU fixtures exercise orchestration explicitly; they return `cpu_fixture_completed`,
have no CUDA timing and cannot pass the screen. The measured CLI requires CUDA and
does not fall back. A zero-byte dense simulated baseline has undefined relative
savings and cannot pass the traffic screen.

## Frozen selectors

Every causal mask is finalized from the normalized FFN input and prior observations
before that token's gate/up computation. Keep exactly ceil(0.9*1120)=1008 groups per
layer. Rank in stable descending score order, breaking ties by packed group index.
Weights remain dense in this diagnostic; independent audits may compute omitted
activations but those values must never feed any causal selector.

* Static: the top calibration-mean group importance, constant through each episode.
* Recency: start from calibration priors. After a token, replace observed selected
  group scores with that token's actual summed abs(z)*down-column-norm score; reset
  unobserved groups to their calibration prior. Use this state on the next token.
* EMA: start from the same priors. Update only observed selected groups by
  state=0.9*state+0.1*observed; keep unobserved state unchanged. No dense audit feedback.
* Learned: per layer, a fixed Gaussian input projection with 64 columns, seed
  1729+layer, scaled by 1/sqrt(hidden). Features concatenate projected input and its
  absolute value. Fit a linear group-importance regressor from calibration only,
  using FP64 sufficient statistics, per-feature mean/std (floor std at 1e-6), an
  intercept and ridge 0.01 on normalized covariance. Solve on CPU in FP64; store
  inference tensors in FP32. No development fitting, hyperparameter search or refit.
* Hindsight: rank the current full abs(z)*down-column-norm group scores. This is a
  noncausal incremental quality control, not an eligible predictor.

Calibration fitting uses the dense popularity layout with the original per-document
prefill inputs; inference uses the causally available current incremental input.
Record this feature-distribution difference. Store sufficient statistics and fitted
tensors with hashes. No labels from development or generated tokens enter fitting.

## Measurements

Run all five selectors on all 16 development articles with state/KV reset per
article. Preserve token-weighted NLL, relative PPL, KL, top-1, document metrics and
actual per-token/layer group bits. Independently record local omitted-output L2
relative to the full FFN output, retained importance fraction and overlap with the
same-token hindsight mask. Report mean and p95, including error >0.1 descriptively.
At 90% retention random set overlap is already high; overlap alone is not quality.
Audit buffers, full dense projections and duplicate output projections are diagnostic
costs, excluded explicitly from selector-only timing, never counted as savings.

Run one additional topic-transition episode formed by concatenating the first two
development token sequences, without resetting KV or selector state at the boundary.
Preserve whole-episode and post-boundary quality/traffic separately. This is a fixed
development transition, not a multi-domain stress test.

For closed-loop inspection use the first 32 tokens of each of the first four
development documents, then generate exactly 64 greedy tokens, including EOS IDs
without early termination. Feed each selector its own generated tokens and state;
save every token sequence and decoded text plus dense controls. Report divergence
from dense, not an external task score. Prompt processing is also incremental.
These are plain-text continuations without a chat template. Preserve generated-path
group masks and independent audits alongside the complete token sequences.

Replay the actual group bits in their measured token/layer order using the existing
equal-layer static cache policy, cold preload and one incoming slot. Charge all
persistent inference selector tensors inside the 2 GiB variable budget, reducing
available FFN slots. Compare warm bytes with the strongest dense static baseline
given the full 2 GiB and the parent grid's widths. Report cold bytes and preload
separately. Learned tensors, state, ranking buffers retained between calls and
down-column norms used by recency/EMA are not free. Training/auditor memory is listed
separately; this remains a simulation without bounded total allocator memory.

Measure selector-only query plus selected-observation update on calibration fixtures
(the first 32 token FFN inputs/activations from the first calibration article, all
28 layers). Use one-token shapes, three warm repetitions and ten measured repetitions,
CUDA events and synchronized CPU wall time. Time the same resident dense FFNs on
those fixtures. Preserve all repetitions, report medians. Observations supplied to
the update are masked to selected groups; dense audits are disabled for this timing.
Do not interpret resident-FFN comparisons as end-to-end paged runtime measurements.

## Development decision

An eligible causal selector must meet aggregate relative PPL <=1.01 and >=10% warm
simulated byte reduction against the strongest dense static baseline, with its
persistent storage charged. Adopt selector CUDA and wall median time <=10% of the
matching resident dense-FFN time as a conservative provisional affordability screen.
Publish every metric even on failure. This screen is not a deployment criterion;
held-out multi-domain quality and actual runtime gates remain open even on a pass.

Freeze no new operating point from failed rows in this run. If none passes all three
screens, document the failed causal opportunity and leave the pager/refinement gates
closed. A failed initial selector family is not proof that no predictor could work;
any new predictor or threshold requires a separate prospective experiment. If a
selector passes, nominate by lowest charged warm bytes, then KL, then name for a
separate bounded execution protocol. Never equate residency misses with silent
omission errors or label closed-loop text inspection as task validation.
