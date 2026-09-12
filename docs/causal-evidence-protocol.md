# Partial-evidence causal repair (prospective)

This bounded diagnostic addresses issue #19. Commit and push the protocol,
configuration and implementation before calibration or evaluation. Preserve all
v0.7-v0.10 results. Follow ADRs 0001 and 0002; Gate A remains independent.

## Hypothesis and fixed sample

Does evidence from initial execution improve quality at the same acquired-byte
budget over a larger one-shot decision? A stronger predictor, resident core or
greater final retention alone does not establish that result.

Use pinned Qwen2.5-1.5B-Instruct FP32/SDPA and the existing Python 3.14.3,
torch 2.10.0+cu130, Transformers 5.13.1 environment, seed 1729, four CPU threads,
TF32 disabled. Reuse the frozen v0.7 popularity layout, width-eight groups,
28 layers, 1120 groups/layer, 147456 weight bytes/group and a 2 GiB FFN working
budget. This phase explicitly reserves batch workspace within that allowance.

Calibration: first four v0.7 calibration documents, first 128 tokens each. Two
fixed collection passes; reset state/KV each document. Evaluation: first four
v0.7 development documents, first 128 tokens each. These are reused development
inputs, not new held-out evidence. Calibration and development IDs are disjoint.
The balanced chat-interface tasks are a separate behavioral control: all 20 fresh
v0.11 tasks, with fixed prompts, greedy decoding, EOS, and scoring from that phase.
They are development tasks reused after interface qualification, not held-out.

## Information and actions

All policies protect and execute a common resident hot core selected solely from
calibration importance. Charge a common conservative controller reservation before
deriving the core, including all predictor tensors, projection matrices, prior,
observed history, age, residency, norms, preserved input and decision scratch.
Reserve 256 MiB acquisition workspace for both sparse and dense paths, including
the gathered device batch, canonical gate/up/down packing buffers and activation
scratch. Assert the largest frozen batch fits this reservation. CPU weight backing
and pinned host staging are recorded diagnostic resources, not device cache hits.
The controller reservation is fixed at 32 MiB and verified against enumerated tensor
storage and conservative scratch before evaluation. Reserve one additional incoming
group slot. Equal reservation across controls
prevents unused predictor storage from changing their acquired-byte budgets.
The strongest dense static baseline pays the same 256 MiB workspace but has no
controller reservation. Both retain the historical additional incoming-group slot
conservatively. This is a new complete-action budget; the old v0.7/v0.8 2 GiB weight
cache comparisons remain unchanged and are not numerically interchangeable.

Initial masks contain 1008 groups: every resident group, then cold groups ranked
from current FFN input and observed history. Recompute this mask before gate/up
on every visit of each policy's own corrected trajectory. Never replay old masks.
Omitted history stays stale; retain age and current/history innovation features.

Use small fixed random projections (eight dimensions each) of normalized input,
observed log-contribution history and selected-group innovation. The current-input
features include the signed projection and its absolute value. The history model
adds projected observed log contributions, projected ages, and a scalar input
change signal. The repair model additionally receives the projected differences
between actually observed initial scores and stored history, their mean absolute
innovation, and the initial output norm. Resident membership determines cost and
eligibility, rather than a post-hoc mask evaluation. This is a simple explicit
estimator, not a Bayesian filter.

Contribution labels are log1p(abs(z)*down-column-norm summed within each group,
divided by that layer's calibration mean group importance). Fit standardized
ridge regression (penalty 0.01) separately per layer: current-input ranker,
input-plus-history ranker, partial-evidence ranker, and a detector predicting
initial relative FFN output omission error. The detector and ranker are distinct.

Pass one uses the calibration prior/history initial ranker and 28 predetermined
calibration-priority additions. Pass two uses the first fitted input/history
ranker and first fitted partial-evidence repair with 28 additions. Both passes
collect dense labels at the actual approximate/corrected states visited. The
final fits use pass two only. No development labels, quality feedback, or dense
audits may update predictors, thresholds or state. Preserve feature/label rows
and fitted tensors so fitting and evaluation decisions can be replayed separately.

During evaluation, the diagnostic still computes dense gate/up and keeps dense
weights. Controller APIs receive only current input, state and selected scores;
dense omitted scores are never passed to a query or update. Compute first and
additional down contributions separately using the preserved FFN input, and add
them before downstream commitment. The dense output is used only for isolated
audits. This is causality/quality evidence, not achieved sparse execution.

## Fixed comparisons

Evaluate these ten policies on all four Wiki prefixes:

- Initial input/history selection, 1008 groups, no correction.
- Larger input/history one-shot, 1036 and 1064 groups.
- Resident-first initial selection plus predetermined calibration-priority cold
  additions, 28 and 56 groups.
- Partial-evidence repair, 28 and 56 groups.
- Current-input-only one-shot, 1036 groups (history ablation).
- Detector/fallback: after the initial pass, predict its error; if >0.05, acquire
  every omitted group in the single corrective round, otherwise acquire 28 groups
  by partial-evidence ranking. No subsequent correction rounds.
- Full completion: initial selection then all 112 omitted groups in one round.

Because every resident group executes and all groups have equal weight bytes,
equal final cardinality gives exactly matched warm cold-weight bytes per layer
visit for the one-shot, predetermined and partial-evidence comparisons. Report
total retained volume too. Larger one-shot predicts before any current execution;
partial repair may use observed evidence. All state updates use only final
executed groups. Predetermined additions do not use partial evidence. No probes
are used in this first diagnostic (probe bytes zero); adding them needs a new
protocol. Full completion is a numerical control and expensive upper endpoint.

Run the 28-addition one-shot, predetermined and partial policies on all 20 chat
tasks, both teacher forcing the frozen answer and free generation on each policy's
own trajectory. Save a native dense reference and full-completion controls with
identical inputs. Report reference fidelity and task success separately. Gate A
failure forbids a preserved-utility claim but does not cancel these math controls.

## Gates and accounting

Gate B: aggregate Wiki relative PPL <=1.01 and every individual Wiki prefix <=1.01.
For a useful behavioral nominee additionally require Gate A, no loss of successful
dense tasks and no new format failures on dense-success tasks. Publish all rows,
individual failures and generation divergence; do not average away invalid values.

Traffic plausibility requires >=10% simulated warm traffic saving against the
strongest dense static comparison, after the common controller reservation.
Report cold preload, initial cold bytes, repair cold bytes, resident computation,
retained volume, correction-round frequency and dense-fallback frequency. Include
the GPU-to-host int64 acquisition-index bytes in total link traffic, separately
from cold-weight bytes. The static dense comparator knows its indices in advance.
On
generation compare bytes per processed input token and actual total sequence
bytes; differing output lengths do not establish lower cost at equivalent utility.

Measure controller query/evidence/update overhead on fixed retained fixtures with
three warmups and ten repetitions, separately from dense resident FFN time. Also
report the cost of the two down products. Keep trace serialization and privileged
audits out of selector timing, but enumerate their diagnostic memory separately.
All GPU work is serialized; no concurrent tests or reviews during timing.

Use v0.9's *gather-transfer-pack-FFN* and resident-FFN operations as the preparation
and execution definitions. Its 512-to-1120 gap is too coarse to decide the present
economics. Before measurement, derive the exact group counts from the fixed cache
reservation, resident cores, initial/final cardinalities and 28/56/112 additions.
Measure these sizes with v0.9's deterministic synthetic input recipe, CPU-reference
integrity check, preallocated buffers, reusable initialized events, three warmups
and ten repetitions. Retain inputs, every raw measurement and medians. This is a
new bounded primitive characterization; v0.9 remains unchanged. Unload the model
before this transfer fixture. Dynamic gathering, staging, transfer, packing and
FFN execution all enter the cold-batch clock; startup allocation does not.

Charge initial and corrective batches separately at their measured exact sizes;
zero-sized batches cost zero. Charge resident computation at its measured exact
size and add measured controller cost. Apply the same gathering/packing treatment
to the dense comparison. Separately time and charge the vector additions that
combine resident and cold contributions, and the extra correction sum when used.
The dense comparator pays the fastest measured one-addition fixture (same output
shape). Report down-product timing separately, without adding it twice to
primitives that already include FFN execution. The controller fixture
replays the first 32 visits of each policy's first development document, including
group-score processing, history updates, GPU-mask nonzero extraction and transfer
of contiguous int64 acquisition IDs to the host for each required round, using
prepared FFN intermediates and
first outputs; preparation is excluded from this component because FFN primitives
are charged separately. No serialization, privileged audit or fit is timed as a
controller action. These synthetic serialized sums do not measure exposed stalls,
overlap or real inference latency. Never substitute preaggregated pinned payloads
while omitting dynamic preparation.

Gate C remains a plausibility screen: >=10% bytes saving and positive serialized
cost headroom after charging controller and both rounds, plus no uncharged action.
Even B+C only justify proposing a bounded physical experiment, not asserting gain.
Claim partial evidence adds value only if it improves aggregate PPL over *both*
equal-cardinality one-shot and predetermined controls, with no worse individual
prefix PPL, and its charged cost does not erase that benefit. Otherwise report
the ablation as negative or mixed, independent of any absolute passing policy.

## Reproducibility

Save clean source/config/protocol snapshots, checkpoint/corpus identities,
calibration feature/label tensors and fits, normalized-input and permitted-evidence
receipts sufficient to replay controller decisions/state, initial/addition/final
masks, isolated error audits, teacher-forced and generation logits/IDs, timing
samples and action-cost calculations. Analyze from receipts without executing
pretrained weights; verify the owning Git source and full row grids. Any numerical
failure is explicit and blocks nomination. Keep raw failure artifacts and use a
fresh run directory for corrected executions. Publish detailed findings, including
negative ablations, and keep broader issues/milestones distinct from this delivery.
