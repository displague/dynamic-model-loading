# v0.31: output-specialist prerequisite screen

Prospective protocol, issue #46. First of six newly authorized releases. Read with
ADRs 0004--0006 and [the course](residency-research-course.md). No checkpoint forward,
fit, diagnostic scoring or candidate replacement before reviewed source is pushed.

## Question and prediction

Can a compact domain-specific output correction of a frozen small draft improve
full-vocabulary agreement with a frozen target beyond one general correction?
Hypothesis: stable domain information may be useful across several positions, but
an output head cannot manufacture missing backbone reasoning. We expect a small
agreement signal at best, with specialization potentially losing to the better-
sampled general head. A negative stops this output-specialist candidate before
physical paging, not all specialist training. There is no acquisition claim yet.

## Frozen inputs and apparatus

`configs/specialist-screen.json` pins all weight/config/tokenizer bytes. Target:
Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306; draft:
Qwen2.5-0.5B-Instruct revision 7ae557604adf67be50417f59c2c2f167def9a775. Both HF
artifacts run FP32, SDPA, four PyTorch CPU threads and no TF32 in the existing `.venv`.
Worker and auditor explicitly enforce four NumPy/OpenBLAS threads and record the
loaded BLAS library hash; PyTorch thread settings alone do not control fitting.
No native code or GGUF substitution. Draft download/hash preparation took 14.155s
outside inference; this is reported separately. Local artifact rehashing, imports,
model loading, collection, fitting, scoring and raw writes are inside a 300s worker.
Independent CPU analysis is separately timed. Sample total GPU <=15000 MiB and
host available >=2048 MiB; publish allocator and RSS peaks, parameter/readback/head
payloads, even for a failed hypothesis. These are feasibility caps, not capacity
improvements. No other GPU work is intentionally launched with the experiment.
Construction payload includes registered rotary buffers as well as parameters.
Both current and original-frequency buffers are registered: 512 B for the target
and 256 B for the draft, as independently checked on allocation-free meta models.
Explicit application-copy accounting is not a trace of every framework/driver DMA.
An error/timeout is an unsuccessful command, not merely an inconclusive JSON row.

Eighteen fresh agent-authored synthetic texts in `specialist-documents.json`:
code/prose/math each have four fit and two diagnostic documents. These are not a
natural benchmark or a claim of unseen-pretraining content. They were authored
before scoring and are distinct from the old Wikipedia fixtures. Plain text,
no chat template or added special tokens; tokenize once and require identical
token IDs for both tokenizers. First 40 tokens are consumed; score next-token
logits at input positions 32..39, eight per document. Thus 96 fit and 48 diagnostic
positions. Both models see the SAME authored prefixes, not generated trajectories.
This tests teacher-forced agreement; it does not measure accepted prefix lengths.
Domain metadata is supplied, not inferred. No online routing claim follows.

## Head and controls

Read the draft's final normalized hidden state (896 components). RMS-normalize it,
project with one seeded Rademacher matrix into 16 features and append a constant.
Use NumPy seed 20260916. For each fit row subtract the mean of the target-minus-
draft full-vocabulary logit residual. Solve ridge regression with lambda 32 on all
17 coefficients, including the intercept. The loss is squared centered-logit
error, not optimized acceptance risk or a calibrated posterior.

Fit one general head on all 96 positions and three domain heads on 32 positions
each, without diagnostic rows. Every head corrects all 151936 logits, and no
shortlist restricts argmax. Head shape is 17x151936; the FP64 archive head is twice
the size of an eventual FP32 runtime representation, which is separately charged
if implemented. Controls: unchanged draft, general head, matching-domain head,
and wrong-domain head (code->prose->math->code). Every condition is fixed here;
do not select or refit after viewing diagnostics. The whole tiny bank fits; there
is no contrived paging/capacity success based on evicting these heads.

## Gates, receipts and limits

Integrity: exact tokenizer compatibility, finite full-vocabulary tensors, frozen
shapes/order/splits, complete source/artifact/resource receipts. Repeated target
and draft forwards on the first diagnostic document must retain all eight argmax
IDs and meet relative-L2 <=1e-6. Reconstruct first fit document's draft readout
from archived final hidden and the actual readout weights; relative-L2 <=1e-6
and identical argmax. These are numerical controls, not independent CPU reruns
of every transformer forward. `use_cache=False`: no production KV qualification.

Hgeneral: general head obtains at least two MORE target argmax matches than base
on 48 diagnostics. Hspecialization: matching-domain head obtains at least three
MORE matches than EACH of base/general/wrong-domain, and is no worse than general
within any domain. These coarse nomination bars are not significance tests and
must not be relaxed after scoring. Hgeneral alone does not admit specialist paging.
Failure stops this candidate. Pass admits only a new short physical protocol.

Archive full target/draft logits and draft hidden states, tokens, metadata, fitted
heads, all predicted IDs, numerical/resource/timing/source receipts and licensing.
An independent model-free auditor refits from fit rows, reproduces all IDs and
gates, verifies source objects and receipt hashes, and checks both tensor byte
counts and resource peaks. It cannot prove each archived neural forward anew.
No generated acceptance, target-scale speedup, task-quality gain, useful capacity,
SSD behavior or native admission is measured. Stock target-only and small-draft
deployment baselines are preserved, not numerically compared with this HF component.

Inspirations: [domain draft alignment](https://arxiv.org/abs/2503.07807) and
[shared/private draft adaptation](https://arxiv.org/abs/2603.09527). Ridge regression
and random projection are ordinary tools; novelty is a hypothesis about useful
acquisition units, not a claim to invent either method or a specialist router.
