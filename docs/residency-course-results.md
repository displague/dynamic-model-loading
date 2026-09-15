# Six deliveries: novel acquisition research, v0.31--v0.36

This completes the second authorized six-delivery course. Each new measurement had
a reviewed, committed/pushed protocol and a <=300s worker. Results include failures;
a completed release is not a satisfied research gate. The stock llama.cpp baseline
was neither modified nor optimized during this course. No native patch or long
performance matrix was launched.

| Delivery | Changed question | Measured outcome |
|---|---|---|
| [v0.31](specialist-screen-results.md) | Do compact output specialists transfer target alignment? | Diagnostic base37/48; matching34/48, same as general. Both gates fail; no specialist pager. |
| [v0.32](sparse-down-screen-results.md) | Can exact observed zeros save physical outgoing-weight loads? | 14.44% H2D saved;4.89965% wall saving misses frozen5%. Outputs match;128-row grain stopped. |
| [v0.33](row-packet-screen-results.md) | Do compact active-row packets improve transport grain? | Corrected screen passes:93.65% H2D and69.36% wall saved vs streaming, all checked logits exact. First numerical failure archived. |
| [v0.34](vocabulary-risk-screen-results.md) | Does full-vocabulary risk buy cheaper actual prefixes? | 13/20 accepts vs15/16 controls;35.847s vs9.532s all35 and more bytes/accept. Stop this ranker. |
| [v0.35](capacity-screen-results.md) | Does packet loading work from startup under a smaller FP32 budget? | All gates pass below4800MiB;93.65% H2D and72.16% wall saved vs stronger contiguous streaming, exact logits. |
| [v0.36](resident-precision-results.md) | Does ordinary resident FP16 already provide that useful access? | Fits and matches16/16 tokens; first episode13.067s misses5s while second takes0.062s. FP32 numerical diagnostic also fails; compound access unresolved. |

## The mechanism that earned further consideration

For an outgoing projection, y=sum_j a_j*w_j+b. Once the fully computed first
projection and ReLU establish a_j=0, that outgoing row's contribution is known to
be zero. It need not be predicted from a weak neighborhood or residual surrogate.
Gather only rows with observed nonzero activation, put their indices and weights
in one physical transfer packet, and scatter into a shared dense GPU workspace.
The first projection stays resident and is fully computed; the outgoing matmul
remains dense. CPU discovery/gather, index bytes, staging, scatter and original
weights all count. This is neither a simulated traffic score nor a llama.cpp flag.

The exact-active-row loader is the strongest result of this course. It survives
a contiguous streaming control and CPU-first construction with no hidden full-GPU
startup. The useful geometry is observed ReLU zeros and physical acquisition grain,
not a learned SwiGLU residual field. [PowerInfer](https://arxiv.org/abs/2312.12456),
[Deja Vu](https://proceedings.mlr.press/v202/liu23am.html), and
[LLM in a Flash](https://arxiv.org/abs/2312.11514) are antecedents. Packing, sparse
algebra and CPU offload are known techniques; the contribution claimed here is
their tested outgoing-only physical implementation and evidence, not priority over
those methods or a new general theorem.

## What remains open

This is pretrained OPT1.3B FP32 under an imposed budget on a16GiB laptop, mostly
two short known continuations. It is not larger-model task quality, long-context
behavior, a physical small-GPU/OOM study, SSD paging or a deployable32B runtime.
v0.35 proves both dense offloading and packets fit; packet's advantage is acquisition
economics within that contract. FP16 also fits and emits the same16 observed tokens.
Its cold first-use stall prevents the frozen useful-access pass, but does not prove
that ordinary lower precision cannot provide useful access after initialization.
No paired warmed FP16/packet comparison or uniquely enabled capacity is established.

The first v0.33 attempt remains a failed numerical run. A prospective correction
restored the original layout/fused linear operation for all conditions and used a
fresh directory without loosening tolerance. Its reused prompts are not an untouched
holdout. All subsequent sparse checks retain that numerical contract. Negatives on
specialists and Gaussian ranking close those candidates, not their entire fields.

622 CPU tests pass at the final source stage; each release separately binds final
clean-tip suite receipts, raw archives and reviews. The release assets retain
full records, including rejected speculation and the first packet failure. Only
the six bounded issues close. Milestone10 and the broader novel-loading research
remain open. The next frontier would need a separately authorized/frozen larger
sparse-model comparison against strong low-memory representations, not another
ranking or stock-flag sweep. No such experiment is started by this conclusion.
