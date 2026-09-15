# v0.33: single-packet exact-active-row acquisition (prospective)

Third of six authorized deliveries, [#48](https://github.com/displague/dynamic-model-loading/issues/48).
Review, validate, commit and push source/config before inference. This is a new
acquisition grain/transport mechanism after v0.32's frozen compound failure, not
a longer run, relaxed gate or new spatial ranker. No checkpoint run has preceded
this protocol. Existing ADRs 0004--0006 and .venv remain mandatory.

## Hypothesis and connection

The outgoing projection is a sum of row contributions x_j W_j. Exact zero x_j
removes a term; this is not a probabilistic guess. v0.32 bought 128-row regions and
used 2,286 copies versus streaming's 384. In transport systems, payload aggregation
trades packing cost and a header for fewer transfers. Test the same trade here:
gather exact active rows plus their indices into one contiguous packet, transfer
it once, scatter rows into the existing dense workspace, then do the same matmul.
Prediction: row granularity should save substantially more bytes than pages;
packet aggregation may repay CPU gather and GPU scatter. It is uncertain whether
that beats PyTorch streaming wall time. No invention of packetization or gathers.

[LLM in a Flash](https://arxiv.org/abs/2312.11514) motivates contiguous acquisition
and row/column bundling; its flash/DRAM tier is different. [PowerInfer](https://arxiv.org/abs/2312.12456)
motivates exploiting activation sparsity but uses heterogeneous hot/cold execution.
Our exact observed-zero contract differs from [Deja Vu's predictions](https://proceedings.mlr.press/v202/liu23am.html).
The experimental contribution being tested is outgoing-only, causal row discovery
plus a physical header/payload packet under explicit residency and faithfulness
accounting, not another predicted neighborhood or sampler.

## Artifacts, workload and order

Same pinned OPT-1.3B revision 3f5c25d0bc631cb57ac65913f76e22c2dfb61d62 as v0.32;
all local checkpoint hashes and committed v0.6 artifact metadata checked. New two
authored prompts in config: first 16 plain-text tokens, no added specials/template;
greedy cap8 or EOS2. No fit data, tuning or warmup/discard. Model FP32, SDPA CUDA,
TF32 off, torch seed20260917, four CPU threads. 300-second worker includes imports,
hashing/loading, all conditions, restore and writes; CPU audit separately timed.

Fixed order: resident A,B; A stream,sparse,packet; B packet,sparse,stream; restore
original dense layout; resident repeats A,B. Ten episodes total. Stream copies each
complete fc2 matrix; sparse is unchanged 128-row exact-zero extent coalescing.
All share the same numerical matmul/layout and fully resident fc1/attention/head.
All candidate executions really transfer from the original host-packed weights.
The two controls are rerun only as controls for this new mechanism on new texts.

Packet condition checks finite x, copies exact activity to CPU, unions activity
across the current forward, and sorts active row IDs in their original order.
For n active rows, the first 8n bytes are int64 IDs and the remaining 8192n bytes
are contiguous FP32 outgoing rows. CPU index_select fills pinned packet storage;
one blocking uint8 H2D copy transfers the entire prefix. GPU index_copy scatters
those rows to the shared workspace. Zero-row packets transfer nothing. Unloaded
finite stale rows remain valid only because every corresponding x is exactly zero.

Keep stream's 64 MiB pinned staging and charge it, plus a **67,174,400 B** pinned
packet buffer and equally sized GPU packet buffer. GPU workspace is another
67,108,864 B. All buffers persist and are charged in every condition; stream/sparse
do not use packet scratch. Thus no lower-memory control or capacity frontier is
claimed. Original host weights remain 1,610,612,736 B. Construction, restoration,
extra pinned/GPU storage, temporary activity/indices, metadata and all copies count.
No claim that this allocation is a minimal or optimized runtime implementation.

## Faithfulness, clocks and gates

Reuse v0.32's independent new-cache per episode contract, full-logit observations,
actual input history, raw activity and serialized call receipts. Same original
greedy IDs/stop reason required; maximum per-position logit relative L2 <=1e-5 and
identical argmax. Stop further episodes on a failed numerical check; preserve raw
files and report inconclusive. No tolerance widening on the new prompts.

Audit every activity-derived packet row/index/byte count; require selection,
packing, transfer, scatter and compute clocks to be ordered and workspace calls
nonoverlapping. Archive logits, activity, row lists and source; do not archive full
checkpoint weights or claim an independent rerun of every neural operation.
Episode clocks include prefix setup, discovery/gather/scatter, readbacks and raw
tensor/call/page writes (the enclosing final episode receipt is outside its own
clock and inside the worker). Report TTFT and prefill/decode traffic separately.
Exact startup/restoration copies, logical payload versus wire counters, first-use
initialization and total resource/allocator peaks remain explicit.

Htraffic: packet **weight plus index H2D** must be at least 50% below **each** of
stream and 128-row sparse controls over both scored episodes. Hruntime: packet
wall must be at least 20% below **each** control. Use inclusive cross multiplication.
All numerical, provenance, raw integrity and memory gates must also pass. 15000 MiB
total GPU/sample and allocator peak caps; 2048 MiB minimum available host RAM.
Report initial and restored resident clocks but never confuse first-use time with
steady-state speedup. A pass permits a separately frozen follow-up, not long-matrix
validation, larger-model access, instruction quality or native admission. A failure
stops this implementation; accepted-prefix risk and useful capacity remain separate
course questions. No llama.cpp change follows automatically.
