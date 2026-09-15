# v0.33 corrected numerical operator: fresh prospective screen

The first [packet protocol](row-packet-screen-protocol.md) was frozen at
`a1119214c711ad7439adfebfb16d8c0b00f2ca2c`. Its sole worker stopped after 18.089s
on document1 packet episode5: all emitted IDs matched, but one full-logit position
had relative L2 1.3045548089693852e-5 against the fixed 1e-5. Six of ten episodes
exist; remaining controls/restoration and compound gates are incomplete. Preserve
the entire failed run `runs/row-packet-screen-20260915-v1`, source and raw logit
diagnostic. Do not reinterpret its partial timing as a passing screen.

This prospective correction changes the numerical operator, NOT the checkpoint,
prompts, subset, order, token counts, precision, thresholds or timeout. The same
two authored prompts are now known/reused diagnostics, not a fresh holdout. The
original complete workload, accounting, citations and gates are restated below;
no extra performance matrix is authorized by the correction.

## Correction and causal rationale

Original OPT uses a contiguous [hidden,neurons] weight and F.linear(x,weight,bias).
The first harness used a contiguous [neurons,hidden] workspace and x@workspace+bias,
changing GEMM layout and bias fusion. Corrected stream, sparse AND packet share
an original-orientation contiguous [2048,8192] weight workspace. Its transposed
neuron-row view has strides [1,8192]. All use the original F.linear operator.

Host weights remain packed neuron rows. Streaming/extent copies now have a strided
GPU destination; any framework rearrangement/staging costs are charged in wall
time and allocator peaks. Packet still gathers IDs and rows into one contiguous
buffer, copies that buffer once, then index_copy scatters into the row view. This
preserves weight positions and the original readout arithmetic. No target logit
substitution, reference-as-output, omitted active contribution or relaxed bound.
Old default paths remain available for reproducing their recorded contracts.

## Fixed screen

Pinned facebook/opt-1.3b revision3f5c25d0bc631cb57ac65913f76e22c2dfb61d62, metadata
and actual file hashes checked. Existing .venv, FP32/SDPA/CUDA, TF32 off, four CPU
threads, seed20260917. Two unchanged authored config prompts, first16 plain tokens
without specials/template; cap8 greedy tokens or EOS2. Fresh cache each episode;
KV length16+j and393216 bytes/token. Order: resident A,B; A stream,sparse,packet;
B packet,sparse,stream; restore original dense weights/methods; repeat residents A,B.
No warmups, discarded episodes, fitting or selection by observed result.

Stream loads full outgoing matrices; sparse discovers exact current-call activity
and coalesces128-neuron pages; packet packs only exact active rows with int64 IDs.
Each packet is n*(8192+8) bytes, one physical H2D. Fully resident first projection,
attention/head/biases, original host weights1,610,612,736 B, workspace67,108,864 B,
stream pinned staging67,108,864 B, packet pinned+GPU buffers67,174,400 B EACH all
remain charged. All ten episodes share these buffers, including unused control
scratch. Temporary framework copies/gather/scatter/selection remain timed and peak
tracked; logical explicit payload counters are not PCIe wire measurements.

Keep every input/argmax, full-logit vector, packed activity, row/page decision,
KV length/bytes, per-step/subphase clocks and source snapshot. Independent CPU
replay checks source/config/tokenizer/raw inventory, arithmetic decisions, actual
accepted GREEDY output histories (there is no speculation), all logical copies,
serialization, buffers and resources. No claim of recomputing all neural operations.
Report prefill/decode traffic and TTFT; episode clock includes discovery/packing/
transfer/scatter/compute and raw tensor/call writes, excluding its own final receipt
but including that write in the total worker. Setup/restoration charged separately.

All hybrid and restored outputs must match original dense IDs/stop reason, and
max per-position relative logit L2 <=1e-5 with identical argmax. Abort subsequent
episodes on failure, preserve partial artifacts, no tolerance changes. Htraffic:
weight+index H2D at least50% below EACH stream/sparse control. Hruntime: wall at
least20% below EACH. Inclusive cross multiplication, all controls mandatory.
15000 MiB total GPU samples AND allocator peaks; minimum2048 MiB available host.
Existing300-second supervised worker includes imports/hashing/loading/conversion/
inference/writes; audit separately timed. Fresh directory only, never overwrite v1.

A pass only nominates a separately frozen follow-up; no long validation matrix,
capacity frontier, instruction-quality or native/llama.cpp claim. Initial resident
includes first-use initialization, restored resident is the warm comparison. Full
resident startup and retained reservations still prohibit a hard-budget claim.
[LLM in a Flash](https://arxiv.org/abs/2312.11514),
[PowerInfer](https://arxiv.org/abs/2312.12456), and
[Deja Vu](https://proceedings.mlr.press/v202/liu23am.html) are inspirations with the
same departures described in v1. Complete review and validation, commit and push
this correction before inference. The first failed attempt remains a release asset.
