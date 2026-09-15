# v0.32 exact-zero physical output-weight screen (prospective)

Second delivery of the authorized six-release course, issue [#47](https://github.com/displague/dynamic-model-loading/issues/47).
Freeze reviewed source/config on pushed main before inference. Existing .venv,
FP32, CUDA SDPA, four CPU threads, TF32 off; 300-second worker including imports,
checkpoint hashing/loading, conversion, inference and writes. Independent CPU audit
is separately timed. No timeout/failure triggers a longer run or a changed tolerance.

## Question and prediction

Does discovering actual ReLU zeros before fetching outgoing FFN weights save useful
physical acquisition? Unlike v0.6's retrospective analysis, this implementation
really evacuates fc2 weights and copies selected host regions into one GPU workspace.
Prediction: 128-neuron pages and the prefill union will dilute the activation zero
rate severely; activity readback and staging may erase latency gains. This is not a
prediction of impossibility for smaller pages or other physical layouts.

Use pinned OPT-1.3B revision 3f5c25d0bc631cb57ac65913f76e22c2dfb61d62,
all file sizes/hashes from the committed v0.6 manifest. Archive that manifest and
bind it to Git; rehash actual local artifacts inside the worker. No checkpoint
download, tuning, training, prompt selection by outcome, or model substitution.
OPT is a pretrained English model, not an instruction-following Qwen replacement.

## Fixed workload and execution

Two new authored plain-text prompts are embedded in config. Tokenize without added
specials or template, use first 16 tokens, generate greedily up to 8 tokens or EOS=2.
No warmup or discarded scored episodes. Order: resident references A,B; streaming A;
sparse A; sparse B; streaming B; restore original dense layout; resident repeats A,B.
Every episode has a new DynamicCache. Prefill consumes all 16 tokens; subsequent
forwards consume only the last emitted token. Cache bytes must equal length times
393216. No speculation, reuse across episodes, crop or rollback claim here.

All attention, embeddings, fc1, biases and head remain resident. Each fc2 host
weight is stored as contiguous neuron rows; original parameter aliases that host
storage. One shared 64 MiB GPU workspace and one 64 MiB pinned staging buffer serve
24 layers. Streaming copies the entire matrix in one operation per layer call.
Sparse checks finite activations, reads the full exact boolean activity to CPU,
unions activity across the current call, chooses 128-neuron pages with any active
row, and coalesces adjacent selected pages. Each extent is staged and synchronously
copied to its workspace offset. Dense PyTorch matrix multiplication remains.
Unloaded finite stale rows are irrelevant only because their observed inputs are
exactly zero. An absent weight or a zero sentinel never certifies inactivity.

Capture every forward's last-position full-vocabulary logits, actual input IDs,
KV lengths/bytes and clocks; sparse records bit-packed activity and physical
extents for every layer call. Full logits are instrumented in every condition;
streaming does not perform candidate-only activation discovery. Charge all explicit
H2D/D2H copies, model construction, conversion/restoration, pinned buffers, original
host weights, workspace, KV, raw arrays and end-to-end episode writes. Setup and
restoration are reported separately, not silently amortized into steady decode.
Episode wall includes prefix construction, callbacks, readbacks and final raw writes;
TTFT is recorded separately. Inner selection/copy/compute clocks are diagnostic only.

## Controls and gates

Every hybrid/repeat output must match the original resident greedy IDs and stopping
reason. At matching histories, full-logit relative L2 must be <=1e-5 and every argmax
identical. Abort subsequent inference on numerical failure. Independent audit checks
all committed source blobs, external artifact metadata, actual tokenizer replay,
raw inventory, call order, activity-derived page/extent decisions, byte accounting,
cache rules, all numerical comparisons and resource/timing receipts. It does not
independently rerun every neural operation. Fresh directory for any corrected run.

Htraffic: sparse outgoing-weight H2D across both full scored episodes is at least
10% below streaming. Hruntime: sparse total episode wall is at least 5% below
streaming. Both require all integrity/numerical/resource controls. A pass permits
only a separately frozen follow-up, not a long suite, deployment, native admission
or confidence of population speedup from two episodes. Report resident timing and
prefill/decode traffic separately even on failure. Total GPU cap 15000 MiB; host
available floor 2048 MiB. The checkpoint already fits: this is not a larger-model
capacity frontier or target-scale comparison against the 32B stock baseline.
Recorded CUDA allocated/reserved high-water marks also must remain within the cap,
including construction/conversion/restoration; samples alone cannot waive a peak.
The replay requires nonoverlapping serialized workspace calls and inclusive gates;
traffic is compared by exact integer cross-multiplication.

## Attribution and scope

[PowerInfer](https://arxiv.org/abs/2312.12456) motivates activity-aware heterogeneous
execution; it computes cold neurons on CPU rather than this outgoing-only pager.
[Deja Vu](https://proceedings.mlr.press/v202/liu23am.html) predicts contextual
sparsity; we instead pay for exact first-projection observations.
[LLM in a Flash](https://arxiv.org/abs/2312.11514) motivates contiguous weight
acquisition and distinguishes tiers; no cold-SSD claim is made here.
[OPT](https://huggingface.co/facebook/opt-1.3b) supplies the unmodified checkpoint.
No invention of ReLU sparsity, offloading or dense PyTorch kernels is claimed.
