# v0.39: FP16 packets preserve outputs and save streaming cost, not resident speed

The28.474-second screen (22.971s worker,5.503s audit) passes same-precision
faithfulness, resources, transport and footprint. Resident-speed fails as predicted.
All checked logits, including declared warmups, are bit-identical to original
resident FP16. All48 scored greedy tokens per condition match. A second raw replay
is identical. Known v0.37 prompts do not become a new holdout by changing precision.

| Scored FP16 condition | Weight + index H2D | Wall,48 tokens | Peak sampled GPU |
|---|---:|---:|---:|
| Ordinary resident | 0 during generation | 0.382103s | 3,413,217,280 B |
| Direct contiguous stream | 38,654,705,664 B | 3.195469s | 2,677,116,928 B |
| Exact observed-row packet | 2,161,006,344 B | 1.526528s | 2,677,116,928 B |

Packets save94.4095% H2D and52.2284% wall versus FP16 streaming. Both offloaded
conditions use21.5662% less sampled GPU memory than full residency. The memory
saving is outgoing-weight OFFLOAD, not a packet-only advantage; packet's advantage
over equal-memory streaming is acquisition. Ordinary warm resident FP16 is about
four times faster than packets and fits here. No unique capacity or resident-speed
win is claimed. Resident references precede hybrid phases; these are observed
screen totals, not a randomized steady-state confidence interval.

GPU non-outgoing parameters1,826,209,792 B; workspace33,554,432 B; packet33,619,968 B
on GPU and equally sized pinned host buffer. Packed outgoing host weights
805,306,368 B plus an equal ORIGINAL-LAYOUT host copy charged to both controls.
No row cache. Original resident parameters2,631,516,160 B move H2D then D2H at
transition; candidate construction H2D1,826,209,792 B. Whole-worker peakRSS
4,899,934,208 B; host minimum12,921,286,656 B. All candidate episodes pass4800MiB;
full reference startup remains explicit and prohibits a CPU-first capacity claim.

FP16 weights and FP16 KV remain consistent throughout. Actual package/CUDA settings,
including reduced-precision reduction, are frozen/recorded. This does not test
FP32-equivalent quality, Q4/Q8, long context, another architecture or deployment.
Initial use and separate warmups are preserved; v0.36's failure is not erased.

## Resulting next step

The component gates nominate a separately frozen SHORT larger sparse-model
capacity test, where resident FP16 parameters can exceed the declared GPU budget.
Include direct dense FP16 streaming, CPU-first construction and a stable dense
same-precision reference. The unmeasured optimized-Q4 comparison must remain an
explicit limitation. Do not extrapolate2.7B/32B performance from this1.3B screen.
No native pivot or hours-scale matrix follows.

[Protocol](precision-packet-protocol.md),
[compact evidence](../results/precision-packet-20260915/summary.json).
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)
motivates same-precision controls; [OPT](https://arxiv.org/abs/2205.01068) supplies
the ReLU model, and [LLM in a Flash](https://arxiv.org/abs/2312.11514) supplies
physical-grain antecedents. No invention of FP16 or packet transport claimed.

650 CPU tests pass; v0.37/v0.38 raw replays remain identical. Review caught the
need for an in-episode memory sample for short resident runs; the fixed sample was
added to ALL FP16 modes before inference. Raw36,303,659 B,155 members, SHA256
`b876cb8a0a70ee59b0d636a06c5cbe91b4df0c101f2c088d938b43bac5d564e3`.
Only issue54 and delivery3 of the authorized4--8 complete; milestone10 remains open.
