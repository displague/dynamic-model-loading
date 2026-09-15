# Residency and useful acquisition: second six-delivery course

The owner authorized another six research releases, v0.31--v0.36, on 2026-09-15.
Priority is (2) draft specialists, (3) sparse architecture, (1) accepted-prefix
risk, then (4) capacity. This is six bounded deliveries, not six positive results.
ADRs 0004--0006, the existing interpreter and stock deployment baselines remain.
Each new inference or fitted-data calculation needs its own reviewed and pushed
protocol. No screen pass alone authorizes a long matrix or native implementation.

The initial questions are: do compact output specialists improve target agreement;
can useful specialization repay real residency/switch costs; can exact-zero
discovery save physical sparse-FFN acquisition; does prefix utility improve paid
decisions; can a tiered representation expand useful bounded capacity; and does
the strongest surviving mechanism survive a distinct bounded stress test?
Failures can change later questions and their order, but never their own verdicts.
If specialization fails, move to sparse discovery rather than expand its pager.
Capacity requires an equal-resource baseline and a declared useful latency bound;
transferred bytes, peak memory, task quality and latency remain separate outcomes.

## Inspirations and departures

- [Domain draft training](https://arxiv.org/abs/2503.07807): target alignment matters,
  not just domain language-model quality. Our initial screen is an inexpensive
  output-only linear distillation test, not a reproduction of that training work.
- [Parameter-efficient draft adaptation](https://arxiv.org/abs/2603.09527): shared
  and private components motivate a resident backbone with separate corrections.
  Here heads are downstream of all KV-producing layers. Specialist switching
  therefore does not itself reinterpret stored backbone KV; token rollback still
  needs explicit verification. No invention of adapters or draft routing is claimed.
- [PowerInfer](https://arxiv.org/abs/2312.12456) and
  [Deja Vu](https://proceedings.mlr.press/v202/liu23am.html): sparse activity is not
  free discovery or free transfer. Exact observed zeros and predicted inactivity
  require different correctness contracts.
- [Speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html):
  accepted prefixes, not isolated token plausibility, produce useful draft work.
  The verified target remains the commit authority.
- [LLM in a Flash](https://arxiv.org/abs/2312.11514): tier-specific transfer volume,
  contiguous acquisition and reuse matter. Ordinary offloading is a comparison,
  not a novel result. Cold SSD reads must be separated from RAM file-cache hits.

## First delivery

[v0.31 protocol](specialist-screen-protocol.md),
[#46](https://github.com/displague/dynamic-model-loading/issues/46): test a small
frozen HF draft plus full-vocabulary low-rank output corrections against the pinned
1.5B target. Fresh authored documents and declared domain labels provide a
controlled prerequisite screen, not an online router or generated acceptance test.
No physical specialist pager follows a failed specialization gate.

The [v0.31 result](specialist-screen-results.md) fails both gates: base/general/
matching/wrong-domain diagnostic agreement is 37/34/34/36 out of 48. Training
improvement does not transfer. Stop this output-only ridge candidate, preserving
its small-data/loss/representation limitations. Next test sparse architecture and
causal exact-zero discovery rather than implement acquisition for these heads.

## Second delivery

[v0.32](sparse-down-screen-results.md) implements physical exact-zero outgoing-weight
acquisition on OPT. Htraffic passes (14.4409% saved); Hruntime narrowly fails
(4.89965% against 5%). All outputs match. Approximately 95.91% activation zeros
do not survive 128-row physical granularity or the prefill union as equivalent byte
savings. The next distinct candidate compacts active rows into one packet rather
than many extent copies, charging gather/metadata/scatter. No threshold change or
longer matrix; startup and retained allocator reservation remain capacity caveats.

## Third delivery

[v0.33](row-packet-screen-results.md) passes after a prospective numerical-layout
correction: exact-active-row packet traffic falls93.65% and wall69.36% vs streaming,
also beating the128-row control. All checked logits match exactly. The first failed
attempt is preserved; prompts/gates unchanged. Packetization and gathers are known
techniques; the tested contribution is their outgoing-only physical acquisition
application. This does not outperform the fully resident model or establish a
capacity frontier. Three releases remain, with accepted-prefix risk and useful
bounded capacity to receive separate protocols. No long matrix or native admission.

## Fourth delivery

[v0.34](vocabulary-risk-screen-results.md) changes the risk event to full-vocabulary
argmax and measures actual prefixes. Faithfulness passes, acquisition and timing
fail:13/20 accepts versus15/16 per control,837.2MB per accept versus644.1MB all35,
35.847s versus9.532s. Stop this ranker; another order of the same35 pages is not
queued. Two deliveries remain for cold-start bounded packet capacity and a stronger
low-memory comparison/stress question. Preserve all four delivered outcomes.
