# Continuing-turn placement tables

Two repeats of one six-request conversation per layout/mode. Continuing metrics exclude the initial request. Times are seconds; GPU is total sampled MiB.

| Layout / mode | Five-turn time | Mean token TTFT | Emitted/request s | Native steps/s | Cap stops /10 | Peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| whole-retained | 33.110 | 0.776 | 8.608 | 9.584 | 8 | 13759.10 |
| attention-retained | 23.985 | 0.623 | 11.883 | 13.418 | 8 | 14447.10 |
| whole-reset | 201.514 | 34.601 | 1.414 | 9.825 | 8 | 13759.10 |
| attention-reset | 172.973 | 30.569 | 1.625 | 13.718 | 6 | 14447.10 |

## Retained turns and cross-layout agreement

Means over two repeats. Prompt/output agreement counts are out of two; equal token counts are not equal text.

| Turn | Whole time | Attention time | Whole TTFT | Attention TTFT | Same prompt | Same output |
|---|---:|---:|---:|---:|---:|---:|
| initial | 40.596 | 34.574 | 31.995 | 28.291 | 2/2 | 2/2 |
| code | 6.153 | 4.504 | 0.378 | 0.292 | 2/2 | 2/2 |
| extraction | 10.112 | 7.293 | 2.302 | 1.903 | 2/2 | 0/2 |
| arithmetic | 6.250 | 4.258 | 0.394 | 0.301 | 0/2 | 0/2 |
| copying | 1.265 | 0.929 | 0.427 | 0.325 | 0/2 | 2/2 |
| topic-change | 9.331 | 7.000 | 0.379 | 0.292 | 0/2 | 0/2 |

## Acceptance and prompt accounting

Counts pool ten continuing requests in each row. Draft-forwarded prompt positions equal target evaluated positions only as a source-derived count. They are not an independent draft KV counter.

| Layout / mode | Cycles | Attempted | Accepted | Target reused | Target evaluated |
|---|---:|---:|---:|---:|---:|
| whole-retained | 148 | 2028 | 420 | 171844 | 1988 |
| whole-reset | 144 | 2012 | 424 | 0 | 173832 |
| attention-retained | 148 | 2056 | 422 | 171844 | 1988 |
| attention-reset | 142 | 2048 | 424 | 0 | 173832 |

All reset requests report zero reuse. Continuing retained target counts are 39, 822, 48, 52 and 33 newly evaluated positions; authored client suffixes contain 38, 821, 47, 51 and 34 tokens. Evaluated counts therefore differ from suffix lengths by +1 on the first four turns and -1 on the EOS-following topic turn. These are observed cache-accounting differences, not independent draft-state measurements.

Every condition/mode reproduces its outputs across repeats (24/24 request pairs). Retained/reset output agreement is 8/12 whole and 10/12 attention, including initial requests; no equality waiver is applied.
