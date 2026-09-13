# Continuing-agent tables

All values are reconstructed from raw SSE, requests, native logs and resource samples.
Initial cold requests are excluded from continuing-turn aggregates. Counts include emitted EOS IDs.

| Configuration | State | Mean five-turn time (s) | Mean client TTFT (s) | Emitted IDs / total request s | Limit stops /10 |
|---|---|---:|---:|---:|---:|
| default | retained | 79.984 | 0.802 | 3.513 | 6 |
| default | reset | 246.449 | 34.708 | 1.104 | 6 |
| offload | retained | 34.669 | 0.766 | 8.221 | 8 |
| offload | reset | 199.099 | 34.087 | 1.366 | 6 |

## Every request

| Run | Turn | Input IDs | Output IDs | Reused | New | TTFT (s) | Whole request (s) | Native prefill (ms) | Native decode (ms) | Stop | Reset agrees |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| offload-r1-retained | initial | 16384 | 128 | 0 | 16384 | 31.375 | 40.243 | 31370.224 | 8856.880 | limit | None |
| offload-r1-retained | code | 16550 | 64 | 16511 | 39 | 0.359 | 6.455 | 354.188 | 6095.185 | limit | None |
| offload-r1-retained | extraction | 17435 | 64 | 16613 | 822 | 2.293 | 9.918 | 2275.279 | 7625.280 | limit | None |
| offload-r1-retained | arithmetic | 17546 | 64 | 17498 | 48 | 0.377 | 6.456 | 372.561 | 6078.397 | limit | None |
| offload-r1-retained | copying | 17661 | 29 | 17609 | 52 | 0.409 | 1.217 | 403.489 | 807.456 | eos | None |
| offload-r1-retained | topic-change | 17724 | 64 | 17691 | 33 | 0.379 | 10.502 | 362.288 | 10121.860 | limit | None |
| offload-r1-reset | initial | 16384 | 128 | 0 | 16384 | 31.451 | 40.330 | 31436.333 | 8867.297 | limit | True |
| offload-r1-reset | code | 16550 | 64 | 0 | 16550 | 31.982 | 37.367 | 31961.373 | 5384.173 | limit | True |
| offload-r1-reset | extraction | 17435 | 64 | 0 | 17435 | 34.334 | 42.345 | 34328.097 | 8010.165 | limit | True |
| offload-r1-reset | arithmetic | 17546 | 64 | 0 | 17546 | 34.473 | 39.667 | 34456.734 | 5192.446 | limit | False |
| offload-r1-reset | copying | 17661 | 29 | 0 | 17661 | 34.602 | 35.430 | 34583.200 | 826.862 | eos | True |
| offload-r1-reset | topic-change | 17724 | 51 | 0 | 17724 | 35.079 | 44.280 | 35073.820 | 9199.542 | eos | False |
| default-r1-retained | initial | 16384 | 128 | 0 | 16384 | 31.537 | 57.239 | 31512.910 | 25691.012 | limit | None |
| default-r1-retained | code | 16550 | 64 | 16511 | 39 | 0.407 | 16.000 | 389.141 | 15592.712 | limit | None |
| default-r1-retained | extraction | 17435 | 64 | 16613 | 822 | 2.298 | 22.956 | 2292.735 | 20656.630 | limit | None |
| default-r1-retained | arithmetic | 17546 | 64 | 17498 | 48 | 0.460 | 16.837 | 434.728 | 16375.443 | limit | None |
| default-r1-retained | copying | 17661 | 29 | 17609 | 52 | 0.457 | 5.465 | 439.503 | 5007.344 | eos | None |
| default-r1-retained | topic-change | 17724 | 60 | 17691 | 33 | 0.402 | 21.151 | 396.774 | 20748.188 | eos | None |
| default-r1-reset | initial | 16384 | 128 | 0 | 16384 | 31.472 | 55.439 | 31467.117 | 23965.084 | limit | True |
| default-r1-reset | code | 16550 | 64 | 0 | 16550 | 32.039 | 45.455 | 32013.613 | 13415.187 | limit | True |
| default-r1-reset | extraction | 17435 | 64 | 0 | 17435 | 37.337 | 56.459 | 37330.863 | 19121.093 | limit | True |
| default-r1-reset | arithmetic | 17546 | 64 | 0 | 17546 | 34.482 | 49.132 | 34459.541 | 14649.084 | limit | True |
| default-r1-reset | copying | 17661 | 29 | 0 | 17661 | 34.631 | 39.355 | 34612.731 | 4723.757 | eos | True |
| default-r1-reset | topic-change | 17724 | 51 | 0 | 17724 | 34.991 | 53.762 | 34968.829 | 18770.557 | eos | False |
| offload-r2-retained | initial | 16384 | 128 | 0 | 16384 | 31.457 | 40.283 | 31452.312 | 8824.600 | limit | None |
| offload-r2-retained | code | 16550 | 64 | 16511 | 39 | 0.368 | 6.500 | 351.561 | 6131.976 | limit | None |
| offload-r2-retained | extraction | 17435 | 64 | 16613 | 822 | 2.272 | 9.931 | 2255.308 | 7658.304 | limit | None |
| offload-r2-retained | arithmetic | 17546 | 64 | 17498 | 48 | 0.397 | 6.518 | 369.262 | 6120.410 | limit | None |
| offload-r2-retained | copying | 17661 | 29 | 17609 | 52 | 0.424 | 1.234 | 419.302 | 808.496 | eos | None |
| offload-r2-retained | topic-change | 17724 | 64 | 17691 | 33 | 0.382 | 10.607 | 365.660 | 10219.424 | limit | None |
| offload-r2-reset | initial | 16384 | 128 | 0 | 16384 | 31.454 | 40.335 | 31448.221 | 8871.306 | limit | True |
| offload-r2-reset | code | 16550 | 64 | 0 | 16550 | 32.009 | 37.427 | 32004.222 | 5417.887 | limit | True |
| offload-r2-reset | extraction | 17435 | 64 | 0 | 17435 | 34.304 | 42.323 | 34299.269 | 8017.539 | limit | True |
| offload-r2-reset | arithmetic | 17546 | 64 | 0 | 17546 | 34.451 | 39.635 | 34445.061 | 5183.310 | limit | False |
| offload-r2-reset | copying | 17661 | 29 | 0 | 17661 | 34.593 | 35.414 | 34587.537 | 819.922 | eos | True |
| offload-r2-reset | topic-change | 17724 | 51 | 0 | 17724 | 35.046 | 44.313 | 35027.689 | 9265.242 | eos | False |
| default-r2-retained | initial | 16384 | 128 | 0 | 16384 | 31.461 | 56.009 | 31438.157 | 24546.711 | limit | None |
| default-r2-retained | code | 16550 | 64 | 16511 | 39 | 0.396 | 14.762 | 390.989 | 14365.366 | limit | None |
| default-r2-retained | extraction | 17435 | 64 | 16613 | 822 | 2.309 | 21.861 | 2289.140 | 19551.063 | limit | None |
| default-r2-retained | arithmetic | 17546 | 64 | 17498 | 48 | 0.413 | 15.304 | 407.994 | 14889.821 | limit | None |
| default-r2-retained | copying | 17661 | 29 | 17609 | 52 | 0.454 | 5.245 | 448.094 | 4790.178 | eos | None |
| default-r2-retained | topic-change | 17724 | 60 | 17691 | 33 | 0.425 | 20.388 | 404.422 | 19961.290 | eos | None |
| default-r2-reset | initial | 16384 | 128 | 0 | 16384 | 31.405 | 56.859 | 31399.157 | 25444.019 | limit | True |
| default-r2-reset | code | 16550 | 64 | 0 | 16550 | 32.029 | 46.137 | 32024.278 | 14106.882 | limit | True |
| default-r2-reset | extraction | 17435 | 64 | 0 | 17435 | 37.541 | 57.505 | 37535.813 | 19962.326 | limit | True |
| default-r2-reset | arithmetic | 17546 | 64 | 0 | 17546 | 34.459 | 49.664 | 34433.533 | 15204.135 | limit | True |
| default-r2-reset | copying | 17661 | 29 | 0 | 17661 | 34.557 | 39.696 | 34537.997 | 5138.294 | eos | True |
| default-r2-reset | topic-change | 17724 | 51 | 0 | 17724 | 35.018 | 55.734 | 35012.896 | 20714.993 | eos | False |

## Independent replay

| Trajectory | Matching IDs | Evaluated IDs |
|---|---:|---:|
| long-code-cache | 128 | 128 |
| long-data-audit | 128 | 128 |
| short-code-parser | 255 | 256 |
| short-code-cache | 255 | 256 |
| short-code-sql | 256 | 256 |
| short-topic-transition | 253 | 256 |
| short-data-audit | 256 | 256 |
| short-prose-debug | 252 | 256 |

## Every discrepancy

Positions are zero-based. Independent top-two margins are diagnostic only; no historical paired verifier distribution is available and no margin waiver is applied.

| Trajectory | Position | Recorded ID | Independent ID | Independent log-softmax margin |
|---|---:|---:|---:|---:|
| short-code-parser | 45 | 5916 | 429 | 0.01053995 |
| short-code-cache | 109 | 1376 | 24758 | 0.00829887 |
| short-topic-transition | 52 | 3210 | 7375 | 0.02950668 |
| short-topic-transition | 77 | 287 | 271 | 0.02627939 |
| short-topic-transition | 157 | 1177 | 608 | 0.02826315 |
| short-prose-debug | 94 | 64547 | 29279 | 0.08716583 |
| short-prose-debug | 98 | 1943 | 7424 | 0.02581799 |
| short-prose-debug | 122 | 311 | 369 | 0.03179169 |
| short-prose-debug | 241 | 6546 | 16953 | 0.03553021 |
