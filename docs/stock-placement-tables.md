# Stock placement tables

Two reused populated16K development prompts, two repeats. Native generation steps exclude the first token per request; emitted IDs and complete request rates use their own denominators. Warmups and allocation/mechanism diagnostics do not enter performance aggregates.

| Study | Threshold | Cold FFNs | Native steps/s | Decode ms/emitted ID | Mean request s | Mean prefill s | Emitted/request s | Historical ID matches | Peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold | 8 | 0 | 14.439 | 68.714 | 40.228 | 31.421 | 3.182 | 4/4 | 13852.39 |
| threshold | 4 | 0 | 15.130 | 65.576 | 39.871 | 31.468 | 3.210 | 4/4 | 13852.39 |
| threshold | 2 | 0 | 15.405 | 64.408 | 39.818 | 31.553 | 3.215 | 4/4 | 13846.39 |
| placement | 2 | 0 | 15.295 | 64.871 | 39.827 | 31.513 | 3.214 | 4/4 | 13846.39 |
| placement | 2 | 32 | 20.568 | 48.240 | 34.451 | 28.254 | 3.715 | 4/4 | 14534.39 |

## Audited original and corrected requests

| Run | Case | Output IDs | Native steps/s | Whole request s | Prefill s | Decode s | Cycles | Stop | Historical IDs match |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| threshold-t8-f0-r1 | warmup | 32 | 18.951 | 1.971 | 0.321 | 1.636 | 6 | limit | None |
| threshold-t8-f0-r1 | long-code-cache | 128 | 14.519 | 39.986 | 31.233 | 8.747 | 22 | limit | True |
| threshold-t8-f0-r1 | long-data-audit | 128 | 14.721 | 40.041 | 31.386 | 8.627 | 22 | limit | True |
| threshold-t4-f0-r1 | warmup | 32 | 18.273 | 2.040 | 0.320 | 1.697 | 6 | limit | None |
| threshold-t4-f0-r1 | long-code-cache | 128 | 15.381 | 39.664 | 31.401 | 8.257 | 22 | limit | True |
| threshold-t4-f0-r1 | long-data-audit | 128 | 15.109 | 39.904 | 31.492 | 8.406 | 22 | limit | True |
| threshold-t2-f0-r1 | warmup | 32 | 18.583 | 2.018 | 0.326 | 1.668 | 6 | limit | None |
| threshold-t2-f0-r1 | long-code-cache | 128 | 15.230 | 39.920 | 31.552 | 8.339 | 22 | limit | True |
| threshold-t2-f0-r1 | long-data-audit | 128 | 15.586 | 39.767 | 31.613 | 8.149 | 22 | limit | True |
| threshold-t2-f0-r2 | warmup | 32 | 18.619 | 2.005 | 0.325 | 1.665 | 6 | limit | None |
| threshold-t2-f0-r2 | long-data-audit | 128 | 15.545 | 39.711 | 31.514 | 8.170 | 22 | limit | True |
| threshold-t2-f0-r2 | long-code-cache | 128 | 15.265 | 39.876 | 31.533 | 8.320 | 22 | limit | True |
| threshold-t4-f0-r2 | warmup | 32 | 18.446 | 2.049 | 0.353 | 1.681 | 6 | limit | None |
| threshold-t4-f0-r2 | long-data-audit | 128 | 14.876 | 40.008 | 31.465 | 8.537 | 22 | limit | True |
| threshold-t4-f0-r2 | long-code-cache | 128 | 15.164 | 39.907 | 31.514 | 8.375 | 22 | limit | True |
| threshold-t8-f0-r2 | warmup | 32 | 20.290 | 1.860 | 0.318 | 1.528 | 6 | limit | None |
| threshold-t8-f0-r2 | long-data-audit | 128 | 14.397 | 40.321 | 31.493 | 8.821 | 22 | limit | True |
| threshold-t8-f0-r2 | long-code-cache | 128 | 14.133 | 40.564 | 31.572 | 8.986 | 22 | limit | True |
| allocation-t2-f36-r1 | warmup | 32 | 18.754 | 2.001 | 0.337 | 1.653 | 6 | limit | None |
| allocation-t2-f36-r1 | long-code-cache | 32 | 14.024 | 31.671 | 29.440 | 2.210 | 7 | limit | None |
| allocation-t2-f34-r1 | warmup | 32 | 19.210 | 1.955 | 0.319 | 1.614 | 6 | limit | None |
| allocation-t2-f34-r1 | long-code-cache | 32 | 14.514 | 30.941 | 28.788 | 2.136 | 7 | limit | None |
| allocation-t2-f32-r1 | warmup | 32 | 20.168 | 1.851 | 0.312 | 1.537 | 6 | limit | None |
| allocation-t2-f32-r1 | long-code-cache | 32 | 15.116 | 30.238 | 28.162 | 2.051 | 7 | limit | None |
| mechanism-t2-f0-r1 | warmup | 32 | 15.977 | 2.322 | 0.362 | 1.940 | 6 | limit | None |
| mechanism-t2-f0-r1 | long-code-cache | 32 | 9.502 | 34.871 | 31.602 | 3.262 | 7 | limit | None |
| mechanism-t2-f32-r1 | warmup | 32 | 17.286 | 2.155 | 0.346 | 1.793 | 6 | limit | None |
| mechanism-t2-f32-r1 | long-code-cache | 32 | 12.966 | 30.648 | 28.252 | 2.391 | 7 | limit | None |
| placement-t2-f0-r1 | warmup | 32 | 18.399 | 2.033 | 0.323 | 1.685 | 6 | limit | None |
| placement-t2-f0-r1 | long-code-cache | 128 | 15.154 | 39.861 | 31.474 | 8.381 | 22 | limit | True |
| placement-t2-f0-r1 | long-data-audit | 128 | 15.417 | 39.762 | 31.519 | 8.237 | 22 | limit | True |
| placement-t2-f32-r1 | warmup | 32 | 20.087 | 1.882 | 0.314 | 1.543 | 6 | limit | None |
| placement-t2-f32-r1 | long-code-cache | 128 | 20.463 | 34.459 | 28.234 | 6.206 | 22 | limit | True |
| placement-t2-f32-r1 | long-data-audit | 128 | 20.728 | 34.446 | 28.293 | 6.127 | 22 | limit | True |
| placement-t2-f32-r2 | warmup | 32 | 20.036 | 1.885 | 0.313 | 1.547 | 6 | limit | None |
| placement-t2-f32-r2 | long-data-audit | 128 | 20.678 | 34.355 | 28.191 | 6.142 | 22 | limit | True |
| placement-t2-f32-r2 | long-code-cache | 128 | 20.405 | 34.545 | 28.300 | 6.224 | 22 | limit | True |
| placement-t2-f0-r2 | warmup | 32 | 18.638 | 2.012 | 0.334 | 1.663 | 6 | limit | None |
| placement-t2-f0-r2 | long-data-audit | 128 | 15.457 | 39.700 | 31.478 | 8.216 | 22 | limit | True |
| placement-t2-f0-r2 | long-code-cache | 128 | 15.157 | 39.983 | 31.581 | 8.379 | 22 | limit | True |

## Backend evidence

Assignments can repeat or be reused. They are not execution counts, transferred bytes or component timers.

- mechanism-t2-f0-r1: 82 target graphs; all64 attention layers CUDA with full192-projection coverage: False. Cold-FFN CUDA projection assignments per graph: 0 to 0.
- mechanism-t2-f32-r1: 82 target graphs; all64 attention layers CUDA with full192-projection coverage: True. Cold-FFN CUDA projection assignments per graph: 0 to 96.
