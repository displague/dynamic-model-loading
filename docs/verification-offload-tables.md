# Verification offload: complete descriptive tables

| Condition | Emitted/request s | Emitted/native decode s | Native decode steps/s | Mean prefill s/request | Mean native decode s/request |
| --- | --- | --- | --- | --- | --- |
| cpu-k4 | 17.6585 | 18.3544 | 18.2827 | 0.541 | 13.948 |
| cpu-k8 | 16.3155 | 16.8801 | 16.8141 | 0.517 | 15.166 |
| cpu-k16 | 11.6805 | 11.8697 | 11.8234 | 0.342 | 21.567 |
| offload-k8 | 19.0336 | 19.6160 | 19.5394 | 0.393 | 13.051 |
| offload-k16 | 19.7385 | 20.5345 | 20.4543 | 0.493 | 12.467 |
| disabled-k16 | 11.3563 | 12.1670 | 12.1195 | 1.495 | 21.041 |

| Condition | Emitted/request s | Emitted/native decode s | Native decode steps/s | Mean prefill s/request | Mean native decode s/request |
| --- | --- | --- | --- | --- | --- |
| target | 1.8018 | 3.4409 | 3.4140 | 33.824 | 37.200 |
| cpu-k4 | 2.1123 | 4.8013 | 4.7638 | 33.927 | 26.660 |
| cpu-k16 | 1.6546 | 2.9246 | 2.9018 | 33.583 | 43.766 |
| offload-k16 | 3.0026 | 13.6047 | 13.4984 | 33.201 | 9.409 |

## Repeats and resource samples

| Stage | Condition | Repeat | Emitted/request s | Emitted/native decode s | Reference IDs match | GPU peak MiB | Minimum host available GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | cpu-k16 | 1 | 1.6444 | 2.9030 | 2/2 | 13921.12 | 8.843 |
| long | cpu-k4 | 1 | 2.1174 | 4.8172 | 2/2 | 13921.12 | 8.803 |
| long | offload-k16 | 1 | 3.0138 | 13.6423 | 2/2 | 13929.12 | 8.787 |
| long | target | 1 | 1.7984 | 3.3930 | 2/2 | 13269.12 | 8.849 |
| long | cpu-k16 | 2 | 1.6649 | 2.9465 | 2/2 | 13921.12 | 9.567 |
| long | cpu-k4 | 2 | 2.1072 | 4.7855 | 2/2 | 13921.12 | 9.596 |
| long | offload-k16 | 2 | 2.9915 | 13.5672 | 2/2 | 13929.12 | 9.456 |
| long | target | 2 | 1.8052 | 3.4902 | 2/2 | 13269.12 | 9.855 |
| short | cpu-k16 | 1 | 11.6269 | 11.8305 | 3/6 | 14689.12 | 11.355 |
| short | cpu-k4 | 1 | 17.8557 | 18.5609 | 3/6 | 14689.12 | 11.113 |
| short | cpu-k8 | 1 | 16.3904 | 17.1354 | 4/6 | 14689.12 | 11.210 |
| short | disabled-k16 | 1 | 11.4128 | 12.2270 | 2/6 | 14611.12 | 11.070 |
| short | offload-k16 | 1 | 19.7741 | 20.6260 | 2/6 | 14719.12 | 11.144 |
| short | offload-k8 | 1 | 19.1062 | 19.6918 | 1/6 | 14721.12 | 11.194 |
| short | cpu-k16 | 2 | 11.7347 | 11.9092 | 3/6 | 14689.12 | 11.162 |
| short | cpu-k4 | 2 | 17.4655 | 18.1524 | 3/6 | 14689.12 | 11.320 |
| short | cpu-k8 | 2 | 16.2412 | 16.6322 | 4/6 | 14689.12 | 11.306 |
| short | disabled-k16 | 2 | 11.3003 | 12.1076 | 2/6 | 14611.12 | 11.228 |
| short | offload-k16 | 2 | 19.7029 | 20.4438 | 2/6 | 14719.12 | 11.290 |
| short | offload-k8 | 2 | 18.9617 | 19.5408 | 1/6 | 14719.12 | 11.282 |

Short identity uses the unchanged v0.14 target reference; long identity uses first long target-only. First differences below are zero-based generated-token positions.

## Individual requests

| Stage | Condition | Repeat | Prompt | Input tokens | Output tokens | Request s | Native prefill s | Native decode s | First reference difference | IDs match earlier same-condition repeat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long | cpu-k16 | 1 | code-cache | 16384 | 128 | 80.9974 | 34.7537 | 46.2351 | same | None |
| long | cpu-k16 | 1 | data-audit | 16384 | 128 | 74.6801 | 32.7227 | 41.9493 | same | None |
| long | cpu-k4 | 1 | code-cache | 16384 | 128 | 61.0949 | 35.0494 | 26.0371 | same | None |
| long | cpu-k4 | 1 | data-audit | 16384 | 128 | 59.8078 | 32.6929 | 27.1062 | same | None |
| long | offload-k16 | 1 | code-cache | 16384 | 128 | 43.0693 | 33.5768 | 9.4836 | same | None |
| long | offload-k16 | 1 | data-audit | 16384 | 128 | 41.8745 | 32.5589 | 9.2815 | same | None |
| long | target | 1 | code-cache | 16384 | 128 | 72.4016 | 34.7702 | 37.5883 | same | None |
| long | target | 1 | data-audit | 16384 | 128 | 69.9492 | 32.0792 | 37.8613 | same | None |
| long | cpu-k16 | 2 | data-audit | 16384 | 128 | 75.3137 | 34.2049 | 41.0999 | same | True |
| long | cpu-k16 | 2 | code-cache | 16384 | 128 | 78.4495 | 32.6495 | 45.7818 | same | True |
| long | cpu-k4 | 2 | data-audit | 16384 | 128 | 63.2366 | 35.2542 | 27.9729 | same | True |
| long | cpu-k4 | 2 | code-cache | 16384 | 128 | 58.2521 | 32.7120 | 25.5219 | same | True |
| long | offload-k16 | 2 | data-audit | 16384 | 128 | 43.2931 | 33.9889 | 9.2954 | same | True |
| long | offload-k16 | 2 | code-cache | 16384 | 128 | 42.2828 | 32.6798 | 9.5736 | same | True |
| long | target | 2 | data-audit | 16384 | 128 | 72.0926 | 35.5952 | 36.4888 | same | True |
| long | target | 2 | code-cache | 16384 | 128 | 69.7217 | 32.8524 | 36.8602 | same | True |
| short | cpu-k16 | 1 | code-parser | 81 | 256 | 25.0636 | 0.8078 | 24.2428 | same | None |
| short | cpu-k16 | 1 | code-cache | 73 | 256 | 14.2437 | 0.2509 | 13.9808 | 222 | None |
| short | cpu-k16 | 1 | code-sql | 85 | 256 | 22.4009 | 0.2495 | 22.1481 | same | None |
| short | cpu-k16 | 1 | topic-transition | 79 | 256 | 31.6972 | 0.2693 | 31.4247 | 52 | None |
| short | cpu-k16 | 1 | data-audit | 103 | 256 | 13.3416 | 0.3534 | 12.9753 | same | None |
| short | cpu-k16 | 1 | prose-debug | 78 | 256 | 25.3605 | 0.2948 | 25.0622 | 136 | None |
| short | cpu-k4 | 1 | code-parser | 81 | 256 | 15.5902 | 1.4219 | 14.1650 | 79 | None |
| short | cpu-k4 | 1 | code-cache | 73 | 256 | 11.4870 | 0.2797 | 11.1804 | same | None |
| short | cpu-k4 | 1 | code-sql | 85 | 256 | 15.4403 | 0.2946 | 15.1417 | same | None |
| short | cpu-k4 | 1 | topic-transition | 79 | 256 | 16.1485 | 0.2782 | 15.8666 | 77 | None |
| short | cpu-k4 | 1 | data-audit | 103 | 256 | 11.5567 | 0.6739 | 10.8703 | same | None |
| short | cpu-k4 | 1 | prose-debug | 78 | 256 | 15.8004 | 0.2662 | 15.5306 | 136 | None |
| short | cpu-k8 | 1 | code-parser | 81 | 256 | 17.0982 | 0.9814 | 16.1128 | same | None |
| short | cpu-k8 | 1 | code-cache | 73 | 256 | 10.7615 | 0.2613 | 10.4969 | same | None |
| short | cpu-k8 | 1 | code-sql | 85 | 256 | 14.6682 | 0.2836 | 14.3725 | same | None |
| short | cpu-k8 | 1 | topic-transition | 79 | 256 | 20.2581 | 0.2812 | 19.9731 | 52 | None |
| short | cpu-k8 | 1 | data-audit | 103 | 256 | 12.7734 | 1.9373 | 10.8319 | same | None |
| short | cpu-k8 | 1 | prose-debug | 78 | 256 | 18.1537 | 0.2892 | 17.8516 | 136 | None |
| short | disabled-k16 | 1 | code-parser | 81 | 256 | 22.8255 | 1.4069 | 21.4154 | 79 | None |
| short | disabled-k16 | 1 | code-cache | 73 | 256 | 14.6130 | 1.3078 | 13.2925 | same | None |
| short | disabled-k16 | 1 | code-sql | 85 | 256 | 22.7573 | 1.5401 | 21.2140 | 239 | None |
| short | disabled-k16 | 1 | topic-transition | 79 | 256 | 32.5319 | 1.4303 | 31.0980 | 38 | None |
| short | disabled-k16 | 1 | data-audit | 103 | 256 | 14.5493 | 1.8305 | 12.7149 | same | None |
| short | disabled-k16 | 1 | prose-debug | 78 | 256 | 27.3091 | 1.4070 | 25.8892 | 94 | None |
| short | offload-k16 | 1 | code-parser | 81 | 256 | 15.2033 | 0.6882 | 14.5017 | 45 | None |
| short | offload-k16 | 1 | code-cache | 73 | 256 | 8.2621 | 0.2363 | 8.0133 | same | None |
| short | offload-k16 | 1 | code-sql | 85 | 256 | 12.5968 | 0.2456 | 12.3478 | same | None |
| short | offload-k16 | 1 | topic-transition | 79 | 256 | 17.1656 | 0.2494 | 16.9130 | 77 | None |
| short | offload-k16 | 1 | data-audit | 103 | 256 | 8.8638 | 1.5011 | 7.3596 | 29 | None |
| short | offload-k16 | 1 | prose-debug | 78 | 256 | 15.5856 | 0.2483 | 15.3337 | 94 | None |
| short | offload-k8 | 1 | code-parser | 81 | 256 | 15.1740 | 0.6530 | 14.5080 | 45 | None |
| short | offload-k8 | 1 | code-cache | 73 | 256 | 9.9518 | 0.2320 | 9.7074 | same | None |
| short | offload-k8 | 1 | code-sql | 85 | 256 | 13.0775 | 0.2405 | 12.8329 | 239 | None |
| short | offload-k8 | 1 | topic-transition | 79 | 256 | 16.8624 | 0.2467 | 16.6126 | 77 | None |
| short | offload-k8 | 1 | data-audit | 103 | 256 | 10.0653 | 0.7402 | 9.3218 | 29 | None |
| short | offload-k8 | 1 | prose-debug | 78 | 256 | 15.2619 | 0.2392 | 15.0192 | 94 | None |
| short | cpu-k16 | 2 | code-cache | 73 | 256 | 13.6945 | 0.4881 | 13.2033 | 222 | True |
| short | cpu-k16 | 2 | prose-debug | 78 | 256 | 24.7767 | 0.2743 | 24.4895 | 136 | True |
| short | cpu-k16 | 2 | code-sql | 85 | 256 | 22.0148 | 0.2842 | 21.7272 | same | True |
| short | cpu-k16 | 2 | code-parser | 81 | 256 | 25.0917 | 0.2780 | 24.8105 | same | True |
| short | cpu-k16 | 2 | data-audit | 103 | 256 | 13.3849 | 0.2875 | 13.0846 | same | True |
| short | cpu-k16 | 2 | topic-transition | 79 | 256 | 31.9313 | 0.2679 | 31.6603 | 52 | True |
| short | cpu-k4 | 2 | code-cache | 73 | 256 | 12.7295 | 0.9020 | 11.8145 | same | True |
| short | cpu-k4 | 2 | prose-debug | 78 | 256 | 16.2519 | 0.2757 | 15.9730 | 136 | True |
| short | cpu-k4 | 2 | code-sql | 85 | 256 | 15.0673 | 0.3242 | 14.7398 | same | True |
| short | cpu-k4 | 2 | code-parser | 81 | 256 | 15.9042 | 0.4500 | 15.4415 | 79 | True |
| short | cpu-k4 | 2 | data-audit | 103 | 256 | 12.2663 | 1.0834 | 11.1796 | same | True |
| short | cpu-k4 | 2 | topic-transition | 79 | 256 | 15.7254 | 0.2445 | 15.4688 | 77 | True |
| short | cpu-k8 | 2 | code-cache | 73 | 256 | 11.6313 | 0.6375 | 10.9906 | same | True |
| short | cpu-k8 | 2 | prose-debug | 78 | 256 | 17.8672 | 0.2709 | 17.5925 | 136 | True |
| short | cpu-k8 | 2 | code-sql | 85 | 256 | 14.9906 | 0.2908 | 14.6872 | same | True |
| short | cpu-k8 | 2 | code-parser | 81 | 256 | 17.2623 | 0.2897 | 16.9441 | same | True |
| short | cpu-k8 | 2 | data-audit | 103 | 256 | 11.7218 | 0.4035 | 11.3143 | same | True |
| short | cpu-k8 | 2 | topic-transition | 79 | 256 | 21.1010 | 0.2755 | 20.8222 | 52 | True |
| short | disabled-k16 | 2 | code-cache | 73 | 256 | 14.3050 | 1.2708 | 13.0309 | same | True |
| short | disabled-k16 | 2 | prose-debug | 78 | 256 | 27.1667 | 1.3926 | 25.7711 | 94 | True |
| short | disabled-k16 | 2 | code-sql | 85 | 256 | 23.0737 | 1.5719 | 21.4985 | 239 | True |
| short | disabled-k16 | 2 | code-parser | 81 | 256 | 23.8513 | 1.4936 | 22.3345 | 79 | True |
| short | disabled-k16 | 2 | data-audit | 103 | 256 | 14.7354 | 1.8422 | 12.8898 | same | True |
| short | disabled-k16 | 2 | topic-transition | 79 | 256 | 32.7932 | 1.4521 | 31.3378 | 38 | True |
| short | offload-k16 | 2 | code-cache | 73 | 256 | 9.0791 | 1.0352 | 8.0307 | same | True |
| short | offload-k16 | 2 | prose-debug | 78 | 256 | 15.7792 | 0.2407 | 15.5149 | 94 | True |
| short | offload-k16 | 2 | code-sql | 85 | 256 | 12.7488 | 0.2422 | 12.5030 | same | True |
| short | offload-k16 | 2 | code-parser | 81 | 256 | 14.9498 | 0.2437 | 14.6872 | 45 | True |
| short | offload-k16 | 2 | data-audit | 103 | 256 | 8.1643 | 0.7542 | 7.3971 | 29 | True |
| short | offload-k16 | 2 | topic-transition | 79 | 256 | 17.2367 | 0.2332 | 17.0000 | 77 | True |
| short | offload-k8 | 2 | code-cache | 73 | 256 | 10.2933 | 0.7628 | 9.5265 | same | True |
| short | offload-k8 | 2 | prose-debug | 78 | 256 | 15.4725 | 0.2450 | 15.2243 | 94 | True |
| short | offload-k8 | 2 | code-sql | 85 | 256 | 13.1697 | 0.2532 | 12.9129 | 239 | True |
| short | offload-k8 | 2 | code-parser | 81 | 256 | 14.8788 | 0.2417 | 14.6165 | 45 | True |
| short | offload-k8 | 2 | data-audit | 103 | 256 | 10.0304 | 0.5976 | 9.4295 | 29 | True |
| short | offload-k8 | 2 | topic-transition | 79 | 256 | 17.1607 | 0.2613 | 16.8952 | 77 | True |

## Accepted-prefix records

Whole-request milliseconds per recorded cycle include prefill and residual direct decoding. They do not isolate a verification cycle. Survival includes short end-of-request proposals and is specific to each tested policy.

| Stage | Condition | Cycles | Attempted | Accepted | Request ms/recorded cycle | P(A >= position) |
| --- | --- | --- | --- | --- | --- | --- |
| short | cpu-k4 | 926 | 3684 | 2132 | 187.870 | 1:0.7667, 2:0.6004, 3:0.5054, 4:0.4298 |
| short | cpu-k8 | 720 | 5716 | 2336 | 261.510 | 1:0.7333, 2:0.5583, 3:0.4583, 4:0.3861, 5:0.3389, 6:0.2972, 7:0.2528, 8:0.2194 |
| short | cpu-k16 | 614 | 9644 | 2442 | 428.341 | 1:0.7134, 2:0.5342, 3:0.4365, 4:0.3648, 5:0.3062, 6:0.2704, 7:0.2248, 8:0.1922, 9:0.1564, 10:0.1531, 11:0.1368, 12:0.1238, 13:0.1042, 14:0.0945, 15:0.0912, 16:0.0749 |
| short | offload-k8 | 708 | 5586 | 2350 | 227.964 | 1:0.7401, 2:0.5904, 3:0.4689, 4:0.3757, 5:0.3446, 6:0.3079, 7:0.2571, 8:0.2345 |
| short | offload-k16 | 618 | 9630 | 2442 | 251.837 | 1:0.7055, 2:0.5405, 3:0.4207, 4:0.3301, 5:0.2977, 6:0.2621, 7:0.2168, 8:0.1942, 9:0.1683, 10:0.1618, 11:0.1424, 12:0.1294, 13:0.1068, 14:0.1003, 15:0.0939, 16:0.0809 |
| short | disabled-k16 | 616 | 9626 | 2442 | 439.142 | 1:0.7110, 2:0.5422, 3:0.4286, 4:0.3506, 5:0.3019, 6:0.2630, 7:0.2208, 8:0.1916, 9:0.1656, 10:0.1591, 11:0.1429, 12:0.1299, 13:0.1039, 14:0.0974, 15:0.0844, 16:0.0714 |
| long | target | 0 | 0 | 0 | — | — |
| long | cpu-k4 | 140 | 546 | 368 | 1731.368 | 1:0.8286, 2:0.6571, 3:0.6286, 4:0.5143 |
| long | cpu-k16 | 88 | 1280 | 420 | 3516.372 | 1:0.7727, 2:0.5000, 3:0.4773, 4:0.3864, 5:0.3636, 6:0.3182, 7:0.2500, 8:0.2273, 9:0.2045, 10:0.2045, 11:0.1818, 12:0.1818, 13:0.1818, 14:0.1818, 15:0.1818, 16:0.1591 |
| long | offload-k16 | 88 | 1280 | 420 | 1937.723 | 1:0.7727, 2:0.5000, 3:0.4773, 4:0.3864, 5:0.3636, 6:0.3182, 7:0.2500, 8:0.2273, 9:0.2045, 10:0.2045, 11:0.1818, 12:0.1818, 13:0.1818, 14:0.1818, 15:0.1818, 16:0.1591 |

## Actual CUDA request windows

| Threshold | Request | H2D bytes | H2D copies | D2H bytes | Kernels | Summed H2D ms |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | calibration-explanation | 6246614648 | 1351 | 204709888 | 97388 | 123.431 |
| 32 | code-cache | 6242370176 | 744 | 94030336 | 42814 | 122.559 |
| 8 | calibration-explanation | 68744509048 | 4701 | 355454976 | 103888 | 1356.149 |
| 8 | code-cache | 24997028480 | 1749 | 145627648 | 44764 | 491.410 |

## Target-only comparisons

| Target-only factor | Requests | Final historical matches | All historical matches | Unique supplied-prefix matches |
| --- | --- | --- | --- | --- |
| prefill | 8 | 4/8 | 4/8 | 4/8 |
| microbatch | 8 | 6/8 | 6/8 | 6/8 |
| incremental-reference | 477 | 8/8 | 477/477 | 358/358 |
| threads | 477 | 8/8 | 477/477 | 358/358 |
| placement | 477 | 7/8 | 473/477 | 355/358 |

| Reference | Candidate | Stratum | Flips/positions | Max probability infinity distance |
| --- | --- | --- | --- | --- |
| incremental-reference | threads | selected historical divergence | 0/4 | 0.000000000 |
| incremental-reference | threads | fixed ordinary position | 0/4 | 0.000000000 |
| incremental-reference | placement | selected historical divergence | 1/4 | 0.014353655 |
| incremental-reference | placement | fixed ordinary position | 0/4 | 0.006212476 |
| incremental-reference | prefill | selected historical divergence | 4/4 | 0.047704415 |
| incremental-reference | prefill | fixed ordinary position | 0/4 | 0.014610095 |
| prefill | microbatch | selected historical divergence | 2/4 | 0.037716540 |
| prefill | microbatch | fixed ordinary position | 0/4 | 0.012519165 |

## Representative native allocation receipts

### long-r1-offload-k16

```text
0.02.220.195 I load_tensors: offloaded 38/65 layers to GPU
0.02.220.198 I load_tensors:        CUDA0 model buffer size = 10939.06 MiB
0.02.220.199 I load_tensors:    CUDA_Host model buffer size =  7986.95 MiB
0.10.465.773 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.10.466.588 I llama_kv_cache:        CPU KV buffer size =  1032.75 MiB
0.10.813.826 I llama_kv_cache:      CUDA0 KV buffer size =  1415.25 MiB
0.11.017.197 I sched_reserve:      CUDA0 compute buffer size =   205.08 MiB
0.11.017.202 I sched_reserve:  CUDA_Host compute buffer size =    19.08 MiB
0.11.768.196 I load_tensors: offloaded 25/25 layers to GPU
0.11.768.200 I load_tensors:        CUDA0 model buffer size =   500.84 MiB
0.11.768.200 I load_tensors:    CUDA_Host model buffer size =   137.94 MiB
0.12.080.197 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.12.080.906 I llama_kv_cache:      CUDA0 KV buffer size =   114.77 MiB
0.12.089.363 I sched_reserve:      CUDA0 compute buffer size =    25.91 MiB
0.12.089.366 I sched_reserve:  CUDA_Host compute buffer size =    10.79 MiB
```

### long-r1-target

```text
0.02.326.174 I load_tensors: offloaded 38/65 layers to GPU
0.02.326.181 I load_tensors:        CUDA0 model buffer size = 10939.06 MiB
0.02.326.183 I load_tensors:    CUDA_Host model buffer size =  7986.95 MiB
0.11.227.267 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.11.228.294 I llama_kv_cache:        CPU KV buffer size =  1032.75 MiB
0.11.539.753 I llama_kv_cache:      CUDA0 KV buffer size =  1415.25 MiB
0.11.746.026 I sched_reserve:      CUDA0 compute buffer size =   205.08 MiB
0.11.746.030 I sched_reserve:  CUDA_Host compute buffer size =    19.08 MiB
```

### short-r1-offload-k16

```text
0.01.963.765 I load_tensors: offloaded 44/65 layers to GPU
0.01.963.769 I load_tensors:        CUDA0 model buffer size = 12581.02 MiB
0.01.963.770 I load_tensors:    CUDA_Host model buffer size =  6344.99 MiB
0.10.171.989 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.10.172.596 I llama_kv_cache:        CPU KV buffer size =   336.00 MiB
0.10.269.936 I llama_kv_cache:      CUDA0 KV buffer size =   688.00 MiB
0.10.374.892 I sched_reserve:      CUDA0 compute buffer size =   171.83 MiB
0.10.374.897 I sched_reserve:  CUDA_Host compute buffer size =    12.01 MiB
0.11.121.459 I load_tensors: offloaded 25/25 layers to GPU
0.11.121.462 I load_tensors:        CUDA0 model buffer size =   500.84 MiB
0.11.121.463 I load_tensors:    CUDA_Host model buffer size =   137.94 MiB
0.11.436.638 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.11.436.849 I llama_kv_cache:      CUDA0 KV buffer size =    48.00 MiB
0.11.445.843 I sched_reserve:      CUDA0 compute buffer size =    18.88 MiB
0.11.445.847 I sched_reserve:  CUDA_Host compute buffer size =     3.76 MiB
```
