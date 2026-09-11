# Per-layer omission study

Status: **layer_study_completed**.

Frozen calibration layouts; teacher-forced development articles; dense masked execution.
Selected FFN volume is hypothetical, with unmasked layers charged in full. No paging speedup is measured.

| Layout | Condition | Layers | Mean KL | Relative perplexity | Top-1 | Whole-FFN weight fraction |
|---|---|---|---:|---:|---:|---:|
| popularity | layer-00 | 0 | 0.00649801 | 1.003543 | 96.94% | 99.1071% |
| popularity | layer-01 | 1 | 0.00225746 | 1.003590 | 98.19% | 99.1071% |
| popularity | layer-02 | 2 | 0.0040824 | 1.002494 | 97.84% | 99.1071% |
| popularity | layer-03 | 3 | 0.00606994 | 0.999225 | 97.28% | 99.1071% |
| popularity | layer-04 | 4 | 0.0073517 | 1.004870 | 97.03% | 99.1071% |
| popularity | layer-05 | 5 | 0.00573396 | 1.004804 | 96.86% | 99.1071% |
| popularity | layer-06 | 6 | 0.00742754 | 1.004355 | 96.40% | 99.1071% |
| popularity | layer-07 | 7 | 0.00660869 | 1.005952 | 96.18% | 99.1071% |
| popularity | layer-08 | 8 | 0.00589315 | 1.002148 | 96.54% | 99.1071% |
| popularity | layer-09 | 9 | 0.00470362 | 1.000450 | 96.96% | 99.1071% |
| popularity | layer-10 | 10 | 0.00477171 | 1.000431 | 96.72% | 99.1071% |
| popularity | layer-11 | 11 | 0.00433417 | 1.001763 | 97.13% | 99.1071% |
| popularity | layer-12 | 12 | 0.00383347 | 0.998745 | 96.91% | 99.1071% |
| popularity | layer-13 | 13 | 0.00316096 | 1.001898 | 97.23% | 99.1071% |
| popularity | layer-14 | 14 | 0.00272312 | 1.002230 | 97.79% | 99.1071% |
| popularity | layer-15 | 15 | 0.00310884 | 1.000943 | 97.16% | 99.1071% |
| popularity | layer-16 | 16 | 0.0030186 | 1.002593 | 97.50% | 99.1071% |
| popularity | layer-17 | 17 | 0.0032678 | 1.003254 | 97.30% | 99.1071% |
| popularity | layer-18 | 18 | 0.00340569 | 1.003660 | 97.16% | 99.1071% |
| popularity | layer-19 | 19 | 0.00426544 | 1.002833 | 96.72% | 99.1071% |
| popularity | layer-20 | 20 | 0.00385665 | 1.003223 | 97.57% | 99.1071% |
| popularity | layer-21 | 21 | 0.00482413 | 1.003543 | 96.42% | 99.1071% |
| popularity | layer-22 | 22 | 0.00491901 | 1.005279 | 96.72% | 99.1071% |
| popularity | layer-23 | 23 | 0.00539751 | 1.005250 | 97.13% | 99.1071% |
| popularity | layer-24 | 24 | 0.00541116 | 1.005855 | 96.96% | 99.1071% |
| popularity | layer-25 | 25 | 0.00688387 | 1.005686 | 97.28% | 99.1071% |
| popularity | layer-26 | 26 | 0.00605415 | 1.004369 | 97.13% | 99.1071% |
| popularity | layer-27 | 27 | 0.00593027 | 1.004303 | 96.37% | 99.1071% |
| popularity | all | 0-27 | 0.154117 | 1.128710 | 83.43% | 75.0000% |
| popularity | block-0 | 0-6 | 0.0442465 | 1.032529 | 91.54% | 93.7500% |
| popularity | block-1 | 7-13 | 0.0353343 | 1.016832 | 91.64% | 93.7500% |
| popularity | block-2 | 14-20 | 0.0265094 | 1.020144 | 92.84% | 93.7500% |
| popularity | block-3 | 21-27 | 0.0412287 | 1.037921 | 91.47% | 93.7500% |
| coactivation | layer-00 | 0 | 0.00675539 | 1.005288 | 96.99% | 99.1071% |
| coactivation | layer-01 | 1 | 0.00274882 | 1.003754 | 98.21% | 99.1071% |
| coactivation | layer-02 | 2 | 0.00386911 | 1.001822 | 97.84% | 99.1071% |
| coactivation | layer-03 | 3 | 0.00651893 | 1.001900 | 97.01% | 99.1071% |
| coactivation | layer-04 | 4 | 0.00730479 | 1.009445 | 96.64% | 99.1071% |
| coactivation | layer-05 | 5 | 0.00548426 | 1.002725 | 96.91% | 99.1071% |
| coactivation | layer-06 | 6 | 0.00787501 | 1.007754 | 95.76% | 99.1071% |
| coactivation | layer-07 | 7 | 0.00722944 | 1.006534 | 96.18% | 99.1071% |
| coactivation | layer-08 | 8 | 0.00670626 | 1.002640 | 96.37% | 99.1071% |
| coactivation | layer-09 | 9 | 0.005334 | 1.003249 | 96.52% | 99.1071% |
| coactivation | layer-10 | 10 | 0.00534336 | 1.009023 | 96.52% | 99.1071% |
| coactivation | layer-11 | 11 | 0.00529161 | 1.000581 | 96.79% | 99.1071% |
| coactivation | layer-12 | 12 | 0.004525 | 1.001850 | 96.20% | 99.1071% |
| coactivation | layer-13 | 13 | 0.00328477 | 1.004212 | 97.21% | 99.1071% |
| coactivation | layer-14 | 14 | 0.00339608 | 1.001586 | 97.16% | 99.1071% |
| coactivation | layer-15 | 15 | 0.00380092 | 0.998397 | 97.08% | 99.1071% |
| coactivation | layer-16 | 16 | 0.00374962 | 1.003365 | 97.13% | 99.1071% |
| coactivation | layer-17 | 17 | 0.0039759 | 1.006230 | 96.47% | 99.1071% |
| coactivation | layer-18 | 18 | 0.00387245 | 1.003183 | 97.25% | 99.1071% |
| coactivation | layer-19 | 19 | 0.00505548 | 1.003751 | 96.42% | 99.1071% |
| coactivation | layer-20 | 20 | 0.00474925 | 1.001808 | 96.59% | 99.1071% |
| coactivation | layer-21 | 21 | 0.00573789 | 1.003819 | 96.62% | 99.1071% |
| coactivation | layer-22 | 22 | 0.0062985 | 1.003239 | 96.74% | 99.1071% |
| coactivation | layer-23 | 23 | 0.00589658 | 1.002594 | 97.21% | 99.1071% |
| coactivation | layer-24 | 24 | 0.00586546 | 1.009013 | 97.23% | 99.1071% |
| coactivation | layer-25 | 25 | 0.00582759 | 1.006169 | 97.11% | 99.1071% |
| coactivation | layer-26 | 26 | 0.00619159 | 1.006391 | 96.91% | 99.1071% |
| coactivation | layer-27 | 27 | 0.00630608 | 1.007918 | 96.69% | 99.1071% |
| coactivation | all | 0-27 | 0.179357 | 1.167550 | 81.13% | 75.0000% |
| coactivation | block-0 | 0-6 | 0.0452979 | 1.041212 | 91.05% | 93.7500% |
| coactivation | block-1 | 7-13 | 0.0404899 | 1.037939 | 90.93% | 93.7500% |
| coactivation | block-2 | 14-20 | 0.0310902 | 1.015668 | 91.23% | 93.7500% |
| coactivation | block-3 | 21-27 | 0.0494143 | 1.048367 | 90.54% | 93.7500% |
