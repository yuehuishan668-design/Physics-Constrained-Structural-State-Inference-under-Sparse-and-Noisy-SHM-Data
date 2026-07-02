# Noise Robustness Summary

## 1. Experiment setup

- Number of noise levels: `6`
- Noise levels: `[0.0, 0.02, 0.05, 0.1, 0.15, 0.2]`
- Each noise level corresponds to an independently generated controlled simulation dataset.

## 2. Best configuration at each noise level

|   noise_percent | feature_set   | model   |   test_mae |   test_rmse |   mae_on_damaged_entries |   n_features |
|----------------:|:--------------|:--------|-----------:|------------:|-------------------------:|-------------:|
|               0 | full          | ridge   |  0.0344682 |   0.0513008 |                0.0526345 |           92 |
|               2 | full          | ridge   |  0.0361334 |   0.053499  |                0.0499892 |           92 |
|               5 | full          | ridge   |  0.0375458 |   0.0557758 |                0.0599427 |           92 |
|              10 | full          | ridge   |  0.0394483 |   0.0591806 |                0.0633879 |           92 |
|              15 | full          | ridge   |  0.0431412 |   0.0645431 |                0.0688998 |           92 |
|              20 | full          | ridge   |  0.0419665 |   0.0649815 |                0.0674867 |           92 |

## 3. Selected preferred configurations

|   noise_percent | feature_set         | model         |   test_mae |   test_rmse |   mae_on_damaged_entries |   mean_prediction_on_zero_entries |   n_features |
|----------------:|:--------------------|:--------------|-----------:|------------:|-------------------------:|----------------------------------:|-------------:|
|               0 | full                | ridge         |  0.0344682 |   0.0513008 |                0.0526345 |                         0.0255206 |           92 |
|               0 | no_meta             | ridge         |  0.0389673 |   0.0557376 |                0.0569686 |                         0.030101  |           86 |
|               0 | full                | elasticnet    |  0.0462801 |   0.062696  |                0.0685551 |                         0.0353088 |           92 |
|               0 | response_basic_only | ridge         |  0.0596605 |   0.0784671 |                0.0902241 |                         0.0446067 |           52 |
|               0 | full                | random_forest |  0.072728  |   0.0914353 |                0.116073  |                         0.0513791 |           92 |
|               2 | full                | ridge         |  0.0361334 |   0.053499  |                0.0499892 |                         0.0301479 |           92 |
|               2 | no_meta             | ridge         |  0.0379261 |   0.055278  |                0.0552885 |                         0.0304258 |           86 |
|               2 | full                | elasticnet    |  0.0456345 |   0.0611194 |                0.0661407 |                         0.0367762 |           92 |
|               2 | response_basic_only | ridge         |  0.0586361 |   0.0769709 |                0.0917002 |                         0.044353  |           52 |
|               2 | full                | random_forest |  0.0703641 |   0.0911861 |                0.115735  |                         0.0507648 |           92 |
|               5 | full                | ridge         |  0.0375458 |   0.0557758 |                0.0599427 |                         0.0280988 |           92 |
|               5 | no_meta             | ridge         |  0.0410009 |   0.0591806 |                0.0641381 |                         0.0312416 |           86 |
|               5 | full                | elasticnet    |  0.0475577 |   0.0648118 |                0.0751361 |                         0.0359251 |           92 |
|               5 | response_basic_only | ridge         |  0.0598627 |   0.0769667 |                0.0939847 |                         0.04547   |           52 |
|               5 | full                | random_forest |  0.0706103 |   0.0889045 |                0.116547  |                         0.0512343 |           92 |
|              10 | full                | ridge         |  0.0394483 |   0.0591806 |                0.0633879 |                         0.0289425 |           92 |
|              10 | no_meta             | ridge         |  0.0434859 |   0.0619536 |                0.0663212 |                         0.0334647 |           86 |
|              10 | full                | elasticnet    |  0.049066  |   0.0682711 |                0.0786077 |                         0.0361017 |           92 |
|              10 | response_basic_only | ridge         |  0.0625949 |   0.0821522 |                0.100434  |                         0.0459894 |           52 |
|              10 | full                | random_forest |  0.0741769 |   0.0927049 |                0.121405  |                         0.0534512 |           92 |
|              15 | full                | ridge         |  0.0431412 |   0.0645431 |                0.0688998 |                         0.0310195 |           92 |
|              15 | no_meta             | ridge         |  0.0464234 |   0.0683406 |                0.0745619 |                         0.0331817 |           86 |
|              15 | full                | elasticnet    |  0.0513633 |   0.0719696 |                0.0841915 |                         0.0359148 |           92 |
|              15 | response_basic_only | ridge         |  0.0610548 |   0.0828784 |                0.0985774 |                         0.0433971 |           52 |
|              15 | full                | random_forest |  0.0725907 |   0.0933722 |                0.120301  |                         0.0501388 |           92 |
|              20 | full                | ridge         |  0.0419665 |   0.0649815 |                0.0674867 |                         0.0306585 |           92 |
|              20 | no_meta             | ridge         |  0.0443345 |   0.0657363 |                0.0720106 |                         0.0320712 |           86 |
|              20 | full                | elasticnet    |  0.0489447 |   0.0699602 |                0.0811241 |                         0.0346861 |           92 |
|              20 | response_basic_only | ridge         |  0.0590041 |   0.0807252 |                0.0963216 |                         0.0424688 |           52 |
|              20 | full                | random_forest |  0.0729452 |   0.0943999 |                0.126552  |                         0.0491921 |           92 |

## 4. Preliminary interpretation

- `full + ridge` should be checked as the main configuration across noise levels.
- Because each noise level uses an independently generated dataset, the trend does not need to be strictly monotonic.
- If the overall trend increases with noise and `full + ridge` remains best or near-best, the result supports noise robustness.
- If high-noise performance collapses, it should be reported as a limitation.

中文解释：

- 重点检查 `full + ridge` 是否在各噪声水平下仍然保持最优或接近最优。
- 由于每个噪声水平对应独立生成的数据集，误差曲线不一定严格单调。
- 如果整体误差随噪声升高而上升，且 `full + ridge` 保持领先，则可支撑噪声鲁棒性结论。
- 如果高噪声下性能崩溃，应作为局限性写入论文。