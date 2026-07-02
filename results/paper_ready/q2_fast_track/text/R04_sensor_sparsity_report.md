# Sensor Sparsity Stress-Test Summary

## 1. Experiment setup

- Sensor sparsity was simulated by zero-masking unavailable response channels.
- The output damage labels remained four-story damage vectors.
- The response tensor shape was preserved to keep downstream feature extraction compatible.

## 2. Best configuration by sensor count

|   sensor_count | sensor_layout   | feature_set   | model         |   test_mae |   test_rmse |   mae_on_damaged_entries |   n_features |
|---------------:|:----------------|:--------------|:--------------|-----------:|------------:|-------------------------:|-------------:|
|              4 | 1-2-3-4         | full          | ridge         |  0.0393357 |   0.0585481 |                0.0634314 |           92 |
|              3 | 1-2-4           | full          | ridge         |  0.0612044 |   0.0835868 |                0.100631  |           92 |
|              2 | 1-4             | full          | random_forest |  0.0763734 |   0.0977175 |                0.126961  |           92 |
|              1 | 4               | full          | random_forest |  0.0801414 |   0.10109   |                0.132297  |           92 |

## 3. Selected preferred configurations

|   sensor_count | sensor_layout   | feature_set         | model         |   test_mae |   test_rmse |   mae_on_damaged_entries |   mean_prediction_on_zero_entries |   n_features |
|---------------:|:----------------|:--------------------|:--------------|-----------:|------------:|-------------------------:|----------------------------------:|-------------:|
|              4 | 1-2-3-4         | full                | ridge         |  0.0393357 |   0.0585481 |                0.0634314 |                         0.0285942 |           92 |
|              4 | 1-2-3-4         | no_meta             | ridge         |  0.0424577 |   0.0617063 |                0.069652  |                         0.030335  |           86 |
|              4 | 1-2-3-4         | full                | elasticnet    |  0.0476007 |   0.0657015 |                0.0780666 |                         0.0340195 |           92 |
|              4 | 1-2-3-4         | response_basic_only | ridge         |  0.0577182 |   0.0773266 |                0.0941736 |                         0.0414671 |           52 |
|              4 | 1-2-3-4         | full                | random_forest |  0.0644156 |   0.0846599 |                0.110377  |                         0.0439267 |           92 |
|              3 | 1-2-4           | full                | ridge         |  0.0612044 |   0.0835868 |                0.100631  |                         0.0436285 |           92 |
|              3 | 1-2-4           | no_meta             | ridge         |  0.0635672 |   0.0857249 |                0.105136  |                         0.0450363 |           86 |
|              3 | 1-2-4           | full                | elasticnet    |  0.0663609 |   0.0871141 |                0.109541  |                         0.0471117 |           92 |
|              3 | 1-2-4           | full                | random_forest |  0.0669101 |   0.0883419 |                0.113735  |                         0.0460362 |           92 |
|              3 | 1-2-4           | response_basic_only | ridge         |  0.073486  |   0.0942698 |                0.121224  |                         0.0522053 |           52 |
|              2 | 1-4             | full                | random_forest |  0.0763734 |   0.0977175 |                0.126961  |                         0.0538225 |           92 |
|              2 | 1-4             | full                | ridge         |  0.0788544 |   0.100172  |                0.13264   |                         0.0548779 |           92 |
|              2 | 1-4             | no_meta             | ridge         |  0.0792538 |   0.100392  |                0.133144  |                         0.0552304 |           86 |
|              2 | 1-4             | full                | elasticnet    |  0.0796994 |   0.100234  |                0.132743  |                         0.0560535 |           92 |
|              2 | 1-4             | response_basic_only | ridge         |  0.0823039 |   0.102433  |                0.135686  |                         0.0585071 |           52 |
|              1 | 4               | full                | random_forest |  0.0801414 |   0.10109   |                0.132297  |                         0.0568912 |           92 |
|              1 | 4               | full                | ridge         |  0.0845468 |   0.10457   |                0.139554  |                         0.0600256 |           92 |
|              1 | 4               | no_meta             | ridge         |  0.0846192 |   0.104525  |                0.139441  |                         0.0601807 |           86 |
|              1 | 4               | full                | elasticnet    |  0.0846891 |   0.104599  |                0.139647  |                         0.0601898 |           92 |
|              1 | 4               | response_basic_only | ridge         |  0.0847817 |   0.104681  |                0.139613  |                         0.0603389 |           52 |

## 4. Preliminary interpretation

- If `full + ridge` remains best or near-best as sensor count decreases, the feature-construction conclusion is robust to sensor sparsity.
- If errors increase as sensors are removed, the experiment supports the sparse-monitoring interpretation.
- If one-sensor performance collapses, this should be reported as the practical limit of the current method.
- This experiment should be described as a zero-masking sensor sparsity stress test, not as an optimal missing-data imputation method.

中文解释：

- 如果 `full + ridge` 在传感器数量减少时仍保持最优或接近最优，说明物理特征构建结论对传感器稀疏性具有稳健性。
- 如果传感器减少导致误差上升，则实验能够支撑 sparse monitoring data 的论文主张。
- 如果单传感器性能明显崩溃，应作为当前方法的实用边界写入论文。
- 本实验应被表述为 zero-masking 传感器稀疏性压力测试，而不是最优缺失数据填补方法。