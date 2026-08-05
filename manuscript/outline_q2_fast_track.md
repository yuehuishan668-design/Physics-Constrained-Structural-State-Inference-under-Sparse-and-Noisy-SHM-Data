# Physics-Informed Descriptor Construction for Structural State Inference under Sparse and Noisy SHM Data

## Working title

Physics-Informed Descriptor Construction for Structural Damage Inference under Sparse and Noisy Structural Health Monitoring Data

Alternative title:

Robust Physics-Informed Feature Construction for Structural Damage Inference under Sparse and Noisy SHM Data

---

# Abstract

## Draft logic

1. Background:
   - Structural health monitoring data are often sparse and noisy.
   - Pure data-driven models may suffer from limited interpretability and unstable reliability under severe damage or limited sensor coverage.

2. Objective:
   - This study proposes and evaluates a physics-informed descriptor construction framework for structural damage inference.

3. Method:
   - A controlled OpenSeesPy-based simulation dataset is generated.
   - Response-derived and excitation-related descriptors are extracted.
   - Feature ablation, repeated-split robustness, damage-stratified reliability, noise robustness, and sensor sparsity stress tests are conducted.

4. Results:
   - The full physics-informed descriptor set with ridge regression achieves the strongest overall performance.
   - Repeated-split analysis confirms that the result is not split-specific.
   - Damage-stratified analysis reveals systematic high-damage underestimation.
   - Noise robustness and sensor sparsity experiments identify practical boundaries.

5. Contribution:
   - The study shows that physics-informed descriptor construction can improve stable damage inference, but average accuracy alone is insufficient for safety-critical SHM evaluation.

## Placeholder abstract

Structural health monitoring data are commonly sparse, noisy, and limited in labeled damage observations, which makes reliable structural state inference challenging. This study develops a physics-informed descriptor construction framework for structural damage inference under sparse and noisy monitoring conditions. A controlled OpenSeesPy-based simulation dataset is generated, and a 92-dimensional descriptor set is constructed from response statistics, frequency-domain information, spatial response patterns, response correlations, and excitation-related metadata. The proposed descriptor set is evaluated through main feature ablation, repeated-split robustness, damage-stratified reliability diagnosis, noise robustness analysis, and sensor sparsity stress testing. On the 3000-case controlled simulation dataset, the full descriptor set combined with ridge regression achieves the best fixed-split performance, with a test MAE of 0.0393 and a test RMSE of 0.0585. Repeated-split analysis over 10 random partitions confirms stable performance, with a mean test MAE of 0.0454 ± 0.0009. However, damage-stratified evaluation reveals that high-damage entries remain systematically underestimated, indicating that average accuracy alone is insufficient for safety-critical evaluation. Noise robustness experiments show that the full-feature ridge model remains the best-performing configuration across 0%–20% noise levels, while sensor sparsity stress tests reveal substantial degradation under extreme sensor reduction. These results demonstrate that physics-informed descriptor construction improves robust structural damage inference under noisy and moderately sparse monitoring conditions, while also identifying reliability limits in high-damage and severely sparse sensing regimes.

---

# Keywords

- Structural health monitoring
- Structural damage inference
- Physics-informed descriptors
- Sparse sensing
- Noisy monitoring data
- Damage-stratified reliability
- OpenSeesPy simulation

---

# 1. Introduction

## 1.1 Background

Structural health monitoring systems provide important information for assessing structural condition, detecting damage, and supporting maintenance decisions. However, practical SHM data are often sparse, noisy, and incomplete. Sensor deployment is constrained by cost, accessibility, and maintenance conditions. Measurement noise and excitation variability further complicate structural state inference.

## 1.2 Problem statement

Purely data-driven models can achieve strong fitting performance when sufficient labeled data are available, but their reliability may degrade under sparse sensing, noisy response measurements, and severe damage regimes. In SHM applications, these limitations are safety-critical because underestimating severe damage can lead to unsafe decisions.

## 1.3 Research gap

Existing studies often focus on average prediction accuracy, such as MAE or RMSE, but average metrics may hide poor reliability in high-damage regimes. In addition, many model comparisons emphasize estimator complexity while paying less attention to physically meaningful descriptor construction and damage-stratified evaluation.

## 1.4 Objective

This study aims to develop and evaluate a physics-informed descriptor construction framework for structural damage inference under sparse and noisy SHM data.

## 1.5 Contributions

The main contributions are:

1. A 92-dimensional physics-informed descriptor set is constructed from response statistics, frequency-domain indicators, spatial response patterns, response correlations, and excitation-related metadata.

2. A controlled OpenSeesPy-based simulation workflow is developed to evaluate structural damage inference under sparse and noisy monitoring conditions.

3. The proposed descriptor set is evaluated through main ablation, repeated-split robustness, noise robustness, sensor sparsity stress testing, and damage-stratified reliability diagnosis.

4. The results show that average accuracy alone is insufficient for safety-critical SHM evaluation, because high-damage entries remain more difficult and can be systematically underestimated.

## Figures and tables used in Introduction

No main result figure required. May include conceptual framework later if needed.

---

# 2. Methodology

## 2.1 Overview of the proposed framework

The proposed workflow consists of:

1. Controlled structural response simulation.
2. Damage label generation.
3. Physics-informed descriptor extraction.
4. Feature ablation and estimator comparison.
5. Robustness and reliability evaluation.

Suggested figure:

- Optional Figure 0: Methodological workflow diagram.

## 2.2 Controlled structural simulation

Use OpenSeesPy to generate structural response data under different damage, excitation, and noise conditions.

Need to describe:

- Structural model type.
- Number of stories / outputs: 4.
- Response length: 2000 time steps.
- Time interval: dt = 0.01 s.
- Response tensor: 3000 × 2000 × 4.
- Damage vector: four-story damage vector.

Relevant table:

- Table 1: Dataset quality summary.
- Source: `T00_dataset_quality_debug_plus_3000.json`.

## 2.3 Damage representation

Damage is represented as a four-dimensional story-level damage vector.

Need to define:

- Healthy cases.
- Damaged cases.
- High-damage entries.
- Damage bins: zero, low, medium, high.

## 2.4 Physics-informed descriptor construction

The full descriptor set contains 92 features.

Descriptor groups:

1. Response basic statistics.
2. Frequency-domain descriptors.
3. Spatial response descriptors.
4. Response correlation descriptors.
5. Excitation-related metadata.
6. Combined full descriptor set.

Relevant table:

- Table 2 or supplementary table: feature summary.
- Source: `T01_physics_feature_summary_debug_plus_3000.json`.

## 2.5 Estimators

Main estimators:

1. Ridge regression.
2. ElasticNet.
3. Random forest.

Important interpretation:

- Estimator complexity is not assumed to be better.
- Regularized linear models are tested against nonlinear tree models.
- Ridge regression is selected when stable descriptor-based inference is desired.

## 2.6 Evaluation metrics

Core metrics:

- Test MAE.
- Test RMSE.
- Damaged-entry MAE.
- Prediction bias.
- Underestimation ratio.
- Mean prediction on zero entries.

Damage-stratified metrics:

- Zero-damage MAE.
- Low-damage MAE.
- Medium-damage MAE.
- High-damage MAE.
- High-damage bias.
- High-damage underestimation ratio.

---

# 3. Experimental Design

## 3.1 Main 3000-case experiment

Dataset:

- `debug_plus_3000`
- 3000 cases.
- Train/validation/test split: 2100 / 450 / 450.

Relevant table:

- `T02_main_ablation_3000.csv`.

Relevant figure:

- `F01_main_ablation_top10_test_mae.png`
- `F01_main_ablation_top10_test_rmse.png`

## 3.2 Repeated-split robustness

Purpose:

- Test whether the fixed-split result is caused by one favorable partition.

Design:

- 10 repeated random seeds.
- Same 3000-case dataset.
- Train/validation/test: 2100 / 450 / 450 for each seed.

Relevant tables:

- `T03_repeated_split_robustness_summary.csv`
- `T04_repeated_split_per_seed_best.csv`

Relevant figures:

- `F02_repeated_split_seed_robustness_test_mae_boxplot.png`
- `F02_repeated_split_seed_robustness_test_mae_mean_std.png`
- `F02_repeated_split_seed_robustness_bias_vs_mae.png`

## 3.3 Damage-stratified reliability diagnosis

Purpose:

- Evaluate whether average accuracy hides poor performance in safety-critical damage regimes.

Relevant tables:

- `T05_damage_stratified_config_summary.csv`
- `T06_damage_stratified_runs.csv`

Relevant figures:

- `F03_damage_stratified_damage_stratified_mae_by_bin.png`
- `F03_damage_stratified_damage_stratified_bias_by_bin.png`
- `F03_damage_stratified_damage_stratified_underestimation_ratio.png`
- `F03_damage_stratified_best_config_true_vs_pred_by_damage_bin.png`

## 3.4 Noise robustness

Purpose:

- Evaluate robustness under noisy monitoring data.

Design:

- Six independently generated 1000-case datasets.
- Noise levels: 0%, 2%, 5%, 10%, 15%, 20%.

Relevant tables:

- `T07_noise_best_by_noise.csv`
- `T08_noise_robustness_summary.csv`

Relevant figures:

- `F04_noise_noise_vs_test_mae.png`
- `F04_noise_noise_vs_test_rmse.png`
- `F04_noise_noise_vs_damaged_mae.png`

Important note:

- Since each noise level corresponds to an independently generated dataset, performance trends need not be strictly monotonic.

## 3.5 Sensor sparsity stress test

Purpose:

- Evaluate sparse SHM data conditions.

Design:

- Zero-masking unavailable response channels.
- Original four-story damage labels preserved.
- Response tensor shape preserved.

Sensor layouts:

- 4 sensors: 1-2-3-4
- 3 sensors: 1-2-4
- 2 sensors: 1-4
- 1 sensor: 4

Relevant tables:

- `T09_sensor_best_by_sensor_count.csv`
- `T10_sensor_sparsity_summary.csv`

Relevant figures:

- `F05_sensor_sparsity_sensor_count_vs_test_mae.png`
- `F05_sensor_sparsity_sensor_count_vs_test_rmse.png`
- `F05_sensor_sparsity_sensor_count_vs_damaged_mae.png`

Important note:

- This is a zero-masking sensor sparsity stress test, not an optimal missing-data imputation method.

---

# 4. Results

## 4.1 Main ablation results

Key result:

- full + ridge:
  - test MAE = 0.0393
  - test RMSE = 0.0585

Main interpretation:

- Full descriptor set is better than reduced feature sets.
- Ridge regression is more stable than ElasticNet and random forest in the full 4-sensor setting.
- Response-basic-only descriptors are insufficient.

Use:

- Table 2: `T02_main_ablation_3000.csv`
- Figure 1: main ablation figure.

Draft paragraph:

On the 3000-case controlled simulation dataset, the full physics-informed feature set combined with ridge regression achieved the best fixed-split performance, with a test MAE of 0.0393 and a test RMSE of 0.0585. Removing metadata-related descriptors increased the test MAE, while response-basic-only descriptors led to substantially larger errors. These results indicate that the full descriptor set provided the most informative representation for structural damage inference under the present sparse and noisy SHM simulation setting.

## 4.2 Repeated-split robustness

Key result:

- full + ridge:
  - mean test MAE = 0.0454
  - std test MAE = 0.0009
  - best in all 10 splits.

Use:

- Table 3.
- Figure 2.

Draft paragraph:

To examine whether the fixed-split ablation result was sensitive to data partitioning, a repeated-split robustness analysis was performed on the 3000-case dataset using 10 random seeds. The full physics-informed feature set combined with ridge regression achieved the best mean test MAE of 0.0454 with a standard deviation of 0.0009, and it was selected as the best-performing configuration in all 10 splits. This confirms that the main conclusion was stable rather than split-specific.

## 4.3 Damage-stratified reliability

Key result:

- full + ridge:
  - overall MAE = 0.0407
  - high-damage MAE = 0.0888
  - high-damage bias = -0.0717
  - high-damage underestimation ratio = 0.8214

Use:

- Table 4.
- Figure 3.

Draft paragraph:

Damage-stratified evaluation further revealed that stable average performance did not imply uniform reliability across damage severities. The full physics-informed feature set with ridge regression achieved the best overall MAE of 0.0407, but its high-damage MAE increased to 0.0888. In the high-damage bin, the mean true damage was 0.2758, whereas the mean predicted damage was 0.2041, resulting in a negative bias of -0.0717 and an underestimation ratio of 0.8214. Therefore, damage-stratified reliability diagnosis is necessary for evaluating SHM-based structural state inference models, especially in safety-critical damage regimes.

## 4.4 Noise robustness

Key result:

- full + ridge is best across all tested noise levels.
- test MAE:
  - 0%: 0.0345
  - 2%: 0.0361
  - 5%: 0.0375
  - 10%: 0.0394
  - 15%: 0.0431
  - 20%: 0.0420

Use:

- Table 5.
- Figure 4.

Draft paragraph:

Noise robustness was evaluated using six independently generated 1000-case datasets with fixed noise levels of 0%, 2%, 5%, 10%, 15%, and 20%. The full physics-informed feature set combined with ridge regression remained the best-performing configuration across all noise levels. Its test MAE increased from 0.0345 at 0% noise to 0.0431 at 15% noise and 0.0420 at 20% noise, while no performance collapse was observed. Although the error trend was not strictly monotonic because each noise level used an independently generated dataset, the overall results indicate that the proposed descriptor set retained robust predictive performance under noisy monitoring conditions.

## 4.5 Sensor sparsity stress test

Key result:

Best configurations:

- 4 sensors: full + ridge, MAE = 0.0393.
- 3 sensors: full + ridge, MAE = 0.0612.
- 2 sensors: full + random forest, MAE = 0.0764.
- 1 sensor: full + random forest, MAE = 0.0801.

Use:

- Table 6.
- Figure 5.

Draft paragraph:

Sensor sparsity was evaluated by zero-masking unavailable response channels while preserving the original four-story damage targets and tensor shape. The full physics-informed feature set remained the strongest descriptor set across all sensor configurations. With four and three sensors, the full-feature ridge model achieved the best test MAE of 0.0393 and 0.0612, respectively. Under more extreme sparsity, the best estimator shifted to random forest, yielding test MAEs of 0.0764 for the two-sensor layout and 0.0801 for the one-sensor layout. These results indicate that the proposed descriptor set remains effective under moderate sensor sparsity, but severe sensor reduction substantially degrades prediction reliability, especially for damaged entries.

---

# 5. Discussion

## 5.1 Descriptor construction matters more than estimator complexity

Across the main ablation, repeated-split robustness, and noise robustness experiments, the full physics-informed descriptor set with ridge regression consistently provided strong performance. This suggests that physically structured descriptor construction can be more important than increasing estimator complexity under sparse and noisy SHM conditions.

## 5.2 Average accuracy is insufficient for SHM reliability

Damage-stratified analysis showed that high-damage entries were systematically underestimated, even when overall MAE was low. This finding is important because SHM applications are safety-critical, and underestimating severe damage can be more consequential than average-case prediction errors.

## 5.3 Metadata provides incremental but not dominant information

The full feature set consistently outperformed no-metadata variants, but the margin was moderate. This indicates that excitation-related metadata provides useful auxiliary information, while the main predictive signal remains response-derived.

## 5.4 Sensor sparsity reveals a practical boundary

The proposed descriptor set remained effective under moderate sensor reduction, but severe reduction to two or one sensor substantially degraded performance. In the one-sensor setting, the advantage of the full descriptor set diminished, suggesting that several physics-informed descriptors rely on sufficient spatial sensing coverage.

---

# 6. Limitations

1. The dataset is generated from controlled simulation rather than real field SHM measurements.
2. The structural model is simplified and should not be interpreted as a direct real-bridge deployment case.
3. Sensor sparsity is simulated through zero-masking, not through an optimized missing-data imputation or sensor placement strategy.
4. Noise robustness datasets are independently generated for each noise level, so error trends are not strictly monotonic.
5. High-damage underestimation remains unresolved and should be addressed by damage-aware training, weighted loss, or targeted high-damage data augmentation in future work.
6. The current study focuses on descriptor construction and classical estimators rather than end-to-end deep learning models.

---

# 7. Conclusion

This study developed and evaluated a physics-informed descriptor construction framework for structural damage inference under sparse and noisy SHM data. The full descriptor set combined response-derived and excitation-related information and achieved the strongest overall performance on the 3000-case controlled simulation dataset. Repeated-split robustness confirmed that the main result was not caused by a favorable single data partition. Noise robustness experiments showed that the proposed descriptor set retained stable performance under increasing measurement noise. Sensor sparsity stress testing demonstrated effectiveness under moderate sensor reduction, but also identified severe sensor sparsity as a practical boundary. Finally, damage-stratified reliability diagnosis revealed that average accuracy alone is insufficient, because high-damage entries remained systematically underestimated. These results suggest that physics-informed descriptor construction is useful for robust SHM-based structural state inference, while reliability-oriented evaluation remains necessary for safety-critical deployment.

---

# 8. Figure list

## Figure 1

Main physics-informed feature ablation.

Source:

- `results/paper_ready/q2_fast_track/figures/F01_main_ablation_top10_test_mae.png`
- `results/paper_ready/q2_fast_track/figures/F01_main_ablation_top10_test_rmse.png`

## Figure 2

Repeated-split robustness.

Source:

- `results/paper_ready/q2_fast_track/figures/F02_repeated_split_seed_robustness_test_mae_boxplot.png`
- `results/paper_ready/q2_fast_track/figures/F02_repeated_split_seed_robustness_test_mae_mean_std.png`

## Figure 3

Damage-stratified reliability diagnosis.

Source:

- `results/paper_ready/q2_fast_track/figures/F03_damage_stratified_damage_stratified_mae_by_bin.png`
- `results/paper_ready/q2_fast_track/figures/F03_damage_stratified_damage_stratified_bias_by_bin.png`
- `results/paper_ready/q2_fast_track/figures/F03_damage_stratified_damage_stratified_underestimation_ratio.png`

## Figure 4

Noise robustness.

Source:

- `results/paper_ready/q2_fast_track/figures/F04_noise_noise_vs_test_mae.png`
- `results/paper_ready/q2_fast_track/figures/F04_noise_noise_vs_test_rmse.png`

## Figure 5

Sensor sparsity stress test.

Source:

- `results/paper_ready/q2_fast_track/figures/F05_sensor_sparsity_sensor_count_vs_test_mae.png`
- `results/paper_ready/q2_fast_track/figures/F05_sensor_sparsity_sensor_count_vs_test_rmse.png`

---

# 9. Table list

## Table 1

Dataset quality summary.

Source:

- `results/paper_ready/q2_fast_track/tables/T00_dataset_quality_debug_plus_3000.json`
- `results/paper_ready/q2_fast_track/tables/T01_physics_feature_summary_debug_plus_3000.json`

## Table 2

Main ablation result.

Source:

- `results/paper_ready/q2_fast_track/tables/T02_main_ablation_3000.csv`

## Table 3

Repeated-split robustness result.

Source:

- `results/paper_ready/q2_fast_track/tables/T03_repeated_split_robustness_summary.csv`
- `results/paper_ready/q2_fast_track/tables/T04_repeated_split_per_seed_best.csv`

## Table 4

Damage-stratified reliability metrics.

Source:

- `results/paper_ready/q2_fast_track/tables/T05_damage_stratified_config_summary.csv`
- `results/paper_ready/q2_fast_track/tables/T06_damage_stratified_runs.csv`

## Table 5

Noise robustness result.

Source:

- `results/paper_ready/q2_fast_track/tables/T07_noise_best_by_noise.csv`
- `results/paper_ready/q2_fast_track/tables/T08_noise_robustness_summary.csv`

## Table 6

Sensor sparsity stress-test result.

Source:

- `results/paper_ready/q2_fast_track/tables/T09_sensor_best_by_sensor_count.csv`
- `results/paper_ready/q2_fast_track/tables/T10_sensor_sparsity_summary.csv`
