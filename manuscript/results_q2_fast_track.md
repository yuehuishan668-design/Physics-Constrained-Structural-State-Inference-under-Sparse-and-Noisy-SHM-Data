# 4. Results

This section presents the experimental results of the proposed physics-informed descriptor construction framework. The evaluation follows five stages: main feature ablation on the 3000-case controlled simulation dataset, repeated-split robustness, damage-stratified reliability diagnosis, noise robustness, and sensor sparsity stress testing. The objective is not only to compare average prediction accuracy, but also to examine whether the proposed descriptor set remains reliable under data partitioning uncertainty, measurement noise, sensor sparsity, and different damage severity regimes.

---

## 4.1 Main ablation results on the 3000-case dataset

The main ablation experiment was conducted on the `debug_plus_3000` controlled simulation dataset. The dataset contained 3000 simulated structural response cases, with each response tensor consisting of 2000 time steps and four monitored structural response channels. The full physics-informed descriptor set contained 92 features, including response-level statistics, frequency-domain indicators, spatial response descriptors, response correlation descriptors, and excitation-related metadata.

Table 2 summarizes the main feature-ablation and estimator-comparison results. The full physics-informed descriptor set combined with ridge regression achieved the best fixed-split performance, with a test MAE of 0.0393 and a test RMSE of 0.0585. The same configuration also achieved a damaged-entry MAE of 0.0634. Removing metadata-related descriptors increased the test MAE to 0.0425, indicating that excitation-related descriptors provided useful but not dominant additional information. In contrast, the response-basic-only feature set produced a substantially larger test MAE of 0.0577, suggesting that simple response statistics alone were insufficient for reliable damage inference.

The estimator comparison also showed that increasing model complexity did not automatically improve performance. Under the full feature set, ridge regression outperformed ElasticNet and random forest in the full four-sensor setting. The full-feature ElasticNet model achieved a test MAE of 0.0476, while the full-feature random forest model produced a larger test MAE of 0.0644. This result indicates that, under the present controlled sparse and noisy SHM simulation setting, structured physics-informed descriptor construction combined with regularized linear regression was more effective than directly using a more complex nonlinear estimator.

These results support two observations. First, the predictive performance was primarily driven by the quality and completeness of the descriptor set rather than by estimator complexity alone. Second, metadata-related descriptors improved the full descriptor set, but the main predictive information was still carried by response-derived physics-informed descriptors.

**Relevant sources**

- Table 2: `results/paper_ready/q2_fast_track/tables/T02_main_ablation_3000.csv`
- Figure 1: `results/paper_ready/q2_fast_track/figures/F01_main_ablation_top10_test_mae.png`
- Figure 1 alternative: `results/paper_ready/q2_fast_track/figures/F01_main_ablation_top10_test_rmse.png`

---

## 4.2 Repeated-split robustness

The fixed-split ablation result may be affected by a favorable or unfavorable train/validation/test partition. To evaluate whether the main conclusion was split-specific, repeated-split robustness analysis was performed using 10 random seeds. For each seed, the 3000-case dataset was repartitioned into 2100 training cases, 450 validation cases, and 450 test cases.

The results in Table 3 show that the full physics-informed descriptor set with ridge regression remained the best-performing configuration across all 10 repeated splits. It achieved a mean test MAE of 0.0454 with a standard deviation of 0.0009, corresponding to a coefficient of variation of approximately 1.90%. The mean test RMSE was 0.0621, and the mean prediction bias was close to zero. The per-seed best-configuration table further confirms that the full-feature ridge model was selected as the best configuration in every repeated split.

The repeated-split result is important because the fixed-split test MAE of 0.0393 was lower than the repeated-split mean test MAE of 0.0454. This difference suggests that the fixed split may have been relatively favorable. Therefore, the repeated-split result provides a more conservative and robust estimate of generalization performance. Nevertheless, the same configuration remained consistently optimal, confirming that the main conclusion was stable rather than caused by a single data partition.

The comparison with the no-metadata variant also remained consistent. The no-metadata ridge model achieved a mean test MAE of 0.0490, which was higher than that of the full-feature ridge model. This again indicates that metadata-related descriptors contributed incremental performance benefits, while the full response-derived and excitation-related descriptor combination provided the strongest overall representation.

**Relevant sources**

- Table 3: `results/paper_ready/q2_fast_track/tables/T03_repeated_split_robustness_summary.csv`
- Table 3 supplementary: `results/paper_ready/q2_fast_track/tables/T04_repeated_split_per_seed_best.csv`
- Figure 2: `results/paper_ready/q2_fast_track/figures/F02_repeated_split_seed_robustness_test_mae_boxplot.png`
- Figure 2 alternative: `results/paper_ready/q2_fast_track/figures/F02_repeated_split_seed_robustness_test_mae_mean_std.png`
- Figure 2 supplementary: `results/paper_ready/q2_fast_track/figures/F02_repeated_split_seed_robustness_bias_vs_mae.png`

---

## 4.3 Damage-stratified reliability diagnosis

Although the full-feature ridge model achieved the best average performance, average error metrics alone do not fully characterize reliability in structural health monitoring applications. In safety-critical settings, errors in severe damage regimes are more consequential than average-case prediction errors. Therefore, damage-stratified reliability diagnosis was performed to evaluate prediction performance across zero-, low-, medium-, and high-damage regimes.

Table 4 shows that the full-feature ridge model achieved the best overall MAE of 0.0407 and an overall RMSE of 0.0600. However, its high-damage MAE increased to 0.0888, which was more than twice the overall MAE. In the high-damage bin, the mean true damage was 0.2758, while the mean predicted damage was only 0.2041. This resulted in a high-damage bias of -0.0717 and an underestimation ratio of 0.8214. These values indicate that the model systematically underestimated severe damage, despite achieving strong average accuracy.

The damage-bin metrics reveal a clear severity-dependent reliability pattern. For the full-feature ridge model, the MAE was 0.0297 in zero-damage entries and 0.0289 in low-damage entries. It increased to 0.0502 in medium-damage entries and further increased to 0.0888 in high-damage entries. Therefore, the model was not failing uniformly across all samples. Instead, the reliability degradation was concentrated in the more severe damage regimes.

The comparison among feature sets also supports the value of the full descriptor set. The no-metadata ridge model produced a high-damage MAE of 0.0948, while the response-basic-only ridge model produced a much larger high-damage MAE of 0.1434. The full-feature ElasticNet model showed even stronger high-damage underestimation, with a high-damage MAE of 0.1787 and an underestimation ratio of 0.9967. These results suggest that reduced feature sets and stronger sparsity-inducing regularization may intensify regression-to-the-mean behavior in severe damage regimes.

Overall, damage-stratified reliability diagnosis demonstrates that stable average prediction accuracy does not imply uniform safety-critical reliability. For SHM-based structural state inference, reporting only overall MAE or RMSE can hide systematic underestimation in high-damage cases. Therefore, damage-stratified evaluation should be considered necessary when assessing data-driven or descriptor-based damage inference models.

**Relevant sources**

- Table 4: `results/paper_ready/q2_fast_track/tables/T05_damage_stratified_config_summary.csv`
- Table 4 supplementary: `results/paper_ready/q2_fast_track/tables/T06_damage_stratified_runs.csv`
- Figure 3: `results/paper_ready/q2_fast_track/figures/F03_damage_stratified_damage_stratified_mae_by_bin.png`
- Figure 3 alternative: `results/paper_ready/q2_fast_track/figures/F03_damage_stratified_damage_stratified_bias_by_bin.png`
- Figure 3 supplementary: `results/paper_ready/q2_fast_track/figures/F03_damage_stratified_damage_stratified_underestimation_ratio.png`
- Figure 3 supplementary: `results/paper_ready/q2_fast_track/figures/F03_damage_stratified_best_config_true_vs_pred_by_damage_bin.png`

---

## 4.4 Noise robustness

Noise robustness was evaluated using six independently generated 1000-case controlled simulation datasets. The fixed noise levels were 0%, 2%, 5%, 10%, 15%, and 20%. For each noise level, the same feature extraction and model comparison procedure was applied. Because each noise level corresponded to an independently generated dataset, the performance trend was not expected to be strictly monotonic.

The results in Table 5 show that the full physics-informed descriptor set with ridge regression remained the best-performing configuration across all tested noise levels. At 0% noise, the full-feature ridge model achieved a test MAE of 0.0345 and a test RMSE of 0.0513. At 2% and 5% noise, the test MAE increased to 0.0361 and 0.0375, respectively. At 10% noise, the test MAE further increased to 0.0394. At 15% and 20% noise, the test MAE was 0.0431 and 0.0420, respectively. Although the 20% noise result was slightly lower than the 15% noise result, the overall trend still indicates that prediction error increased as the monitoring data became noisier.

The damaged-entry MAE followed a similar overall pattern. For the full-feature ridge model, the damaged-entry MAE was 0.0526 at 0% noise, 0.0599 at 5% noise, 0.0634 at 10% noise, and approximately 0.068–0.069 at 15%–20% noise. This indicates that damaged entries were more sensitive to noise than the overall average performance, which is consistent with the damage-stratified reliability findings.

The full descriptor set also consistently outperformed the no-metadata and response-basic-only variants. Across all noise levels, the no-metadata ridge model produced higher MAE than the full-feature ridge model, while the response-basic-only ridge model showed substantially larger errors. This suggests that combining multiple physics-informed descriptor groups improved robustness under noisy monitoring conditions.

The noise robustness experiment therefore supports the use of the proposed descriptor set under noisy SHM data. However, the results should be interpreted as robustness under controlled simulated noise rather than direct evidence of field deployment performance. Real monitoring noise may include sensor drift, environmental effects, missing data, nonstationary operational variability, and other uncertainties not fully represented by the current controlled simulation design.

**Relevant sources**

- Table 5: `results/paper_ready/q2_fast_track/tables/T07_noise_best_by_noise.csv`
- Table 5 supplementary: `results/paper_ready/q2_fast_track/tables/T08_noise_robustness_summary.csv`
- Figure 4: `results/paper_ready/q2_fast_track/figures/F04_noise_noise_vs_test_mae.png`
- Figure 4 alternative: `results/paper_ready/q2_fast_track/figures/F04_noise_noise_vs_test_rmse.png`
- Figure 4 supplementary: `results/paper_ready/q2_fast_track/figures/F04_noise_noise_vs_damaged_mae.png`

---

## 4.5 Sensor sparsity stress test

Sensor sparsity was evaluated by zero-masking unavailable response channels while preserving the original four-story damage labels and the original response tensor shape. This design allowed the downstream feature extraction and evaluation workflow to remain compatible across different sensor layouts. The tested layouts included a four-sensor setting using sensors 1-2-3-4, a three-sensor setting using sensors 1-2-4, a two-sensor setting using sensors 1-4, and a one-sensor setting using sensor 4 only.

The results in Table 6 show that prediction performance degraded substantially as the number of available sensors decreased. In the full four-sensor setting, the best configuration was full + ridge, with a test MAE of 0.0393 and a test RMSE of 0.0585. In the three-sensor setting, full + ridge remained the best configuration, but the test MAE increased to 0.0612 and the test RMSE increased to 0.0836. This indicates that even moderate sensor reduction caused a clear loss of information, although the full-feature ridge model remained the strongest configuration.

Under more extreme sensor sparsity, the best estimator shifted from ridge regression to random forest. In the two-sensor setting, the best configuration was full + random forest, with a test MAE of 0.0764 and a test RMSE of 0.0977. In the one-sensor setting, full + random forest again achieved the lowest test MAE, with a value of 0.0801. This estimator shift suggests that when spatial sensing coverage becomes severely limited, the linear relationship captured by ridge regression becomes less sufficient, and nonlinear estimators may capture residual patterns more effectively.

Nevertheless, the full descriptor set remained the strongest descriptor set across all sensor configurations. The best-performing configuration in each sensor setting used the full feature set. This indicates that the proposed physics-informed descriptor construction remained valuable even when sensor availability was reduced. However, the performance gap between full descriptors and reduced descriptors diminished in the one-sensor setting. For example, in the one-sensor layout, the full-feature ridge model achieved a test MAE of 0.0845, while the response-basic-only ridge model achieved a similar test MAE of 0.0848. This suggests that several physics-informed descriptor groups rely on sufficient spatial sensing coverage and become less informative under extreme sensor sparsity.

The damaged-entry MAE also increased as sensors were removed. For the best configuration in each sensor setting, the damaged-entry MAE increased from 0.0634 in the four-sensor setting to 0.1006 in the three-sensor setting, 0.1270 in the two-sensor setting, and 0.1323 in the one-sensor setting. This confirms that damaged entries are particularly sensitive to sparse sensing conditions.

Overall, the sensor sparsity stress test supports the sparse-monitoring interpretation of the study. The proposed descriptor set remained effective under moderate sensor sparsity, but severe reduction to one or two sensors substantially degraded prediction reliability. Therefore, the method should not be interpreted as being insensitive to sensor availability. Instead, the results identify a practical boundary: physics-informed descriptor construction can improve structural state inference under sparse monitoring, but sufficient spatial sensing coverage remains important for reliable damage estimation.

**Relevant sources**

- Table 6: `results/paper_ready/q2_fast_track/tables/T09_sensor_best_by_sensor_count.csv`
- Table 6 supplementary: `results/paper_ready/q2_fast_track/tables/T10_sensor_sparsity_summary.csv`
- Figure 5: `results/paper_ready/q2_fast_track/figures/F05_sensor_sparsity_sensor_count_vs_test_mae.png`
- Figure 5 alternative: `results/paper_ready/q2_fast_track/figures/F05_sensor_sparsity_sensor_count_vs_test_rmse.png`
- Figure 5 supplementary: `results/paper_ready/q2_fast_track/figures/F05_sensor_sparsity_sensor_count_vs_damaged_mae.png`

---

## 4.6 Summary of results

The experimental results provide a coherent evaluation of the proposed physics-informed descriptor construction framework. The main ablation experiment showed that the full descriptor set combined with ridge regression achieved the strongest fixed-split performance on the 3000-case dataset. Repeated-split robustness confirmed that this conclusion was stable across random partitions. Noise robustness experiments showed that the full-feature ridge model remained the best-performing configuration across 0%–20% simulated noise levels. Sensor sparsity stress testing further showed that the full descriptor set remained the strongest descriptor set across all sensor configurations, although severe sensor reduction substantially degraded reliability and changed the best estimator under extreme sparsity.

At the same time, the damage-stratified analysis revealed a critical limitation. Strong average performance did not eliminate high-damage underestimation. The full-feature ridge model systematically underestimated severe damage entries, indicating that average MAE and RMSE are insufficient for safety-critical SHM evaluation. Therefore, the proposed framework is best interpreted as a robust descriptor construction approach for controlled sparse and noisy SHM data, with explicitly identified reliability boundaries in high-damage and severely sparse sensing regimes.

