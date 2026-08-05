# Abstract, Keywords, and Conclusion

---

# Abstract

Structural health monitoring (SHM) data are commonly sparse, noisy, and limited in labeled damage observations, which makes reliable structural state inference challenging. This study develops a physics-informed descriptor construction framework for structural damage inference under sparse and noisy SHM-like response data. A controlled OpenSeesPy-based simulation workflow is used to generate structural response histories with four-story damage labels. A 92-dimensional descriptor set is constructed from response statistics, frequency-domain indicators, spatial response patterns, response correlations, and excitation-related metadata. The framework is evaluated through main feature ablation, repeated-split robustness analysis, damage-stratified reliability diagnosis, noise robustness analysis, and sensor sparsity stress testing.

On the 3000-case controlled simulation dataset, the full physics-informed descriptor set combined with ridge regression achieves the best fixed-split performance, with a test MAE of 0.0393 and a test RMSE of 0.0585. Repeated-split analysis over 10 random partitions confirms that this result is stable rather than split-specific, with a mean test MAE of 0.0454 ± 0.0009. Noise robustness experiments show that the full-feature ridge model remains the best-performing configuration across fixed noise levels from 0% to 20%, with no performance collapse. Sensor sparsity stress testing shows that the full descriptor set remains the strongest descriptor set across all sensor layouts, although severe reduction to one or two sensors substantially degrades prediction reliability.

However, damage-stratified evaluation reveals that strong average accuracy does not imply uniform safety-critical reliability. The high-damage MAE of the best overall model increases to 0.0888, with a high-damage bias of -0.0717 and an underestimation ratio of 0.8214. These results indicate that severe damage entries are systematically underestimated despite strong average performance. The study demonstrates that physics-informed descriptor construction can improve robust structural damage inference under controlled noisy and moderately sparse monitoring conditions, while also showing that damage-stratified reliability diagnosis is necessary for safety-critical SHM evaluation.

---

# Keywords

Structural health monitoring; Structural damage inference; Physics-informed descriptors; Sparse sensing; Noisy monitoring data; Damage-stratified reliability; OpenSeesPy simulation

---

# 6. Conclusion

This study developed and evaluated a physics-informed descriptor construction framework for structural damage inference under sparse and noisy SHM-like data. The framework transforms structural response histories into a structured descriptor set that integrates response statistics, frequency-domain indicators, spatial response patterns, response correlations, and excitation-related metadata. A controlled OpenSeesPy-based simulation workflow was used to generate structural response data with four-story damage labels, and the proposed descriptor set was evaluated through main ablation, repeated-split robustness, damage-stratified reliability diagnosis, noise robustness analysis, and sensor sparsity stress testing.

The main ablation experiment on the 3000-case controlled simulation dataset showed that the full physics-informed descriptor set combined with ridge regression achieved the best fixed-split performance, with a test MAE of 0.0393 and a test RMSE of 0.0585. Reduced feature sets, including no-metadata and response-basic-only variants, produced larger errors, indicating that the full descriptor set provided the most informative representation. The comparison among estimators further showed that increasing model complexity did not automatically improve performance under the full four-sensor setting. Instead, regularized linear regression on a physically structured descriptor space provided the most stable result.

Repeated-split robustness analysis confirmed that the main finding was not caused by a favorable data partition. Across 10 random train/validation/test splits, the full-feature ridge model remained the best-performing configuration, achieving a mean test MAE of 0.0454 with a standard deviation of 0.0009. Noise robustness experiments further showed that the same configuration remained the best-performing model across fixed noise levels from 0% to 20%. Although the error increased overall with noise level, no performance collapse was observed under the tested controlled noise conditions.

The sensor sparsity stress test showed that the full descriptor set remained the strongest descriptor set across all tested sensor layouts. In the four-sensor and three-sensor settings, the full-feature ridge model achieved the best performance. Under more extreme two-sensor and one-sensor conditions, the best estimator shifted to random forest, with best test MAEs of 0.0764 and 0.0801, respectively. This indicates that physics-informed descriptor construction remains useful under moderate sensor sparsity, but severe reduction in spatial sensing coverage creates a practical reliability boundary.

The damage-stratified reliability diagnosis revealed the most important safety-related limitation. Although the full-feature ridge model achieved the best overall accuracy, its high-damage MAE increased to 0.0888, and its high-damage bias was -0.0717. The underestimation ratio in the high-damage regime reached 0.8214, indicating systematic underestimation of severe damage. This result shows that average MAE and RMSE alone are insufficient for evaluating SHM-based structural state inference models. Safety-critical evaluation should include damage-stratified metrics, prediction bias, and underestimation behavior.

Overall, the results suggest that physics-informed descriptor construction is a useful strategy for robust structural damage inference under controlled sparse and noisy SHM-like data. The proposed descriptor set improves stable average performance, remains robust under tested noise levels, and retains value under moderate sensor sparsity. At the same time, the study identifies clear reliability boundaries, especially under severe damage and extreme sensor sparsity. Future work should validate the framework using field SHM data, more complex structural systems, realistic environmental and operational variability, optimized sensor placement, missing-data reconstruction, damage-aware training, and uncertainty quantification.

