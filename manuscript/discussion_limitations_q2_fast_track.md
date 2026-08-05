# 5. Discussion and Limitations

This section discusses the main findings of the proposed physics-informed descriptor construction framework. The discussion focuses on five aspects: the role of descriptor construction, the limitation of average accuracy metrics, high-damage underestimation, robustness under noisy monitoring, sensor sparsity boundaries, and methodological limitations.

---

## 5.1 Physics-informed descriptor construction is more important than estimator complexity

The main ablation, repeated-split robustness, and noise robustness experiments consistently show that the full physics-informed descriptor set provides the strongest representation for structural damage inference. In the full four-sensor setting, the full descriptor set combined with ridge regression achieved the best fixed-split performance, and the same configuration remained optimal across all 10 repeated splits. It also remained the best-performing configuration across the tested noise levels from 0% to 20%.

This finding suggests that, under the present controlled sparse and noisy SHM simulation setting, the construction of physically meaningful descriptors can be more important than increasing estimator complexity. Ridge regression is a relatively simple regularized linear estimator, yet it outperformed ElasticNet and random forest in the main four-sensor and noise robustness experiments. This indicates that the full descriptor set made the regression problem sufficiently structured for a stable regularized linear model to perform well.

This result should not be interpreted as evidence that linear models are generally superior to nonlinear models in SHM. Rather, it shows that when descriptor construction captures relevant response magnitude, frequency-domain behavior, spatial response patterns, response correlations, and excitation-related information, a simple estimator may provide strong and stable performance. In contrast, increasing model flexibility without improving the physical informativeness of the input representation may not improve reliability.

The sensor sparsity stress test further refines this interpretation. Under four-sensor and three-sensor settings, ridge regression remained the best estimator. However, under more extreme two-sensor and one-sensor settings, the best estimator shifted to random forest. This suggests that estimator choice may depend on sensing coverage. When sufficient spatial response information is available, the descriptor space appears structured enough for ridge regression. When spatial information becomes severely limited, nonlinear estimators may capture residual patterns that regularized linear models cannot fully represent.

Therefore, the main methodological implication is not that one estimator is universally optimal. Instead, the results indicate that physics-informed descriptor construction is the primary source of robust performance, while estimator selection should be considered in relation to sensing conditions.

---

## 5.2 Average accuracy is insufficient for safety-critical SHM evaluation

The damage-stratified reliability results demonstrate that average error metrics can hide safety-critical weaknesses. The full-feature ridge model achieved the best overall MAE, but its high-damage MAE was much larger than its overall MAE. More importantly, the high-damage bias was negative, and the high-damage underestimation ratio was high. This means that the model did not merely produce random large errors in severe damage cases; it systematically underestimated high damage.

This finding is important for SHM applications. In ordinary regression tasks, low average MAE or RMSE may be considered sufficient evidence of good predictive performance. In structural condition assessment, however, the consequence of prediction error is asymmetric. Underestimating severe damage is more dangerous than overestimating minor damage because it may delay inspection, maintenance, strengthening, or emergency intervention.

The results therefore support a reliability-oriented evaluation perspective. A model should not be evaluated only by average accuracy over all entries. It should also be evaluated by damage regime, especially in high-damage regions. Metrics such as high-damage MAE, high-damage bias, and underestimation ratio provide additional safety-relevant information that is not captured by overall MAE or RMSE.

The results also show that the model's weakness is not uniform across all damage regimes. The full-feature ridge model performed well in zero- and low-damage entries, but error increased substantially in medium- and high-damage entries. This pattern suggests that the average-case mapping learned from the descriptor space is more reliable for mild structural states than for severe damage states. For safety-critical SHM tasks, this distinction is essential.

---

## 5.3 High-damage underestimation indicates a methodological boundary

High-damage underestimation is one of the most important limitations identified by this study. Even the best overall configuration underestimated high-damage entries. This behavior may be related to regression-to-the-mean effects, data distribution imbalance, limited high-damage representation, and the use of average-error-based model selection.

When a model is optimized mainly for average error, it may favor predictions close to common or moderate damage levels. If severe damage cases are less frequent or have more diverse response patterns, the model may reduce global error by under-predicting high damage. This behavior can produce acceptable overall MAE while still being unreliable in safety-critical regimes.

The comparison among estimators supports this interpretation. ElasticNet showed particularly strong high-damage underestimation, suggesting that stronger sparsity-inducing regularization may suppress damage-sensitive descriptors. Response-basic-only features also produced larger high-damage errors, indicating that simple response statistics are insufficient for severe damage inference.

This limitation suggests several future methodological directions. First, damage-aware training objectives could assign greater weight to high-damage entries. Second, stratified or balanced sampling could increase the influence of severe damage cases during model training. Third, specialized high-damage calibration models could be developed to correct systematic underestimation. Fourth, uncertainty quantification could be incorporated so that high-risk underestimation is explicitly flagged instead of hidden behind point predictions.

In the current study, high-damage underestimation is not treated as a solved problem. Instead, it is reported as an identified reliability boundary of the proposed descriptor-based framework.

---

## 5.4 Metadata provides consistent but secondary information

The full descriptor set consistently outperformed the no-metadata variant in the main ablation, repeated-split robustness, and noise robustness experiments. This indicates that excitation-related metadata provides useful auxiliary information. Including excitation information can help distinguish response changes caused by damage from response changes caused by different input conditions.

However, the performance gap between the full and no-metadata variants was moderate rather than dominant. This suggests that the main predictive signal was carried by response-derived descriptors, while metadata contributed incremental improvement. This distinction is important because it prevents overinterpreting metadata as the central driver of the model.

From a practical SHM perspective, this result is reasonable. Structural responses are influenced by both structural state and input excitation. If excitation-related information is available, it should be used because it can improve robustness. However, a damage inference framework should not depend entirely on metadata, since input information may be incomplete or uncertain in field monitoring applications.

Therefore, the results support a balanced interpretation: excitation-related descriptors are useful, but the core contribution remains the construction of response-derived physics-informed descriptors.

---

## 5.5 Noise robustness supports controlled noisy-monitoring applicability

The noise robustness experiment shows that the full-feature ridge model remained the best-performing configuration across all tested noise levels. Its error increased overall as the fixed noise level increased, but no performance collapse was observed up to 20% simulated noise.

This result supports the applicability of the descriptor set under controlled noisy monitoring conditions. The combination of response statistics, frequency descriptors, spatial descriptors, correlation descriptors, and metadata appears to provide a representation that is not overly sensitive to moderate measurement noise.

However, this interpretation should remain bounded. The noise robustness datasets were independently generated for each fixed noise level, so the error trend was not strictly monotonic. In addition, the simulated noise used in this study is a controlled representation of measurement uncertainty. Real SHM noise may include sensor drift, environmental effects, operational variability, missing data, temperature effects, nonstationary boundary conditions, and instrumentation faults.

Therefore, the noise robustness result should be interpreted as evidence of robustness under controlled simulated noise, not as a claim of field-ready performance under all real monitoring uncertainties.

---

## 5.6 Sensor sparsity reveals the importance of spatial sensing coverage

The sensor sparsity stress test shows that reducing the number of available sensors substantially degrades prediction performance. The full descriptor set remained the strongest descriptor set across all sensor configurations, but the error increased as sensors were removed.

This result has two implications. First, physics-informed descriptor construction can still be useful under moderate sensor sparsity. In the three-sensor setting, the full-feature ridge model remained the best configuration, although its error increased compared with the four-sensor baseline. This suggests that the descriptor set retains useful information when some spatial coverage remains.

Second, severe sensor reduction creates a practical boundary. In the two-sensor and one-sensor settings, prediction errors increased substantially, and the best estimator shifted to random forest. In the one-sensor setting, the performance gap between full descriptors and response-basic-only descriptors became very small. This indicates that several physics-informed descriptor groups rely on sufficient spatial sensing coverage. When spatial information is nearly absent, correlation and spatial descriptors lose much of their value.

This result should be interpreted carefully. The sensor sparsity experiment used zero-masking to simulate unavailable channels. It was designed as a stress test, not as an optimal missing-data imputation method or sensor placement strategy. In practice, better sparse-sensing performance may be obtained through optimized sensor placement, missing-data reconstruction, graph-based spatial modeling, or models explicitly trained for missing sensor patterns.

Nevertheless, the current result is useful because it identifies the practical sensing boundary of the proposed descriptor framework. The method is not insensitive to sensor availability. Reliable damage inference still requires sufficient spatial monitoring coverage.

---

# 6. Limitations

## 6.1 Controlled simulation rather than field SHM validation

The dataset used in this study was generated from a controlled OpenSeesPy-based simulation workflow. This design allows systematic evaluation of damage levels, noise conditions, sensor sparsity, and feature ablation. However, it does not reproduce all complexities of real field SHM data.

Field monitoring data may include environmental variability, aging effects, nonstationary boundary conditions, unknown excitation, sensor drift, missing data, and modeling uncertainty. Therefore, the results should be interpreted as controlled simulation-based evidence rather than direct proof of field deployment readiness.

Future work should validate the proposed descriptor construction framework using field monitoring datasets, laboratory benchmark structures, or hybrid simulation-experimental datasets.

---

## 6.2 Simplified structural system and damage representation

The present study uses a simplified four-story structural representation and a four-dimensional story-level damage vector. This controlled setting is useful for developing and evaluating the proposed framework, but it cannot cover the full diversity of civil infrastructure systems.

Real structures may have complex geometry, material nonlinearities, soil-structure interaction, connection damage, local component failures, nonstructural interactions, and uncertain boundary conditions. The story-level damage vector used in this study is therefore a simplified representation of structural damage.

Future research should test the framework on more complex structural systems, including bridges, spatial frames, irregular buildings, and structures with localized component-level damage.

---

## 6.3 Zero-masking is not an optimal sparse-sensing strategy

The sensor sparsity experiment uses zero-masking to represent unavailable response channels. This approach preserves tensor shape and keeps the feature extraction workflow compatible across sensor layouts, but it is not an optimal missing-data strategy.

Zero-masking can introduce artificial patterns that may not fully represent realistic missing sensor conditions. It also does not account for optimized sensor placement or reconstruction from partial observations. Therefore, the sensor sparsity results should be interpreted as a stress test rather than a final solution to sparse SHM sensing.

Future work should compare zero-masking with missing-data imputation, graph-based reconstruction, sensor placement optimization, and models trained with random sensor dropout.

---

## 6.4 Noise model is controlled and simplified

The noise robustness experiment uses controlled fixed noise levels. This allows systematic comparison across noise intensities, but real SHM noise is more complex. Field noise may be non-Gaussian, nonstationary, temperature-dependent, sensor-specific, or correlated with operational conditions.

Moreover, each noise level in this study corresponds to an independently generated dataset. This design supports controlled robustness testing, but the resulting error curve does not need to be strictly monotonic.

Future work should evaluate the framework under more realistic noise mechanisms, including sensor drift, environmental variability, operational variability, and missing or corrupted measurements.

---

## 6.5 High-damage underestimation remains unresolved

The proposed descriptor set improves average accuracy and robustness, but high-damage underestimation remains unresolved. This is the most important safety-related limitation of the current study.

The current framework reports high-damage underestimation rather than hiding it. Future work should directly address this issue through damage-aware loss functions, weighted training, high-damage oversampling, synthetic severe-damage augmentation, calibration models, and uncertainty quantification.

For safety-critical SHM applications, a model should not only predict accurately on average but also avoid systematic underestimation of severe damage.

---

## 6.6 Limited estimator family

This study focuses on ridge regression, ElasticNet, and random forest regression. These estimators were selected to evaluate whether structured physics-informed descriptors can support stable inference without relying on complex end-to-end neural networks.

However, other model families may provide additional benefits, including Gaussian process regression, gradient boosting, graph neural networks, temporal convolutional networks, transformers, neural operators, and physics-guided neural networks.

Future work may compare the proposed descriptor set with deep learning models and hybrid physics-guided architectures. However, such comparisons should be designed carefully to avoid conflating the effects of descriptor construction, model capacity, data size, and hyperparameter tuning.

---

## 6.7 Dataset size and distribution

Although the main dataset contains 3000 simulated cases, this remains limited compared with the diversity of real structural states, excitation conditions, and damage mechanisms. In addition, the damage distribution is controlled by the simulation design and should not be interpreted as the natural distribution of real structural damage.

Future work should evaluate the framework under broader distributions of damage location, damage intensity, excitation type, noise mechanism, and structural configuration.

---

# 7. Future work

Based on the identified limitations, future work should focus on the following directions:

1. Field validation using real SHM datasets or laboratory benchmark structures.
2. Damage-aware learning objectives to reduce high-damage underestimation.
3. Weighted or stratified training strategies for severe damage regimes.
4. Uncertainty quantification for safety-critical decision support.
5. Sensor placement optimization for sparse monitoring conditions.
6. Missing-data reconstruction and graph-based spatial inference.
7. Extension to more complex structural systems and component-level damage.
8. Comparison with deep learning and physics-guided neural architectures.
9. Environmental and operational variability modeling.
10. Development of reliability-oriented SHM evaluation protocols.

---

# 8. Discussion summary

The results demonstrate that physics-informed descriptor construction can provide a robust and interpretable representation for structural damage inference under controlled sparse and noisy SHM-like data. The full descriptor set consistently outperformed reduced feature sets, and ridge regression provided strong performance when spatial sensing coverage was sufficient. However, the results also show clear reliability boundaries. High-damage entries were systematically underestimated, and severe sensor sparsity substantially degraded prediction performance.

Therefore, the proposed framework should be interpreted as a useful descriptor construction and evaluation strategy, not as a complete field-ready SHM solution. Its main value lies in showing that physically motivated descriptors can improve stable damage inference while also revealing why average accuracy alone is insufficient for safety-critical structural health monitoring.

