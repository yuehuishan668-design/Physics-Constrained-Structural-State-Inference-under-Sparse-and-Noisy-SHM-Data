# 2. Methodology

This section describes the controlled simulation workflow, damage representation, physics-informed descriptor construction, estimator settings, evaluation metrics, and robustness-test design used in this study. The proposed framework is designed for structural damage inference under sparse and noisy SHM-like response data. The focus is on descriptor construction and reliability-oriented evaluation rather than on developing a new end-to-end deep learning architecture.

---

## 2.1 Overview of the proposed framework

The proposed workflow consists of six main stages:

1. Controlled structural response simulation using OpenSeesPy.
2. Story-level damage label generation.
3. Train/validation/test splitting and response normalization.
4. Physics-informed descriptor extraction from structural responses and excitation-related information.
5. Estimator training and feature-ablation comparison.
6. Robustness and reliability evaluation under repeated splits, damage stratification, noise variation, and sensor sparsity.

The overall objective is to examine whether physics-informed descriptor construction can provide a stable and interpretable representation for structural damage inference when monitoring data are noisy, sparse, and limited in labeled damage observations.

Unlike purely end-to-end data-driven approaches, the proposed framework first converts raw response histories into structured descriptors. These descriptors are designed to encode response magnitude, frequency characteristics, spatial response distribution, inter-story response relationships, and excitation-related information. Classical estimators are then trained on the descriptor space. This design allows the study to separate the contribution of feature construction from the contribution of estimator complexity.

---

## 2.2 Controlled structural response simulation

A controlled simulation dataset was generated using an OpenSeesPy-based structural model. The simulated structure contains four story-level response channels, and the damage target is represented as a four-dimensional story-level damage vector. Each simulated case contains a response history with 2000 time steps and four response channels. The time step is 0.01 s.

For the main dataset, 3000 simulated cases were generated. The response tensor therefore has the form:

\[
X \in \mathbb{R}^{3000 \times 2000 \times 4}
\]

where the first dimension corresponds to simulated cases, the second dimension corresponds to time steps, and the third dimension corresponds to monitored story-level response channels.

The corresponding damage target is represented as:

\[
y \in \mathbb{R}^{3000 \times 4}
\]

where each component represents the damage level associated with one story. The generated dataset includes healthy cases, single-story damaged cases, and multi-story damaged cases. The dataset is intended as a controlled benchmark for method development and should not be interpreted as direct field SHM measurements from a specific real bridge or building.

The main generated dataset is denoted as `debug_plus_3000`. Its train/validation/test split contains 2100, 450, and 450 cases, respectively. The response inputs are normalized using training-set statistics only, and the same normalization parameters are applied to validation and test sets.

---

## 2.3 Damage representation and damage regimes

The damage target is defined as a four-dimensional vector:

\[
y_i = [d_{i,1}, d_{i,2}, d_{i,3}, d_{i,4}]
\]

where \(d_{i,j}\) denotes the damage level of the \(j\)-th story in the \(i\)-th simulated case. A zero value represents no damage in the corresponding story, while larger values represent higher damage intensity.

For reliability-oriented evaluation, damage entries are divided into four regimes:

- zero damage;
- low damage;
- medium damage;
- high damage.

The exact bin boundaries follow the implementation used in the evaluation scripts. The purpose of the binning is not to redefine the regression target, but to evaluate whether prediction reliability changes across damage severities. This is important because overall error metrics can hide systematic errors in safety-critical high-damage entries.

In addition to overall MAE and RMSE, the study reports damaged-entry MAE, high-damage MAE, prediction bias, and underestimation ratio. These metrics are used to determine whether the model tends to underestimate severe structural damage.

---

## 2.4 Physics-informed descriptor construction

The raw structural response history is transformed into a structured descriptor vector. The full descriptor set contains 92 features. These features are grouped into several physically motivated categories.

### 2.4.1 Response basic statistics

Basic response descriptors summarize the magnitude and distribution of response histories. They include statistical indicators such as response amplitude, dispersion, and other time-domain summary quantities. These features provide a baseline representation of structural response intensity.

However, basic response statistics alone may not be sufficient for damage inference because they do not fully capture frequency shifts, spatial response redistribution, or inter-story relationships. Therefore, the response-basic-only feature set is treated as a reduced baseline in the ablation study.

### 2.4.2 Frequency-domain descriptors

Frequency-domain descriptors are extracted to capture changes in dynamic characteristics. Structural damage can alter stiffness and therefore affect modal and frequency-related response behavior. Frequency-domain descriptors are included to represent these dynamic changes in a compact feature form.

These descriptors are not intended to replace full modal identification. Instead, they provide practical frequency-sensitive indicators that can be used in descriptor-based damage inference.

### 2.4.3 Spatial response descriptors

Spatial descriptors characterize how response quantities are distributed across the four monitored story-level channels. Damage may change the relative response pattern among stories, so spatial response information can provide useful clues for localizing and quantifying damage.

These descriptors are particularly relevant when multiple sensors are available. Under severe sensor sparsity, the usefulness of spatial descriptors may diminish because the spatial response pattern becomes under-observed.

### 2.4.4 Response correlation descriptors

Correlation descriptors summarize relationships among response channels. Inter-story correlation patterns may change when structural damage modifies load transfer, stiffness distribution, or dynamic coupling. These descriptors are designed to capture cross-channel response relationships beyond independent channel-wise statistics.

### 2.4.5 Excitation-related metadata

The full descriptor set also includes excitation-related metadata. These descriptors provide information about input motion conditions, such as excitation amplitude or frequency-related parameters. Including excitation-related information can help distinguish whether response changes are caused by structural damage or by different input conditions.

To evaluate the contribution of metadata, a no-metadata feature set is also tested. Comparing the full feature set against the no-metadata variant allows the study to separate response-derived information from excitation-related auxiliary information.

### 2.4.6 Feature-set variants

The following feature-set variants are used in the ablation study:

1. `full`: all physics-informed response-derived and excitation-related descriptors.
2. `no_meta`: full descriptor set excluding metadata-related descriptors.
3. `physics_no_meta_core`: equivalent or near-equivalent response-derived physics descriptor subset without metadata.
4. `response_basic_only`: basic response statistics only.
5. `response_frequency`: frequency-related descriptor subset.
6. `response_spatial`: spatial response descriptor subset.
7. `response_correlation`: response correlation descriptor subset.

These variants are used to evaluate whether the full descriptor set provides additional predictive value beyond individual or reduced descriptor groups.

---

## 2.5 Estimators

This study uses classical estimators rather than end-to-end neural networks. The purpose is to evaluate whether physically structured descriptors can support stable damage inference without relying on highly complex model architectures.

The main estimators are:

1. Ridge regression.
2. ElasticNet regression.
3. Random forest regression.

### 2.5.1 Ridge regression

Ridge regression is used as the primary estimator because it provides regularized linear regression in the descriptor space. It is suitable when the descriptor set is informative and when stability is preferred over highly flexible nonlinear fitting.

Ridge regression can be interpreted as testing whether the constructed descriptors make the regression problem sufficiently structured for a simple regularized estimator to perform well.

### 2.5.2 ElasticNet regression

ElasticNet combines \(L_1\) and \(L_2\) regularization. It is included to test whether sparsity-inducing regularization improves or degrades damage inference. If ElasticNet performs worse than ridge regression, it may indicate that removing or shrinking groups of descriptors too strongly can suppress damage-sensitive information.

### 2.5.3 Random forest regression

Random forest regression is included as a nonlinear baseline. It tests whether a more flexible nonlinear estimator improves performance over regularized linear models. The comparison is used to examine whether estimator complexity is more important than physics-informed descriptor construction.

The results show that random forest is not consistently superior. It performs poorly in the full four-sensor main setting, but becomes competitive under extreme sensor sparsity. This suggests that estimator choice may depend on sensing conditions.

---

## 2.6 Evaluation metrics

The primary regression metrics are test MAE and test RMSE.

The test MAE is defined as:

\[
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N} |\hat{y}_i - y_i|
\]

The test RMSE is defined as:

\[
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}
\]

where \(y_i\) and \(\hat{y}_i\) denote true and predicted damage values, respectively. In the multi-output setting, these metrics are computed over story-level damage entries.

In addition to overall MAE and RMSE, the following reliability-oriented metrics are used:

1. Damaged-entry MAE.
2. Mean prediction on zero-damage entries.
3. Prediction bias.
4. High-damage MAE.
5. High-damage underestimation ratio.

The prediction bias is calculated as the mean predicted damage minus the mean true damage:

\[
\mathrm{Bias} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)
\]

A negative bias in high-damage entries indicates systematic underestimation, which is safety-critical in structural condition assessment.

The underestimation ratio is defined as the proportion of entries for which the predicted damage is lower than the true damage:

\[
\mathrm{Underestimation\ Ratio} =
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i < y_i)
\]

This metric is particularly important for high-damage entries because underestimating severe damage can lead to unsafe maintenance or intervention decisions.

---

## 2.7 Main ablation experiment

The main ablation experiment is performed on the 3000-case dataset. The objective is to compare descriptor groups and estimators under the same fixed train/validation/test split.

The experiment compares the full descriptor set against reduced descriptor sets, including no-metadata descriptors, response-basic-only descriptors, frequency-related descriptors, spatial descriptors, and correlation descriptors. Each feature set is evaluated using the selected estimators.

The main questions are:

1. Does the full descriptor set outperform reduced feature subsets?
2. Does metadata provide useful additional information?
3. Does a more complex estimator outperform ridge regression?
4. Are simple response statistics sufficient for damage inference?

The main ablation experiment provides the baseline result for the subsequent robustness and reliability analyses.

---

## 2.8 Repeated-split robustness analysis

To evaluate whether the fixed-split result is sensitive to a particular data partition, repeated-split robustness analysis is conducted using 10 random seeds. For each seed, the 3000-case dataset is repartitioned into training, validation, and test subsets.

The same feature extraction and estimator-comparison workflow is applied to each split. The mean, standard deviation, and coefficient of variation of test MAE are reported for each configuration.

This analysis is used to determine whether the best-performing configuration is stable across different data partitions, rather than being selected only because of one favorable split.

---

## 2.9 Damage-stratified reliability analysis

Damage-stratified reliability analysis is conducted to evaluate whether model performance is uniform across damage severities. Predictions are grouped into zero-, low-, medium-, and high-damage regimes.

For each regime, the study reports MAE, RMSE, prediction bias, mean true damage, mean predicted damage, and underestimation ratio. This analysis is motivated by the safety-critical nature of SHM: a model with low overall MAE may still be unreliable if it systematically underestimates high-damage entries.

The damage-stratified analysis therefore complements average prediction metrics and provides a more reliability-oriented assessment.

---

## 2.10 Noise robustness analysis

Noise robustness is evaluated using six independently generated 1000-case datasets. Each dataset is generated with a fixed noise level:

- 0%;
- 2%;
- 5%;
- 10%;
- 15%;
- 20%.

For each noise level, the same descriptor extraction and model-comparison process is applied. The full descriptor set, no-metadata descriptors, response-basic-only descriptors, and selected estimators are compared.

Because each noise level corresponds to an independently generated controlled simulation dataset, the performance trend is not expected to be strictly monotonic. The purpose is to evaluate whether the proposed descriptor set remains competitive and avoids performance collapse under increasing measurement noise.

---

## 2.11 Sensor sparsity stress test

Sensor sparsity is evaluated using a zero-masking strategy. Unavailable response channels are set to zero while preserving the original tensor shape and four-story damage targets. This design keeps the downstream feature extraction and evaluation scripts compatible across different sensor layouts.

The tested sensor layouts are:

1. Four sensors: 1-2-3-4.
2. Three sensors: 1-2-4.
3. Two sensors: 1-4.
4. One sensor: 4.

This experiment should be interpreted as a sensor sparsity stress test, not as an optimal missing-data imputation method. The objective is to evaluate how the descriptor set and estimators behave when spatial sensing coverage is progressively reduced.

The sensor sparsity experiment addresses three questions:

1. Does the full descriptor set remain useful as sensor count decreases?
2. Does prediction error increase under sparse sensing?
3. Does the best estimator change under extreme sensor sparsity?

---

## 2.12 Reproducibility and implementation notes

All data generation, preprocessing, feature extraction, and evaluation scripts are preserved in the project repository. Large generated `.npz` datasets are not versioned because of file-size constraints. Instead, the full data generation and preprocessing workflows are provided to support reproducibility.

The main reproducibility scripts include:

- `scripts/run_q2_generate_debug_plus_3000.sh`
- `scripts/run_q2_3000_seed_robustness.sh`
- `scripts/run_q2_3000_damage_stratified.sh`
- `scripts/run_q2_noise_robustness_1000.sh`
- `scripts/run_q2_sensor_sparsity_3000.sh`
- `scripts/build_q2_paper_ready_package.sh`

The paper-ready tables, figures, captions, and text snippets are collected under:

- `results/paper_ready/q2_fast_track/`

---

## 2.13 Methodological scope

The present study is a controlled simulation-based method evaluation. It does not claim direct field deployment readiness. The structural model, excitation conditions, damage distributions, noise levels, and sensor sparsity settings are designed for systematic testing rather than for reproducing one specific real structure.

The proposed framework should therefore be interpreted as a physics-informed descriptor construction and evaluation strategy for sparse and noisy SHM-like data. Future work should validate the framework using field monitoring data, more diverse structural systems, environmental variability, and realistic missing-data mechanisms.

