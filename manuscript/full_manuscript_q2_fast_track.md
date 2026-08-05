# Physics-Informed Descriptor Construction for Structural Damage Inference under Sparse and Noisy SHM-like Data

## Abstract

Structural health monitoring (SHM) can support condition assessment and maintenance decision-making, but monitored responses are often sparse, noisy, and difficult to connect to labelled damage states. These constraints make structural damage inference vulnerable to unstable model selection, weak interpretability, and safety-critical errors that are hidden by average accuracy metrics. Here, we develop and evaluate a physics-informed descriptor construction framework for story-level structural damage inference under controlled sparse and noisy SHM-like response data. A controlled OpenSeesPy-based workflow generated structural response histories for a four-story frame, and each case was represented by a 92-dimensional descriptor vector combining response statistics, frequency-domain indicators, spatial response patterns, response correlations, and excitation-related metadata.

The framework was evaluated through main feature ablation, repeated-split robustness analysis, damage-stratified reliability diagnosis, noise robustness testing, and sensor sparsity stress testing. On the 3000-case controlled simulation dataset, the full descriptor set with ridge regression achieved the best fixed-split performance, with a test MAE of 0.0393, a test RMSE of 0.0585, and a damaged-entry MAE of 0.0634. Across 10 repeated train/validation/test partitions, the same configuration remained best in every split and achieved a mean test MAE of 0.0454 +/- 0.0009. Under independently generated 1000-case datasets with fixed noise levels from 0% to 20%, full + ridge remained the best-performing configuration at each tested noise level.

The reliability analysis also exposed important boundaries. For the best overall configuration, the high-damage MAE increased to 0.0888, the high-damage bias was -0.0717, and the high-damage underestimation ratio reached 0.8214. Sensor sparsity further degraded performance: full + ridge remained best with four and three sensors, but the best estimator shifted to random forest under two- and one-sensor layouts. These results indicate that physics-informed descriptor construction can support stable average damage inference under controlled noisy and moderately sparse monitoring conditions, while average MAE alone is insufficient for safety-critical SHM evaluation. Field validation, realistic missing-data mechanisms, and damage-aware calibration remain necessary before deployment-oriented claims can be made.

## Keywords

Structural health monitoring; structural damage inference; physics-informed descriptors; sparse sensing; noisy monitoring data; OpenSeesPy simulation; damage-stratified reliability

## Terminology Ledger

| Canonical term | First-use definition | Notes for consistency |
| --- | --- | --- |
| Structural health monitoring (SHM) | Sensor-based monitoring for structural condition assessment | Define once, then use SHM. |
| SHM-like response data | Controlled simulated responses designed to mimic monitored structural response channels | Do not call this field SHM data. |
| Physics-informed descriptor construction | Transformation of response histories into physically motivated summary descriptors | Use descriptor, not feature, in the main manuscript unless referring to feature ablation. |
| Full descriptor set | The 92-dimensional descriptor set including response-derived and excitation-related descriptors | Use full descriptor set or full + ridge consistently. |
| Ridge regression | Regularized linear estimator selected as the main four-sensor model | Do not claim universal superiority. |
| Damage-stratified reliability diagnosis | Evaluation by zero, low, medium, and high damage regimes | Use to distinguish average accuracy from safety-critical reliability. |
| Sensor sparsity stress test | Zero-masking of unavailable response channels while preserving the four-story damage target | Do not describe as optimized missing-data imputation. |

## 1. Introduction

Structural health monitoring is increasingly used to support condition assessment, damage detection, and maintenance planning for civil infrastructure [CITATION NEEDED: representative SHM overview]. Yet the translation from monitored response histories to reliable structural state information remains difficult. In many monitoring scenarios, only a limited number of response channels are available, measurements are contaminated by noise, and labelled damage observations are scarce. These constraints are especially problematic when the task is not only to detect abnormal behaviour, but to infer story- or component-level damage severity.

Machine learning has been widely explored for SHM-related damage identification and state inference [CITATION NEEDED: representative data-driven SHM review]. Flexible models can approximate complex relationships between structural responses and damage labels when training data are abundant and representative. In civil infrastructure applications, however, severe damage states are rarely observed, field data are affected by changing excitation and environmental conditions, and labelled examples may be expensive or impossible to obtain. Under these conditions, increasing model complexity alone may not increase reliability. A model can achieve a low average error while still underestimating the damage regimes that matter most for safety.

Physics-informed or physics-guided machine learning has been proposed as one route for improving interpretability and data efficiency in engineering systems [CITATION NEEDED: physics-guided or physics-informed ML for SHM]. In the present setting, the central move is not to impose a physics-informed neural network formulation, but to construct descriptors that encode physically meaningful information from structural responses. Response magnitudes, frequency-domain indicators, spatial response redistribution, inter-channel correlations, and excitation-related metadata can provide a compact representation of structural behaviour before estimator training. This approach separates the contribution of representation from the contribution of estimator complexity.

A second gap concerns evaluation. Mean absolute error (MAE) and root mean squared error (RMSE) are useful global metrics, but they can conceal systematic errors in severe damage regimes. For SHM-oriented decision support, underestimating high damage is more consequential than producing small average errors on healthy or low-damage entries. A defensible evaluation therefore needs damage-stratified metrics, prediction bias, underestimation behaviour, and stress tests under monitoring constraints such as measurement noise and sparse sensing.

This study develops and evaluates a physics-informed descriptor construction framework for structural damage inference under controlled sparse and noisy SHM-like data. A controlled OpenSeesPy-based simulation workflow generated a 3000-case main dataset with four-story damage labels, and the response histories were transformed into a 92-dimensional descriptor set. The framework was tested through feature ablation, repeated-split robustness, damage-stratified reliability diagnosis, noise robustness analysis, and sensor sparsity stress testing. The contribution is a bounded method-and-diagnosis result: physically motivated descriptors can support stable average inference under controlled conditions, but high-damage underestimation and extreme sensor sparsity remain reliability boundaries.

## 2. Methodology

### 2.1 Study scope and task formulation

The task was multi-output regression from structural response histories to story-level damage vectors. For each simulated case, the input was a time history with 2000 time steps and four response channels, and the output was a four-dimensional damage vector. Each output component represented the stiffness degradation ratio associated with one story. A value of zero denoted no damage in that story, whereas larger values denoted greater stiffness degradation.

The study was designed as a controlled simulation-based evaluation of descriptor construction. It was not intended to represent a calibrated field SHM dataset, a bridge deployment study, or an end-to-end deep learning architecture. The scope was to test whether physically motivated descriptors support stable damage inference under controlled noisy and sparse monitoring conditions, and to identify where the resulting inference remains unreliable.

### 2.2 Controlled OpenSeesPy structural simulation

Structural responses were generated using a two-dimensional OpenSeesPy frame model. The model represented a four-story, two-bay elastic frame. Story-level damage was introduced by reducing the Young's modulus of the column elements in the corresponding story according to the damage ratio. This produced a controlled mapping between damage labels and structural dynamic response changes.

Each dynamic analysis used a sine-wave ground excitation with randomized amplitude, frequency, and phase. The analysis recorded floor response histories, including acceleration response channels, and the response tensor used for inference had shape 3000 x 2000 x 4 for the main dataset. The time step was 0.01 s. The controlled excitation design allowed systematic generation of labelled cases, but it also bounds the interpretation: the data are SHM-like simulation responses, not measured field monitoring data.

### 2.3 Dataset construction and splitting

The main dataset, denoted debug_plus_3000, contained 3000 simulated cases. The corresponding damage target had shape 3000 x 4. The dataset included healthy cases, single-story damaged cases, and multi-story damaged cases. The train/validation/test split contained 2100, 450, and 450 cases, respectively.

The response inputs were normalized using training-set statistics, and the same normalization parameters were applied to validation and test subsets. Damage-bin labels were used for stratified splitting and reliability diagnosis. The damage regimes were zero, low, medium, and high, with high-damage entries defined by the implementation used in the evaluation scripts. This binning was used only for evaluation; the primary learning task remained continuous damage regression.

### 2.4 Physics-informed descriptor construction

Each response history was transformed into a 92-dimensional descriptor vector. The descriptor set combined five groups of information. First, response statistics summarized story-wise magnitude and distribution, including mean, standard deviation, maximum absolute response, root-mean-square response, peak-to-peak response, and crest factor. Second, frequency-domain descriptors captured dominant frequency, spectral centroid, and band-energy information. Third, spatial descriptors represented inter-story response distribution, amplification, and adjacent-story ratios. Fourth, response correlation descriptors summarized relationships between floor responses and between floor and input signals. Fifth, excitation-related metadata encoded input amplitude, input frequency, noise level, and proximity to the healthy first-mode frequency.

The descriptors were standardized using training-set statistics only. This prevented validation and test data from influencing the feature scale. The full descriptor set was compared with reduced descriptor groups, including no-metadata descriptors, response-basic-only descriptors, frequency-related descriptors, spatial descriptors, correlation descriptors, and physics-no-metadata core descriptors.

### 2.5 Estimators and model selection

The estimator set included ridge regression, ElasticNet regression, and random forest regression. Ridge regression served as the primary estimator because it tests whether the descriptor space is sufficiently informative for a regularized linear model. ElasticNet tested whether sparsity-inducing regularization changed performance. Random forest regression provided a nonlinear baseline for evaluating whether estimator flexibility outweighed descriptor structure.

Candidate hyperparameters were selected using validation-set mean squared error. Predictions were clipped to the physically admissible range used in the scripts. The main comparisons were reported on held-out test data. This design was used to compare descriptor groups and estimator families without presenting estimator complexity as the main contribution.

### 2.6 Evaluation protocol

The evaluation followed five stages. First, a fixed-split main ablation on the 3000-case dataset compared descriptor groups and estimators. Second, a repeated-split robustness analysis repartitioned the 3000 cases across 10 random seeds to test whether the main conclusion depended on one favourable split. Third, damage-stratified reliability diagnosis evaluated zero-, low-, medium-, and high-damage entries, including bias and underestimation ratio. Fourth, noise robustness was tested using six independently generated 1000-case datasets with fixed noise levels of 0%, 2%, 5%, 10%, 15%, and 20%. Fifth, sensor sparsity was tested by zero-masking unavailable response channels while preserving the original four-story damage labels and response tensor shape.

The primary metrics were test MAE and test RMSE over story-level damage entries. Additional reliability metrics included damaged-entry MAE, prediction bias, mean prediction on zero-damage entries, high-damage MAE, and high-damage underestimation ratio. The underestimation ratio was defined as the proportion of entries for which predicted damage was lower than true damage.

## 3. Results

### 3.1 Main ablation on the 3000-case dataset

The full physics-informed descriptor set with ridge regression achieved the strongest fixed-split performance on the main 3000-case dataset. The full + ridge configuration produced a test MAE of 0.0393, a test RMSE of 0.0585, and a damaged-entry MAE of 0.0634 (Table 2; Figure 1). This result established the main empirical basis for using the full descriptor set as the central representation in the manuscript.

Reduced descriptor sets produced larger errors. Removing metadata increased the test MAE to 0.0425, indicating that excitation-related descriptors contributed useful but secondary information. The response-basic-only ridge model produced a test MAE of 0.0577, showing that simple time-domain response statistics were insufficient to capture the damage-relevant information in the controlled setting.

Estimator comparison showed that greater model flexibility did not automatically increase performance. Under the full descriptor set, ElasticNet achieved a test MAE of 0.0476, and random forest achieved a test MAE of 0.0644. Thus, the best four-sensor result came from a regularized linear estimator applied to a physically structured descriptor space, rather than from the most flexible estimator tested.

### 3.2 Repeated-split robustness

The main fixed-split conclusion was stable across repeated random partitions. Across 10 train/validation/test splits of the 3000-case dataset, full + ridge remained the best-performing configuration in every split (Table 3; Figure 2). Its mean test MAE was 0.0454 with a standard deviation of 0.0009, and its mean test RMSE was 0.0621.

The repeated-split mean was higher than the fixed-split MAE of 0.0393, which indicates that the fixed split was relatively favourable. This distinction is important for interpretation: the fixed-split result shows the best main ablation outcome under the selected protocol, whereas the repeated-split result provides a more conservative estimate of generalization. Both protocols selected the same descriptor-estimator pairing.

The no-metadata ridge model remained second in the repeated-split summary, with a mean test MAE of 0.0490. This reinforced the fixed-split interpretation that metadata added incremental performance benefits, while response-derived physics-informed descriptors carried the main predictive signal.

### 3.3 Damage-stratified reliability diagnosis

Damage-stratified evaluation showed that strong average accuracy did not imply uniform reliability across damage regimes. The full + ridge configuration achieved the best overall MAE of 0.0407 and an overall RMSE of 0.0600 in the damage-stratified repeated analysis (Table 4; Figure 3). However, its high-damage MAE increased to 0.0888, more than twice the overall MAE.

The high-damage entries were systematically underestimated. In the high-damage bin, the mean true damage was 0.2758, whereas the mean predicted damage was 0.2041. This produced a high-damage bias of -0.0717 and an underestimation ratio of 0.8214. These values show that the best average model still tended to underpredict severe damage.

The reliability degradation was severity-dependent. For full + ridge, the MAE was 0.0297 for zero-damage entries and 0.0289 for low-damage entries, then increased to 0.0502 for medium-damage entries and 0.0888 for high-damage entries. The no-metadata ridge model and the response-basic-only ridge model showed larger high-damage MAEs of 0.0948 and 0.1434, respectively. Full + ElasticNet showed stronger high-damage underestimation, with a high-damage MAE of 0.1787 and an underestimation ratio of 0.9967. These comparisons indicate that average accuracy and high-damage reliability must be evaluated separately.

### 3.4 Noise robustness

The full + ridge configuration remained the best-performing model across all tested noise levels. Noise robustness was evaluated using independently generated 1000-case datasets at 0%, 2%, 5%, 10%, 15%, and 20% fixed noise levels (Table 5; Figure 4). The corresponding full + ridge test MAEs were 0.0345, 0.0361, 0.0375, 0.0394, 0.0431, and 0.0420.

The overall trend was consistent with increasing error under noisier responses, but the curve did not increase at every adjacent noise step because each noise level used an independently generated dataset. The 20% noise MAE was slightly lower than the 15% noise MAE, although both were higher than the lower-noise settings. This design supports a controlled robustness interpretation rather than a deterministic noise-degradation law.

Damaged-entry errors were more sensitive to noise than the global average. For full + ridge, damaged-entry MAE increased from 0.0526 at 0% noise to approximately 0.068-0.069 at 15%-20% noise. The full descriptor set also outperformed no-metadata and response-basic-only variants across the tested noise levels. These results support robustness under controlled simulated noise, but they do not establish performance under field monitoring uncertainty such as sensor drift, environmental effects, missing data, or nonstationary operational conditions.

### 3.5 Sensor sparsity stress test

Sensor sparsity substantially degraded damage inference performance. The stress test zero-masked unavailable response channels while preserving the four-story damage labels and the original tensor shape. The tested layouts were four sensors (1-2-3-4), three sensors (1-2-4), two sensors (1-4), and one sensor (4) (Table 6; Figure 5).

With four sensors, full + ridge was best, matching the main fixed-split result with a test MAE of 0.0393 and a test RMSE of 0.0585. With three sensors, full + ridge remained best, but the test MAE increased to 0.0612 and the test RMSE increased to 0.0836. This shows that even moderate sensor reduction removed useful spatial information.

Under more extreme sparsity, the best estimator shifted from ridge regression to random forest. The two-sensor layout achieved its best result with full + random forest, producing a test MAE of 0.0764 and a test RMSE of 0.0977. The one-sensor layout also selected full + random forest, with a test MAE of 0.0801 and a test RMSE of 0.1011. Damaged-entry MAE increased from 0.0634 with four sensors to 0.1006, 0.1270, and 0.1323 with three, two, and one sensors, respectively.

The full descriptor set remained the strongest descriptor set across all sensor layouts, but its advantage narrowed under one-sensor conditions. In the one-sensor setting, full + ridge achieved a test MAE of 0.0845, while response-basic-only + ridge achieved a similar test MAE of 0.0848. This indicates that several physics-informed descriptor groups depend on sufficient spatial sensing coverage.

### 3.6 Result synthesis

The experiments support a coherent but bounded conclusion. Physics-informed descriptor construction supported and stabilized average damage inference under the controlled four-sensor setting, and the full descriptor set remained useful under tested noise and sensor-sparsity conditions. At the same time, the reliability diagnosis revealed a systematic high-damage underestimation pattern, and the sensor sparsity test showed a clear degradation under one- and two-sensor layouts. The central finding is therefore not that the framework solves sparse and noisy SHM, but that it provides a stable descriptor-based route for controlled damage inference while exposing safety-relevant limitations that average metrics would hide.

## 4. Discussion

The main methodological implication is that physically structured representation can matter more than estimator complexity under limited, noisy SHM-like data. In the full four-sensor setting, ridge regression on the full descriptor set outperformed ElasticNet and random forest. This does not mean that linear models are generally superior for SHM. Rather, it suggests that a descriptor space encoding response magnitude, frequency content, spatial distribution, correlations, and excitation information can make the regression problem sufficiently structured for a simple regularized estimator.

The estimator shift under severe sensor sparsity refines this interpretation. Ridge regression remained best with four and three sensors, but random forest became best with two and one sensors. When spatial response coverage was reduced, the relationship between descriptors and damage labels may have become less well approximated by a linear mapping. This result argues against a universal model-choice claim. Estimator selection should be interpreted together with sensing coverage.

The damage-stratified analysis is the most important reliability result. The best average model underestimated high-damage entries, with a high-damage bias of -0.0717 and an underestimation ratio of 0.8214. This pattern is consistent with a regression-to-the-mean failure mode, where a model optimized for average error favours moderate predictions when severe damage entries are harder to identify. For SHM-oriented state inference, this behaviour is safety-relevant because underestimating severe damage can delay inspection, strengthening, or intervention.

The results also clarify the role of excitation-related metadata. The full descriptor set outperformed the no-metadata variant in the main ablation, repeated-split analysis, and noise robustness tests. However, the performance gain was moderate rather than dominant. This suggests that metadata helped separate structural response changes from input-condition variation, but the core predictive signal remained in response-derived physics-informed descriptors.

The noise robustness results support controlled noisy-monitoring applicability, but they should not be generalized beyond the tested design. The noise datasets used fixed noise levels and were generated independently. Real monitoring uncertainty can include sensor drift, environmental variability, operational variability, missing channels, correlated noise, and changing boundary conditions. The present results therefore show that the descriptor set retained performance under controlled simulated noise, not that it is ready for field deployment.

The sensor sparsity stress test identifies a practical sensing boundary. Zero-masking showed that the full descriptor set remained useful when one sensor was removed, but performance degraded strongly with two or one sensors. This is expected because spatial descriptors and correlation descriptors require sufficient response coverage. Future sparse-sensing work should compare zero-masking with optimized sensor placement, missing-data reconstruction, graph-based spatial modelling, and training schemes designed for random sensor dropout.

## 5. Limitations and Future Work

This study used controlled OpenSeesPy-based simulation rather than field SHM validation. The controlled setting is useful for isolating descriptor construction, noise effects, sensor sparsity, and damage severity, but it does not reproduce all sources of uncertainty in field monitoring. Environmental variability, temperature effects, aging, soil-structure interaction, sensor drift, unknown excitation, and nonstationary boundary conditions remain outside the present evidence.

The structural model and damage representation were simplified. The simulated system was a four-story elastic frame, and damage was represented as story-level stiffness degradation. This abstraction supports reproducible method evaluation, but it does not capture the full diversity of bridges, irregular frames, local component failures, connection damage, nonlinear material behaviour, or spatially distributed damage mechanisms.

The sensor sparsity experiment used zero-masking rather than optimized missing-data treatment. This choice preserved the tensor shape and kept the feature extraction pipeline consistent across sensor layouts, but it is a stress-test design. Future work should evaluate sensor placement, imputation, graph-based reconstruction, and models trained explicitly for missing-sensor patterns.

High-damage underestimation remains unresolved. The current study reports this limitation rather than hiding it. Future work should test damage-aware objective functions, high-damage weighting, stratified sampling, severe-damage augmentation, calibration models, and uncertainty quantification. For safety-critical SHM, a useful model should not only reduce average error, but also avoid systematic underestimation in severe damage regimes.

The estimator family was intentionally limited to ridge regression, ElasticNet, and random forest regression. This choice allowed the study to focus on descriptor construction rather than deep architecture design. Future comparisons with gradient boosting, Gaussian processes, graph neural networks, temporal convolutional networks, transformers, neural operators, or physics-guided neural architectures should be designed so that data size, hyperparameter effort, and representation effects are not conflated.

## 6. Conclusion

This study developed and evaluated a physics-informed descriptor construction framework for story-level structural damage inference under controlled sparse and noisy SHM-like data. The framework transformed response histories into a 92-dimensional descriptor set combining response statistics, frequency-domain indicators, spatial response patterns, response correlations, and excitation-related metadata.

The full descriptor set with ridge regression achieved the best main four-sensor result, with a fixed-split test MAE of 0.0393 and a test RMSE of 0.0585. Repeated-split analysis confirmed that the same configuration remained best across 10 random partitions, with a mean test MAE of 0.0454 +/- 0.0009. Noise robustness testing further showed that full + ridge remained best across fixed noise levels from 0% to 20%.

The reliability analyses identified the limits of the approach. High-damage entries were systematically underestimated, with a high-damage MAE of 0.0888, a bias of -0.0717, and an underestimation ratio of 0.8214. Sensor sparsity also degraded performance, and the best estimator shifted to random forest under two- and one-sensor layouts. These findings show that average accuracy is insufficient for safety-critical SHM evaluation. Physics-informed descriptor construction can support controlled damage inference, but field validation, sparse-sensing strategies, and damage-aware reliability calibration are required before deployment-oriented conclusions can be drawn.

## Data Availability Statement

The numerical results used for this draft are summarized in `results/paper_ready/q2_fast_track/`. Large generated `.npz` datasets are local experimental artifacts and are not included in this manuscript draft. A final submission should state which processed data, scripts, and paper-ready tables will be made available, and whether large simulation datasets will be archived separately.

## Code Availability Statement

The project contains scripts for data generation, preprocessing, descriptor extraction, feature ablation, repeated-split analysis, damage-stratified diagnosis, noise robustness, sensor sparsity testing, and paper-ready package generation. A final submission should specify the repository URL or archival DOI once the code-release plan is fixed.

## Ethics Statement

This study used controlled simulation data and did not involve human participants, animal subjects, or private field-monitoring records. If a target journal requires a formal ethics statement, this should be adapted to its exact wording.

## Funding

[FUNDING INFORMATION NEEDED: add grant numbers, institutional support, or state that no specific funding was received.]

## Competing Interests

[COMPETING INTERESTS STATEMENT NEEDED: add the authors' declaration.]

## Author Contributions

[AUTHOR CONTRIBUTIONS NEEDED: add CRediT-style roles after the author list is fixed.]

## AI-use Disclosure

[AI-USE DISCLOSURE NEEDED: revise according to the target journal policy and the author's actual use of AI tools during drafting, editing, and code assistance.]

## References

[REFERENCES PLACEHOLDER: Do not submit without replacing this section with verified references. Candidate literature should be drawn from the user's Zotero `first paper` collection or another verified BibTeX/PDF source. Do not invent author names, years, DOIs, or citation keys.]
