# Draft Result Section

## 1. Main physics-informed feature ablation

The feature-ablation experiment shows that the most defensible main configuration is `full + ridge + PhysicsSklearnRegressor`. This configuration achieved a test MAE of `0.04710682195088334` and a test RMSE of `0.07141816467590416` on the 500-case dataset. Compared with reduced response-feature subsets, the full physics-informed descriptor set provides the strongest empirical support for the value of combining response amplitude, frequency-domain, spatial, correlation, and metadata-related descriptors.

This result should be presented as the main positive finding of the study. The key claim is not that a complex nonlinear model is required, but that interpretable physics-informed descriptors can provide stable damage-inference performance when combined with a simple regularized estimator.

## 2. Damage-stratified reliability diagnosis

The damage-stratified analysis should be used to show the reliability boundary of the method. The current experiments indicate that high-damage cases remain more difficult than zero, low, and medium damage cases. This means the paper should avoid claiming fully reliable high-severity damage quantification.

The correct paper-level interpretation is that the proposed feature construction improves overall state inference, but high-damage underestimation remains a systematic limitation that must be explicitly reported.

## 3. Repeated-split robustness

Repeated-split experiments are used to determine whether the main result is stable or merely caused by a favorable train/test split. These results should be described as robustness evidence rather than as a separate method.

## 4. Paired healthy-baseline diagnosis

The paired healthy-baseline experiments should not be used as the main method. The clean input-only matching control shows that healthy-baseline normalization does not sufficiently solve high-damage underestimation or zero/damaged separability. Therefore, this part should be framed as a diagnostic/control experiment.

## 5. Final paper-level conclusion

The paper should be framed as a method-and-diagnosis study. The defensible contribution is:

1. construction of physics-informed response descriptors for sparse and noisy SHM-based structural state inference;
2. ablation-based validation of feature groups and estimator choices;
3. repeated-split robustness analysis;
4. damage-stratified reliability diagnosis showing both the value and the limitation of the proposed descriptors.

The current evidence does not support changing the topic into a pure healthy-baseline normalization paper or a pure damage classification paper.
