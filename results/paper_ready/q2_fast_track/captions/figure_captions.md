# Figure Captions

## Figure 1. Main physics-informed feature ablation

Performance comparison of different physics-informed feature subsets and estimators on the 3000-case controlled simulation dataset. The full physics-informed descriptor set combined with ridge regression achieved the strongest overall predictive performance, indicating the benefit of combining response-derived and excitation-related descriptors.

## Figure 2. Repeated-split robustness

Repeated-split robustness analysis over 10 random train/validation/test partitions. The full physics-informed feature set with ridge regression remained the best-performing configuration across all splits, demonstrating that the main conclusion was not caused by a favorable single data partition.

## Figure 3. Damage-stratified reliability diagnosis

Damage-stratified evaluation of prediction reliability across zero-, low-, medium-, and high-damage regimes. Although the full-feature ridge model achieved the best average accuracy, high-damage entries exhibited larger errors and systematic underestimation, highlighting the need for reliability-oriented evaluation beyond average metrics.

## Figure 4. Noise robustness

Noise robustness analysis under fixed noise levels of 0%, 2%, 5%, 10%, 15%, and 20%. The full physics-informed feature set with ridge regression remained the best-performing configuration across all tested noise levels, with no performance collapse under high-noise conditions.

## Figure 5. Sensor sparsity stress test

Sensor sparsity stress test using zero-masked response channels. The full physics-informed feature set remained the strongest descriptor set across all sensor configurations, while severe sensor reduction substantially degraded prediction reliability, especially for damaged entries.
