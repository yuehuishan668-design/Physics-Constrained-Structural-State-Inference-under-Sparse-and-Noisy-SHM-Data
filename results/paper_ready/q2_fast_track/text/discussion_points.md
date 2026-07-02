# Paper-ready Discussion Points

## Point 1. Average accuracy is insufficient

The results show that average prediction accuracy alone can mask reliability limitations in safety-critical regimes. Although the full-feature ridge model achieved the best overall accuracy, high-damage entries showed larger errors and systematic underestimation. This supports the need for damage-stratified reliability diagnosis in SHM-oriented state inference.

## Point 2. Physics-informed descriptors are more important than model complexity

Across main ablation, repeated-split, and noise robustness experiments, regularized linear regression on the full physics-informed descriptor set was consistently stronger than more complex alternatives under the current controlled simulation setting. This suggests that structured descriptor construction can be more important than estimator complexity when data are sparse and noisy.

## Point 3. Metadata provides incremental but not dominant benefit

The full descriptor set consistently outperformed no-metadata variants, but the improvement was moderate rather than dominant. This indicates that excitation-related descriptors provide useful auxiliary information, while the main predictive signal remains response-derived.

## Point 4. Extreme sensor sparsity is a practical boundary

The sensor sparsity stress test showed that the proposed descriptor set remains useful under moderate sensor reduction, but severe reduction to two or one sensor substantially degrades reliability. This should be reported as a practical boundary of the current method rather than hidden.
