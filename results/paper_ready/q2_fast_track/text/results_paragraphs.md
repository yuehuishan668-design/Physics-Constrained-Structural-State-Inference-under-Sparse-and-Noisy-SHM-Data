# Paper-ready Results Paragraphs

## 1. Main ablation

On the 3000-case controlled simulation dataset, the full physics-informed feature set combined with ridge regression achieved the best fixed-split performance, with a test MAE of 0.0393 and a test RMSE of 0.0585. Removing metadata-related descriptors increased the test MAE, while response-basic-only descriptors led to substantially larger errors. These results indicate that the full descriptor set provided the most informative representation for structural damage inference under the present sparse and noisy SHM simulation setting.

## 2. Repeated-split robustness

To examine whether the fixed-split ablation result was sensitive to data partitioning, a repeated-split robustness analysis was performed on the 3000-case dataset using 10 random seeds. The full physics-informed feature set combined with ridge regression achieved the best mean test MAE of 0.0454 with a standard deviation of 0.0009, and it was selected as the best-performing configuration in all 10 splits. This confirms that the main conclusion was stable rather than split-specific.

## 3. Damage-stratified reliability

Damage-stratified evaluation further revealed that stable average performance did not imply uniform reliability across damage severities. The full physics-informed feature set with ridge regression achieved the best overall MAE of 0.0407, but its high-damage MAE increased to 0.0888. In the high-damage bin, the mean true damage was 0.2758, whereas the mean predicted damage was 0.2041, resulting in a negative bias of -0.0717 and an underestimation ratio of 0.8214. Therefore, damage-stratified reliability diagnosis is necessary for evaluating SHM-based structural state inference models, especially in safety-critical damage regimes.

## 4. Noise robustness

Noise robustness was evaluated using six independently generated 1000-case datasets with fixed noise levels of 0%, 2%, 5%, 10%, 15%, and 20%. The full physics-informed feature set combined with ridge regression remained the best-performing configuration across all noise levels. Its test MAE increased from 0.0345 at 0% noise to 0.0431 at 15% noise and 0.0420 at 20% noise, while no performance collapse was observed. Although the error trend was not strictly monotonic because each noise level used an independently generated dataset, the overall results indicate that the proposed descriptor set retained robust predictive performance under noisy monitoring conditions.

## 5. Sensor sparsity

Sensor sparsity was evaluated by zero-masking unavailable response channels while preserving the original four-story damage targets and tensor shape. The full physics-informed feature set remained the strongest descriptor set across all sensor configurations. With four and three sensors, the full-feature ridge model achieved the best test MAE of 0.0393 and 0.0612, respectively. Under more extreme sparsity, the best estimator shifted to random forest, yielding test MAEs of 0.0764 for the two-sensor layout and 0.0801 for the one-sensor layout. These results indicate that the proposed descriptor set remains effective under moderate sensor sparsity, but severe sensor reduction substantially degrades prediction reliability, especially for damaged entries.
