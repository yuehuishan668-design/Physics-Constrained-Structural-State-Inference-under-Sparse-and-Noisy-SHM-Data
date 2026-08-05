# Q2 Numerical Consistency Check

This report checks whether the key paper-ready tables match the canonical numerical claims used in the manuscript drafts.

## 1. Dataset and feature summary
- **PASS** `dataset n_cases`: actual=3000.000000, expected=3000.000000, diff=0.000000
- **PASS** `dataset n_stories`: actual=4.000000, expected=4.000000, diff=0.000000
- **PASS** `response time steps`: actual=2000.000000, expected=2000.000000, diff=0.000000
- **PASS** `feature n_features`: actual=92.000000, expected=92.000000, diff=0.000000
- **PASS** `F_train size`: actual=2100.000000, expected=2100.000000, diff=0.000000
- **PASS** `F_val size`: actual=450.000000, expected=450.000000, diff=0.000000
- **PASS** `F_test size`: actual=450.000000, expected=450.000000, diff=0.000000

## 2. Main fixed-split ablation
- **PASS** `main full + ridge test_mae`: actual=0.039336, expected=0.039336, diff=0.000000
- **PASS** `main full + ridge test_rmse`: actual=0.058548, expected=0.058548, diff=0.000000
- **PASS** `main full + ridge damaged-entry MAE`: actual=0.063431, expected=0.063431, diff=0.000000
- **PASS** `main no_meta + ridge test_mae`: actual=0.042458, expected=0.042458, diff=0.000000
- **PASS** `main response_basic_only + ridge test_mae`: actual=0.057718, expected=0.057718, diff=0.000000
- **PASS** `main full + elasticnet test_mae`: actual=0.047601, expected=0.047601, diff=0.000000
- **PASS** `main full + random_forest test_mae`: actual=0.064416, expected=0.064416, diff=0.000000

## 3. Repeated-split robustness
- **PASS** `repeated full + ridge test_mae_mean`: actual=0.045426, expected=0.045426, diff=0.000000
- **PASS** `repeated full + ridge test_mae_std`: actual=0.000865, expected=0.000865, diff=0.000000
- **PASS** `repeated full + ridge test_rmse_mean`: actual=0.062085, expected=0.062085, diff=0.000000
- **PASS** `repeated no_meta + ridge test_mae_mean`: actual=0.049016, expected=0.049016, diff=0.000000
- **PASS** `per-seed best full + ridge count`: actual=10.000000, expected=10.000000, diff=0.000000

## 4. Damage-stratified reliability
- **PASS** `damage full + ridge overall_mae_mean`: actual=0.040679, expected=0.040679, diff=0.000000
- **PASS** `damage full + ridge overall_rmse_mean`: actual=0.060010, expected=0.060010, diff=0.000000
- **PASS** `damage full + ridge high_damage_mae_mean`: actual=0.088806, expected=0.088806, diff=0.000000
- **PASS** `damage full + ridge high_damage_underestimation_ratio_mean`: actual=0.821399, expected=0.821399, diff=0.000000
- **WARN** `damage high bin bias_mean` not found in wide-format T06 table.
- **WARN** `damage high bin mean_true_mean` not found in wide-format T06 table.
- **WARN** `damage high bin mean_pred_mean` not found in wide-format T06 table.
- **WARN** T06_damage_stratified_runs.csv has no `bin` column and no recognized high-damage bias/mean columns.
- **INFO** Available T06 columns: ['seed', 'feature_set', 'model', 'best_candidate', 'n_total_samples', 'n_train', 'n_val', 'n_test', 'n_features', 'best_val_mse', 'overall_mse', 'overall_mae', 'overall_rmse', 'overall_bias', 'negative_prediction_ratio_raw', 'negative_prediction_ratio', 'mean_true_damage', 'mean_pred_damage', 'mae_damaged', 'mae_high_damage', 'selected_feature_indices']
- **INFO** Core damage-stratified checks from T05 passed above; detailed high-bin bias values should be verified from R02_damage_stratified_report.md if needed.

## 5. Noise robustness
- **PASS** noise 0% best config: full + ridge
- **PASS** `noise 0% full + ridge test_mae`: actual=0.034468, expected=0.034468, diff=0.000000
- **PASS** noise 2% best config: full + ridge
- **PASS** `noise 2% full + ridge test_mae`: actual=0.036133, expected=0.036133, diff=0.000000
- **PASS** noise 5% best config: full + ridge
- **PASS** `noise 5% full + ridge test_mae`: actual=0.037546, expected=0.037546, diff=0.000000
- **PASS** noise 10% best config: full + ridge
- **PASS** `noise 10% full + ridge test_mae`: actual=0.039448, expected=0.039448, diff=0.000000
- **PASS** noise 15% best config: full + ridge
- **PASS** `noise 15% full + ridge test_mae`: actual=0.043141, expected=0.043141, diff=0.000000
- **PASS** noise 20% best config: full + ridge
- **PASS** `noise 20% full + ridge test_mae`: actual=0.041966, expected=0.041966, diff=0.000000

## 6. Sensor sparsity
- **PASS** 4-sensor best config: actual=full + ridge, expected=full + ridge
- **PASS** `4-sensor best test_mae`: actual=0.039336, expected=0.039336, diff=0.000000
- **PASS** 3-sensor best config: actual=full + ridge, expected=full + ridge
- **PASS** `3-sensor best test_mae`: actual=0.061204, expected=0.061204, diff=0.000000
- **PASS** 2-sensor best config: actual=full + random_forest, expected=full + random_forest
- **PASS** `2-sensor best test_mae`: actual=0.076373, expected=0.076373, diff=0.000000
- **PASS** 1-sensor best config: actual=full + random_forest, expected=full + random_forest
- **PASS** `1-sensor best test_mae`: actual=0.080141, expected=0.080141, diff=0.000000

## 7. Required rounded values in manuscript text
- **PASS** `0.0393` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0585` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0454` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0009` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0407` found in: manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0600` found in: manuscript/results_q2_fast_track.md
- **PASS** `0.0888` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `-0.0717` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.8214` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0345` found in: manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0431` found in: manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0420` found in: manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0612` found in: manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0764` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `0.0801` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `3000` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/discussion_limitations_q2_fast_track.md, manuscript/introduction_q2_fast_track.md, manuscript/methodology_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `92` found in: manuscript/abstract_conclusion_q2_fast_track.md, manuscript/introduction_q2_fast_track.md, manuscript/methodology_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `2100` found in: manuscript/methodology_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md
- **PASS** `450` found in: manuscript/methodology_q2_fast_track.md, manuscript/outline_q2_fast_track.md, manuscript/results_q2_fast_track.md

## 8. Risky phrase scan
- **CHECK** risky phrase `prove` found in `manuscript/abstract_conclusion_q2_fast_track.md`
- **CHECK** risky phrase `strictly monotonic` found in `manuscript/discussion_limitations_q2_fast_track.md`
- **CHECK** risky phrase `field-ready` found in `manuscript/discussion_limitations_q2_fast_track.md`
- **CHECK** risky phrase `prove` found in `manuscript/discussion_limitations_q2_fast_track.md`
- **CHECK** risky phrase `prove` found in `manuscript/introduction_q2_fast_track.md`
- **CHECK** risky phrase `strictly monotonic` found in `manuscript/methodology_q2_fast_track.md`
- **CHECK** risky phrase `prove` found in `manuscript/methodology_q2_fast_track.md`
- **CHECK** risky phrase `strictly monotonic` found in `manuscript/outline_q2_fast_track.md`
- **CHECK** risky phrase `prove` found in `manuscript/outline_q2_fast_track.md`
- **CHECK** risky phrase `strictly monotonic` found in `manuscript/results_q2_fast_track.md`
- **CHECK** risky phrase `prove` found in `manuscript/results_q2_fast_track.md`