# Final Evidence Synthesis for Paper Results

- Generated at: `2026-06-30T16:08:28`

## 1. Final technical decision

The current evidence supports using `full physics features + ridge` as the main paper model.

Paired healthy-baseline normalization should not be used as the main method. The clean paired-baseline control shows that input-only matching cannot sufficiently solve high-damage underestimation or zero/damaged separability.

## 2. Paper-level decision table

| evidence_block | best_or_key_result | main_metrics | paper_role | decision |
| --- | --- | --- | --- | --- |
| Physics feature ablation | full + ridge | test_MAE=0.0471068; test_RMSE=0.0714182; n_features=92 | Main positive result | Use as the main empirical evidence for physics-informed feature construction. |
| Dirty paired healthy-baseline diagnosis | extra_trees | test_MAE=0.0861017; high_MAE=0.0894426; high_under=0.97561; classifier_AUC=0.666045; zero_FA=0.5 | Diagnostic result | Do not use as main method because response-feature matching may contaminate baseline selection. |
| Clean paired healthy-baseline diagnosis | extra_trees | test_MAE=0.0909449; high_MAE=0.092507; high_under=0.97561; classifier_AUC=0.628731; zero_FA=1 | Negative/control evidence | Use as limitation evidence: clean input-only matching does not solve high-damage underestimation or zero/damaged separation. |
| Final method selection | full physics features + ridge | Selected by interpretability, stability, and ablation consistency rather than by paired-baseline variants. | Final paper conclusion | Main paper should focus on physics-informed feature ablation and damage-stratified reliability diagnosis. |

## 3. Key physics-ablation result

- source: `results/tables/physics_ablation/debug_plus_500/ablation_model_comparison.csv`
- best_feature_set: `full`
- best_estimator: `ridge`
- best_test_mae: `0.0471068`
- best_test_rmse: `0.0714182`
- best_n_features: `92`
- best_mean_true_damage: `0.0645186`
- best_mean_pred_damage: `0.0794733`

## 4. Paired-baseline diagnostic snapshot

- best_regression_result__test_mae: `0.0861017`
- best_regression_result__high_mae: `0.0894426`
- best_regression_result__high_bias_pred_minus_true: `-0.0863528`
- best_regression_result__high_underestimation_ratio: `0.97561`
- best_zero_vs_damaged_classifier_result__roc_auc: `0.666045`
- best_zero_vs_damaged_classifier_result__zero_false_alarm_ratio: `0.5`

## 5. Clean paired-baseline control snapshot

- best_regression_result__test_mae: `0.0909449`
- best_regression_result__high_mae: `0.092507`
- best_regression_result__high_bias_pred_minus_true: `-0.0914998`
- best_regression_result__high_underestimation_ratio: `0.97561`
- best_zero_vs_damaged_classifier_result__roc_auc: `0.628731`
- best_zero_vs_damaged_classifier_result__balanced_accuracy: `0.5`
- best_zero_vs_damaged_classifier_result__zero_false_alarm_ratio: `1`
- best_zero_vs_damaged_classifier_result__damaged_miss_ratio: `0`

## 6. Paper-ready interpretation

1. The main positive result is not that complex classifiers or paired normalization dominate. The defensible result is that physics-informed response descriptors combined with a simple regularized model provide stable damage-inference performance under sparse and noisy SHM conditions.

2. The damage-stratified diagnosis reveals a systematic limitation: high-damage cases are consistently underestimated. This should be presented as a reliability limitation, not hidden.

3. The clean paired-baseline experiment is useful as a control experiment. It shows that strict input-only healthy matching is insufficient, which prevents the paper from overclaiming healthy-baseline normalization.

4. The final paper should be framed as a method-and-diagnosis study: physics-informed feature construction, ablation validation, repeated-split robustness, and damage-stratified reliability analysis.

## 7. Available metric files detected

| relative_path | n_rows | metric_columns | best_by_test_mae | best_feature_set | best_estimator | best_by_high_mae |
| --- | --- | --- | --- | --- | --- | --- |
| results/tables/all_model_comparison_after_sklearn.csv | 12 | test_mae, test_rmse | 0.081699 |  | unknown | nan |
| results/tables/case_level_any_damage/debug_plus_500/case_level_any_damage_metrics_threshold_0p5.csv | 12 | roc_auc, pr_auc, balanced_accuracy, zero_false_alarm_ratio, damaged_miss_ratio | nan | nan | nan | nan |
| results/tables/case_level_any_damage/debug_plus_500/case_level_any_damage_selected_threshold_metrics.csv | 4 | roc_auc, pr_auc, balanced_accuracy, zero_false_alarm_ratio, damaged_miss_ratio | nan | nan | nan | nan |
| results/tables/clean_paired_baseline/debug_plus_500/clean_paired_classifier_results.csv | 3 | roc_auc, pr_auc, balanced_accuracy, zero_false_alarm_ratio, damaged_miss_ratio | nan | nan | nan | nan |
| results/tables/clean_paired_baseline/debug_plus_500/clean_paired_regression_results.csv | 4 | test_mae, test_rmse, high_mae, high_underestimation_ratio | 0.0909449 |  | extra_trees | 0.0896078 |
| results/tables/damage_aware_weighting/debug_plus_500/damage_aware_model_comparison.csv | 108 | val_mae, val_rmse, val_high_mae, val_zero_mae, test_mae, test_rmse, test_zero_mae, test_low_mae, test_medium_mae, test_high_mae, test_damaged_mae | 0.0491702 | full | ridge | nan |
| results/tables/damage_stratified/debug_plus_500/damage_stratified_runs.csv | 90 | overall_mae, overall_rmse | nan | nan | nan | nan |
| results/tables/fair_baseline_500_comparison.csv | 12 | test_mae, test_rmse | 0.0471068 |  | ridge | nan |
| results/tables/gated_calibration/debug_plus_500/gated_calibration_model_comparison.csv | 30 | test_mae, test_rmse, test_damaged_mae, test_zero_mae, test_low_mae, test_medium_mae, test_high_mae, val_mae, val_rmse, val_damaged_mae, val_zero_mae, val_low_mae | 0.0463787 | full |  | nan |
| results/tables/model_comparison_physics_feature_step.csv | 8 | test_mae, test_rmse | 0.081699 |  | unknown | nan |
| results/tables/paired_baseline/debug_plus_500/paired_baseline_regression_metrics.csv | 9 | test_mae, test_rmse, high_mae, high_underestimation_ratio | 0.0521111 |  | random_forest | 0.0501343 |
| results/tables/paired_baseline/debug_plus_500/paired_baseline_zero_damaged_classification_metrics.csv | 9 | roc_auc, pr_auc, balanced_accuracy, zero_false_alarm_ratio, damaged_miss_ratio | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/ablation_model_comparison.csv | 28 | test_mae, test_rmse | 0.0815053 | ridge | ridge | nan |
| results/tables/physics_ablation/debug_plus_100/ablation_model_comparison_fixed.csv | 28 | test_mae, test_rmse | 0.0815053 | response_correlation | ridge | nan |
| results/tables/physics_ablation/debug_plus_100/full/elasticnet/candidate_validation_records.csv | 20 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/full/gradient_boosting/candidate_validation_records.csv | 18 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/full/model_comparison.csv | 4 | test_mae, test_rmse | 0.0817229 |  | random_forest | nan |
| results/tables/physics_ablation/debug_plus_100/full/random_forest/candidate_validation_records.csv | 12 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/full/ridge/candidate_validation_records.csv | 6 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/no_meta/elasticnet/candidate_validation_records.csv | 20 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/no_meta/gradient_boosting/candidate_validation_records.csv | 18 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/no_meta/model_comparison.csv | 4 | test_mae, test_rmse | 0.0816557 |  | random_forest | nan |
| results/tables/physics_ablation/debug_plus_100/no_meta/random_forest/candidate_validation_records.csv | 12 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/no_meta/ridge/candidate_validation_records.csv | 6 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/physics_no_meta_core/elasticnet/candidate_validation_records.csv | 20 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/physics_no_meta_core/gradient_boosting/candidate_validation_records.csv | 18 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/physics_no_meta_core/model_comparison.csv | 4 | test_mae, test_rmse | 0.0816557 |  | random_forest | nan |
| results/tables/physics_ablation/debug_plus_100/physics_no_meta_core/random_forest/candidate_validation_records.csv | 12 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/physics_no_meta_core/ridge/candidate_validation_records.csv | 6 | val_mae | nan | nan | nan | nan |
| results/tables/physics_ablation/debug_plus_100/response_basic_only/elasticnet/candidate_validation_records.csv | 20 | val_mae | nan | nan | nan | nan |

## 8. Result-file inventory

| relative_path | suffix | size_kb |
| --- | --- | --- |
| results/tables/all_model_comparison_after_sklearn.csv | .csv | 4.52 |
| results/tables/case_level_any_damage/debug_plus_500/case_level_any_damage_metrics_threshold_0p5.csv | .csv | 2.42 |
| results/tables/case_level_any_damage/debug_plus_500/case_level_any_damage_report.md | .md | 4.34 |
| results/tables/case_level_any_damage/debug_plus_500/case_level_any_damage_selected_threshold_metrics.csv | .csv | 1.29 |
| results/tables/case_level_any_damage/debug_plus_500/case_level_any_damage_summary.json | .json | 0.39 |
| results/tables/case_level_any_damage/debug_plus_500/case_level_label_counts.csv | .csv | 0.15 |
| results/tables/case_level_any_damage/debug_plus_500/resolved_feature_names.csv | .csv | 2.61 |
| results/tables/clean_paired_baseline/debug_plus_500/clean_matching_features.csv | .csv | 0.28 |
| results/tables/clean_paired_baseline/debug_plus_500/clean_paired_baseline_feature_names.csv | .csv | 10.66 |
| results/tables/clean_paired_baseline/debug_plus_500/clean_paired_baseline_report.md | .md | 3.83 |
| results/tables/clean_paired_baseline/debug_plus_500/clean_paired_classifier_results.csv | .csv | 0.7 |
| results/tables/clean_paired_baseline/debug_plus_500/clean_paired_regression_results.csv | .csv | 0.83 |
| results/tables/damage_aware_weighting/debug_plus_500/damage_aware_bin_summary.csv | .csv | 102.89 |
| results/tables/damage_aware_weighting/debug_plus_500/damage_aware_model_comparison.csv | .csv | 66.53 |
| results/tables/damage_aware_weighting/debug_plus_500/damage_aware_report.md | .md | 7.46 |
| results/tables/damage_aware_weighting/debug_plus_500/damage_aware_run_summary.json | .json | 0.65 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__elasticnet__balanced_bins/predictions_test.csv | .csv | 30.42 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__elasticnet__continuous/predictions_test.csv | .csv | 29.67 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__elasticnet__high_only/predictions_test.csv | .csv | 29.63 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__elasticnet__moderate/predictions_test.csv | .csv | 29.24 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__elasticnet__none/predictions_test.csv | .csv | 27.96 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__elasticnet__strong/predictions_test.csv | .csv | 28.66 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__random_forest__balanced_bins/predictions_test.csv | .csv | 33.11 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__random_forest__continuous/predictions_test.csv | .csv | 32.07 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__random_forest__high_only/predictions_test.csv | .csv | 31.8 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__random_forest__moderate/predictions_test.csv | .csv | 31.49 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__random_forest__none/predictions_test.csv | .csv | 30.42 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__random_forest__strong/predictions_test.csv | .csv | 30.9 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__ridge__balanced_bins/predictions_test.csv | .csv | 28.22 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__ridge__continuous/predictions_test.csv | .csv | 27.8 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__ridge__high_only/predictions_test.csv | .csv | 27.74 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__ridge__moderate/predictions_test.csv | .csv | 27.51 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__ridge__none/predictions_test.csv | .csv | 25.72 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/full__ridge__strong/predictions_test.csv | .csv | 26.81 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__elasticnet__balanced_bins/predictions_test.csv | .csv | 30.9 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__elasticnet__continuous/predictions_test.csv | .csv | 30.64 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__elasticnet__high_only/predictions_test.csv | .csv | 30.44 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__elasticnet__moderate/predictions_test.csv | .csv | 30.07 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__elasticnet__none/predictions_test.csv | .csv | 28.44 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__elasticnet__strong/predictions_test.csv | .csv | 29.49 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__random_forest__balanced_bins/predictions_test.csv | .csv | 34.04 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__random_forest__continuous/predictions_test.csv | .csv | 32.96 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__random_forest__high_only/predictions_test.csv | .csv | 32.65 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__random_forest__moderate/predictions_test.csv | .csv | 32.36 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__random_forest__none/predictions_test.csv | .csv | 31.35 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__random_forest__strong/predictions_test.csv | .csv | 31.85 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__ridge__balanced_bins/predictions_test.csv | .csv | 28.73 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__ridge__continuous/predictions_test.csv | .csv | 28.74 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__ridge__high_only/predictions_test.csv | .csv | 28.53 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__ridge__moderate/predictions_test.csv | .csv | 28.01 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__ridge__none/predictions_test.csv | .csv | 26.36 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/no_meta__ridge__strong/predictions_test.csv | .csv | 27.83 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/response_basic_only__elasticnet__balanced_bins/predictions_test.csv | .csv | 36.03 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/response_basic_only__elasticnet__continuous/predictions_test.csv | .csv | 34.98 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/response_basic_only__elasticnet__high_only/predictions_test.csv | .csv | 34.72 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/response_basic_only__elasticnet__moderate/predictions_test.csv | .csv | 34.62 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/response_basic_only__elasticnet__none/predictions_test.csv | .csv | 33.27 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/response_basic_only__elasticnet__strong/predictions_test.csv | .csv | 33.97 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/response_basic_only__random_forest__balanced_bins/predictions_test.csv | .csv | 37.56 |
| results/tables/damage_aware_weighting/debug_plus_500/predictions/response_basic_only__random_forest__continuous/predictions_test.csv | .csv | 36.49 |
