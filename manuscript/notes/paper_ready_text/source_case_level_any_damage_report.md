# Case-level Any-damage Diagnosis

## 1. Input files
- Features: `data_processed/debug_plus_500_physics_features_mlp.npz`
- Split dataset: `data_processed/debug_plus_500_split_normalized.npz`
- Feature names: `data_processed/debug_plus_500_physics_feature_names.csv`
- Feature shape example: `(350, 92)`

## 2. Case-level any-damage counts
| split | zero_cases | damaged_cases | total_cases | damaged_ratio |
| --- | --- | --- | --- | --- |
| train | 65 | 285 | 350 | 0.814286 |
| val | 11 | 64 | 75 | 0.853333 |
| test | 8 | 67 | 75 | 0.893333 |

## 3. Metrics at threshold 0.5
| model | split | threshold | n | zero_n | damaged_n | roc_auc | pr_auc | balanced_accuracy | precision | recall | f1 | zero_false_alarm_ratio | damaged_miss_ratio | mean_proba_zero | mean_proba_damaged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extra_trees_balanced | test | 0.5 | 75 | 8 | 67 | 0.63806 | 0.933992 | 0.602612 | 0.914286 | 0.955224 | 0.934307 | 0.75 | 0.0447761 | 0.674 | 0.741193 |
| random_forest_balanced | test | 0.5 | 75 | 8 | 67 | 0.572761 | 0.917282 | 0.485075 | 0.890411 | 0.970149 | 0.928571 | 1 | 0.0298507 | 0.761573 | 0.767648 |
| gradient_boosting | test | 0.5 | 75 | 8 | 67 | 0.527985 | 0.908143 | 0.517724 | 0.897059 | 0.910448 | 0.903704 | 0.875 | 0.0895522 | 0.835836 | 0.834304 |
| logistic_l2_balanced | test | 0.5 | 75 | 8 | 67 | 0.501866 | 0.89398 | 0.508396 | 0.895833 | 0.641791 | 0.747826 | 0.625 | 0.358209 | 0.557814 | 0.578985 |
| random_forest_balanced | train | 0.5 | 350 | 65 | 285 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.322453 | 0.888535 |
| extra_trees_balanced | train | 0.5 | 350 | 65 | 285 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.100162 | 0.899838 |
| gradient_boosting | train | 0.5 | 350 | 65 | 285 | 1 | 1 | 0.992308 | 0.996503 | 1 | 0.998249 | 0.0153846 | 0 | 0.275271 | 0.937058 |
| logistic_l2_balanced | train | 0.5 | 350 | 65 | 285 | 0.819055 | 0.953202 | 0.723617 | 0.922374 | 0.708772 | 0.801587 | 0.261538 | 0.291228 | 0.383682 | 0.614097 |
| logistic_l2_balanced | val | 0.5 | 75 | 11 | 64 | 0.775568 | 0.955524 | 0.737216 | 0.954545 | 0.65625 | 0.777778 | 0.181818 | 0.34375 | 0.372684 | 0.572969 |
| gradient_boosting | val | 0.5 | 75 | 11 | 64 | 0.713068 | 0.940184 | 0.52983 | 0.861111 | 0.96875 | 0.911765 | 0.909091 | 0.03125 | 0.759014 | 0.868378 |
| random_forest_balanced | val | 0.5 | 75 | 11 | 64 | 0.573864 | 0.909723 | 0.545455 | 0.864865 | 1 | 0.927536 | 0.909091 | 0 | 0.731386 | 0.768975 |
| extra_trees_balanced | val | 0.5 | 75 | 11 | 64 | 0.569602 | 0.898151 | 0.514205 | 0.857143 | 0.9375 | 0.895522 | 0.909091 | 0.0625 | 0.69747 | 0.73394 |

## 4. Validation-selected threshold and test result
| model | split | threshold | n | zero_n | damaged_n | roc_auc | pr_auc | balanced_accuracy | precision | recall | f1 | zero_false_alarm_ratio | damaged_miss_ratio | mean_proba_zero | mean_proba_damaged | selected_threshold | selection_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extra_trees_balanced | test_selected_threshold | 0.8 | 75 | 8 | 67 | 0.63806 | 0.933992 | 0.653918 | 0.966667 | 0.432836 | 0.597938 | 0.125 | 0.567164 | 0.674 | 0.741193 | 0.8 | no_feasible_threshold_choose_lowest_false_alarm |
| random_forest_balanced | test_selected_threshold | 0.8 | 75 | 8 | 67 | 0.572761 | 0.917282 | 0.56903 | 0.928571 | 0.38806 | 0.547368 | 0.25 | 0.61194 | 0.761573 | 0.767648 | 0.8 | no_feasible_threshold_choose_lowest_false_alarm |
| gradient_boosting | test_selected_threshold | 0.8 | 75 | 8 | 67 | 0.527985 | 0.908143 | 0.465485 | 0.885246 | 0.80597 | 0.84375 | 0.875 | 0.19403 | 0.835836 | 0.834304 | 0.8 | no_feasible_threshold_choose_lowest_false_alarm |
| logistic_l2_balanced | test_selected_threshold | 0.7 | 75 | 8 | 67 | 0.501866 | 0.89398 | 0.384328 | 0.818182 | 0.268657 | 0.404494 | 0.5 | 0.731343 | 0.557814 | 0.578985 | 0.7 | feasible_under_false_alarm_limit |

## 5. Best diagnostic model
- Best model by test PR-AUC: `extra_trees_balanced`

## 6. Preliminary interpretation
- Case-level any-damage separability is weak: the best test ROC-AUC is below 0.65.
- Under a low false-alarm operating point, damaged recall is usable; story-local alignment should be investigated next.
- Compare this result with the previous story-level zero-vs-damaged result to decide whether the bottleneck is global damage detection or story localization.
