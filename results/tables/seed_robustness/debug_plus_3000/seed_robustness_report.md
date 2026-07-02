# Seed Robustness Summary

## 1. Dataset

- Feature file: `data_processed/debug_plus_3000_physics_features_mlp.npz`
- Source layout: `F_train/F_val/F_test`
- Number of samples: `3000`
- Train/val/test per seed: `2100` / `450` / `450`
- Number of repeated seeds: `10`

## 2. Overall best configuration

- Best feature set: `full`
- Best model: `ridge`
- Mean Test MAE: `0.045426`
- Std Test MAE: `0.000865`
- Mean Test RMSE: `0.062085`
- Mean prediction bias: `0.000223`

## 3. Summary ranked by mean Test MAE

| rank_by_mean_test_mae | feature_set | model | n_runs | test_mae_mean | test_mae_std | test_mae_cv_pct | test_rmse_mean | bias_mean | mae_damaged_mean | mae_high_damage_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | full | ridge | 10 | 0.045426 | 0.000865 | 1.903908 | 0.062085 | 0.000223 | 0.064184 | 0.078755 |
| 2 | no_meta | ridge | 10 | 0.049016 | 0.000736 | 1.501785 | 0.065168 | 0.000389 | 0.068649 | 0.084106 |
| 3 | full | elasticnet | 10 | 0.057278 | 0.000787 | 1.373134 | 0.074120 | -0.000075 | 0.091342 | 0.118898 |
| 4 | no_meta | elasticnet | 10 | 0.059577 | 0.000707 | 1.186860 | 0.076630 | -0.000089 | 0.094902 | 0.123796 |
| 5 | full | random_forest | 10 | 0.081415 | 0.000743 | 0.912056 | 0.100497 | -0.000829 | 0.135757 | 0.183644 |
| 6 | no_meta | random_forest | 10 | 0.081416 | 0.000744 | 0.913946 | 0.100498 | -0.000829 | 0.135762 | 0.183646 |

## 4. Per-seed best model

| seed | feature_set | model | best_candidate | test_mae | test_rmse | bias_mean_pred_minus_true | mae_on_high_damage_entries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | full | ridge | ridge_alpha_0.01 | 0.044157 | 0.061155 | 0.000045 | 0.076301 |
| 1 | full | ridge | ridge_alpha_0.01 | 0.044979 | 0.060937 | 0.001108 | 0.077342 |
| 2 | full | ridge | ridge_alpha_0.01 | 0.046709 | 0.064222 | 0.001992 | 0.081180 |
| 3 | full | ridge | ridge_alpha_0.01 | 0.044791 | 0.060764 | 0.000660 | 0.078865 |
| 4 | full | ridge | ridge_alpha_0.01 | 0.046562 | 0.063529 | -0.001126 | 0.082300 |
| 5 | full | ridge | ridge_alpha_0.01 | 0.045741 | 0.062552 | 0.000140 | 0.077481 |
| 6 | full | ridge | ridge_alpha_0.01 | 0.045018 | 0.062117 | -0.000112 | 0.079903 |
| 7 | full | ridge | ridge_alpha_0.01 | 0.044961 | 0.062097 | 0.000851 | 0.077838 |
| 8 | full | ridge | ridge_alpha_0.01 | 0.046373 | 0.062783 | -0.000645 | 0.082671 |
| 9 | full | ridge | ridge_alpha_0.01 | 0.044970 | 0.060691 | -0.000681 | 0.073666 |

## 5. Preliminary interpretation template

- If the same feature set and model ranks first across most seeds, the conclusion is stable rather than split-specific.
- If the full feature set is only marginally better than no_meta, external metadata contributes limited incremental information.
- If high-damage MAE remains high, the next methodological problem is large-damage underestimation rather than average-case fitting.
- If Ridge remains competitive against nonlinear models, this supports using regularized physics-informed descriptors under limited-data SHM settings.

中文解释：

- 如果同一模型在多数 seed 下排名第一，说明结论不是某一次数据划分造成的偶然结果。
- 如果 full 和 no_meta 差距很小，说明外部元数据贡献有限，主体信息来自结构响应特征。
- 如果大损伤样本误差仍高，下一步问题不是继续堆模型，而是处理大损伤低估偏差。
- 如果 Ridge 稳定优于非线性模型，可以支撑“正则化物理特征模型适合小样本 noisy SHM”的论文主张。
