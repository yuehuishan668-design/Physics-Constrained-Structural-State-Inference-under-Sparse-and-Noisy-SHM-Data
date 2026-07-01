# Damage-stratified Evaluation Summary

## 1. Dataset

- Source layout: `F_train/F_val/F_test`
- Number of samples: `3000`
- Number of features: `92`
- Number of stories / outputs: `4`
- Number of seeds: `10`

## 2. Overall best configuration

- Best configuration: `full + ridge`
- Overall MAE mean: `0.040679`
- Overall MAE std: `0.001233`
- Overall RMSE mean: `0.060010`
- Mean bias: `0.005768`

## 3. Overall ranking

| rank | feature_set | model | n_runs | overall_mae_mean | overall_mae_std | overall_rmse_mean | overall_bias_mean | high_damage_mae_mean | high_damage_underestimation_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | full | ridge | 10 | 0.040679 | 0.001233 | 0.060010 | 0.005768 | 0.088806 | 0.821399 |
| 2 | no_meta | ridge | 10 | 0.043867 | 0.001149 | 0.063090 | 0.006052 | 0.094808 | 0.849505 |
| 3 | response_basic_only | ridge | 10 | 0.059847 | 0.001553 | 0.079326 | 0.003314 | 0.143386 | 0.956037 |
| 4 | full | elasticnet | 10 | 0.070943 | 0.001220 | 0.089584 | -0.000438 | 0.178716 | 0.996729 |

## 4. Damage-bin metrics

| feature_set | model | bin | n_entries_mean | mae_mean | rmse_mean | bias_mean | underestimation_ratio_mean | mean_true_mean | mean_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | elasticnet | all | 1804.000000 | 0.070943 | 0.089584 | -0.000438 | 0.284202 | 0.060588 | 0.060149 |
| full | elasticnet | damaged | 541.500000 | 0.119736 | 0.141167 | -0.118090 | 0.946768 | 0.201871 | 0.083781 |
| full | elasticnet | high | 275.700000 | 0.178716 | 0.185310 | -0.178464 | 0.996729 | 0.275775 | 0.097311 |
| full | elasticnet | low | 92.100000 | 0.021018 | 0.026199 | -0.012956 | 0.724879 | 0.075454 | 0.062498 |
| full | elasticnet | medium | 173.700000 | 0.078251 | 0.084880 | -0.077778 | 0.984936 | 0.151389 | 0.073610 |
| full | elasticnet | zero | 1262.500000 | 0.050018 | 0.054017 | 0.050018 | 0.000000 | 0.000000 | 0.050018 |
| full | ridge | all | 1804.000000 | 0.040679 | 0.060010 | 0.005768 | 0.235477 | 0.060588 | 0.066355 |
| full | ridge | damaged | 541.500000 | 0.066294 | 0.087938 | -0.049991 | 0.784524 | 0.201871 | 0.151881 |
| full | ridge | high | 275.700000 | 0.088806 | 0.110392 | -0.071656 | 0.821399 | 0.275775 | 0.204119 |
| full | ridge | low | 92.100000 | 0.028854 | 0.040132 | -0.004324 | 0.591345 | 0.075454 | 0.071131 |
| full | ridge | medium | 173.700000 | 0.050235 | 0.061939 | -0.039574 | 0.828299 | 0.151389 | 0.111814 |
| full | ridge | zero | 1262.500000 | 0.029680 | 0.042712 | 0.029680 | 0.000000 | 0.000000 | 0.029680 |
| no_meta | ridge | all | 1804.000000 | 0.043867 | 0.063090 | 0.006052 | 0.237472 | 0.060588 | 0.066640 |
| no_meta | ridge | damaged | 541.500000 | 0.071058 | 0.091680 | -0.054891 | 0.790965 | 0.201871 | 0.146981 |
| no_meta | ridge | high | 275.700000 | 0.094808 | 0.114577 | -0.079874 | 0.849505 | 0.275775 | 0.195900 |
| no_meta | ridge | low | 92.100000 | 0.031656 | 0.043273 | -0.002988 | 0.552016 | 0.075454 | 0.072466 |
| no_meta | ridge | medium | 173.700000 | 0.054065 | 0.065596 | -0.042445 | 0.824167 | 0.151389 | 0.108944 |
| no_meta | ridge | zero | 1262.500000 | 0.032190 | 0.045560 | 0.032190 | 0.000000 | 0.000000 | 0.032190 |
| response_basic_only | ridge | all | 1804.000000 | 0.059847 | 0.079326 | 0.003314 | 0.265576 | 0.060588 | 0.063902 |
| response_basic_only | ridge | damaged | 541.500000 | 0.099517 | 0.119799 | -0.088852 | 0.884809 | 0.201871 | 0.113020 |
| response_basic_only | ridge | high | 275.700000 | 0.143386 | 0.154357 | -0.134040 | 0.956037 | 0.275775 | 0.141735 |
| response_basic_only | ridge | low | 92.100000 | 0.027344 | 0.035850 | -0.007855 | 0.604262 | 0.075454 | 0.067600 |
| response_basic_only | ridge | medium | 173.700000 | 0.067866 | 0.078496 | -0.059720 | 0.919838 | 0.151389 | 0.091669 |
| response_basic_only | ridge | zero | 1262.500000 | 0.042827 | 0.053230 | 0.042827 | 0.000000 | 0.000000 | 0.042827 |

## 5. High-damage ranking

| feature_set | model | mae_mean | rmse_mean | bias_mean | underestimation_ratio_mean | mean_true_mean | mean_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | ridge | 0.088806 | 0.110392 | -0.071656 | 0.821399 | 0.275775 | 0.204119 |
| no_meta | ridge | 0.094808 | 0.114577 | -0.079874 | 0.849505 | 0.275775 | 0.195900 |
| response_basic_only | ridge | 0.143386 | 0.154357 | -0.134040 | 0.956037 | 0.275775 | 0.141735 |
| full | elasticnet | 0.178716 | 0.185310 | -0.178464 | 0.996729 | 0.275775 | 0.097311 |

## 6. Story-level summary

| feature_set | model | story | mae_mean | rmse_mean | bias_mean | underestimation_ratio_mean | mean_true | mean_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | elasticnet | 1 | 0.071650 | 0.091206 | -0.002297 | 0.290022 | 0.059366 | 0.057069 |
| full | elasticnet | 2 | 0.075615 | 0.094607 | -0.000827 | 0.283814 | 0.061421 | 0.060594 |
| full | elasticnet | 3 | 0.070851 | 0.088882 | 0.001141 | 0.274723 | 0.057483 | 0.058624 |
| full | elasticnet | 4 | 0.065656 | 0.083061 | 0.000231 | 0.288248 | 0.064080 | 0.064311 |
| full | ridge | 1 | 0.054733 | 0.075119 | 0.004353 | 0.255654 | 0.059366 | 0.063719 |
| full | ridge | 2 | 0.042184 | 0.060996 | 0.007635 | 0.235698 | 0.061421 | 0.069056 |
| full | ridge | 3 | 0.031096 | 0.047895 | 0.005999 | 0.219512 | 0.057483 | 0.063482 |
| full | ridge | 4 | 0.034703 | 0.052164 | 0.005083 | 0.231042 | 0.064080 | 0.069163 |
| no_meta | ridge | 1 | 0.056674 | 0.077205 | 0.003651 | 0.254545 | 0.059366 | 0.063017 |
| no_meta | ridge | 2 | 0.047496 | 0.066386 | 0.008038 | 0.237694 | 0.061421 | 0.069460 |
| no_meta | ridge | 3 | 0.033841 | 0.049938 | 0.006989 | 0.219290 | 0.057483 | 0.064472 |
| no_meta | ridge | 4 | 0.037458 | 0.055095 | 0.005530 | 0.238359 | 0.064080 | 0.069610 |
| response_basic_only | ridge | 1 | 0.061215 | 0.081553 | 0.000955 | 0.265188 | 0.059366 | 0.060321 |
| response_basic_only | ridge | 2 | 0.061291 | 0.081060 | 0.004597 | 0.261197 | 0.061421 | 0.066018 |
| response_basic_only | ridge | 3 | 0.055901 | 0.073852 | 0.004583 | 0.256541 | 0.057483 | 0.062066 |
| response_basic_only | ridge | 4 | 0.060980 | 0.080390 | 0.003122 | 0.279379 | 0.064080 | 0.067202 |

## 7. Preliminary interpretation

- If the high-damage MAE is much larger than the overall MAE, the model is not failing on average-case fitting; it is failing mainly on severe-damage estimation.
- If the high-damage bias is negative, the model has systematic high-damage underestimation, which is important for structural safety interpretation.
- If `full + ridge` remains best overall but not best in the high-damage bin, the next paper argument should distinguish average accuracy from safety-critical damage sensitivity.
- If all configurations underestimate high damage, the next methodological step should be weighted training, damage-aware loss, or data rebalancing.

中文解释：

- 如果高损伤 MAE 明显大于 overall MAE，问题不是平均拟合失败，而是严重损伤识别不足。
- 如果高损伤 bias 为负，说明模型存在高损伤低估风险，这对结构安全判断非常关键。
- 如果 full + ridge 仍是整体最优，但高损伤区间不是最优，论文中需要区分平均精度与安全关键区间敏感性。
- 如果所有模型都低估高损伤，下一步应考虑加权训练、损伤感知损失函数或数据重平衡。
