#!/usr/bin/env bash
set -euo pipefail

# English:
# Run damage-stratified evaluation on the 500-case physics-feature dataset.
#
# 中文：
# 在 500-case 物理特征数据集上运行“按损伤等级分层评价”。
# 请在项目根目录执行：
# bash scripts/run_500_damage_stratified.sh

echo "Step 0: check required files"
test -f data_processed/debug_plus_500_physics_features_mlp.npz
test -f data_processed/debug_plus_500_physics_feature_names.csv
echo "Required files found."

echo ""
echo "Step 1: run damage-stratified evaluation"
python -m src.evaluation.run_damage_stratified_analysis \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --configs \
    full:ridge \
    no_meta:ridge \
    full:elasticnet \
    no_meta:elasticnet \
    response_basic_only:ridge \
    response_basic_only:elasticnet \
    response_spatial:ridge \
    response_frequency:ridge \
    response_correlation:ridge \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --train-size 0.70 \
  --val-size 0.15 \
  --max-damage 0.5 \
  --high-threshold 0.20 \
  --output-dir results/tables/damage_stratified/debug_plus_500 \
  --figures-dir results/figures/damage_stratified/debug_plus_500

echo ""
echo "Step 2: show key outputs"
echo "Report:"
echo "results/tables/damage_stratified/debug_plus_500/damage_stratified_report.md"

echo ""
echo "Top rows of config summary:"
python - <<'PY'
import pandas as pd
path = "results/tables/damage_stratified/debug_plus_500/damage_stratified_config_summary.csv"
df = pd.read_csv(path)
cols = [
    "feature_set",
    "model",
    "overall_mae_mean",
    "overall_mae_std",
    "overall_rmse_mean",
    "overall_bias_mean",
    "high_damage_mae_mean",
    "high_damage_underestimation_ratio_mean",
]
print(df[cols].head(20).to_string(index=False))
PY

echo ""
echo "Step 3: list generated figures"
find results/figures/damage_stratified/debug_plus_500 -maxdepth 1 -type f | sort

echo ""
echo "Damage-stratified experiment finished."
