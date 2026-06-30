#!/usr/bin/env bash
set -euo pipefail

# Repeated-split robustness experiment for the 500-case physics feature dataset.
# 中文说明：本脚本用于运行 500-case 的多随机种子稳健性验证。
# 运行位置：项目根目录，即能看到 src/、data_processed/、results/ 的那一层目录。

echo "Step 0: check required files"
test -f data_processed/debug_plus_500_physics_features_mlp.npz
test -f data_processed/debug_plus_500_physics_feature_names.csv
echo "Required files found."

echo
echo "Step 1: run repeated-split robustness experiment"
python -m src.evaluation.run_seed_robustness \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --output-dir results/tables/seed_robustness/debug_plus_500 \
  --figures-dir results/figures/seed_robustness/debug_plus_500 \
  --models ridge elasticnet random_forest \
  --feature-sets full no_meta response_basic_only response_frequency response_correlation response_spatial \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --clip-predictions \
  --max-damage 0.5

echo
echo "Step 2: print key outputs"
echo "----- seed_robustness_summary.csv -----"
cat results/tables/seed_robustness/debug_plus_500/seed_robustness_summary.csv

echo
echo "----- seed_robustness_per_seed_best.csv -----"
cat results/tables/seed_robustness/debug_plus_500/seed_robustness_per_seed_best.csv

echo
echo "----- seed_robustness_report.md -----"
cat results/tables/seed_robustness/debug_plus_500/seed_robustness_report.md

echo
echo "Step 3: list generated figures"
find results/figures/seed_robustness/debug_plus_500 -maxdepth 1 -type f | sort

echo
echo "Done."
