#!/usr/bin/env bash
set -euo pipefail

# Damage-aware weighted training for the 500-case dataset.
# 500 样本数据集的损伤感知加权训练脚本。

echo "Step 0: check required 500-case feature files"

required_files=(
  "data_processed/debug_plus_500_physics_features_mlp.npz"
  "data_processed/debug_plus_500_physics_feature_names.csv"
)

for f in "${required_files[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Missing required file: $f"
    exit 1
  fi
  echo "$f"
done

echo ""
echo "Step 1: run damage-aware weighted training"

python -m src.evaluation.run_damage_aware_weighting \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --output-dir results/tables/damage_aware_weighting/debug_plus_500 \
  --figures-dir results/figures/damage_aware_weighting/debug_plus_500 \
  --feature-sets full no_meta response_spatial response_basic_only response_frequency response_correlation \
  --models ridge elasticnet random_forest \
  --weight-schemes none moderate strong high_only continuous balanced_bins \
  --clip-predictions \
  --max-damage 0.5 \
  --low-threshold 0.10 \
  --high-threshold 0.20 \
  --high-mae-weight 0.50 \
  --high-under-bias-weight 0.50 \
  --zero-mae-weight 0.10 \
  --top-k 20 \
  --random-seed 42

echo ""
echo "Step 2: show main report"

cat results/tables/damage_aware_weighting/debug_plus_500/damage_aware_report.md

echo ""
echo "Step 3: show top comparison rows"

head -n 30 results/tables/damage_aware_weighting/debug_plus_500/damage_aware_model_comparison.csv

echo ""
echo "Damage-aware weighting experiment completed."
echo "Please send back:"
echo "1) terminal output from the top configurations section"
echo "2) results/tables/damage_aware_weighting/debug_plus_500/damage_aware_report.md"
echo "3) head -n 30 results/tables/damage_aware_weighting/debug_plus_500/damage_aware_model_comparison.csv"
echo "4) these figures:"
echo "   results/figures/damage_aware_weighting/debug_plus_500/top_overall_test_mae.png"
echo "   results/figures/damage_aware_weighting/debug_plus_500/top_high_damage_test_mae.png"
echo "   results/figures/damage_aware_weighting/debug_plus_500/overall_vs_high_damage_tradeoff.png"
echo "   results/figures/damage_aware_weighting/debug_plus_500/high_damage_bias_underestimation.png"
echo "   results/figures/damage_aware_weighting/debug_plus_500/best_overall_true_vs_predicted.png"
echo "   results/figures/damage_aware_weighting/debug_plus_500/best_high_damage_true_vs_predicted.png"
