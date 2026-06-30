#!/usr/bin/env bash
set -euo pipefail

echo "Step 0: check required 500-case physics feature files"

REQUIRED_FILES=(
  "data_processed/debug_plus_500_physics_features_mlp.npz"
  "data_processed/debug_plus_500_physics_feature_names.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Missing required file: $file"
    echo "Please run the 500-case physics feature extraction step first."
    exit 1
  fi
  echo "$file"
done

echo ""
echo "Step 1: run damage-gated calibration"

python -m src.evaluation.run_damage_gated_calibration \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --output-dir results/tables/gated_calibration/debug_plus_500 \
  --figures-dir results/figures/gated_calibration/debug_plus_500 \
  --dataset-tag debug_plus_500 \
  --feature-sets full no_meta response_spatial response_basic_only response_correlation \
  --classifiers logistic_balanced random_forest_balanced \
  --calibrators direct_ridge residual_ridge high_sensitive_residual_ridge \
  --clip-predictions \
  --max-damage 0.5 \
  --random-seed 42 \
  --top-k 20

echo ""
echo "Step 2: show generated summary files"
ls -lh results/tables/gated_calibration/debug_plus_500/gated_calibration_model_comparison.csv
ls -lh results/tables/gated_calibration/debug_plus_500/gated_calibration_report.md

echo ""
echo "Step 3: print markdown report"
cat results/tables/gated_calibration/debug_plus_500/gated_calibration_report.md

echo ""
echo "Step 4: list generated figures"
find results/figures/gated_calibration/debug_plus_500 -maxdepth 1 -type f | sort

echo ""
echo "Damage-gated calibration shell script completed."
