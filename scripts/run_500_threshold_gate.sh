#!/usr/bin/env bash
set -euo pipefail

# Threshold-gated calibration on the 500-case physics-feature dataset.
# 在 500 工况物理特征数据集上运行阈值门控校准实验。

FEATURES="data_processed/debug_plus_500_physics_features_mlp.npz"
FEATURE_NAMES="data_processed/debug_plus_500_physics_feature_names.csv"
OUTPUT_DIR="results/tables/threshold_gated_calibration/debug_plus_500"
FIGURES_DIR="results/figures/threshold_gated_calibration/debug_plus_500"

printf "Step 0: check required files\n"
test -f "$FEATURES"
test -f "$FEATURE_NAMES"
ls -lh "$FEATURES" "$FEATURE_NAMES"

printf "\nStep 1: run threshold-gated calibration\n"
python -m src.evaluation.run_threshold_gated_calibration \
  --features "$FEATURES" \
  --feature-names "$FEATURE_NAMES" \
  --output-dir "$OUTPUT_DIR" \
  --figures-dir "$FIGURES_DIR" \
  --feature-sets full no_meta response_spatial \
  --classifiers logistic_balanced random_forest_balanced \
  --calibrators residual_ridge direct_ridge \
  --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 \
  --high-weights 2 4 6 8 \
  --ridge-alpha 300.0 \
  --max-damage 0.5 \
  --target-overall-mae 0.050 \
  --target-high-mae 0.085 \
  --target-zero-fa 0.250 \
  --random-seed 42

printf "\nStep 2: print concise report\n"
cat "$OUTPUT_DIR/threshold_gated_report.md"

printf "\nStep 3: show generated outputs\n"
find "$OUTPUT_DIR" -maxdepth 1 -type f | sort
find "$FIGURES_DIR" -maxdepth 1 -type f | sort

printf "\nDone.\n"
