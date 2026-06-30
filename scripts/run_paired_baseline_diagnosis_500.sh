#!/usr/bin/env bash
set -e

echo "Step 0: check required files"

test -f data_processed/debug_plus_500_physics_features_mlp.npz
test -f data_processed/debug_plus_500_split_normalized.npz
test -f data_processed/debug_plus_500_physics_feature_names.csv

echo "Required files found."

echo ""
echo "Step 1: run paired healthy-baseline feature diagnosis"

python -m src.evaluation.run_paired_baseline_diagnosis \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --split data_processed/debug_plus_500_split_normalized.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --paired-output data_processed/debug_plus_500_paired_baseline_features.npz \
  --paired-feature-names-output data_processed/debug_plus_500_paired_baseline_feature_names.csv \
  --output-dir results/tables/paired_baseline/debug_plus_500 \
  --figures-dir results/figures/paired_baseline/debug_plus_500 \
  --k-neighbors 5 \
  --damage-threshold 0.05 \
  --high-threshold 0.20 \
  --seed 42

echo ""
echo "Step 2: print generated report"

cat results/tables/paired_baseline/debug_plus_500/paired_baseline_diagnosis_report.md

echo ""
echo "Paired baseline diagnosis finished."
