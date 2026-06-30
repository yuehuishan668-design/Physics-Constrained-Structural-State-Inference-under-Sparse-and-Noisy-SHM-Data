#!/usr/bin/env bash
set -euo pipefail

echo "Step 0: check required inputs"
test -f data_processed/debug_plus_500_physics_features_mlp.npz
test -f data_processed/debug_plus_500_physics_feature_names.csv

echo "Step 1: run zero-vs-damaged feature separability diagnosis"
python -m src.evaluation.diagnose_zero_damage_separability \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --output-dir results/tables/zero_damage_diagnosis/debug_plus_500 \
  --figures-dir results/figures/zero_damage_diagnosis/debug_plus_500 \
  --zero-eps 1e-8 \
  --low-cut 0.1 \
  --medium-cut 0.2 \
  --max-zero-false-alarm 0.05 \
  --top-k 30 \
  --random-state 42

echo "Step 2: print the main report"
cat results/tables/zero_damage_diagnosis/debug_plus_500/zero_damage_separability_report.md

echo "Step 3: list generated outputs"
find results/tables/zero_damage_diagnosis/debug_plus_500 -maxdepth 1 -type f | sort
find results/figures/zero_damage_diagnosis/debug_plus_500 -maxdepth 1 -type f | sort

echo "Zero-damage separability diagnosis completed."
