#!/usr/bin/env bash
set -euo pipefail

echo "Step 0: check required input files"
test -f data_processed/debug_plus_500_physics_features_mlp.npz
test -f data_processed/debug_plus_500_physics_feature_names.csv

echo "Step 1: run zero-aware hierarchical experiment"
python -m src.evaluation.run_zero_aware_hierarchical \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --output-dir results/tables/zero_aware_hierarchical/debug_plus_500 \
  --figures-dir results/figures/zero_aware_hierarchical/debug_plus_500 \
  --thresholds 0.30,0.40,0.50,0.60,0.70,0.80 \
  --high-weights 1,2,4,6,8 \
  --zero-eps 1e-8 \
  --low-cut 0.10 \
  --medium-cut 0.20 \
  --max-damage 0.50 \
  --seed 42 \
  --top-k 20

echo "Step 2: print report"
cat results/tables/zero_aware_hierarchical/debug_plus_500/zero_aware_hierarchical_report.md

echo "Step 3: print selected configs"
cat results/tables/zero_aware_hierarchical/debug_plus_500/selected_zero_aware_configs.json

echo "Step 4: list generated tables"
find results/tables/zero_aware_hierarchical/debug_plus_500 -maxdepth 1 -type f | sort

echo "Step 5: list generated figures"
find results/figures/zero_aware_hierarchical/debug_plus_500 -maxdepth 1 -type f | sort

echo "Zero-aware hierarchical pipeline completed."
