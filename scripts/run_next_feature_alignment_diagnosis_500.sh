#!/usr/bin/env bash
set -e

# English:
#   Run the next diagnostic step for debug_plus_500.
# 中文：
#   运行 debug_plus_500 的下一步诊断实验。

echo "Step 0: check required files"
test -f data_processed/debug_plus_500_physics_features_mlp.npz
test -f data_processed/debug_plus_500_physics_feature_names.csv
test -f data_processed/debug_plus_500_split_normalized.npz

echo "Step 1: case-level any-damage diagnosis"
python -m src.evaluation.run_case_level_any_damage_diagnosis \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --split data_processed/debug_plus_500_split_normalized.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --tag debug_plus_500 \
  --output-dir results/tables/case_level_any_damage/debug_plus_500 \
  --figures-dir results/figures/case_level_any_damage/debug_plus_500 \
  --max-zero-false-alarm 0.05 \
  --thresholds 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --seed 42

echo "Step 2: story-local feature alignment diagnosis"
python -m src.evaluation.run_story_local_feature_diagnosis \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --split data_processed/debug_plus_500_split_normalized.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --tag debug_plus_500 \
  --output-dir results/tables/story_local_alignment/debug_plus_500 \
  --figures-dir results/figures/story_local_alignment/debug_plus_500 \
  --include-neighbor \
  --include-story-one-hot \
  --classification-threshold 0.5 \
  --seed 42

echo "Step 3: print key reports"
echo ""
echo "===== Case-level any-damage report ====="
cat results/tables/case_level_any_damage/debug_plus_500/case_level_any_damage_report.md

echo ""
echo "===== Story-local feature alignment report ====="
cat results/tables/story_local_alignment/debug_plus_500/story_local_feature_alignment_report.md

echo ""
echo "Next diagnostic step completed."
echo "Please send back:"
echo "1) terminal output after running this script"
echo "2) case_level_any_damage_report.md content"
echo "3) story_local_feature_alignment_report.md content"
echo "4) the generated figures listed in the assistant instructions"
