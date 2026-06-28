#!/usr/bin/env bash
set -euo pipefail

# English:
# Run this script only after debug_plus_500_dataset.npz and debug_plus_500_split_normalized.npz exist.
#
# 中文：
# 只有在 data_processed/debug_plus_500_dataset.npz 和
# data_processed/debug_plus_500_split_normalized.npz 已经生成之后，才运行本脚本。

TAG="${1:-debug_plus_500}"

RAW_DATASET="data_processed/${TAG}_dataset.npz"
SPLIT_DATASET="data_processed/${TAG}_split_normalized.npz"
FEATURES="data_processed/${TAG}_physics_features_mlp.npz"
FEATURE_NAMES="data_processed/${TAG}_physics_feature_names.csv"

QUALITY_JSON="results/tables/dataset_quality/${TAG}_quality_summary.json"
FEATURE_SUMMARY_JSON="results/tables/${TAG}_physics_feature_summary.json"

ABLATION_ROOT="results/tables/physics_ablation/${TAG}"
ABLATION_FIGURES_ROOT="results/figures/physics_ablation/${TAG}"
FIXED_COMPARISON_CSV="${ABLATION_ROOT}/ablation_model_comparison_fixed.csv"

echo "Step 1/5: checking dataset files..."
test -f "${RAW_DATASET}" || { echo "Missing raw dataset: ${RAW_DATASET}"; exit 1; }
test -f "${SPLIT_DATASET}" || { echo "Missing split dataset: ${SPLIT_DATASET}"; exit 1; }

echo "Step 2/5: checking dataset quality..."
python -m src.evaluation.check_damage_dataset_quality \
  --raw-dataset "${RAW_DATASET}" \
  --split-dataset "${SPLIT_DATASET}" \
  --output-json "${QUALITY_JSON}"

echo "Step 3/5: extracting physics features..."
python -m src.preprocessing.extract_physics_features \
  --raw-dataset "${RAW_DATASET}" \
  --split-dataset "${SPLIT_DATASET}" \
  --output "${FEATURES}" \
  --feature-name-csv "${FEATURE_NAMES}" \
  --summary-json "${FEATURE_SUMMARY_JSON}"

echo "Step 4/5: running selected physics feature ablation..."
python -m src.experiments.run_physics_feature_ablation \
  --features "${FEATURES}" \
  --feature-names "${FEATURE_NAMES}" \
  --dataset-tag "${TAG}" \
  --models ridge random_forest elasticnet \
  --feature-sets full no_meta response_basic_only response_correlation \
  --output-root "results/tables/physics_ablation" \
  --figures-root "results/figures/physics_ablation"

echo "Step 5/5: fixing and exporting ablation comparison table..."
python -m src.evaluation.compare_ablation_metrics \
  --experiment-root "${ABLATION_ROOT}" \
  --output "${FIXED_COMPARISON_CSV}" \
  --sort-by test_mae \
  --print-top 30

echo ""
echo "Scale-up workflow completed."
echo "Quality JSON: ${QUALITY_JSON}"
echo "Feature summary JSON: ${FEATURE_SUMMARY_JSON}"
echo "Fixed comparison CSV: ${FIXED_COMPARISON_CSV}"
echo "Figures root: ${ABLATION_FIGURES_ROOT}"
