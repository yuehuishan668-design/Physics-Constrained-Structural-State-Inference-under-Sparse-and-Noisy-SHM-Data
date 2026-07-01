#!/usr/bin/env bash
set -euo pipefail

TAG="debug_plus_3000"
N_CASES=3000
SEED=20260701

RAW_DATASET="data_processed/${TAG}_dataset.npz"
INDEX_CSV="data_processed/${TAG}_dataset_index.csv"
SPLIT_DATASET="data_processed/${TAG}_split_normalized.npz"
FEATURES="data_processed/${TAG}_physics_features_mlp.npz"
FEATURE_NAMES="data_processed/${TAG}_physics_feature_names.csv"

QUALITY_JSON="results/tables/dataset_quality/${TAG}_quality_summary.json"
FEATURE_SUMMARY_JSON="results/tables/${TAG}_physics_feature_summary.json"

ABLATION_ROOT="results/tables/physics_ablation/${TAG}"
ABLATION_FIGURES_ROOT="results/figures/physics_ablation/${TAG}"
FIXED_COMPARISON_CSV="${ABLATION_ROOT}/ablation_model_comparison_fixed.csv"

echo "============================================================"
echo "Q2 fast-track Step 1: generate ${TAG}"
echo "============================================================"

echo ""
echo "Step 1/6: generate raw ${N_CASES}-case dataset"
python -m src.data_generation.generate_dataset \
  --n-cases "${N_CASES}" \
  --seed "${SEED}" \
  --output-prefix "${TAG}" \
  --output-processed "${RAW_DATASET}" \
  --index-csv "${INDEX_CSV}"

echo ""
echo "Step 2/6: create split-normalized dataset"
python -m src.preprocessing.create_split_normalized_dataset \
  --raw-dataset "${RAW_DATASET}" \
  --output "${SPLIT_DATASET}" \
  --seed "${SEED}" \
  --train-ratio 0.70 \
  --val-ratio 0.15

echo ""
echo "Step 3/6: check dataset quality"
mkdir -p "results/tables/dataset_quality"
python -m src.evaluation.check_damage_dataset_quality \
  --raw-dataset "${RAW_DATASET}" \
  --split-dataset "${SPLIT_DATASET}" \
  --output-json "${QUALITY_JSON}"

echo ""
echo "Step 4/6: extract physics features"
python -m src.preprocessing.extract_physics_features \
  --raw-dataset "${RAW_DATASET}" \
  --split-dataset "${SPLIT_DATASET}" \
  --output "${FEATURES}" \
  --feature-name-csv "${FEATURE_NAMES}" \
  --summary-json "${FEATURE_SUMMARY_JSON}"

echo ""
echo "Step 5/6: run selected physics feature ablation"
python -m src.experiments.run_physics_feature_ablation \
  --base-features "${FEATURES}" \
  --feature-names "${FEATURE_NAMES}" \
  --tag "${TAG}" \
  --models ridge random_forest elasticnet \
  --output-root "results/tables/physics_ablation" \
  --figures-root "results/figures/physics_ablation" \
  --plot

echo ""
echo "Step 6/6: summarize ablation metrics"
python -m src.evaluation.compare_ablation_metrics \
  --experiment-root "${ABLATION_ROOT}" \
  --output "${FIXED_COMPARISON_CSV}" \
  --sort-by test_mae \
  --print-top 30

echo ""
echo "============================================================"
echo "Q2 debug_plus_3000 workflow completed."
echo "============================================================"
echo "Raw dataset: ${RAW_DATASET}"
echo "Index CSV: ${INDEX_CSV}"
echo "Split dataset: ${SPLIT_DATASET}"
echo "Physics features: ${FEATURES}"
echo "Feature names: ${FEATURE_NAMES}"
echo "Quality JSON: ${QUALITY_JSON}"
echo "Feature summary JSON: ${FEATURE_SUMMARY_JSON}"
echo "Ablation root: ${ABLATION_ROOT}"
echo "Ablation figures: ${ABLATION_FIGURES_ROOT}"
echo "Fixed comparison CSV: ${FIXED_COMPARISON_CSV}"
