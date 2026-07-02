#!/usr/bin/env bash
set -euo pipefail

N_CASES=1000
BASE_SEED=20260710

NOISE_LEVELS=("0.00" "0.02" "0.05" "0.10" "0.15" "0.20")
NOISE_TAGS=("000" "002" "005" "010" "015" "020")

SUMMARY_OUTPUT_DIR="results/tables/noise_robustness/q2_noise_1000"
SUMMARY_FIGURES_DIR="results/figures/noise_robustness/q2_noise_1000"

TABLES=()
TAGS=()

echo "============================================================"
echo "Q2 fast-track Step 4: noise robustness experiment"
echo "Cases per noise level: ${N_CASES}"
echo "============================================================"

for i in "${!NOISE_LEVELS[@]}"; do
  NOISE="${NOISE_LEVELS[$i]}"
  NOISE_TAG="${NOISE_TAGS[$i]}"
  TAG="debug_noise_${NOISE_TAG}_${N_CASES}"
  SEED=$((BASE_SEED + i))

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

  echo ""
  echo "------------------------------------------------------------"
  echo "Noise level: ${NOISE} | TAG: ${TAG}"
  echo "------------------------------------------------------------"

  echo "Step 1: generate raw dataset"
  python -m src.data_generation.generate_dataset \
    --n-cases "${N_CASES}" \
    --seed "${SEED}" \
    --output-prefix "${TAG}" \
    --output-processed "${RAW_DATASET}" \
    --index-csv "${INDEX_CSV}" \
    --noise-levels "${NOISE}"

  echo "Step 2: create split-normalized dataset"
  python -m src.preprocessing.create_split_normalized_dataset \
    --raw-dataset "${RAW_DATASET}" \
    --output "${SPLIT_DATASET}" \
    --seed "${SEED}" \
    --train-ratio 0.70 \
    --val-ratio 0.15

  echo "Step 3: check dataset quality"
  mkdir -p "results/tables/dataset_quality"
  python -m src.evaluation.check_damage_dataset_quality \
    --raw-dataset "${RAW_DATASET}" \
    --split-dataset "${SPLIT_DATASET}" \
    --output-json "${QUALITY_JSON}"

  echo "Step 4: extract physics features"
  python -m src.preprocessing.extract_physics_features \
    --raw-dataset "${RAW_DATASET}" \
    --split-dataset "${SPLIT_DATASET}" \
    --output "${FEATURES}" \
    --feature-name-csv "${FEATURE_NAMES}" \
    --summary-json "${FEATURE_SUMMARY_JSON}"

  echo "Step 5: run physics feature ablation with selected models"
  python -m src.experiments.run_physics_feature_ablation \
    --base-features "${FEATURES}" \
    --feature-names "${FEATURE_NAMES}" \
    --tag "${TAG}" \
    --models ridge elasticnet random_forest \
    --output-root "results/tables/physics_ablation" \
    --figures-root "results/figures/physics_ablation" \
    --plot

  echo "Step 6: summarize ablation metrics"
  python -m src.evaluation.compare_ablation_metrics \
    --experiment-root "${ABLATION_ROOT}" \
    --output "${FIXED_COMPARISON_CSV}" \
    --sort-by test_mae \
    --print-top 30

  TABLES+=("${FIXED_COMPARISON_CSV}")
  TAGS+=("${TAG}")

  echo "Completed TAG: ${TAG}"
  echo "Comparison CSV: ${FIXED_COMPARISON_CSV}"
  echo "Figures root: ${ABLATION_FIGURES_ROOT}"
done

echo ""
echo "================================================------------"
echo "Summarizing all noise robustness results"
echo "================================================------------"

python -m src.evaluation.summarize_noise_robustness \
  --tables "${TABLES[@]}" \
  --noise-levels "${NOISE_LEVELS[@]}" \
  --tags "${TAGS[@]}" \
  --output-dir "${SUMMARY_OUTPUT_DIR}" \
  --figures-dir "${SUMMARY_FIGURES_DIR}"

echo ""
echo "============================================================"
echo "Q2 noise robustness workflow completed."
echo "============================================================"
echo "Summary output: ${SUMMARY_OUTPUT_DIR}"
echo "Summary figures: ${SUMMARY_FIGURES_DIR}"

echo ""
echo "Main summary:"
cat "${SUMMARY_OUTPUT_DIR}/noise_robustness_report.md"
