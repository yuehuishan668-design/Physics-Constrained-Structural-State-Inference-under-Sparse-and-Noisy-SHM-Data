#!/usr/bin/env bash
set -euo pipefail

BASE_TAG="debug_plus_3000"
RAW_BASE="data_processed/${BASE_TAG}_dataset.npz"
SPLIT_BASE="data_processed/${BASE_TAG}_split_normalized.npz"
FEATURES_BASE="data_processed/${BASE_TAG}_physics_features_mlp.npz"
FEATURE_NAMES_BASE="data_processed/${BASE_TAG}_physics_feature_names.csv"

TMP_DIR="data_processed/_tmp_sensor_sparsity"

CONFIG_NAMES=("sensors_1234" "sensors_124" "sensors_14" "sensors_4")
SENSOR_COUNTS=("4" "3" "2" "1")
SENSOR_LAYOUTS=("1-2-3-4" "1-2-4" "1-4" "4")
KEEP_SENSOR_LISTS=("1 2 3 4" "1 2 4" "1 4" "4")

SUMMARY_OUTPUT_DIR="results/tables/sensor_sparsity/q2_sensor_3000"
SUMMARY_FIGURES_DIR="results/figures/sensor_sparsity/q2_sensor_3000"

TABLES=()
TAGS=()

echo "============================================================"
echo "Q2 fast-track Step 5: sensor sparsity stress test"
echo "Base dataset: ${BASE_TAG}"
echo "============================================================"

test -f "${RAW_BASE}" || { echo "Missing raw base dataset: ${RAW_BASE}"; exit 1; }
test -f "${SPLIT_BASE}" || { echo "Missing split base dataset: ${SPLIT_BASE}"; exit 1; }
test -f "${FEATURES_BASE}" || { echo "Missing base features: ${FEATURES_BASE}"; exit 1; }
test -f "${FEATURE_NAMES_BASE}" || { echo "Missing base feature names: ${FEATURE_NAMES_BASE}"; exit 1; }

mkdir -p "${TMP_DIR}"

for i in "${!CONFIG_NAMES[@]}"; do
  CONFIG="${CONFIG_NAMES[$i]}"
  SENSOR_COUNT="${SENSOR_COUNTS[$i]}"
  SENSOR_LAYOUT="${SENSOR_LAYOUTS[$i]}"
  KEEP_SENSORS="${KEEP_SENSOR_LISTS[$i]}"

  TAG="debug_${CONFIG}_3000"

  RAW_MASKED="${TMP_DIR}/${TAG}_dataset.npz"
  SPLIT_MASKED="${TMP_DIR}/${TAG}_split_normalized.npz"

  FEATURES="data_processed/${TAG}_physics_features_mlp.npz"
  FEATURE_NAMES="data_processed/${TAG}_physics_feature_names.csv"
  FEATURE_SUMMARY_JSON="results/tables/${TAG}_physics_feature_summary.json"
  QUALITY_JSON="results/tables/dataset_quality/${TAG}_quality_summary.json"

  ABLATION_ROOT="results/tables/physics_ablation/${TAG}"
  ABLATION_FIGURES_ROOT="results/figures/physics_ablation/${TAG}"
  FIXED_COMPARISON_CSV="${ABLATION_ROOT}/ablation_model_comparison_fixed.csv"

  echo ""
  echo "------------------------------------------------------------"
  echo "Sensor config: ${SENSOR_LAYOUT} | count=${SENSOR_COUNT} | tag=${TAG}"
  echo "------------------------------------------------------------"

  if [ "${SENSOR_COUNT}" = "4" ]; then
    echo "Using original 4-sensor feature file without masking."
    FEATURES="${FEATURES_BASE}"
    FEATURE_NAMES="${FEATURE_NAMES_BASE}"
  else
    echo "Step 1: create zero-masked sensor dataset"
    python -m src.preprocessing.create_sensor_masked_dataset \
      --raw-dataset "${RAW_BASE}" \
      --split-dataset "${SPLIT_BASE}" \
      --output-raw "${RAW_MASKED}" \
      --output-split "${SPLIT_MASKED}" \
      --keep-sensors ${KEEP_SENSORS}

    echo "Step 2: check masked dataset quality"
    mkdir -p "results/tables/dataset_quality"
    python -m src.evaluation.check_damage_dataset_quality \
      --raw-dataset "${RAW_MASKED}" \
      --split-dataset "${SPLIT_MASKED}" \
      --output-json "${QUALITY_JSON}"

    echo "Step 3: extract physics features from masked sensor dataset"
    python -m src.preprocessing.extract_physics_features \
      --raw-dataset "${RAW_MASKED}" \
      --split-dataset "${SPLIT_MASKED}" \
      --output "${FEATURES}" \
      --feature-name-csv "${FEATURE_NAMES}" \
      --summary-json "${FEATURE_SUMMARY_JSON}"

    echo "Step 4: remove temporary large masked raw/split datasets"
    rm -f "${RAW_MASKED}" "${SPLIT_MASKED}"
  fi

  echo "Step 5: run physics feature ablation"
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

  echo "Completed sensor config: ${SENSOR_LAYOUT}"
  echo "Comparison CSV: ${FIXED_COMPARISON_CSV}"
  echo "Figures root: ${ABLATION_FIGURES_ROOT}"
done

echo ""
echo "============================================================"
echo "Summarizing sensor sparsity stress-test results"
echo "============================================================"

python -m src.evaluation.summarize_sensor_sparsity \
  --tables "${TABLES[@]}" \
  --sensor-counts "${SENSOR_COUNTS[@]}" \
  --sensor-layouts "${SENSOR_LAYOUTS[@]}" \
  --tags "${TAGS[@]}" \
  --output-dir "${SUMMARY_OUTPUT_DIR}" \
  --figures-dir "${SUMMARY_FIGURES_DIR}"

echo ""
echo "============================================================"
echo "Q2 sensor sparsity workflow completed."
echo "============================================================"
echo "Summary output: ${SUMMARY_OUTPUT_DIR}"
echo "Summary figures: ${SUMMARY_FIGURES_DIR}"

echo ""
echo "Main report:"
cat "${SUMMARY_OUTPUT_DIR}/sensor_sparsity_report.md"
