#!/usr/bin/env bash
set -euo pipefail

TAG="debug_plus_3000"

FEATURES="data_processed/${TAG}_physics_features_mlp.npz"
FEATURE_NAMES="data_processed/${TAG}_physics_feature_names.csv"

OUTPUT_DIR="results/tables/damage_stratified/${TAG}"
FIGURES_DIR="results/figures/damage_stratified/${TAG}"

echo "============================================================"
echo "Q2 fast-track Step 3: damage-stratified reliability analysis"
echo "Dataset tag: ${TAG}"
echo "============================================================"

echo ""
echo "Step 0: check required files"
test -f "${FEATURES}" || { echo "Missing features: ${FEATURES}"; exit 1; }
test -f "${FEATURE_NAMES}" || { echo "Missing feature names: ${FEATURE_NAMES}"; exit 1; }

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${FIGURES_DIR}"

echo ""
echo "Step 1: run damage-stratified analysis"
python -m src.evaluation.run_damage_stratified_analysis \
  --features "${FEATURES}" \
  --feature-names "${FEATURE_NAMES}" \
  --output-dir "${OUTPUT_DIR}" \
  --figures-dir "${FIGURES_DIR}"

echo ""
echo "Step 2: list output tables"
find "${OUTPUT_DIR}" -maxdepth 1 -type f | sort

echo ""
echo "Step 3: list output figures"
find "${FIGURES_DIR}" -maxdepth 1 -type f | sort

echo ""
echo "Step 4: show report if available"
if [ -f "${OUTPUT_DIR}/damage_stratified_report.md" ]; then
  cat "${OUTPUT_DIR}/damage_stratified_report.md"
else
  echo "Missing ${OUTPUT_DIR}/damage_stratified_report.md"
fi

echo ""
echo "Step 5: show config summary if available"
if [ -f "${OUTPUT_DIR}/damage_stratified_config_summary.csv" ]; then
  cat "${OUTPUT_DIR}/damage_stratified_config_summary.csv"
else
  echo "Missing ${OUTPUT_DIR}/damage_stratified_config_summary.csv"
fi

echo ""
echo "Step 6: show main runs if available"
if [ -f "${OUTPUT_DIR}/damage_stratified_runs.csv" ]; then
  head -n 40 "${OUTPUT_DIR}/damage_stratified_runs.csv"
else
  echo "Missing ${OUTPUT_DIR}/damage_stratified_runs.csv"
fi

echo ""
echo "============================================================"
echo "Q2 debug_plus_3000 damage-stratified analysis completed."
echo "============================================================"
echo "Output tables: ${OUTPUT_DIR}"
echo "Output figures: ${FIGURES_DIR}"
