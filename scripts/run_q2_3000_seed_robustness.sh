#!/usr/bin/env bash
set -euo pipefail

TAG="debug_plus_3000"

FEATURES="data_processed/${TAG}_physics_features_mlp.npz"
FEATURE_NAMES="data_processed/${TAG}_physics_feature_names.csv"

OUTPUT_DIR="results/tables/seed_robustness/${TAG}"
FIGURES_DIR="results/figures/seed_robustness/${TAG}"

echo "============================================================"
echo "Q2 fast-track Step 2: repeated-split robustness for ${TAG}"
echo "============================================================"

echo ""
echo "Step 0: check required files"
test -f "${FEATURES}" || { echo "Missing features: ${FEATURES}"; exit 1; }
test -f "${FEATURE_NAMES}" || { echo "Missing feature names: ${FEATURE_NAMES}"; exit 1; }

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${FIGURES_DIR}"

echo ""
echo "Step 1: run repeated-split robustness"
python -m src.evaluation.run_seed_robustness \
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
echo "Step 4: show summary CSV"
if [ -f "${OUTPUT_DIR}/seed_robustness_summary.csv" ]; then
  cat "${OUTPUT_DIR}/seed_robustness_summary.csv"
else
  echo "Missing ${OUTPUT_DIR}/seed_robustness_summary.csv"
fi

echo ""
echo "Step 5: show per-seed best CSV"
if [ -f "${OUTPUT_DIR}/seed_robustness_per_seed_best.csv" ]; then
  cat "${OUTPUT_DIR}/seed_robustness_per_seed_best.csv"
else
  echo "Missing ${OUTPUT_DIR}/seed_robustness_per_seed_best.csv"
fi

echo ""
echo "Step 6: show markdown report"
if [ -f "${OUTPUT_DIR}/seed_robustness_report.md" ]; then
  cat "${OUTPUT_DIR}/seed_robustness_report.md"
else
  echo "Missing ${OUTPUT_DIR}/seed_robustness_report.md"
fi

echo ""
echo "============================================================"
echo "Q2 debug_plus_3000 seed robustness completed."
echo "============================================================"
echo "Output tables: ${OUTPUT_DIR}"
echo "Output figures: ${FIGURES_DIR}"
