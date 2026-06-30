#!/usr/bin/env bash
set -e

echo "Step 1: run final evidence synthesis"

python -m src.evaluation.synthesize_final_evidence \
  --dataset-tag debug_plus_500 \
  --output-dir results/tables/final_evidence/debug_plus_500

echo ""
echo "Step 2: show generated files"
find results/tables/final_evidence/debug_plus_500 -maxdepth 1 -type f | sort

echo ""
echo "Step 3: preview final paper synthesis report"
cat results/tables/final_evidence/debug_plus_500/final_paper_result_synthesis.md
