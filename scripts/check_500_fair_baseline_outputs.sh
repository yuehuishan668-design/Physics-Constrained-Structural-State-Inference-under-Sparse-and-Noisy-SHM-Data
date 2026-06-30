#!/usr/bin/env bash
set -euo pipefail

echo "Check metric JSON files:"
find results/tables \
  \( -path "*500*" -o -path "*debug_plus_500*" \) \
  -name "*.json" \
  | sort

echo ""
echo "Check figure files:"
find results/figures \
  \( -path "*500*" -o -path "*debug_plus_500*" \) \
  -name "*.png" \
  | sort

echo ""
echo "Main comparison:"
ls -lh results/tables/fair_baseline_500_comparison.csv
head -n 40 results/tables/fair_baseline_500_comparison.csv
