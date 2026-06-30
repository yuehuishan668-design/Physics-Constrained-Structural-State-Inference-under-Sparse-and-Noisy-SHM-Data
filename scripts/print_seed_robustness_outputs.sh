#!/usr/bin/env bash
set -euo pipefail

# Print the outputs that should be sent back for analysis.
# 中文说明：运行完成后，用这个脚本统一打印需要回传给我的关键结果。

echo "===== Summary CSV ====="
cat results/tables/seed_robustness/debug_plus_500/seed_robustness_summary.csv

echo
echo "===== Per-seed best CSV ====="
cat results/tables/seed_robustness/debug_plus_500/seed_robustness_per_seed_best.csv

echo
echo "===== Markdown report ====="
cat results/tables/seed_robustness/debug_plus_500/seed_robustness_report.md

echo
echo "===== Generated figures ====="
find results/figures/seed_robustness/debug_plus_500 -maxdepth 1 -type f | sort
