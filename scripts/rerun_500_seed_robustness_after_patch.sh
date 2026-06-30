#!/usr/bin/env bash
set -euo pipefail

# Patch and rerun seed robustness.
# 中文说明：先修复 boxplot 标签参数，再重新运行稳健性验证脚本。

python scripts/patch_seed_robustness_boxplot.py
bash scripts/run_500_seed_robustness.sh
