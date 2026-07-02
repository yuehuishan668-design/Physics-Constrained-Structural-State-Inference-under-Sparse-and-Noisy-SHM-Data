#!/usr/bin/env bash
set -euo pipefail

OUT="results/paper_ready/q2_fast_track"

TABLES="${OUT}/tables"
FIGURES="${OUT}/figures"
CAPTIONS="${OUT}/captions"
TEXT="${OUT}/text"

echo "============================================================"
echo "Build Q2 fast-track paper-ready package"
echo "============================================================"

rm -rf "${OUT}"
mkdir -p "${TABLES}" "${FIGURES}" "${CAPTIONS}" "${TEXT}"

echo ""
echo "Step 1: copy core tables"

copy_if_exists() {
  src="$1"
  dst="$2"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "Copied: $src -> $dst"
  else
    echo "Missing: $src"
  fi
}

copy_if_exists "results/tables/dataset_quality/debug_plus_3000_quality_summary.json" \
  "${TABLES}/T00_dataset_quality_debug_plus_3000.json"

copy_if_exists "results/tables/debug_plus_3000_physics_feature_summary.json" \
  "${TABLES}/T01_physics_feature_summary_debug_plus_3000.json"

copy_if_exists "results/tables/physics_ablation/debug_plus_3000/ablation_model_comparison_fixed.csv" \
  "${TABLES}/T02_main_ablation_3000.csv"

copy_if_exists "results/tables/seed_robustness/debug_plus_3000/seed_robustness_summary.csv" \
  "${TABLES}/T03_repeated_split_robustness_summary.csv"

copy_if_exists "results/tables/seed_robustness/debug_plus_3000/seed_robustness_per_seed_best.csv" \
  "${TABLES}/T04_repeated_split_per_seed_best.csv"

copy_if_exists "results/tables/damage_stratified/debug_plus_3000/damage_stratified_config_summary.csv" \
  "${TABLES}/T05_damage_stratified_config_summary.csv"

copy_if_exists "results/tables/damage_stratified/debug_plus_3000/damage_stratified_runs.csv" \
  "${TABLES}/T06_damage_stratified_runs.csv"

copy_if_exists "results/tables/noise_robustness/q2_noise_1000/noise_robustness_best_by_noise.csv" \
  "${TABLES}/T07_noise_best_by_noise.csv"

copy_if_exists "results/tables/noise_robustness/q2_noise_1000/noise_robustness_summary.csv" \
  "${TABLES}/T08_noise_robustness_summary.csv"

copy_if_exists "results/tables/sensor_sparsity/q2_sensor_3000/sensor_sparsity_best_by_sensor_count.csv" \
  "${TABLES}/T09_sensor_best_by_sensor_count.csv"

copy_if_exists "results/tables/sensor_sparsity/q2_sensor_3000/sensor_sparsity_summary.csv" \
  "${TABLES}/T10_sensor_sparsity_summary.csv"


echo ""
echo "Step 2: copy reports"

copy_if_exists "results/tables/seed_robustness/debug_plus_3000/seed_robustness_report.md" \
  "${TEXT}/R01_repeated_split_robustness_report.md"

copy_if_exists "results/tables/damage_stratified/debug_plus_3000/damage_stratified_report.md" \
  "${TEXT}/R02_damage_stratified_report.md"

copy_if_exists "results/tables/noise_robustness/q2_noise_1000/noise_robustness_report.md" \
  "${TEXT}/R03_noise_robustness_report.md"

copy_if_exists "results/tables/sensor_sparsity/q2_sensor_3000/sensor_sparsity_report.md" \
  "${TEXT}/R04_sensor_sparsity_report.md"


echo ""
echo "Step 3: copy core figures if available"

copy_figures_from_dir() {
  src_dir="$1"
  prefix="$2"

  if [ -d "$src_dir" ]; then
    n=1
    find "$src_dir" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" -o -name "*.svg" \) | sort | while read f
    do
      ext="${f##*.}"
      base="$(basename "$f")"
      cp "$f" "${FIGURES}/${prefix}_${base}"
      echo "Copied figure: $f -> ${FIGURES}/${prefix}_${base}"
      n=$((n+1))
    done
  else
    echo "Missing figure directory: $src_dir"
  fi
}

copy_figures_from_dir "results/figures/physics_ablation/debug_plus_3000" "F01_main_ablation"
copy_figures_from_dir "results/figures/seed_robustness/debug_plus_3000" "F02_repeated_split"
copy_figures_from_dir "results/figures/damage_stratified/debug_plus_3000" "F03_damage_stratified"
copy_figures_from_dir "results/figures/noise_robustness/q2_noise_1000" "F04_noise"
copy_figures_from_dir "results/figures/sensor_sparsity/q2_sensor_3000" "F05_sensor_sparsity"


echo ""
echo "Step 4: create captions"

cat > "${CAPTIONS}/figure_captions.md" <<'EOF'
# Figure Captions

## Figure 1. Main physics-informed feature ablation

Performance comparison of different physics-informed feature subsets and estimators on the 3000-case controlled simulation dataset. The full physics-informed descriptor set combined with ridge regression achieved the strongest overall predictive performance, indicating the benefit of combining response-derived and excitation-related descriptors.

## Figure 2. Repeated-split robustness

Repeated-split robustness analysis over 10 random train/validation/test partitions. The full physics-informed feature set with ridge regression remained the best-performing configuration across all splits, demonstrating that the main conclusion was not caused by a favorable single data partition.

## Figure 3. Damage-stratified reliability diagnosis

Damage-stratified evaluation of prediction reliability across zero-, low-, medium-, and high-damage regimes. Although the full-feature ridge model achieved the best average accuracy, high-damage entries exhibited larger errors and systematic underestimation, highlighting the need for reliability-oriented evaluation beyond average metrics.

## Figure 4. Noise robustness

Noise robustness analysis under fixed noise levels of 0%, 2%, 5%, 10%, 15%, and 20%. The full physics-informed feature set with ridge regression remained the best-performing configuration across all tested noise levels, with no performance collapse under high-noise conditions.

## Figure 5. Sensor sparsity stress test

Sensor sparsity stress test using zero-masked response channels. The full physics-informed feature set remained the strongest descriptor set across all sensor configurations, while severe sensor reduction substantially degraded prediction reliability, especially for damaged entries.
EOF

cat > "${CAPTIONS}/table_captions.md" <<'EOF'
# Table Captions

## Table 1. Dataset quality summary

Summary statistics of the 3000-case controlled simulation dataset, including response dimensions, damage distribution, damaged-story distribution, and response magnitude ranges.

## Table 2. Main ablation results

Main feature-ablation and model-comparison results on the 3000-case dataset. The table reports test MAE, test RMSE, damaged-entry MAE, feature count, and selected estimator settings.

## Table 3. Repeated-split robustness summary

Repeated-split robustness results over 10 random seeds. The table reports mean and standard deviation of test MAE and RMSE for each selected configuration.

## Table 4. Damage-stratified reliability metrics

Damage-stratified evaluation metrics for zero-, low-, medium-, and high-damage regimes, including MAE, RMSE, bias, and underestimation ratio.

## Table 5. Noise robustness summary

Performance comparison under fixed noise levels of 0%, 2%, 5%, 10%, 15%, and 20%, used to evaluate robustness under noisy monitoring conditions.

## Table 6. Sensor sparsity stress-test summary

Performance comparison under different sensor layouts, including four-, three-, two-, and one-sensor settings, used to evaluate sparse monitoring conditions.
EOF


echo ""
echo "Step 5: create paper-ready text snippets"

cat > "${TEXT}/results_paragraphs.md" <<'EOF'
# Paper-ready Results Paragraphs

## 1. Main ablation

On the 3000-case controlled simulation dataset, the full physics-informed feature set combined with ridge regression achieved the best fixed-split performance, with a test MAE of 0.0393 and a test RMSE of 0.0585. Removing metadata-related descriptors increased the test MAE, while response-basic-only descriptors led to substantially larger errors. These results indicate that the full descriptor set provided the most informative representation for structural damage inference under the present sparse and noisy SHM simulation setting.

## 2. Repeated-split robustness

To examine whether the fixed-split ablation result was sensitive to data partitioning, a repeated-split robustness analysis was performed on the 3000-case dataset using 10 random seeds. The full physics-informed feature set combined with ridge regression achieved the best mean test MAE of 0.0454 with a standard deviation of 0.0009, and it was selected as the best-performing configuration in all 10 splits. This confirms that the main conclusion was stable rather than split-specific.

## 3. Damage-stratified reliability

Damage-stratified evaluation further revealed that stable average performance did not imply uniform reliability across damage severities. The full physics-informed feature set with ridge regression achieved the best overall MAE of 0.0407, but its high-damage MAE increased to 0.0888. In the high-damage bin, the mean true damage was 0.2758, whereas the mean predicted damage was 0.2041, resulting in a negative bias of -0.0717 and an underestimation ratio of 0.8214. Therefore, damage-stratified reliability diagnosis is necessary for evaluating SHM-based structural state inference models, especially in safety-critical damage regimes.

## 4. Noise robustness

Noise robustness was evaluated using six independently generated 1000-case datasets with fixed noise levels of 0%, 2%, 5%, 10%, 15%, and 20%. The full physics-informed feature set combined with ridge regression remained the best-performing configuration across all noise levels. Its test MAE increased from 0.0345 at 0% noise to 0.0431 at 15% noise and 0.0420 at 20% noise, while no performance collapse was observed. Although the error trend was not strictly monotonic because each noise level used an independently generated dataset, the overall results indicate that the proposed descriptor set retained robust predictive performance under noisy monitoring conditions.

## 5. Sensor sparsity

Sensor sparsity was evaluated by zero-masking unavailable response channels while preserving the original four-story damage targets and tensor shape. The full physics-informed feature set remained the strongest descriptor set across all sensor configurations. With four and three sensors, the full-feature ridge model achieved the best test MAE of 0.0393 and 0.0612, respectively. Under more extreme sparsity, the best estimator shifted to random forest, yielding test MAEs of 0.0764 for the two-sensor layout and 0.0801 for the one-sensor layout. These results indicate that the proposed descriptor set remains effective under moderate sensor sparsity, but severe sensor reduction substantially degrades prediction reliability, especially for damaged entries.
EOF

cat > "${TEXT}/discussion_points.md" <<'EOF'
# Paper-ready Discussion Points

## Point 1. Average accuracy is insufficient

The results show that average prediction accuracy alone can mask reliability limitations in safety-critical regimes. Although the full-feature ridge model achieved the best overall accuracy, high-damage entries showed larger errors and systematic underestimation. This supports the need for damage-stratified reliability diagnosis in SHM-oriented state inference.

## Point 2. Physics-informed descriptors are more important than model complexity

Across main ablation, repeated-split, and noise robustness experiments, regularized linear regression on the full physics-informed descriptor set was consistently stronger than more complex alternatives under the current controlled simulation setting. This suggests that structured descriptor construction can be more important than estimator complexity when data are sparse and noisy.

## Point 3. Metadata provides incremental but not dominant benefit

The full descriptor set consistently outperformed no-metadata variants, but the improvement was moderate rather than dominant. This indicates that excitation-related descriptors provide useful auxiliary information, while the main predictive signal remains response-derived.

## Point 4. Extreme sensor sparsity is a practical boundary

The sensor sparsity stress test showed that the proposed descriptor set remains useful under moderate sensor reduction, but severe reduction to two or one sensor substantially degrades reliability. This should be reported as a practical boundary of the current method rather than hidden.
EOF


echo ""
echo "Step 6: create README"

cat > "${OUT}/README.md" <<'EOF'
# Q2 Fast-track Paper-ready Package

This folder collects the core tables, figures, captions, and text snippets for the Q2 fast-track manuscript.

## Experimental chain

1. 3000-case controlled simulation dataset
2. Main physics-informed feature ablation
3. Repeated-split robustness
4. Damage-stratified reliability diagnosis
5. Noise robustness
6. Sensor sparsity stress test

## Folder structure

- `tables/`: core paper tables and machine-readable summaries
- `figures/`: copied paper-ready figures from experimental outputs
- `captions/`: draft figure and table captions
- `text/`: draft Results and Discussion paragraphs

## Notes

Large generated `.npz` datasets are not included in version control. All data generation, preprocessing, feature extraction, and evaluation scripts are preserved for reproducibility.

The sensor sparsity experiment should be described as a zero-masking stress test, not as a missing-data imputation method.

Noise robustness datasets were independently generated for each fixed noise level; therefore, performance trends need not be strictly monotonic.
EOF


echo ""
echo "Step 7: show package contents"

find "${OUT}" -maxdepth 3 -type f | sort

echo ""
echo "============================================================"
echo "Paper-ready package created:"
echo "${OUT}"
echo "============================================================"
