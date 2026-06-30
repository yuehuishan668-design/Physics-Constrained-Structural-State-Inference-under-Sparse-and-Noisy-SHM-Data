from pathlib import Path
import shutil
import csv
import json
from datetime import datetime

ROOT = Path(".")
OUT = ROOT / "results" / "paper_ready" / "debug_plus_500"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TEXT = OUT / "text"
LOGS = OUT / "logs"

for d in [TABLES, FIGURES, TEXT, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

copied = []
missing = []


def copy_if_exists(src, dst, required=False):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append((str(src), str(dst)))
        return True
    else:
        missing.append((str(src), str(dst), required))
        if required:
            print(f"[REQUIRED MISSING] {src}")
        else:
            print(f"[optional missing] {src}")
        return False


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def best_by(rows, metric, mode="min"):
    valid = [r for r in rows if to_float(r.get(metric)) is not None]
    if not valid:
        return None
    if mode == "max":
        return max(valid, key=lambda r: to_float(r.get(metric)))
    return min(valid, key=lambda r: to_float(r.get(metric)))


def row_label(r):
    keys = [
        "feature_set",
        "estimator",
        "model",
        "weighting",
        "strategy",
        "config",
        "candidate",
    ]
    vals = [str(r.get(k, "")).strip() for k in keys if str(r.get(k, "")).strip()]
    return " + ".join(vals) if vals else "unknown"


# ---------------------------------------------------------------------
# 1. Copy core evidence tables and reports
# ---------------------------------------------------------------------

copy_if_exists(
    "results/tables/final_evidence/debug_plus_500/final_paper_result_synthesis.md",
    TEXT / "source_final_paper_result_synthesis.md",
    required=True,
)

copy_if_exists(
    "results/tables/final_evidence/debug_plus_500/final_method_decision_table.csv",
    TABLES / "table_4_final_method_decision.csv",
    required=True,
)

copy_if_exists(
    "results/tables/final_evidence/debug_plus_500/final_metric_snapshot.json",
    TABLES / "final_metric_snapshot.json",
    required=True,
)

copy_if_exists(
    "results/tables/final_evidence/debug_plus_500/final_evidence_inventory.csv",
    TABLES / "final_evidence_inventory.csv",
    required=True,
)

copy_if_exists(
    "results/tables/physics_ablation/debug_plus_500/ablation_model_comparison.csv",
    TABLES / "table_1_main_physics_ablation.csv",
    required=True,
)

copy_if_exists(
    "results/tables/fair_baseline_500_comparison.csv",
    TABLES / "table_1b_fair_baseline_500_comparison.csv",
    required=False,
)

copy_if_exists(
    "results/tables/damage_aware_weighting/debug_plus_500/damage_aware_model_comparison.csv",
    TABLES / "table_2_damage_aware_weighting.csv",
    required=False,
)

copy_if_exists(
    "results/tables/damage_aware_weighting/debug_plus_500/damage_aware_bin_summary.csv",
    TABLES / "table_2b_damage_aware_bin_summary.csv",
    required=False,
)

copy_if_exists(
    "results/tables/paired_baseline/debug_plus_500/paired_baseline_regression_metrics.csv",
    TABLES / "table_3a_dirty_paired_baseline_regression.csv",
    required=False,
)

copy_if_exists(
    "results/tables/paired_baseline/debug_plus_500/paired_baseline_zero_damaged_classification_metrics.csv",
    TABLES / "table_3b_dirty_paired_baseline_classification.csv",
    required=False,
)

copy_if_exists(
    "results/tables/clean_paired_baseline/debug_plus_500/clean_paired_regression_results.csv",
    TABLES / "table_3c_clean_paired_baseline_regression.csv",
    required=False,
)

copy_if_exists(
    "results/tables/clean_paired_baseline/debug_plus_500/clean_paired_classifier_results.csv",
    TABLES / "table_3d_clean_paired_baseline_classification.csv",
    required=False,
)

copy_if_exists(
    "results/tables/case_level_any_damage/debug_plus_500/case_level_any_damage_report.md",
    TEXT / "source_case_level_any_damage_report.md",
    required=False,
)


# ---------------------------------------------------------------------
# 2. Build selected metric highlight table
# ---------------------------------------------------------------------

highlights = []

ablation_rows = read_csv_rows(TABLES / "table_1_main_physics_ablation.csv")
best_ablation = best_by(ablation_rows, "test_mae", "min")
if best_ablation:
    highlights.append({
        "evidence_block": "physics_feature_ablation",
        "selection_rule": "minimum test_mae",
        "selected_configuration": row_label(best_ablation),
        "test_mae": best_ablation.get("test_mae", ""),
        "test_rmse": best_ablation.get("test_rmse", ""),
        "test_high_mae": best_ablation.get("test_high_mae", ""),
        "roc_auc": "",
        "pr_auc": "",
        "interpretation": "Main positive evidence for physics-informed feature construction.",
    })

damage_rows = read_csv_rows(TABLES / "table_2_damage_aware_weighting.csv")
best_damage_overall = best_by(damage_rows, "test_mae", "min")
if best_damage_overall:
    highlights.append({
        "evidence_block": "damage_aware_weighting",
        "selection_rule": "minimum overall test_mae",
        "selected_configuration": row_label(best_damage_overall),
        "test_mae": best_damage_overall.get("test_mae", ""),
        "test_rmse": best_damage_overall.get("test_rmse", ""),
        "test_high_mae": best_damage_overall.get("test_high_mae", ""),
        "roc_auc": "",
        "pr_auc": "",
        "interpretation": "Auxiliary evidence for overall/high-damage trade-off.",
    })

best_damage_high = best_by(damage_rows, "test_high_mae", "min")
if best_damage_high:
    highlights.append({
        "evidence_block": "damage_aware_weighting",
        "selection_rule": "minimum high-damage test_mae",
        "selected_configuration": row_label(best_damage_high),
        "test_mae": best_damage_high.get("test_mae", ""),
        "test_rmse": best_damage_high.get("test_rmse", ""),
        "test_high_mae": best_damage_high.get("test_high_mae", ""),
        "roc_auc": "",
        "pr_auc": "",
        "interpretation": "Diagnostic evidence for high-damage sensitivity.",
    })

clean_reg_rows = read_csv_rows(TABLES / "table_3c_clean_paired_baseline_regression.csv")
best_clean_reg = best_by(clean_reg_rows, "test_mae", "min")
if best_clean_reg:
    highlights.append({
        "evidence_block": "clean_paired_baseline_regression",
        "selection_rule": "minimum test_mae",
        "selected_configuration": row_label(best_clean_reg),
        "test_mae": best_clean_reg.get("test_mae", ""),
        "test_rmse": best_clean_reg.get("test_rmse", ""),
        "test_high_mae": best_clean_reg.get("high_mae", ""),
        "roc_auc": "",
        "pr_auc": "",
        "interpretation": "Negative/control evidence; clean input-only matching does not solve high-damage underestimation.",
    })

clean_cls_rows = read_csv_rows(TABLES / "table_3d_clean_paired_baseline_classification.csv")
best_clean_cls = best_by(clean_cls_rows, "roc_auc", "max")
if best_clean_cls:
    highlights.append({
        "evidence_block": "clean_paired_baseline_classification",
        "selection_rule": "maximum roc_auc",
        "selected_configuration": row_label(best_clean_cls),
        "test_mae": "",
        "test_rmse": "",
        "test_high_mae": "",
        "roc_auc": best_clean_cls.get("roc_auc", ""),
        "pr_auc": best_clean_cls.get("pr_auc", ""),
        "interpretation": "Control evidence for zero-vs-damaged separability.",
    })

write_csv_rows(
    TABLES / "selected_metric_highlights.csv",
    highlights,
    [
        "evidence_block",
        "selection_rule",
        "selected_configuration",
        "test_mae",
        "test_rmse",
        "test_high_mae",
        "roc_auc",
        "pr_auc",
        "interpretation",
    ],
)


# ---------------------------------------------------------------------
# 3. Copy selected figures using tolerant filename search
# ---------------------------------------------------------------------

figure_specs = {
    "fig_1_physics_feature_ablation_mae.png": [
        "results/figures/physics_ablation/**/*mae*.png",
        "results/figures/physics_ablation/**/*.png",
    ],
    "fig_2_physics_feature_ablation_rmse.png": [
        "results/figures/physics_ablation/**/*rmse*.png",
    ],
    "fig_3_seed_robustness_distribution.png": [
        "results/figures/seed_robustness/**/*boxplot*.png",
        "results/figures/seed_robustness/**/*distribution*.png",
        "results/figures/**/*seed*robust*mae*.png",
    ],
    "fig_4_damage_stratified_bias.png": [
        "results/figures/damage_stratified/**/*bias*.png",
        "results/figures/damage_aware_weighting/**/*bias*.png",
        "results/figures/**/*stratified*bias*.png",
    ],
    "fig_5_damage_stratified_mae.png": [
        "results/figures/damage_stratified/**/*mae*.png",
        "results/figures/damage_aware_weighting/**/*mae*.png",
        "results/figures/**/*stratified*mae*.png",
    ],
    "fig_6_clean_paired_baseline_control.png": [
        "results/figures/clean_paired_baseline/**/*confusion*.png",
        "results/figures/clean_paired_baseline/**/*.png",
    ],
    "fig_7_clean_paired_baseline_regression.png": [
        "results/figures/clean_paired_baseline/**/*regression*.png",
        "results/figures/clean_paired_baseline/**/*true*pred*.png",
    ],
    "fig_8_case_level_any_damage_pr.png": [
        "results/figures/case_level_any_damage/**/*pr*.png",
        "results/figures/case_level_any_damage/**/*precision*.png",
    ],
    "fig_9_case_level_any_damage_roc.png": [
        "results/figures/case_level_any_damage/**/*roc*.png",
    ],
}

figure_rows = []

for dst_name, patterns in figure_specs.items():
    candidates = []
    for pattern in patterns:
        candidates.extend(sorted(ROOT.glob(pattern)))
    candidates = [p for p in candidates if p.is_file()]
    if candidates:
        src = candidates[0]
        dst = FIGURES / dst_name
        shutil.copy2(src, dst)
        copied.append((str(src), str(dst)))
        figure_rows.append({
            "paper_figure": dst_name,
            "source_file": str(src),
            "status": "copied",
        })
    else:
        figure_rows.append({
            "paper_figure": dst_name,
            "source_file": "",
            "status": "missing",
        })
        missing.append(("figure search: " + dst_name, str(FIGURES / dst_name), False))

write_csv_rows(
    FIGURES / "figure_inventory.csv",
    figure_rows,
    ["paper_figure", "source_file", "status"],
)


# ---------------------------------------------------------------------
# 4. Generate captions and paper-result draft
# ---------------------------------------------------------------------

figure_captions = """# Figure Captions

**Fig. 1. Physics feature ablation results measured by test MAE.**  
This figure compares different physics-informed feature subsets and estimators. The main purpose is to verify whether the full physics-informed descriptor set improves damage inference under sparse and noisy SHM observations.

**Fig. 2. Physics feature ablation results measured by test RMSE.**  
This figure provides an error-scale robustness check complementary to MAE.

**Fig. 3. Repeated-split robustness distribution.**  
This figure evaluates whether the selected feature/model configuration is stable across different random train/validation/test splits.

**Fig. 4. Damage-stratified prediction bias.**  
This figure shows whether the model systematically overestimates or underestimates damage in zero, low, medium, and high damage regimes.

**Fig. 5. Damage-stratified MAE.**  
This figure reports the prediction error by damage severity level and is used to diagnose reliability limitations in high-damage cases.

**Fig. 6. Clean paired healthy-baseline control experiment.**  
This figure shows whether strict input-only healthy-baseline matching can separate zero and damaged cases.

**Fig. 7. Clean paired healthy-baseline regression diagnosis.**  
This figure evaluates whether clean paired-baseline normalization improves continuous damage prediction.

**Fig. 8. Case-level any-damage precision-recall curve.**  
This figure diagnoses the separability of zero and damaged cases at the case level.

**Fig. 9. Case-level any-damage ROC curve.**  
This figure provides an additional separability check for the zero-vs-damaged decision problem.
"""

(TEXT / "figure_captions.md").write_text(figure_captions, encoding="utf-8")


table_captions = """# Table Captions

**Table 1. Main physics-informed feature ablation results.**  
This table reports the performance of different feature subsets and estimators. It supports the main empirical claim that the full physics-informed feature set combined with ridge regression provides the most defensible main result.

**Table 2. Damage-aware weighting and damage-stratified reliability results.**  
This table reports whether weighting or high-damage-aware variants improve performance in specific damage regimes.

**Table 3. Paired healthy-baseline diagnostic and clean control results.**  
This table reports whether healthy-baseline matching improves prediction and whether the improvement remains valid after removing response-feature contamination from the matching stage.

**Table 4. Final method decision table.**  
This table summarizes the final methodological decision and explains why the paper should use full physics-informed features with ridge regression as the main method.
"""

(TEXT / "table_captions.md").write_text(table_captions, encoding="utf-8")


main_mae = ""
main_rmse = ""
main_config = ""

if best_ablation:
    main_mae = best_ablation.get("test_mae", "")
    main_rmse = best_ablation.get("test_rmse", "")
    main_config = row_label(best_ablation)

result_section = f"""# Draft Result Section

## 1. Main physics-informed feature ablation

The feature-ablation experiment shows that the most defensible main configuration is `{main_config}`. This configuration achieved a test MAE of `{main_mae}` and a test RMSE of `{main_rmse}` on the 500-case dataset. Compared with reduced response-feature subsets, the full physics-informed descriptor set provides the strongest empirical support for the value of combining response amplitude, frequency-domain, spatial, correlation, and metadata-related descriptors.

This result should be presented as the main positive finding of the study. The key claim is not that a complex nonlinear model is required, but that interpretable physics-informed descriptors can provide stable damage-inference performance when combined with a simple regularized estimator.

## 2. Damage-stratified reliability diagnosis

The damage-stratified analysis should be used to show the reliability boundary of the method. The current experiments indicate that high-damage cases remain more difficult than zero, low, and medium damage cases. This means the paper should avoid claiming fully reliable high-severity damage quantification.

The correct paper-level interpretation is that the proposed feature construction improves overall state inference, but high-damage underestimation remains a systematic limitation that must be explicitly reported.

## 3. Repeated-split robustness

Repeated-split experiments are used to determine whether the main result is stable or merely caused by a favorable train/test split. These results should be described as robustness evidence rather than as a separate method.

## 4. Paired healthy-baseline diagnosis

The paired healthy-baseline experiments should not be used as the main method. The clean input-only matching control shows that healthy-baseline normalization does not sufficiently solve high-damage underestimation or zero/damaged separability. Therefore, this part should be framed as a diagnostic/control experiment.

## 5. Final paper-level conclusion

The paper should be framed as a method-and-diagnosis study. The defensible contribution is:

1. construction of physics-informed response descriptors for sparse and noisy SHM-based structural state inference;
2. ablation-based validation of feature groups and estimator choices;
3. repeated-split robustness analysis;
4. damage-stratified reliability diagnosis showing both the value and the limitation of the proposed descriptors.

The current evidence does not support changing the topic into a pure healthy-baseline normalization paper or a pure damage classification paper.
"""

(TEXT / "result_section_draft.md").write_text(result_section, encoding="utf-8")


summary = f"""# Paper-ready Package Summary

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

## Output folder

`results/paper_ready/debug_plus_500/`

## Core decision

The current paper-ready package is built around:

`full physics features + ridge`

This is selected as the main paper method because it is supported by feature ablation, interpretability, and final evidence synthesis.

## Main use in paper

- Main result: physics-informed feature ablation.
- Supporting result: repeated-split robustness.
- Diagnostic result: damage-stratified reliability and high-damage underestimation.
- Control result: paired healthy-baseline and clean paired healthy-baseline diagnosis.

## Files copied

Total copied files: `{len(copied)}`

## Missing optional files

Total missing optional/required entries: `{len(missing)}`

See:

- `logs/copied_files.csv`
- `logs/missing_files.csv`
- `tables/selected_metric_highlights.csv`
- `figures/figure_inventory.csv`
"""

(TEXT / "paper_ready_summary.md").write_text(summary, encoding="utf-8")


# ---------------------------------------------------------------------
# 5. Write logs
# ---------------------------------------------------------------------

write_csv_rows(
    LOGS / "copied_files.csv",
    [{"source": s, "destination": d} for s, d in copied],
    ["source", "destination"],
)

write_csv_rows(
    LOGS / "missing_files.csv",
    [{"source": s, "destination": d, "required": str(r)} for s, d, r in missing],
    ["source", "destination", "required"],
)

print("\nDONE: paper-ready package created.")
print(f"Output: {OUT}")
print(f"Copied files: {len(copied)}")
print(f"Missing entries: {len(missing)}")
print("\nCheck summary:")
print(TEXT / "paper_ready_summary.md")
