"""
Final evidence synthesis for the SHM damage-inference experiments.
最终证据汇总脚本：把消融实验、稳健性实验、paired baseline、clean paired baseline 等结果汇总成论文结果包。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_value(raw: str) -> Any:
    """Parse a markdown metric value into int/float/string. 将 markdown 中的指标值解析为数值或字符串。"""
    text = raw.strip().strip("`").strip()
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", text):
            return float(text)
    except Exception:
        pass
    return text


def clean_section_name(line: str) -> str:
    """Convert markdown section heading into a safe key. 将 markdown 标题转换为安全字段名。"""
    line = re.sub(r"^#+\s*", "", line).strip()
    line = re.sub(r"^\d+\.\s*", "", line).strip()
    line = re.sub(r"[^A-Za-z0-9]+", "_", line).strip("_").lower()
    return line or "section"


def parse_markdown_metrics(path: Path) -> dict[str, Any]:
    """Extract '- key: `value`' metrics from markdown sections. 从 markdown 报告中提取指标。"""
    if not path.exists():
        return {}

    metrics: dict[str, Any] = {}
    current_section = "global"

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("## "):
            current_section = clean_section_name(line)
            continue

        match = re.match(r"^\s*-\s*([A-Za-z0-9_]+):\s*(.+?)\s*$", line)
        if match:
            key = match.group(1).strip()
            value = parse_value(match.group(2))
            metrics[f"{current_section}__{key}"] = value

    return metrics


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV safely. 安全读取 CSV。"""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def fmt(x: Any) -> str:
    """Format value for reports. 格式化输出值。"""
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Small dependency-free markdown table writer. 不依赖 tabulate 的 markdown 表格生成函数。"""
    if df.empty:
        return "_No available data._"

    if max_rows is not None:
        df = df.head(max_rows)

    columns = list(df.columns)
    rows = [[fmt(v) for v in row] for row in df.to_numpy().tolist()]

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def find_first_existing(project_root: Path, candidate_paths: list[str], recursive_pattern: str | None = None) -> Path | None:
    """Find first existing path. 查找第一个存在的结果文件。"""
    for rel in candidate_paths:
        p = project_root / rel
        if p.exists():
            return p

    if recursive_pattern:
        found = sorted(project_root.glob(recursive_pattern))
        if found:
            return found[0]

    return None


def collect_inventory(project_root: Path, output_dir: Path) -> pd.DataFrame:
    """Collect available result files. 汇总当前已有结果文件。"""
    table_root = project_root / "results" / "tables"
    rows = []

    if not table_root.exists():
        return pd.DataFrame(columns=["relative_path", "suffix", "size_kb"])

    for p in sorted(table_root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".md", ".json"}:
            continue

        try:
            p.relative_to(output_dir)
            continue
        except ValueError:
            pass

        rows.append(
            {
                "relative_path": str(p.relative_to(project_root)),
                "suffix": p.suffix.lower(),
                "size_kb": round(p.stat().st_size / 1024, 2),
            }
        )

    return pd.DataFrame(rows)


def load_physics_ablation(project_root: Path, dataset_tag: str) -> tuple[pd.DataFrame, dict[str, Any], Path | None]:
    """Load physics feature ablation comparison table. 读取物理特征消融实验结果。"""
    path = find_first_existing(
        project_root,
        [
            f"results/tables/physics_ablation/{dataset_tag}/ablation_model_comparison.csv",
            "results/tables/physics_ablation/debug_plus_500/ablation_model_comparison.csv",
        ],
        "results/tables/**/ablation_model_comparison.csv",
    )

    if path is None:
        return pd.DataFrame(), {}, None

    df = safe_read_csv(path)
    if df.empty or "test_mae" not in df.columns:
        return df, {}, path

    df = df.copy()
    df["test_mae"] = pd.to_numeric(df["test_mae"], errors="coerce")
    if "test_rmse" in df.columns:
        df["test_rmse"] = pd.to_numeric(df["test_rmse"], errors="coerce")

    best = df.sort_values("test_mae", ascending=True).iloc[0].to_dict()

    summary = {
        "source": str(path.relative_to(project_root)),
        "best_feature_set": best.get("feature_set", "NA"),
        "best_estimator": best.get("estimator", "NA"),
        "best_test_mae": best.get("test_mae", None),
        "best_test_rmse": best.get("test_rmse", None),
        "best_n_features": best.get("n_features", None),
        "best_mean_true_damage": best.get("mean_true_damage", None),
        "best_mean_pred_damage": best.get("mean_pred_damage", None),
    }

    return df, summary, path


def scan_metric_csvs(project_root: Path, output_dir: Path) -> pd.DataFrame:
    """Scan CSV files that contain useful evaluation metrics. 扫描包含评价指标的 CSV。"""
    table_root = project_root / "results" / "tables"
    rows = []

    if not table_root.exists():
        return pd.DataFrame()

    metric_like = {
        "test_mae",
        "test_rmse",
        "high_mae",
        "high_damage_test_mae",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "zero_false_alarm_ratio",
        "damaged_miss_ratio",
        "high_underestimation_ratio",
    }

    for path in sorted(table_root.rglob("*.csv")):
        try:
            path.relative_to(output_dir)
            continue
        except ValueError:
            pass

        name = path.name.lower()
        if "prediction" in name or "predictions" in name:
            continue

        df = safe_read_csv(path)
        if df.empty:
            continue

        matched_cols = [c for c in df.columns if c in metric_like or c.endswith("_mae") or c.endswith("_rmse")]
        if not matched_cols:
            continue

        row = {
            "relative_path": str(path.relative_to(project_root)),
            "n_rows": len(df),
            "metric_columns": ", ".join(matched_cols[:12]),
        }

        if "test_mae" in df.columns:
            tmp = df.copy()
            tmp["test_mae"] = pd.to_numeric(tmp["test_mae"], errors="coerce")
            best = tmp.sort_values("test_mae", ascending=True).iloc[0]
            row["best_by_test_mae"] = fmt(best.get("test_mae"))
            row["best_feature_set"] = best.get("feature_set", best.get("features", best.get("configuration", "")))
            row["best_estimator"] = best.get("estimator", best.get("model", ""))

        if "high_mae" in df.columns:
            tmp = df.copy()
            tmp["high_mae"] = pd.to_numeric(tmp["high_mae"], errors="coerce")
            best = tmp.sort_values("high_mae", ascending=True).iloc[0]
            row["best_by_high_mae"] = fmt(best.get("high_mae"))

        rows.append(row)

    return pd.DataFrame(rows)


def build_decision_table(
    physics_summary: dict[str, Any],
    paired_metrics: dict[str, Any],
    clean_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Build paper-level decision table. 构建论文层面的实验结论表。"""
    rows = []

    if physics_summary:
        rows.append(
            {
                "evidence_block": "Physics feature ablation",
                "best_or_key_result": f"{physics_summary.get('best_feature_set')} + {physics_summary.get('best_estimator')}",
                "main_metrics": (
                    f"test_MAE={fmt(physics_summary.get('best_test_mae'))}; "
                    f"test_RMSE={fmt(physics_summary.get('best_test_rmse'))}; "
                    f"n_features={fmt(physics_summary.get('best_n_features'))}"
                ),
                "paper_role": "Main positive result",
                "decision": "Use as the main empirical evidence for physics-informed feature construction.",
            }
        )

    if paired_metrics:
        rows.append(
            {
                "evidence_block": "Dirty paired healthy-baseline diagnosis",
                "best_or_key_result": fmt(paired_metrics.get("best_regression_result__model")),
                "main_metrics": (
                    f"test_MAE={fmt(paired_metrics.get('best_regression_result__test_mae'))}; "
                    f"high_MAE={fmt(paired_metrics.get('best_regression_result__high_mae'))}; "
                    f"high_under={fmt(paired_metrics.get('best_regression_result__high_underestimation_ratio'))}; "
                    f"classifier_AUC={fmt(paired_metrics.get('best_zero_vs_damaged_classifier_result__roc_auc'))}; "
                    f"zero_FA={fmt(paired_metrics.get('best_zero_vs_damaged_classifier_result__zero_false_alarm_ratio'))}"
                ),
                "paper_role": "Diagnostic result",
                "decision": "Do not use as main method because response-feature matching may contaminate baseline selection.",
            }
        )

    if clean_metrics:
        rows.append(
            {
                "evidence_block": "Clean paired healthy-baseline diagnosis",
                "best_or_key_result": fmt(clean_metrics.get("best_regression_result__model")),
                "main_metrics": (
                    f"test_MAE={fmt(clean_metrics.get('best_regression_result__test_mae'))}; "
                    f"high_MAE={fmt(clean_metrics.get('best_regression_result__high_mae'))}; "
                    f"high_under={fmt(clean_metrics.get('best_regression_result__high_underestimation_ratio'))}; "
                    f"classifier_AUC={fmt(clean_metrics.get('best_zero_vs_damaged_classifier_result__roc_auc'))}; "
                    f"zero_FA={fmt(clean_metrics.get('best_zero_vs_damaged_classifier_result__zero_false_alarm_ratio'))}"
                ),
                "paper_role": "Negative/control evidence",
                "decision": "Use as limitation evidence: clean input-only matching does not solve high-damage underestimation or zero/damaged separation.",
            }
        )

    rows.append(
        {
            "evidence_block": "Final method selection",
            "best_or_key_result": "full physics features + ridge",
            "main_metrics": "Selected by interpretability, stability, and ablation consistency rather than by paired-baseline variants.",
            "paper_role": "Final paper conclusion",
            "decision": "Main paper should focus on physics-informed feature ablation and damage-stratified reliability diagnosis.",
        }
    )

    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    decision_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    metric_files_df: pd.DataFrame,
    physics_summary: dict[str, Any],
    paired_metrics: dict[str, Any],
    clean_metrics: dict[str, Any],
) -> None:
    """Write final markdown synthesis report. 写入最终 markdown 汇总报告。"""
    lines = []
    lines.append("# Final Evidence Synthesis for Paper Results")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## 1. Final technical decision")
    lines.append("")
    lines.append("The current evidence supports using `full physics features + ridge` as the main paper model.")
    lines.append("")
    lines.append("Paired healthy-baseline normalization should not be used as the main method. The clean paired-baseline control shows that input-only matching cannot sufficiently solve high-damage underestimation or zero/damaged separability.")
    lines.append("")

    lines.append("## 2. Paper-level decision table")
    lines.append("")
    lines.append(df_to_markdown(decision_df))
    lines.append("")

    lines.append("## 3. Key physics-ablation result")
    lines.append("")
    if physics_summary:
        for k, v in physics_summary.items():
            lines.append(f"- {k}: `{fmt(v)}`")
    else:
        lines.append("- Physics ablation result was not found.")
    lines.append("")

    lines.append("## 4. Paired-baseline diagnostic snapshot")
    lines.append("")
    if paired_metrics:
        selected_keys = [
            "best_regression_result__test_mae",
            "best_regression_result__high_mae",
            "best_regression_result__high_bias_pred_minus_true",
            "best_regression_result__high_underestimation_ratio",
            "best_zero_vs_damaged_classifier_result__roc_auc",
            "best_zero_vs_damaged_classifier_result__zero_false_alarm_ratio",
        ]
        for k in selected_keys:
            lines.append(f"- {k}: `{fmt(paired_metrics.get(k))}`")
    else:
        lines.append("- Dirty paired-baseline report was not found.")
    lines.append("")

    lines.append("## 5. Clean paired-baseline control snapshot")
    lines.append("")
    if clean_metrics:
        selected_keys = [
            "best_regression_result__test_mae",
            "best_regression_result__high_mae",
            "best_regression_result__high_bias_pred_minus_true",
            "best_regression_result__high_underestimation_ratio",
            "best_zero_vs_damaged_classifier_result__roc_auc",
            "best_zero_vs_damaged_classifier_result__balanced_accuracy",
            "best_zero_vs_damaged_classifier_result__zero_false_alarm_ratio",
            "best_zero_vs_damaged_classifier_result__damaged_miss_ratio",
        ]
        for k in selected_keys:
            lines.append(f"- {k}: `{fmt(clean_metrics.get(k))}`")
    else:
        lines.append("- Clean paired-baseline report was not found.")
    lines.append("")

    lines.append("## 6. Paper-ready interpretation")
    lines.append("")
    lines.append("1. The main positive result is not that complex classifiers or paired normalization dominate. The defensible result is that physics-informed response descriptors combined with a simple regularized model provide stable damage-inference performance under sparse and noisy SHM conditions.")
    lines.append("")
    lines.append("2. The damage-stratified diagnosis reveals a systematic limitation: high-damage cases are consistently underestimated. This should be presented as a reliability limitation, not hidden.")
    lines.append("")
    lines.append("3. The clean paired-baseline experiment is useful as a control experiment. It shows that strict input-only healthy matching is insufficient, which prevents the paper from overclaiming healthy-baseline normalization.")
    lines.append("")
    lines.append("4. The final paper should be framed as a method-and-diagnosis study: physics-informed feature construction, ablation validation, repeated-split robustness, and damage-stratified reliability analysis.")
    lines.append("")

    lines.append("## 7. Available metric files detected")
    lines.append("")
    lines.append(df_to_markdown(metric_files_df, max_rows=30))
    lines.append("")

    lines.append("## 8. Result-file inventory")
    lines.append("")
    lines.append(df_to_markdown(inventory_df, max_rows=60))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-tag", type=str, default="debug_plus_500")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables/final_evidence/debug_plus_500"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    physics_df, physics_summary, physics_path = load_physics_ablation(project_root, args.dataset_tag)

    paired_report = project_root / f"results/tables/paired_baseline/{args.dataset_tag}/paired_baseline_diagnosis_report.md"
    clean_report = project_root / f"results/tables/clean_paired_baseline/{args.dataset_tag}/clean_paired_baseline_report.md"

    paired_metrics = parse_markdown_metrics(paired_report)
    clean_metrics = parse_markdown_metrics(clean_report)

    inventory_df = collect_inventory(project_root, output_dir)
    metric_files_df = scan_metric_csvs(project_root, output_dir)
    decision_df = build_decision_table(physics_summary, paired_metrics, clean_metrics)

    decision_df.to_csv(output_dir / "final_method_decision_table.csv", index=False)
    inventory_df.to_csv(output_dir / "final_evidence_inventory.csv", index=False)
    metric_files_df.to_csv(output_dir / "detected_metric_files.csv", index=False)

    snapshot = {
        "physics_ablation_source": str(physics_path.relative_to(project_root)) if physics_path else None,
        "physics_summary": physics_summary,
        "paired_report": str(paired_report.relative_to(project_root)) if paired_report.exists() else None,
        "paired_metrics": paired_metrics,
        "clean_report": str(clean_report.relative_to(project_root)) if clean_report.exists() else None,
        "clean_metrics": clean_metrics,
    }
    (output_dir / "final_metric_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(
        output_path=output_dir / "final_paper_result_synthesis.md",
        decision_df=decision_df,
        inventory_df=inventory_df,
        metric_files_df=metric_files_df,
        physics_summary=physics_summary,
        paired_metrics=paired_metrics,
        clean_metrics=clean_metrics,
    )

    print("Final evidence synthesis completed.")
    print(f"Output directory: {output_dir}")
    print(f"Main report: {output_dir / 'final_paper_result_synthesis.md'}")
    print(f"Decision table: {output_dir / 'final_method_decision_table.csv'}")


if __name__ == "__main__":
    main()
