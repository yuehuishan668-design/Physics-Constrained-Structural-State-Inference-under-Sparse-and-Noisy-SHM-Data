"""
Summarize noise robustness experiments.

English:
This script reads physics-ablation comparison tables generated under different
fixed noise levels and summarizes how model performance changes as noise increases.

中文：
本脚本读取不同固定噪声水平下的物理特征消融结果，并汇总模型性能随噪声增加
的变化趋势，用于支撑论文中的 noise robustness analysis。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def find_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Cannot find any candidate columns: {candidates}. Existing={list(df.columns)}")
    return None


def read_table(table_path: Path, noise_level: float, tag: str) -> pd.DataFrame:
    df = pd.read_csv(table_path)

    feature_col = find_column(df, ["feature_set", "base_feature_set", "features"])

    # Important:
    # In this project, "estimator" is ridge / elasticnet / random_forest.
    # The "model" column is often PhysicsSklearnRegressor, which is not the algorithm label.
    # 中文：本项目中 estimator 才是 ridge / elasticnet / random_forest，model 通常只是包装器名称。
    model_col = find_column(df, ["estimator", "regressor", "model"])

    mae_col = find_column(df, ["test_mae", "mae_test", "overall_mae", "test_mean_absolute_error"])
    rmse_col = find_column(df, ["test_rmse", "rmse_test", "overall_rmse", "test_root_mean_squared_error"], required=False)
    n_features_col = find_column(df, ["n_features", "feature_count"], required=False)
    damaged_mae_col = find_column(df, ["mae_on_damaged_entries", "mae_damaged", "damaged_mae"], required=False)
    zero_pred_col = find_column(df, ["mean_prediction_on_zero_entries"], required=False)

    out = pd.DataFrame({
        "noise_level": [noise_level] * len(df),
        "noise_percent": [noise_level * 100.0] * len(df),
        "tag": [tag] * len(df),
        "feature_set": df[feature_col].astype(str),
        "model": df[model_col].astype(str),
        "test_mae": pd.to_numeric(df[mae_col], errors="coerce"),
        "source_table": [str(table_path)] * len(df),
    })

    if rmse_col is not None:
        out["test_rmse"] = pd.to_numeric(df[rmse_col], errors="coerce")
    else:
        out["test_rmse"] = pd.NA

    if n_features_col is not None:
        out["n_features"] = pd.to_numeric(df[n_features_col], errors="coerce")
    else:
        out["n_features"] = pd.NA

    if damaged_mae_col is not None:
        out["mae_on_damaged_entries"] = pd.to_numeric(df[damaged_mae_col], errors="coerce")
    else:
        out["mae_on_damaged_entries"] = pd.NA

    if zero_pred_col is not None:
        out["mean_prediction_on_zero_entries"] = pd.to_numeric(df[zero_pred_col], errors="coerce")
    else:
        out["mean_prediction_on_zero_entries"] = pd.NA

    out["config_label"] = out["feature_set"] + " + " + out["model"]

    return out


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, sub in df.groupby("config_label"):
        sub = sub.sort_values("noise_percent")
        ax.plot(sub["noise_percent"], sub[metric], marker="o", label=label)

    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--noise-levels", nargs="+", required=True, type=float)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    args = parser.parse_args()

    if not (len(args.tables) == len(args.noise_levels) == len(args.tags)):
        raise ValueError("--tables, --noise-levels, and --tags must have the same length.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for table, noise, tag in zip(args.tables, args.noise_levels, args.tags):
        table_path = Path(table)
        if not table_path.exists():
            raise FileNotFoundError(f"Missing table: {table_path}")
        rows.append(read_table(table_path, noise_level=noise, tag=tag))

    all_results = pd.concat(rows, ignore_index=True)
    all_results = all_results.sort_values(["noise_percent", "test_mae"])

    all_path = args.output_dir / "noise_robustness_all_results.csv"
    all_results.to_csv(all_path, index=False)

    preferred = all_results[
        all_results["feature_set"].isin([
            "full",
            "no_meta",
            "physics_no_meta_core",
            "response_basic_only",
            "response_frequency",
            "response_correlation",
            "response_spatial",
        ])
        & all_results["model"].isin(["ridge", "elasticnet", "random_forest"])
    ].copy()

    preferred_path = args.output_dir / "noise_robustness_summary.csv"
    preferred.to_csv(preferred_path, index=False)

    best_by_noise = (
        all_results.sort_values(["noise_percent", "test_mae"])
        .groupby("noise_percent", as_index=False)
        .first()
    )

    best_path = args.output_dir / "noise_robustness_best_by_noise.csv"
    best_by_noise.to_csv(best_path, index=False)

    plot_df = preferred[
        preferred["config_label"].isin([
            "full + ridge",
            "no_meta + ridge",
            "response_basic_only + ridge",
            "full + elasticnet",
            "full + random_forest",
        ])
    ].copy()

    if not plot_df.empty:
        plot_metric(
            plot_df,
            metric="test_mae",
            output_path=args.figures_dir / "noise_vs_test_mae.png",
            title="Noise robustness measured by test MAE",
        )

        if plot_df["test_rmse"].notna().any():
            plot_metric(
                plot_df,
                metric="test_rmse",
                output_path=args.figures_dir / "noise_vs_test_rmse.png",
                title="Noise robustness measured by test RMSE",
            )

        if plot_df["mae_on_damaged_entries"].notna().any():
            plot_metric(
                plot_df,
                metric="mae_on_damaged_entries",
                output_path=args.figures_dir / "noise_vs_damaged_mae.png",
                title="Noise robustness measured by damaged-entry MAE",
            )

    lines = []
    lines.append("# Noise Robustness Summary")
    lines.append("")
    lines.append("## 1. Experiment setup")
    lines.append("")
    lines.append(f"- Number of noise levels: `{len(args.noise_levels)}`")
    lines.append(f"- Noise levels: `{args.noise_levels}`")
    lines.append("- Each noise level corresponds to an independently generated controlled simulation dataset.")
    lines.append("")
    lines.append("## 2. Best configuration at each noise level")
    lines.append("")
    lines.append(best_by_noise[[
        "noise_percent", "feature_set", "model", "test_mae", "test_rmse",
        "mae_on_damaged_entries", "n_features"
    ]].to_markdown(index=False))
    lines.append("")
    lines.append("## 3. Selected preferred configurations")
    lines.append("")
    compact = preferred[
        preferred["config_label"].isin([
            "full + ridge",
            "no_meta + ridge",
            "response_basic_only + ridge",
            "full + elasticnet",
            "full + random_forest",
        ])
    ].copy()
    lines.append(compact[[
        "noise_percent", "feature_set", "model", "test_mae", "test_rmse",
        "mae_on_damaged_entries", "mean_prediction_on_zero_entries", "n_features"
    ]].sort_values(["noise_percent", "test_mae"]).to_markdown(index=False))
    lines.append("")
    lines.append("## 4. Preliminary interpretation")
    lines.append("")
    lines.append("- `full + ridge` should be checked as the main configuration across noise levels.")
    lines.append("- Because each noise level uses an independently generated dataset, the trend does not need to be strictly monotonic.")
    lines.append("- If the overall trend increases with noise and `full + ridge` remains best or near-best, the result supports noise robustness.")
    lines.append("- If high-noise performance collapses, it should be reported as a limitation.")
    lines.append("")
    lines.append("中文解释：")
    lines.append("")
    lines.append("- 重点检查 `full + ridge` 是否在各噪声水平下仍然保持最优或接近最优。")
    lines.append("- 由于每个噪声水平对应独立生成的数据集，误差曲线不一定严格单调。")
    lines.append("- 如果整体误差随噪声升高而上升，且 `full + ridge` 保持领先，则可支撑噪声鲁棒性结论。")
    lines.append("- 如果高噪声下性能崩溃，应作为局限性写入论文。")

    report_path = args.output_dir / "noise_robustness_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Noise robustness summary completed.")
    print(f"All results: {all_path}")
    print(f"Preferred summary: {preferred_path}")
    print(f"Best by noise: {best_path}")
    print(f"Report: {report_path}")
    print(f"Figures: {args.figures_dir}")
    print("")
    print("Best by noise:")
    print(best_by_noise[["noise_percent", "feature_set", "model", "test_mae", "test_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
