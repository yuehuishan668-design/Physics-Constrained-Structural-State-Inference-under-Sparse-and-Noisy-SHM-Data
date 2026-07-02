"""
Summarize sensor sparsity stress-test results.

English:
This script reads physics-ablation comparison tables under different sensor layouts
and summarizes the degradation caused by reducing available monitoring channels.

中文：
本脚本读取不同传感器布设条件下的物理特征消融结果，并汇总传感器数量减少
导致的性能退化，用于支撑 sparse SHM data 的论文实验。
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
        raise KeyError(f"Cannot find columns {candidates}. Existing columns={list(df.columns)}")
    return None


def read_table(table_path: Path, sensor_count: int, sensor_layout: str, tag: str) -> pd.DataFrame:
    df = pd.read_csv(table_path)

    feature_col = find_column(df, ["feature_set", "base_feature_set", "features"])
    model_col = find_column(df, ["estimator", "regressor", "model"])
    mae_col = find_column(df, ["test_mae", "mae_test", "overall_mae"])
    rmse_col = find_column(df, ["test_rmse", "rmse_test", "overall_rmse"], required=False)
    n_features_col = find_column(df, ["n_features", "feature_count"], required=False)
    damaged_mae_col = find_column(df, ["mae_on_damaged_entries", "mae_damaged", "damaged_mae"], required=False)
    zero_pred_col = find_column(df, ["mean_prediction_on_zero_entries"], required=False)

    out = pd.DataFrame({
        "sensor_count": [sensor_count] * len(df),
        "sensor_layout": [sensor_layout] * len(df),
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
        sub = sub.sort_values("sensor_count")
        ax.plot(sub["sensor_count"], sub[metric], marker="o", label=label)

    ax.set_xlabel("Number of available sensors")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(sorted(df["sensor_count"].unique()))
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--sensor-counts", nargs="+", required=True, type=int)
    parser.add_argument("--sensor-layouts", nargs="+", required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    args = parser.parse_args()

    if not (
        len(args.tables)
        == len(args.sensor_counts)
        == len(args.sensor_layouts)
        == len(args.tags)
    ):
        raise ValueError("--tables, --sensor-counts, --sensor-layouts, and --tags must have equal length.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for table, count, layout, tag in zip(args.tables, args.sensor_counts, args.sensor_layouts, args.tags):
        path = Path(table)
        if not path.exists():
            raise FileNotFoundError(f"Missing table: {path}")
        rows.append(read_table(path, sensor_count=count, sensor_layout=layout, tag=tag))

    all_results = pd.concat(rows, ignore_index=True)
    all_results = all_results.sort_values(["sensor_count", "test_mae"], ascending=[False, True])

    all_path = args.output_dir / "sensor_sparsity_all_results.csv"
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

    summary_path = args.output_dir / "sensor_sparsity_summary.csv"
    preferred.to_csv(summary_path, index=False)

    best_by_sensor = (
        all_results.sort_values(["sensor_count", "test_mae"], ascending=[False, True])
        .groupby("sensor_count", as_index=False)
        .first()
        .sort_values("sensor_count", ascending=False)
    )

    best_path = args.output_dir / "sensor_sparsity_best_by_sensor_count.csv"
    best_by_sensor.to_csv(best_path, index=False)

    compact = preferred[
        preferred["config_label"].isin([
            "full + ridge",
            "no_meta + ridge",
            "response_basic_only + ridge",
            "full + elasticnet",
            "full + random_forest",
        ])
    ].copy()

    if not compact.empty:
        plot_metric(
            compact,
            metric="test_mae",
            output_path=args.figures_dir / "sensor_count_vs_test_mae.png",
            title="Sensor sparsity stress test measured by test MAE",
        )

        if compact["test_rmse"].notna().any():
            plot_metric(
                compact,
                metric="test_rmse",
                output_path=args.figures_dir / "sensor_count_vs_test_rmse.png",
                title="Sensor sparsity stress test measured by test RMSE",
            )

        if compact["mae_on_damaged_entries"].notna().any():
            plot_metric(
                compact,
                metric="mae_on_damaged_entries",
                output_path=args.figures_dir / "sensor_count_vs_damaged_mae.png",
                title="Sensor sparsity stress test measured by damaged-entry MAE",
            )

    lines = []
    lines.append("# Sensor Sparsity Stress-Test Summary")
    lines.append("")
    lines.append("## 1. Experiment setup")
    lines.append("")
    lines.append("- Sensor sparsity was simulated by zero-masking unavailable response channels.")
    lines.append("- The output damage labels remained four-story damage vectors.")
    lines.append("- The response tensor shape was preserved to keep downstream feature extraction compatible.")
    lines.append("")
    lines.append("## 2. Best configuration by sensor count")
    lines.append("")
    lines.append(best_by_sensor[[
        "sensor_count", "sensor_layout", "feature_set", "model",
        "test_mae", "test_rmse", "mae_on_damaged_entries", "n_features"
    ]].to_markdown(index=False))
    lines.append("")
    lines.append("## 3. Selected preferred configurations")
    lines.append("")
    lines.append(compact[[
        "sensor_count", "sensor_layout", "feature_set", "model",
        "test_mae", "test_rmse", "mae_on_damaged_entries",
        "mean_prediction_on_zero_entries", "n_features"
    ]].sort_values(["sensor_count", "test_mae"], ascending=[False, True]).to_markdown(index=False))
    lines.append("")
    lines.append("## 4. Preliminary interpretation")
    lines.append("")
    lines.append("- If `full + ridge` remains best or near-best as sensor count decreases, the feature-construction conclusion is robust to sensor sparsity.")
    lines.append("- If errors increase as sensors are removed, the experiment supports the sparse-monitoring interpretation.")
    lines.append("- If one-sensor performance collapses, this should be reported as the practical limit of the current method.")
    lines.append("- This experiment should be described as a zero-masking sensor sparsity stress test, not as an optimal missing-data imputation method.")
    lines.append("")
    lines.append("中文解释：")
    lines.append("")
    lines.append("- 如果 `full + ridge` 在传感器数量减少时仍保持最优或接近最优，说明物理特征构建结论对传感器稀疏性具有稳健性。")
    lines.append("- 如果传感器减少导致误差上升，则实验能够支撑 sparse monitoring data 的论文主张。")
    lines.append("- 如果单传感器性能明显崩溃，应作为当前方法的实用边界写入论文。")
    lines.append("- 本实验应被表述为 zero-masking 传感器稀疏性压力测试，而不是最优缺失数据填补方法。")

    report_path = args.output_dir / "sensor_sparsity_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Sensor sparsity summary completed.")
    print(f"All results: {all_path}")
    print(f"Preferred summary: {summary_path}")
    print(f"Best by sensor count: {best_path}")
    print(f"Report: {report_path}")
    print(f"Figures: {args.figures_dir}")
    print("")
    print("Best by sensor count:")
    print(best_by_sensor[[
        "sensor_count", "sensor_layout", "feature_set", "model", "test_mae", "test_rmse"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
