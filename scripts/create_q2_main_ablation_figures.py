"""
Create paper-ready main ablation figures from T02_main_ablation_3000.csv.

English:
This script creates compact paper-ready figures for the main 3000-case ablation
result using the already generated ablation comparison table.

中文：
本脚本基于已经生成的 3000-case 主消融结果表，生成论文可直接使用的
主消融图像，不重新运行实验。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = Path("results/paper_ready/q2_fast_track/tables/T02_main_ablation_3000.csv")
FIGURE_DIR = Path("results/paper_ready/q2_fast_track/figures")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find columns {candidates}. Existing columns={list(df.columns)}")


def make_top_metric_plot(df: pd.DataFrame, metric: str, output_name: str, title: str) -> None:
    feature_col = find_col(df, ["feature_set", "base_feature_set", "features"])
    model_col = find_col(df, ["estimator", "regressor", "model"])

    if metric not in df.columns:
        raise KeyError(f"Missing metric column: {metric}")

    plot_df = df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    plot_df["config"] = plot_df[feature_col].astype(str) + " + " + plot_df[model_col].astype(str)

    top = plot_df.sort_values(metric, ascending=True).head(10).copy()
    top = top.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["config"], top[metric])
    ax.set_xlabel(metric)
    ax.set_ylabel("Configuration")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / output_name
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print(f"Saved: {out}")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    make_top_metric_plot(
        df,
        metric="test_mae",
        output_name="F01_main_ablation_top10_test_mae.png",
        title="Main ablation on the 3000-case dataset: top configurations by test MAE",
    )

    make_top_metric_plot(
        df,
        metric="test_rmse",
        output_name="F01_main_ablation_top10_test_rmse.png",
        title="Main ablation on the 3000-case dataset: top configurations by test RMSE",
    )


if __name__ == "__main__":
    main()
