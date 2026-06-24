"""
File location:
    src/evaluation/plot_mlp_predictions.py

Purpose:
    Visualize MLP prediction results.

Input:
    results/tables/mlp_debug_predictions_test.csv

Outputs:
    results/figures/mlp_predictions/
        test_true_vs_pred_scatter.png
        test_error_heatmap.png
        test_case_<case_id>_true_pred_bar.png
        test_sample_mae_distribution.png

中文说明：
    本文件用于诊断 MLP 预测结果：
        1. 真实值 vs 预测值散点图；
        2. 每个样本、每层楼的误差热力图；
        3. 每个 case 的真实值-预测值柱状图；
        4. 样本级 MAE 分布；
        5. 负值预测比例。
"""

from __future__ import annotations
# 延迟类型注解解析。

import argparse
# argparse：读取命令行参数。

import csv
# csv：读取预测结果表格。

from pathlib import Path
# Path：处理文件路径。

from typing import Dict, List
# Dict、List：类型注解。

import matplotlib.pyplot as plt
# matplotlib：绘图。

import numpy as np
# numpy：数组计算。


def read_prediction_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read prediction CSV. 中文说明：读取 train_mlp.py 输出的预测 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")

    rows: List[dict[str, str]] = []
    # rows：保存 CSV 每一行。
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        raise ValueError(f"No rows found in {path}")

    story_indices = []
    # story_indices：楼层编号，例如 [1,2,3,4]。
    for key in rows[0].keys():
        if key.startswith("y_true_story_"):
            story_indices.append(int(key.replace("y_true_story_", "")))
    story_indices = sorted(story_indices)

    y_true, y_pred, errors = [], [], []
    sample_mae, sample_rmse = [], []
    case_ids, sample_ids = [], []

    for row in rows:
        true_row, pred_row, error_row = [], [], []
        for story_idx in story_indices:
            true_value = float(row[f"y_true_story_{story_idx}"])
            pred_value = float(row[f"y_pred_story_{story_idx}"])
            error_value = pred_value - true_value
            true_row.append(true_value)
            pred_row.append(pred_value)
            error_row.append(error_value)

        y_true.append(true_row)
        y_pred.append(pred_row)
        errors.append(error_row)
        case_ids.append(int(row.get("case_id", row.get("sample_id", 0))))
        sample_ids.append(int(row.get("sample_id", len(sample_ids))))
        sample_mae.append(float(row.get("sample_mae", np.mean(np.abs(error_row)))))
        sample_rmse.append(float(row.get("sample_rmse", np.sqrt(np.mean(np.array(error_row) ** 2)))))

    return {
        "y_true": np.array(y_true, dtype=float),
        "y_pred": np.array(y_pred, dtype=float),
        "error": np.array(errors, dtype=float),
        "sample_mae": np.array(sample_mae, dtype=float),
        "sample_rmse": np.array(sample_rmse, dtype=float),
        "case_id": np.array(case_ids, dtype=int),
        "sample_id": np.array(sample_ids, dtype=int),
        "story_indices": np.array(story_indices, dtype=int),
    }


def plot_true_vs_pred_scatter(data: Dict[str, np.ndarray], output_path: Path, split_name: str) -> None:
    """Plot true-pred scatter. 中文说明：绘制真实值与预测值散点图。"""
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    story_indices = data["story_indices"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    for local_idx, story_idx in enumerate(story_indices):
        plt.scatter(y_true[:, local_idx], y_pred[:, local_idx], label=f"Story {story_idx}")

    min_value = min(float(y_true.min()), float(y_pred.min()), 0.0)
    max_value = max(float(y_true.max()), float(y_pred.max()), 0.35)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", label="Ideal y=x")
    plt.xlabel("True stiffness degradation ratio")
    plt.ylabel("Predicted stiffness degradation ratio")
    plt.title(f"{split_name}: true vs predicted damage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_error_heatmap(data: Dict[str, np.ndarray], output_path: Path, split_name: str) -> None:
    """Plot error heatmap. 中文说明：绘制误差热力图。"""
    error = data["error"]
    case_ids = data["case_id"]
    story_indices = data["story_indices"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, max(3, 0.45 * error.shape[0] + 2)))
    image = plt.imshow(error, aspect="auto")
    plt.colorbar(image, label="Prediction error")
    plt.xticks(np.arange(len(story_indices)), [f"Story {idx}" for idx in story_indices])
    plt.yticks(np.arange(len(case_ids)), [f"case {case_id}" for case_id in case_ids])
    plt.xlabel("Story")
    plt.ylabel("Case ID")
    plt.title(f"{split_name}: prediction error heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_case_bar_charts(data: Dict[str, np.ndarray], output_dir: Path, split_name: str) -> None:
    """Plot true vs predicted bars for each case. 中文说明：为每个 case 绘制柱状图。"""
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    case_ids = data["case_id"]
    story_indices = data["story_indices"]
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(story_indices))
    width = 0.35
    for i, case_id in enumerate(case_ids):
        plt.figure(figsize=(7, 4))
        plt.bar(x - width / 2, y_true[i], width, label="True")
        plt.bar(x + width / 2, y_pred[i], width, label="Predicted")
        plt.xticks(x, [f"Story {idx}" for idx in story_indices])
        plt.xlabel("Story")
        plt.ylabel("Stiffness degradation ratio")
        max_y = max(float(y_true[i].max()), float(y_pred[i].max()), 0.35)
        min_y = min(float(y_pred[i].min()), 0.0)
        plt.ylim(min_y - 0.03, max_y + 0.05)
        plt.title(f"{split_name}: case {case_id} true vs predicted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{split_name}_case_{case_id:04d}_true_pred_bar.png", dpi=200)
        plt.close()


def plot_sample_mae_distribution(data: Dict[str, np.ndarray], output_path: Path, split_name: str) -> None:
    """Plot sample MAE distribution. 中文说明：绘制样本级 MAE 分布。"""
    sample_mae = data["sample_mae"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(sample_mae, bins=min(10, max(3, len(sample_mae))))
    plt.xlabel("Sample MAE")
    plt.ylabel("Count")
    plt.title(f"{split_name}: sample-level MAE distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_summary(data: Dict[str, np.ndarray], split_name: str) -> None:
    """Print summary. 中文说明：打印诊断摘要。"""
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    error = y_pred - y_true
    print(f"\nPrediction diagnostics for split: {split_name}")
    print(f"Number of samples: {y_true.shape[0]}")
    print(f"Number of stories: {y_true.shape[1]}")
    print(f"MAE overall: {np.mean(np.abs(error)):.6f}")
    print(f"RMSE overall: {np.sqrt(np.mean(error ** 2)):.6f}")
    print(f"Negative prediction ratio: {np.mean(y_pred < 0.0):.6f}")
    print(f"Case IDs: {data['case_id'].tolist()}")


def run(csv_path: Path, output_dir: Path, split_name: str) -> None:
    """Run plotting pipeline. 中文说明：执行完整预测可视化流程。"""
    data = read_prediction_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_true_vs_pred_scatter(data, output_dir / f"{split_name}_true_vs_pred_scatter.png", split_name)
    plot_error_heatmap(data, output_dir / f"{split_name}_error_heatmap.png", split_name)
    plot_case_bar_charts(data, output_dir, split_name)
    plot_sample_mae_distribution(data, output_dir / f"{split_name}_sample_mae_distribution.png", split_name)
    print_summary(data, split_name)
    print(f"Figures saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/tables/mlp_debug_predictions_test.csv")
    parser.add_argument("--output-dir", type=str, default="results/figures/mlp_predictions")
    parser.add_argument("--split-name", type=str, default="test")
    args = parser.parse_args()
    run(Path(args.csv), Path(args.output_dir), args.split_name)


if __name__ == "__main__":
    main()
