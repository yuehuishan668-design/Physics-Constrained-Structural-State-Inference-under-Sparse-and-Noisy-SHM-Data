"""
File location:
    src/evaluation/compare_metrics_table.py

Purpose:
    Compare metrics from multiple experiment folders.

中文说明：
    本文件用于把多个实验目录中的指标 JSON 汇总成一个 CSV 对比表。
    后续论文实验表格可以基于这个脚本继续扩展。
"""

from __future__ import annotations
# English: Postpone type-hint evaluation.
# 中文：延迟类型注解解析。

import argparse
# English: Parse command-line arguments.
# 中文：解析命令行参数。

import csv
# English: Write CSV tables.
# 中文：写入 CSV 表格。

import json
# English: Read JSON metrics.
# 中文：读取 JSON 指标文件。

from pathlib import Path
# English: Handle paths.
# 中文：路径处理。

from typing import Any, Dict, List
# English: Type hints.
# 中文：类型标注。


def read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON file. / 读取一个 JSON 文件。"""
    with path.open("r", encoding="utf-8") as f:
        # English: Open JSON file.
        # 中文：打开 JSON 文件。
        return json.load(f)
        # English: Parse JSON content.
        # 中文：解析 JSON 内容。


def find_metrics_json(experiment_dir: Path) -> Path | None:
    """Find a metrics JSON file in one experiment directory. / 在实验目录中查找指标 JSON。"""
    candidates = [
        "mlp_debug_metrics.json",
        "lstm_metrics.json",
        "two_head_lstm_metrics.json",
        "metrics.json",
    ]
    # English: Common metric file names used in this project.
    # 中文：本项目中常见的指标文件名。
    for name in candidates:
        path = experiment_dir / name
        # English: Build candidate path.
        # 中文：构造候选路径。
        if path.exists():
            return path
            # English: Return the first valid metrics file.
            # 中文：返回第一个存在的指标文件。
    json_files = sorted(experiment_dir.glob("*metrics*.json"))
    # English: Fallback search.
    # 中文：兜底搜索包含 metrics 的 JSON 文件。
    if json_files:
        return json_files[0]
    return None
    # English: Return None if no metrics file is found.
    # 中文：如果没有找到则返回 None。


def get_test_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract test metrics across schema versions. / 兼容不同脚本的测试集指标结构。"""
    if "test_metrics" in metrics:
        return metrics["test_metrics"]
        # English: Newer scripts store metrics here.
        # 中文：较新脚本将测试指标放在 test_metrics 中。
    if "test" in metrics:
        return metrics["test"]
        # English: Alternative schema.
        # 中文：另一种可能的结构。
    return metrics
    # English: Fallback.
    # 中文：兜底返回整个字典。


def extract_row(name: str, metrics_path: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract one row for comparison. / 提取一行对比结果。"""
    test = get_test_metrics(metrics)
    # English: Get test metrics.
    # 中文：获取测试集指标。
    return {
        "experiment": name,
        "metrics_path": str(metrics_path),
        "model": metrics.get("model", "unknown"),
        "condition_mode": metrics.get("condition_mode", "not_applicable"),
        "output_mode": metrics.get("output_mode", "not_applicable"),
        "final_mode": metrics.get("final_mode", "not_applicable"),
        "prob_threshold": metrics.get("prob_threshold", "not_applicable"),
        "best_epoch": metrics.get("best_epoch", "not_applicable"),
        "best_val_mse_or_loss": metrics.get(
            "best_val_loss_mse",
            metrics.get("best_val_mse", metrics.get("best_val_total_loss", metrics.get("best_val_loss", "not_applicable"))),
        ),
        "test_mae": test.get("mae_overall", test.get("test_mae", "not_applicable")),
        "test_rmse": test.get("rmse_overall", test.get("test_rmse", "not_applicable")),
        "negative_prediction_ratio": test.get("negative_prediction_ratio", "not_applicable"),
        "predicted_positive_ratio": test.get("predicted_positive_ratio", "not_applicable"),
        "true_positive_ratio": test.get("true_positive_ratio", "not_applicable"),
        "precision_macro": test.get("precision_macro", "not_applicable"),
        "recall_macro": test.get("recall_macro", "not_applicable"),
        "f1_macro": test.get("f1_macro", "not_applicable"),
        "exact_mask_match_ratio": test.get("exact_damage_mask_match_ratio", "not_applicable"),
    }
    # English: Return standardized row.
    # 中文：返回标准化后的实验指标行。


def parse_experiment_arg(item: str) -> tuple[str, Path]:
    """Parse name=path or path. / 解析 name=path 或纯路径格式。"""
    if "=" in item:
        name, path_str = item.split("=", 1)
        return name, Path(path_str)
        # English: Explicit experiment name.
        # 中文：用户显式指定实验名。
    path = Path(item)
    return path.name, path
    # English: Use folder name as experiment name.
    # 中文：默认使用文件夹名作为实验名。


def main() -> None:
    """Run comparison. / 执行实验对比。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment dirs, format: name=path or path")
    parser.add_argument("--output", type=str, default="results/tables/model_comparison_summary.csv")
    args = parser.parse_args()
    # English: Parse command-line arguments.
    # 中文：解析命令行参数。

    rows: List[Dict[str, Any]] = []
    # English: Store comparison rows.
    # 中文：保存对比表行。
    for item in args.experiments:
        name, experiment_dir = parse_experiment_arg(item)
        # English: Parse experiment name and directory.
        # 中文：解析实验名称和目录。
        metrics_path = find_metrics_json(experiment_dir)
        # English: Locate metrics JSON.
        # 中文：查找指标 JSON。
        if metrics_path is None:
            print(f"{name}: missing metrics file in {experiment_dir}")
            continue
        metrics = read_json(metrics_path)
        # English: Read metrics.
        # 中文：读取指标。
        rows.append(extract_row(name, metrics_path, metrics))
        # English: Add one row.
        # 中文：添加一行。

    if not rows:
        raise RuntimeError("No valid experiment metrics found.")
        # English: Stop if nothing was found.
        # 中文：如果没有任何有效实验，则报错。

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # English: Create output directory.
    # 中文：创建输出目录。
    fieldnames = list(rows[0].keys())
    # English: CSV columns.
    # 中文：CSV 列名。
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # English: Write table.
        # 中文：写入对比表。

    print("Model comparison completed.")
    print(f"Saved to: {output_path}")
    print("")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[key]) for key in fieldnames))
    # English: Print the table to terminal as well.
    # 中文：同时在终端打印对比表。


if __name__ == "__main__":
    main()
    # English: Execute main when run as module/script.
    # 中文：作为模块或脚本运行时执行 main。
