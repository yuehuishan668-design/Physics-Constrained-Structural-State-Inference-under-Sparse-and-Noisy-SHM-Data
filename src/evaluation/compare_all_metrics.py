"""
Compare multiple model metric files.

中文说明：
把 MLP、LSTM、two-head LSTM、sklearn physics models 等实验结果汇总到一个 CSV。
只要目录下存在 metrics.json、mlp_debug_metrics.json、lstm_metrics.json、two_head_lstm_metrics.json 等文件，脚本会自动尝试读取。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

COMMON_METRIC_NAMES = [
    "metrics.json",
    "mlp_debug_metrics.json",
    "lstm_metrics.json",
    "two_head_lstm_metrics.json",
]


def find_metric_file(directory: Path) -> Optional[Path]:
    """Find a metric JSON in an experiment directory. / 查找实验指标文件。"""
    if not directory.exists():
        return None
    for name in COMMON_METRIC_NAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    candidates = sorted(directory.glob("*metrics*.json"))
    return candidates[0] if candidates else None


def extract_row(experiment_dir: Path, metric_path: Path) -> Dict[str, object]:
    """Extract one row from one metric JSON. / 从一个指标文件中抽取一行。"""
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    test = metrics.get("test_metrics", {})
    return {
        "experiment": experiment_dir.name,
        "experiment_dir": str(experiment_dir),
        "metric_file": str(metric_path),
        "model": metrics.get("model", metrics.get("estimator", "unknown")),
        "estimator": metrics.get("estimator", metrics.get("model", "unknown")),
        "condition_mode": metrics.get("condition_mode", "not_applicable"),
        "output_mode": metrics.get("output_mode", metrics.get("final_mode", "not_applicable")),
        "best_candidate": metrics.get("best_candidate", "not_applicable"),
        "best_epoch": metrics.get("best_epoch", "not_applicable"),
        "best_val_mse": metrics.get("best_val_loss_mse", metrics.get("best_validation_mse", metrics.get("best_val_mse", "not_available"))),
        "test_mse": metrics.get("test_loss_mse", test.get("mse_overall", metrics.get("test_mse", "not_available"))),
        "test_mae": test.get("mae_overall", metrics.get("test_mae", "not_available")),
        "test_rmse": test.get("rmse_overall", metrics.get("test_rmse", "not_available")),
        "negative_prediction_ratio": test.get("negative_prediction_ratio", metrics.get("negative_prediction_ratio", "not_available")),
        "mae_on_damaged_entries": test.get("mae_on_damaged_entries", "not_available"),
        "mean_prediction_on_zero_entries": test.get("mean_prediction_on_zero_entries", "not_available"),
        "precision_macro": test.get("precision_macro", metrics.get("precision_macro", "not_applicable")),
        "recall_macro": test.get("recall_macro", metrics.get("recall_macro", "not_applicable")),
        "f1_macro": test.get("f1_macro", metrics.get("f1_macro", "not_applicable")),
    }


def sortable_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model metrics from multiple experiment directories.")
    parser.add_argument("--experiment-dirs", nargs="+", type=Path, required=True, help="Experiment directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    missing: List[str] = []

    for directory in args.experiment_dirs:
        metric_path = find_metric_file(directory)
        if metric_path is None:
            missing.append(str(directory))
            continue
        rows.append(extract_row(directory, metric_path))

    rows.sort(key=lambda row: sortable_float(row.get("test_mae")))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("Metric comparison completed.")
    print(f"Output CSV: {args.output}")
    print("\nAvailable experiments sorted by Test MAE:")
    for row in rows:
        print(f"{row['experiment']} | test_mae={row['test_mae']} | test_rmse={row['test_rmse']} | neg={row['negative_prediction_ratio']}")

    if missing:
        print("\nMissing metric files:")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
