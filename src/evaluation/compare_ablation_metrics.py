"""
Compare ablation experiment metrics.

English:
This script scans experiment folders, reads metrics.json files, fixes the feature_set
extraction bug, and writes one clean comparison CSV.

中文：
本脚本用于扫描消融实验文件夹，读取各模型的 metrics.json，修复 feature_set
被错误写成模型名的问题，并输出一份干净的对比 CSV 表格。

Expected folder structure:
results/tables/physics_ablation/debug_plus_100/full/ridge/metrics.json
results/tables/physics_ablation/debug_plus_100/response_correlation/random_forest/metrics.json

Typical command:
python -m src.evaluation.compare_ablation_metrics \
  --experiment-root results/tables/physics_ablation/debug_plus_100 \
  --output results/tables/physics_ablation/debug_plus_100/ablation_model_comparison_fixed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


KNOWN_FEATURE_SETS = {
    "full",
    "no_meta",
    "physics_no_meta_core",
    "response_basic_only",
    "response_spatial",
    "response_frequency",
    "response_correlation",
}


OUTPUT_COLUMNS = [
    "feature_set",
    "estimator",
    "experiment_dir",
    "metrics_path",
    "model",
    "best_candidate",
    "condition_mode",
    "output_mode",
    "final_mode",
    "n_features",
    "best_epoch",
    "best_val_mse",
    "test_mse",
    "test_mae",
    "test_rmse",
    "negative_prediction_ratio",
    "mean_prediction_on_zero_entries",
    "mae_on_damaged_entries",
    "mean_true_damage",
    "mean_pred_damage",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "exact_mask_match_ratio",
]


def safe_get(d: Dict[str, Any], keys: Iterable[str], default: Any = "not_available") -> Any:
    """
    English:
    Return the first available key from a dictionary.

    中文：
    从字典中按顺序读取候选字段，返回第一个存在的值。
    """
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def read_json(path: Path) -> Dict[str, Any]:
    """
    English:
    Read a UTF-8 JSON file.

    中文：
    读取 UTF-8 编码的 JSON 文件。
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_metrics_files(experiment_root: Optional[Path], experiment_dirs: List[Path]) -> List[Path]:
    """
    English:
    Collect metrics.json files from either one root folder or a list of experiment folders.

    中文：
    从一个总实验目录或多个实验目录中收集 metrics.json 文件。
    """
    metrics_files: List[Path] = []

    if experiment_root is not None:
        metrics_files.extend(sorted(experiment_root.rglob("metrics.json")))

    for d in experiment_dirs:
        if d.is_file() and d.name.endswith(".json"):
            metrics_files.append(d)
        elif d.is_dir():
            metrics_files.extend(sorted(d.rglob("metrics.json")))

    # Remove duplicates while preserving order.
    # 去重，同时保持原始顺序。
    unique: List[Path] = []
    seen = set()
    for p in metrics_files:
        rp = str(p.resolve())
        if rp not in seen:
            unique.append(p)
            seen.add(rp)

    return unique


def infer_feature_set(metrics_path: Path, experiment_root: Optional[Path]) -> str:
    """
    English:
    Infer the feature set name from the folder path.

    中文：
    从路径中推断特征组名称。这里是本次需要修复的关键点：
    feature_set 应该是 full / no_meta / response_correlation 等，
    而不是 ridge / random_forest / elasticnet。
    """
    if experiment_root is not None:
        try:
            rel_parts = metrics_path.parent.relative_to(experiment_root).parts
            if len(rel_parts) >= 1:
                return rel_parts[0]
        except ValueError:
            pass

    parts = metrics_path.parts
    for part in reversed(parts):
        if part in KNOWN_FEATURE_SETS:
            return part

    # Fallback:
    # Expected pattern: .../<feature_set>/<estimator>/metrics.json
    # 兜底规则：通常 metrics.json 的父目录是 estimator，爷爷目录是 feature_set。
    if len(metrics_path.parents) >= 2:
        return metrics_path.parents[1].name

    return "unknown_feature_set"


def infer_estimator(metrics: Dict[str, Any], metrics_path: Path) -> str:
    """
    English:
    Infer estimator/model type.

    中文：
    推断估计器/模型类型。
    """
    estimator = safe_get(metrics, ["estimator", "model_name"], default=None)
    if estimator not in (None, "not_available"):
        return str(estimator)

    # Expected pattern: .../<feature_set>/<estimator>/metrics.json
    # 通常 metrics.json 的父目录就是 ridge / random_forest / elasticnet。
    return metrics_path.parent.name


def flatten_metrics(metrics: Dict[str, Any], metrics_path: Path, experiment_root: Optional[Path]) -> Dict[str, Any]:
    """
    English:
    Convert different metric JSON schemas into one flat row.

    中文：
    将不同训练脚本输出的指标 JSON 统一整理成一行表格。
    """
    test_metrics = metrics.get("test_metrics", {})
    if not isinstance(test_metrics, dict):
        test_metrics = {}

    row = {
        "feature_set": infer_feature_set(metrics_path, experiment_root),
        "estimator": infer_estimator(metrics, metrics_path),
        "experiment_dir": str(metrics_path.parent),
        "metrics_path": str(metrics_path),
        "model": safe_get(metrics, ["model", "model_class", "model_name"], default="not_available"),
        "best_candidate": safe_get(metrics, ["best_candidate", "candidate"], default="not_available"),
        "condition_mode": safe_get(metrics, ["condition_mode"], default="not_applicable"),
        "output_mode": safe_get(metrics, ["output_mode"], default="not_available"),
        "final_mode": safe_get(metrics, ["final_mode"], default="not_applicable"),
        "n_features": safe_get(metrics, ["n_features", "input_dim"], default="not_available"),
        "best_epoch": safe_get(metrics, ["best_epoch"], default="not_available"),
        "best_val_mse": safe_get(metrics, ["best_val_mse", "best_val_loss_mse", "best_validation_mse"], default="not_available"),
        "test_mse": safe_get(test_metrics, ["mse_overall", "test_mse"], default=safe_get(metrics, ["test_mse"], default="not_available")),
        "test_mae": safe_get(test_metrics, ["mae_overall", "test_mae"], default=safe_get(metrics, ["test_mae"], default="not_available")),
        "test_rmse": safe_get(test_metrics, ["rmse_overall", "test_rmse"], default=safe_get(metrics, ["test_rmse"], default="not_available")),
        "negative_prediction_ratio": safe_get(
            test_metrics,
            ["negative_prediction_ratio"],
            default=safe_get(metrics, ["negative_prediction_ratio"], default="not_available"),
        ),
        "mean_prediction_on_zero_entries": safe_get(
            test_metrics,
            ["mean_prediction_on_zero_entries"],
            default=safe_get(metrics, ["mean_prediction_on_zero_entries"], default="not_available"),
        ),
        "mae_on_damaged_entries": safe_get(
            test_metrics,
            ["mae_on_damaged_entries"],
            default=safe_get(metrics, ["mae_on_damaged_entries"], default="not_available"),
        ),
        "mean_true_damage": safe_get(
            test_metrics,
            ["mean_true_damage"],
            default=safe_get(metrics, ["mean_true_damage"], default="not_available"),
        ),
        "mean_pred_damage": safe_get(
            test_metrics,
            ["mean_pred_damage"],
            default=safe_get(metrics, ["mean_pred_damage"], default="not_available"),
        ),
        "precision_macro": safe_get(
            test_metrics,
            ["precision_macro"],
            default=safe_get(metrics, ["precision_macro"], default="not_available"),
        ),
        "recall_macro": safe_get(
            test_metrics,
            ["recall_macro"],
            default=safe_get(metrics, ["recall_macro"], default="not_available"),
        ),
        "f1_macro": safe_get(
            test_metrics,
            ["f1_macro"],
            default=safe_get(metrics, ["f1_macro"], default="not_available"),
        ),
        "exact_mask_match_ratio": safe_get(
            test_metrics,
            ["exact_mask_match_ratio"],
            default=safe_get(metrics, ["exact_mask_match_ratio"], default="not_available"),
        ),
    }

    return row


def numeric_or_inf(value: Any) -> float:
    """
    English:
    Convert a value to float for sorting. Non-numeric values are sorted to the end.

    中文：
    将字段转换为浮点数用于排序；不能转换的值排到最后。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def write_csv(rows: List[Dict[str, Any]], output: Path) -> None:
    """
    English:
    Write rows to a CSV file.

    中文：
    将结果行写入 CSV 文件。
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "not_available") for col in OUTPUT_COLUMNS})


def parse_args() -> argparse.Namespace:
    """
    English:
    Parse command-line arguments.

    中文：
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Compare ablation experiment metrics and fix feature_set extraction.")
    parser.add_argument("--experiment-root", type=Path, default=None, help="Root folder containing feature-set subfolders.")
    parser.add_argument("--experiment-dirs", type=Path, nargs="*", default=[], help="Experiment folders or metric JSON paths.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--sort-by", type=str, default="test_mae", help="Column used for sorting.")
    parser.add_argument("--print-top", type=int, default=30, help="Number of sorted rows to print.")
    return parser.parse_args()


def main() -> None:
    """
    English:
    Main execution function.

    中文：
    主执行函数。
    """
    args = parse_args()

    metrics_files = find_metrics_files(args.experiment_root, args.experiment_dirs)
    if not metrics_files:
        raise FileNotFoundError("No metrics.json files found. Check --experiment-root or --experiment-dirs.")

    rows: List[Dict[str, Any]] = []
    for path in metrics_files:
        metrics = read_json(path)
        rows.append(flatten_metrics(metrics, path, args.experiment_root))

    rows.sort(key=lambda r: numeric_or_inf(r.get(args.sort_by)))

    write_csv(rows, args.output)

    print("Metric comparison completed.")
    print(f"Output CSV: {args.output}")
    print(f"Number of rows: {len(rows)}")
    print("")
    print(f"Available experiments sorted by {args.sort_by}:")
    for row in rows[: args.print_top]:
        print(
            f"{row['feature_set']} | {row['estimator']} | "
            f"test_mae={row['test_mae']} | "
            f"test_rmse={row['test_rmse']} | "
            f"neg={row['negative_prediction_ratio']}"
        )


if __name__ == "__main__":
    main()
