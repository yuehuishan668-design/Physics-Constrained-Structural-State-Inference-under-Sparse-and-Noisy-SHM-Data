#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare ablation experiment metrics.

This script recursively collects model metrics from experiment directories and
writes one sorted CSV table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_feature_set_from_path(path: Path, root: Path) -> str:
    """Infer feature-set name from a metrics path."""
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 3:
            return parts[0]
    except ValueError:
        pass
    return path.parent.name


def collect_metrics(root_dirs: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for root in root_dirs:
        if not root.exists():
            print(f"Warning: missing root directory: {root}")
            continue

        for metrics_path in sorted(root.rglob("metrics.json")):
            metrics = read_json(metrics_path)
            test = metrics.get("test_metrics", metrics.get("test", {}))

            feature_set = metrics.get("feature_set")
            if feature_set in (None, "", "not_available"):
                feature_set = infer_feature_set_from_path(metrics_path, root)

            row = {
                "feature_set": feature_set,
                "experiment_dir": str(metrics_path.parent),
                "metrics_path": str(metrics_path),
                "model": metrics.get("model", metrics.get("estimator", metrics_path.parent.name)),
                "estimator": metrics.get("estimator", metrics.get("model", metrics_path.parent.name)),
                "best_candidate": metrics.get("best_candidate", metrics.get("best_model", "not_available")),
                "condition_mode": metrics.get("condition_mode", "not_applicable"),
                "output_mode": metrics.get("output_mode", "not_available"),
                "final_mode": metrics.get("final_mode", "not_applicable"),
                "n_features": metrics.get("n_features", "not_available"),
                "best_epoch": metrics.get("best_epoch", "not_available"),
                "best_val_mse": metrics.get(
                    "best_val_loss_mse",
                    metrics.get("best_val_loss", metrics.get("best_validation_mse", "not_available")),
                ),
                "test_mse": metrics.get("test_loss_mse", metrics.get("test_mse", "not_available")),
                "test_mae": test.get("mae_overall", metrics.get("test_mae", "not_available")),
                "test_rmse": test.get("rmse_overall", metrics.get("test_rmse", "not_available")),
                "negative_prediction_ratio": test.get(
                    "negative_prediction_ratio",
                    metrics.get("negative_prediction_ratio", "not_available"),
                ),
                "mean_prediction_on_zero_entries": metrics.get(
                    "mean_prediction_on_zero_entries",
                    test.get("mean_prediction_on_zero_entries", "not_available"),
                ),
                "mae_on_damaged_entries": metrics.get(
                    "mae_on_damaged_entries",
                    test.get("mae_on_damaged_entries", "not_available"),
                ),
                "mean_true_damage": metrics.get("mean_true_damage", "not_available"),
                "mean_pred_damage": metrics.get("mean_pred_damage", "not_available"),
                "precision_macro": test.get("precision_macro", "not_available"),
                "recall_macro": test.get("recall_macro", "not_available"),
                "f1_macro": test.get("f1_macro", "not_available"),
                "exact_mask_match_ratio": test.get("exact_mask_match_ratio", "not_available"),
            }
            rows.append(row)

    def sort_key(row: Dict[str, Any]):
        try:
            return float(row["test_mae"])
        except Exception:
            return float("inf")

    return sorted(rows, key=sort_key)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("warning\nNo metrics.json files found\n")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare metrics from ablation experiments.")
    parser.add_argument("--experiment-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--print-top-k", type=int, default=20)
    args = parser.parse_args()

    rows = collect_metrics(args.experiment_dirs)
    write_csv(args.output, rows)

    print("Metric comparison completed.")
    print(f"Output CSV: {args.output}")
    print(f"Number of rows: {len(rows)}")

    if rows:
        print("\nAvailable experiments sorted by Test MAE:")
        for row in rows[: args.print_top_k]:
            print(
                f"{row['feature_set']} | {row['estimator']} | "
                f"test_mae={row['test_mae']} | "
                f"test_rmse={row['test_rmse']} | "
                f"neg={row['negative_prediction_ratio']}"
            )


if __name__ == "__main__":
    main()
