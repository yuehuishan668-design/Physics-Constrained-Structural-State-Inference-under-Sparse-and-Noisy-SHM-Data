#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run physics-feature ablation experiments.

This script automates the next experiment stage:
    1. Create filtered feature datasets for several feature groups.
    2. Train sklearn baselines on each feature set.
    3. Collect all metrics into one ablation comparison CSV.
    4. Optionally generate prediction diagnostic figures.

This script assumes the following modules already exist in the project:
    - src.preprocessing.filter_physics_features
    - src.training.train_physics_sklearn
    - src.evaluation.compare_ablation_metrics
    - src.evaluation.plot_mlp_predictions
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


ABLATION_PLANS = {
    "full": {
        "include_groups": "",
        "exclude_groups": "",
        "description": "All extracted physics features.",
    },
    "no_meta": {
        "include_groups": "",
        "exclude_groups": "meta",
        "description": "All physics features except generation-condition metadata.",
    },
    "response_basic_only": {
        "include_groups": "response_basic",
        "exclude_groups": "meta",
        "description": "Per-story time-domain response statistics only.",
    },
    "response_spatial": {
        "include_groups": "response_basic spatial ratio",
        "exclude_groups": "meta",
        "description": "Response statistics plus spatial-fraction and ratio features.",
    },
    "response_frequency": {
        "include_groups": "response_basic frequency",
        "exclude_groups": "meta",
        "description": "Response statistics plus frequency-domain features.",
    },
    "response_correlation": {
        "include_groups": "response_basic correlation",
        "exclude_groups": "meta",
        "description": "Response statistics plus inter-story correlation features.",
    },
    "physics_no_meta_core": {
        "include_groups": "response_basic spatial frequency correlation ratio amplification ground",
        "exclude_groups": "meta",
        "description": "Core physics features without metadata.",
    },
}


def run_command(cmd: Sequence[str], dry_run: bool = False) -> None:
    """Run one command and stop immediately if it fails."""
    printable = " ".join(str(x) for x in cmd)
    print("\n" + "=" * 100)
    print(printable)
    print("=" * 100)
    if dry_run:
        return
    subprocess.run(list(cmd), check=True)


def write_plan_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_set", "include_groups", "exclude_groups", "description"])
        for name, cfg in ABLATION_PLANS.items():
            writer.writerow([name, cfg["include_groups"], cfg["exclude_groups"], cfg["description"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run physics feature ablation experiment.")
    parser.add_argument("--base-features", required=True, type=Path)
    parser.add_argument("--feature-names", required=True, type=Path)
    parser.add_argument("--tag", default="debug_plus_100")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "elasticnet", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--max-damage", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=Path("results/tables/physics_ablation"))
    parser.add_argument("--features-root", type=Path, default=Path("data_processed/physics_ablation_features"))
    parser.add_argument("--figures-root", type=Path, default=Path("results/figures/physics_ablation"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.base_features.exists():
        raise FileNotFoundError(f"Base feature dataset not found: {args.base_features}")
    if not args.feature_names.exists():
        raise FileNotFoundError(f"Feature-name CSV not found: {args.feature_names}")

    experiment_root = args.output_root / args.tag
    feature_root = args.features_root / args.tag
    figures_root = args.figures_root / args.tag

    write_plan_csv(experiment_root / "ablation_plan.csv")
    experiment_dirs: List[str] = []

    for feature_set, cfg in ABLATION_PLANS.items():
        feature_npz = feature_root / feature_set / f"{args.tag}_{feature_set}_features.npz"
        feature_csv = feature_root / feature_set / f"{args.tag}_{feature_set}_feature_names.csv"
        feature_summary = feature_root / feature_set / f"{args.tag}_{feature_set}_feature_summary.json"

        run_command(
            [
                sys.executable,
                "-m",
                "src.preprocessing.filter_physics_features",
                "--input",
                str(args.base_features),
                "--feature-names",
                str(args.feature_names),
                "--output",
                str(feature_npz),
                "--output-feature-names",
                str(feature_csv),
                "--summary-json",
                str(feature_summary),
                "--include-groups",
                cfg["include_groups"],
                "--exclude-groups",
                cfg["exclude_groups"],
            ],
            dry_run=args.dry_run,
        )

        out_dir = experiment_root / feature_set
        run_command(
            [
                sys.executable,
                "-m",
                "src.training.train_physics_sklearn",
                "--features",
                str(feature_npz),
                "--feature-names",
                str(feature_csv),
                "--models",
                *args.models,
                "--clip-predictions",
                "--max-damage",
                str(args.max_damage),
                "--output-dir",
                str(out_dir),
                "--random-seed",
                "42",
            ],
            dry_run=args.dry_run,
        )
        experiment_dirs.append(str(out_dir))

        if args.plot:
            for model_name in ["random_forest", "ridge"]:
                pred_csv = out_dir / model_name / "predictions_test.csv"
                plot_dir = figures_root / feature_set / f"{model_name}_predictions"
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.evaluation.plot_mlp_predictions",
                        "--csv",
                        str(pred_csv),
                        "--output-dir",
                        str(plot_dir),
                        "--split-name",
                        f"test_{feature_set}_{model_name}",
                    ],
                    dry_run=args.dry_run,
                )

    comparison_csv = experiment_root / "ablation_model_comparison.csv"
    run_command(
        [
            sys.executable,
            "-m",
            "src.evaluation.compare_ablation_metrics",
            "--experiment-dirs",
            *experiment_dirs,
            "--output",
            str(comparison_csv),
            "--print-top",
            "50",
        ],
        dry_run=args.dry_run,
    )

    print("\nAblation experiment completed.")
    print(f"Ablation plan: {experiment_root / 'ablation_plan.csv'}")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Filtered feature datasets: {feature_root}")
    if args.plot:
        print(f"Prediction figures: {figures_root}")


if __name__ == "__main__":
    main()
