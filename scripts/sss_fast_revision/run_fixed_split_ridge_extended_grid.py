"""
Evaluate the four canonical descriptor sets using an extended Ridge grid.

This experiment preserves the frozen fixed-split protocol except for
expanding the alpha search below 0.01.

扩展Ridge正则化参数搜索范围，检查旧实验中的alpha=0.01是否为边界解。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.sss_fast_revision.run_fixed_split_ridge_baseline import (  # noqa: E402
    DESCRIPTOR_DATASETS,
    calculate_metrics,
    clip_damage_prediction,
    confirm_shared_partitions,
    load_dataset,
    mean_absolute_error,
    mean_squared_error,
)


EXTENDED_RIDGE_ALPHAS = [
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    300.0,
]


def train_extended_ridge(
    descriptor_set: str,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Select alpha using clipped validation MSE."""
    x_train = arrays["F_train"]
    x_val = arrays["F_val"]
    x_test = arrays["F_test"]

    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]

    candidate_rows: list[dict[str, Any]] = []
    candidate_models: dict[float, Ridge] = {}

    for alpha in EXTENDED_RIDGE_ALPHAS:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="auto",
        )

        model.fit(x_train, y_train)

        validation_prediction = clip_damage_prediction(
            model.predict(x_val)
        )

        validation_mse = mean_squared_error(
            y_val,
            validation_prediction,
        )

        validation_mae = mean_absolute_error(
            y_val,
            validation_prediction,
        )

        candidate_rows.append(
            {
                "descriptor_set": descriptor_set,
                "n_features": int(x_train.shape[1]),
                "alpha": float(alpha),
                "val_mse": validation_mse,
                "val_mae": validation_mae,
                "selected": False,
            }
        )

        candidate_models[float(alpha)] = model

    selected_row = min(
        candidate_rows,
        key=lambda row: (
            row["val_mse"],
            row["alpha"],
        ),
    )

    selected_alpha = float(selected_row["alpha"])

    for row in candidate_rows:
        row["selected"] = (
            float(row["alpha"]) == selected_alpha
        )

    selected_model = candidate_models[selected_alpha]

    test_prediction = clip_damage_prediction(
        selected_model.predict(x_test)
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prediction,
    )

    summary = {
        "descriptor_set": descriptor_set,
        "n_features": int(x_train.shape[1]),
        "selected_alpha": selected_alpha,
        "selected_alpha_at_lower_boundary": (
            selected_alpha == min(EXTENDED_RIDGE_ALPHAS)
        ),
        "selected_alpha_at_upper_boundary": (
            selected_alpha == max(EXTENDED_RIDGE_ALPHAS)
        ),
        "val_mse": float(selected_row["val_mse"]),
        "val_mae": float(selected_row["val_mae"]),
        **test_metrics,
    }

    return summary, candidate_rows, test_prediction


def load_frozen_grid_summary(
    path: Path,
) -> pd.DataFrame | None:
    """Load the previously reproduced old-grid results if available."""
    if not path.is_file():
        return None

    required_columns = {
        "descriptor_set",
        "selected_alpha",
        "test_mae",
        "test_rmse",
        "damaged_entry_mae",
    }

    frame = pd.read_csv(path)

    missing = sorted(required_columns - set(frame.columns))

    if missing:
        raise ValueError(
            f"Frozen-grid summary is missing columns: {missing}"
        )

    return frame


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data_processed/sss_fast_revision/"
            "descriptor_sets"
        ),
    )

    parser.add_argument(
        "--frozen-grid-summary",
        type=Path,
        default=Path(
            "results/sss_fast_revision/"
            "fixed_split_ridge/"
            "fixed_split_ridge_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/sss_fast_revision/"
            "fixed_split_ridge_extended_grid"
        ),
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    reference_arrays: dict[str, np.ndarray] | None = None

    for descriptor_set, relative_path in DESCRIPTOR_DATASETS.items():
        dataset_path = args.dataset_root / relative_path
        arrays = load_dataset(dataset_path)

        if reference_arrays is None:
            reference_arrays = arrays
        else:
            confirm_shared_partitions(
                reference=reference_arrays,
                candidate=arrays,
                candidate_name=descriptor_set,
            )

        summary, candidate_rows, prediction = train_extended_ridge(
            descriptor_set=descriptor_set,
            arrays=arrays,
        )

        summaries.append(summary)
        candidates.extend(candidate_rows)

        np.savez_compressed(
            args.output_root
            / f"{descriptor_set}_test_predictions.npz",
            y_test=arrays["y_test"],
            y_prediction=prediction,
            selected_alpha=np.asarray(
                summary["selected_alpha"],
                dtype=np.float64,
            ),
            descriptor_set=np.asarray(descriptor_set),
            test_idx=arrays.get(
                "test_idx",
                np.arange(arrays["y_test"].shape[0]),
            ),
            case_id_test=arrays.get(
                "case_id_test",
                np.arange(arrays["y_test"].shape[0]),
            ),
        )

    summary_frame = pd.DataFrame(summaries)
    candidate_frame = pd.DataFrame(candidates)

    frozen_frame = load_frozen_grid_summary(
        args.frozen_grid_summary
    )

    if frozen_frame is not None:
        frozen_comparison = frozen_frame[
            [
                "descriptor_set",
                "selected_alpha",
                "test_mae",
                "test_rmse",
                "damaged_entry_mae",
            ]
        ].rename(
            columns={
                "selected_alpha": "old_grid_alpha",
                "test_mae": "old_grid_test_mae",
                "test_rmse": "old_grid_test_rmse",
                "damaged_entry_mae": (
                    "old_grid_damaged_entry_mae"
                ),
            }
        )

        summary_frame = summary_frame.merge(
            frozen_comparison,
            on="descriptor_set",
            how="left",
            validate="one_to_one",
        )

        summary_frame["test_mae_change_vs_old_grid"] = (
            summary_frame["test_mae"]
            - summary_frame["old_grid_test_mae"]
        )

        summary_frame[
            "test_mae_change_vs_old_grid_percent"
        ] = (
            summary_frame["test_mae_change_vs_old_grid"]
            / summary_frame["old_grid_test_mae"]
            * 100.0
        )

    oracle_mae = float(
        summary_frame.loc[
            summary_frame["descriptor_set"] == "oracle_full",
            "test_mae",
        ].iloc[0]
    )

    summary_frame[
        "mae_increase_vs_oracle_percent"
    ] = (
        (
            summary_frame["test_mae"]
            - oracle_mae
        )
        / oracle_mae
        * 100.0
    )

    summary_frame = summary_frame.sort_values(
        "test_mae",
        ascending=True,
    ).reset_index(drop=True)

    candidate_frame = candidate_frame.sort_values(
        ["descriptor_set", "alpha"]
    ).reset_index(drop=True)

    summary_path = (
        args.output_root
        / "extended_grid_ridge_summary.csv"
    )

    candidate_path = (
        args.output_root
        / "extended_grid_ridge_candidates.csv"
    )

    report_path = (
        args.output_root
        / "extended_grid_ridge_report.json"
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    lower_boundary_sets = summary_frame.loc[
        summary_frame[
            "selected_alpha_at_lower_boundary"
        ],
        "descriptor_set",
    ].tolist()

    upper_boundary_sets = summary_frame.loc[
        summary_frame[
            "selected_alpha_at_upper_boundary"
        ],
        "descriptor_set",
    ].tolist()

    report = {
        "protocol": {
            "estimator": "Ridge",
            "alpha_candidates": EXTENDED_RIDGE_ALPHAS,
            "selection_criterion": (
                "clipped validation MSE"
            ),
            "prediction_clip": [0.0, 0.5],
            "dataset_split": "frozen 2100/450/450",
        },
        "selected_alpha_lower_boundary_sets": (
            lower_boundary_sets
        ),
        "selected_alpha_upper_boundary_sets": (
            upper_boundary_sets
        ),
        "summary_csv": str(summary_path),
        "candidate_csv": str(candidate_path),
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    display_columns = [
        "descriptor_set",
        "n_features",
        "selected_alpha",
        "selected_alpha_at_lower_boundary",
        "val_mse",
        "test_mae",
        "test_rmse",
        "damaged_entry_mae",
        "high_damage_mae",
        "high_damage_bias",
        "high_damage_underestimation_ratio",
        "mae_increase_vs_oracle_percent",
    ]

    optional_columns = [
        "old_grid_alpha",
        "old_grid_test_mae",
        "test_mae_change_vs_old_grid",
        "test_mae_change_vs_old_grid_percent",
    ]

    display_columns.extend(
        column
        for column in optional_columns
        if column in summary_frame.columns
    )

    print("===== EXTENDED-GRID RIDGE RESULTS =====")
    print(
        summary_frame[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("===== SELECTED ALPHA CHECK =====")
    print(
        "Lower-boundary selections:",
        lower_boundary_sets,
    )
    print(
        "Upper-boundary selections:",
        upper_boundary_sets,
    )

    if lower_boundary_sets:
        print(
            "WARNING: at least one descriptor set selected "
            "the smallest tested alpha. An OLS/near-OLS check "
            "will be required."
        )
    else:
        print(
            "CHECK PASSED: no descriptor set selected the "
            "lower alpha boundary."
        )

    print()
    print("Summary:", summary_path)
    print("Candidates:", candidate_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
