"""
OLS and near-OLS boundary check for the 59-dimensional
structural-response-only descriptor set.

The model is selected using clipped validation MSE. Test data are
evaluated only after the validation-based selection is complete.

针对59维纯结构响应描述符，检查OLS及近OLS区域。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.sss_fast_revision.run_fixed_split_ridge_baseline import (  # noqa: E402
    calculate_metrics,
    clip_damage_prediction,
    load_dataset,
    mean_absolute_error,
    mean_squared_error,
)


CANDIDATE_RIDGE_ALPHAS = [
    1.0e-12,
    1.0e-10,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
]


def analyse_design_matrix(
    matrix: np.ndarray,
) -> dict[str, Any]:
    """
    Analyse rank and numerical conditioning after centring.

    分析设计矩阵的秩和数值条件。
    """
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    centred = matrix - np.mean(
        matrix,
        axis=0,
        keepdims=True,
    )

    singular_values = np.linalg.svd(
        centred,
        full_matrices=False,
        compute_uv=False,
    )

    if singular_values.size == 0:
        raise ValueError(
            "No singular values were produced."
        )

    tolerance = (
        np.finfo(np.float64).eps
        * max(centred.shape)
        * singular_values[0]
    )

    nonzero = singular_values[
        singular_values > tolerance
    ]

    numerical_rank = int(nonzero.size)

    if numerical_rank == 0:
        condition_number = float("inf")
        smallest_nonzero = float("nan")
    else:
        smallest_nonzero = float(nonzero[-1])
        condition_number = float(
            nonzero[0] / nonzero[-1]
        )

    return {
        "sample_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "numerical_rank": numerical_rank,
        "full_column_rank": (
            numerical_rank == matrix.shape[1]
        ),
        "largest_singular_value": float(
            singular_values[0]
        ),
        "smallest_nonzero_singular_value": (
            smallest_nonzero
        ),
        "condition_number_nonzero_spectrum": (
            condition_number
        ),
        "rank_tolerance": float(tolerance),
    }


def model_coefficient_metrics(
    model: LinearRegression | Ridge,
) -> dict[str, float]:
    """Summarise fitted coefficient magnitude."""
    coefficients = np.asarray(
        model.coef_,
        dtype=np.float64,
    )

    intercept = np.asarray(
        model.intercept_,
        dtype=np.float64,
    )

    return {
        "coefficient_l2_norm": float(
            np.linalg.norm(coefficients)
        ),
        "coefficient_max_abs": float(
            np.max(np.abs(coefficients))
        ),
        "intercept_l2_norm": float(
            np.linalg.norm(intercept)
        ),
    }


def fit_candidate(
    candidate_name: str,
    estimator: LinearRegression | Ridge,
    arrays: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    """Fit one candidate and evaluate validation and test predictions."""
    x_train = arrays["F_train"]
    x_val = arrays["F_val"]
    x_test = arrays["F_test"]

    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]

    estimator.fit(
        x_train,
        y_train,
    )

    validation_prediction = clip_damage_prediction(
        estimator.predict(x_val)
    )

    test_prediction = clip_damage_prediction(
        estimator.predict(x_test)
    )

    validation_mse = mean_squared_error(
        y_val,
        validation_prediction,
    )

    validation_mae = mean_absolute_error(
        y_val,
        validation_prediction,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prediction,
    )

    coefficient_metrics = (
        model_coefficient_metrics(estimator)
    )

    row = {
        "candidate_name": candidate_name,
        "estimator": (
            estimator.__class__.__name__
        ),
        "alpha": (
            float(estimator.alpha)
            if isinstance(estimator, Ridge)
            else np.nan
        ),
        "val_mse": validation_mse,
        "val_mae": validation_mae,
        **test_metrics,
        **coefficient_metrics,
        "selected_by_validation": False,
    }

    return (
        row,
        validation_prediction,
        test_prediction,
    )


def relative_difference_percent(
    value: float,
    reference: float,
) -> float:
    """Return percentage difference relative to a reference."""
    if reference == 0.0:
        return float("nan")

    return float(
        (value - reference)
        / reference
        * 100.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_processed/sss_fast_revision/"
            "descriptor_sets/"
            "structural_response_only/"
            "debug_plus_3000_"
            "structural_response_only_features.npz"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/sss_fast_revision/"
            "structural_response_ols_check"
        ),
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = load_dataset(
        args.input
    )

    if arrays["F_train"].shape[1] != 59:
        raise ValueError(
            "Expected 59 structural-response-only "
            f"features, found {arrays['F_train'].shape[1]}."
        )

    design_diagnostics = analyse_design_matrix(
        arrays["F_train"]
    )

    candidates: list[
        tuple[
            str,
            LinearRegression | Ridge,
        ]
    ] = [
        (
            "ordinary_least_squares",
            LinearRegression(
                fit_intercept=True,
            ),
        )
    ]

    for alpha in CANDIDATE_RIDGE_ALPHAS:
        candidates.append(
            (
                f"ridge_alpha_{alpha:.0e}",
                Ridge(
                    alpha=alpha,
                    fit_intercept=True,
                    solver="auto",
                ),
            )
        )

    rows: list[dict[str, Any]] = []
    validation_predictions: dict[
        str,
        np.ndarray,
    ] = {}
    test_predictions: dict[
        str,
        np.ndarray,
    ] = {}

    for candidate_name, estimator in candidates:
        (
            row,
            validation_prediction,
            test_prediction,
        ) = fit_candidate(
            candidate_name=candidate_name,
            estimator=estimator,
            arrays=arrays,
        )

        rows.append(row)

        validation_predictions[
            candidate_name
        ] = validation_prediction

        test_predictions[
            candidate_name
        ] = test_prediction

    # Model selection uses validation MSE only.
    #
    # 模型选择只使用验证集MSE。
    selected_row = min(
        rows,
        key=lambda row: (
            row["val_mse"],
            row["candidate_name"],
        ),
    )

    selected_name = str(
        selected_row["candidate_name"]
    )

    for row in rows:
        row["selected_by_validation"] = (
            row["candidate_name"]
            == selected_name
        )

    frame = pd.DataFrame(rows)

    ols_row = frame.loc[
        frame["candidate_name"]
        == "ordinary_least_squares"
    ].iloc[0]

    ridge_1e6_row = frame.loc[
        frame["candidate_name"]
        == "ridge_alpha_1e-06"
    ].iloc[0]

    reference_prediction = (
        test_predictions[
            "ordinary_least_squares"
        ]
    )

    frame[
        "test_prediction_max_abs_difference_vs_ols"
    ] = [
        float(
            np.max(
                np.abs(
                    test_predictions[
                        candidate_name
                    ]
                    - reference_prediction
                )
            )
        )
        for candidate_name in frame[
            "candidate_name"
        ]
    ]

    frame[
        "test_prediction_mean_abs_difference_vs_ols"
    ] = [
        float(
            np.mean(
                np.abs(
                    test_predictions[
                        candidate_name
                    ]
                    - reference_prediction
                )
            )
        )
        for candidate_name in frame[
            "candidate_name"
        ]
    ]

    frame = frame.sort_values(
        [
            "val_mse",
            "candidate_name",
        ],
        ascending=True,
    ).reset_index(drop=True)

    candidate_output = (
        args.output_root
        / "structural_response_ols_candidates.csv"
    )

    frame.to_csv(
        candidate_output,
        index=False,
    )

    selected_test_prediction = (
        test_predictions[selected_name]
    )

    prediction_output = (
        args.output_root
        / "selected_test_predictions.npz"
    )

    np.savez_compressed(
        prediction_output,
        y_test=arrays["y_test"],
        y_prediction=selected_test_prediction,
        selected_candidate=np.asarray(
            selected_name
        ),
        test_idx=arrays.get(
            "test_idx",
            np.arange(
                arrays["y_test"].shape[0]
            ),
        ),
        case_id_test=arrays.get(
            "case_id_test",
            np.arange(
                arrays["y_test"].shape[0]
            ),
        ),
    )

    selected_frame_row = frame.loc[
        frame["candidate_name"]
        == selected_name
    ].iloc[0]

    summary = {
        "input_dataset": str(args.input),
        "descriptor_set": (
            "structural_response_only"
        ),
        "feature_count": 59,
        "selection_criterion": (
            "clipped validation MSE"
        ),
        "prediction_clip": [
            0.0,
            0.5,
        ],
        "design_matrix_diagnostics": (
            design_diagnostics
        ),
        "selected_candidate": selected_name,
        "selected_candidate_metrics": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (bool, np.bool_),
                )
                else float(value)
                if isinstance(
                    value,
                    (
                        float,
                        int,
                        np.floating,
                        np.integer,
                    ),
                )
                else value
            )
            for key, value in (
                selected_frame_row.to_dict().items()
            )
        },
        "ols_vs_ridge_1e_minus_6": {
            "validation_mse_difference": float(
                ols_row["val_mse"]
                - ridge_1e6_row["val_mse"]
            ),
            "validation_mse_difference_percent": (
                relative_difference_percent(
                    value=float(
                        ols_row["val_mse"]
                    ),
                    reference=float(
                        ridge_1e6_row["val_mse"]
                    ),
                )
            ),
            "test_mae_difference": float(
                ols_row["test_mae"]
                - ridge_1e6_row["test_mae"]
            ),
            "test_mae_difference_percent": (
                relative_difference_percent(
                    value=float(
                        ols_row["test_mae"]
                    ),
                    reference=float(
                        ridge_1e6_row["test_mae"]
                    ),
                )
            ),
            "test_rmse_difference": float(
                ols_row["test_rmse"]
                - ridge_1e6_row["test_rmse"]
            ),
            "high_damage_mae_difference": float(
                ols_row["high_damage_mae"]
                - ridge_1e6_row[
                    "high_damage_mae"
                ]
            ),
        },
        "candidate_csv": str(
            candidate_output
        ),
        "selected_prediction_npz": str(
            prediction_output
        ),
    }

    summary_output = (
        args.output_root
        / "structural_response_ols_report.json"
    )

    with summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    display_columns = [
        "candidate_name",
        "alpha",
        "val_mse",
        "val_mae",
        "test_mae",
        "test_rmse",
        "damaged_entry_mae",
        "high_damage_mae",
        "high_damage_bias",
        "high_damage_underestimation_ratio",
        "coefficient_l2_norm",
        "coefficient_max_abs",
        "selected_by_validation",
    ]

    print(
        "===== STRUCTURAL-RESPONSE OLS CHECK ====="
    )

    print(
        frame[display_columns].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.12f}"
            ),
        )
    )

    print()
    print(
        "===== DESIGN MATRIX DIAGNOSTICS ====="
    )

    for key, value in (
        design_diagnostics.items()
    ):
        print(f"{key}: {value}")

    print()
    print(
        "===== OLS VS RIDGE 1E-6 ====="
    )

    for key, value in summary[
        "ols_vs_ridge_1e_minus_6"
    ].items():
        print(f"{key}: {value}")

    print()
    print(
        "SELECTED BY VALIDATION:",
        selected_name,
    )
    print(
        "Candidate table:",
        candidate_output,
    )
    print(
        "Report:",
        summary_output,
    )
    print(
        "Prediction file:",
        prediction_output,
    )


if __name__ == "__main__":
    main()
