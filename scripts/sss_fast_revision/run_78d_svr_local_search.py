"""
Local RBF-SVR hyperparameter refinement on the 78D
signal-derived-with-ground descriptor set.

Protocol
--------
- Frozen 2100 / 450 / 450 split.
- Standardised 78D descriptors.
- Candidate selection using clipped validation MSE only.
- Test set is evaluated only once, after validation selection.
- Predictions are clipped to [0, 0.5].
- Current C=100, gamma=0.01, epsilon=0.03 result is reproduced
  as an internal anchor.

78维主描述符集RBF-SVR局部扩展调参。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


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


EXPECTED_FEATURE_COUNT = 78

C_VALUES = [
    30.0,
    100.0,
    300.0,
    1000.0,
]

GAMMA_VALUES: list[str | float] = [
    0.003,
    0.01,
    0.03,
    "scale",
]

EPSILON_VALUES = [
    0.01,
    0.02,
    0.03,
    0.05,
]

CURRENT_ANCHOR = {
    "C": 100.0,
    "gamma": 0.01,
    "epsilon": 0.03,
    "val_mse": 0.0028005556,
}

ANCHOR_TOLERANCE = 5.0e-11


def gamma_label(
    gamma: str | float,
) -> str:
    """Generate a stable display label."""
    if isinstance(gamma, str):
        return gamma

    return f"{gamma:g}".replace(
        ".",
        "p",
    )


def build_candidate_name(
    C: float,
    gamma: str | float,
    epsilon: float,
) -> str:
    """Create an unambiguous candidate name."""
    return (
        f"rbf_svr_C_{C:g}"
        f"_gamma_{gamma_label(gamma)}"
        f"_eps_{epsilon:g}"
    )


def fit_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    C: float,
    gamma: str | float,
    epsilon: float,
) -> tuple[
    MultiOutputRegressor,
    dict[str, Any],
]:
    """
    Fit one four-output RBF-SVR candidate and evaluate validation only.

    此函数绝不读取测试集。
    """
    estimator = MultiOutputRegressor(
        SVR(
            kernel="rbf",
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            tol=1.0e-4,
            cache_size=1024,
        ),
        n_jobs=1,
    )

    start = time.perf_counter()

    estimator.fit(
        x_train,
        y_train,
    )

    fit_seconds = float(
        time.perf_counter()
        - start
    )

    val_prediction = clip_damage_prediction(
        estimator.predict(
            x_val
        )
    )

    val_mse = mean_squared_error(
        y_val,
        val_prediction,
    )

    val_mae = mean_absolute_error(
        y_val,
        val_prediction,
    )

    support_vector_counts = [
        int(
            len(
                fitted_estimator.support_
            )
        )
        for fitted_estimator
        in estimator.estimators_
    ]

    support_vector_ratios = [
        count
        / x_train.shape[0]
        for count in support_vector_counts
    ]

    row = {
        "candidate": build_candidate_name(
            C=C,
            gamma=gamma,
            epsilon=epsilon,
        ),
        "C": float(C),
        "gamma": gamma,
        "gamma_display": str(gamma),
        "epsilon": float(epsilon),
        "val_mse": float(
            val_mse
        ),
        "val_mae": float(
            val_mae
        ),
        "fit_seconds": fit_seconds,
        "support_vectors_story_1": (
            support_vector_counts[0]
        ),
        "support_vectors_story_2": (
            support_vector_counts[1]
        ),
        "support_vectors_story_3": (
            support_vector_counts[2]
        ),
        "support_vectors_story_4": (
            support_vector_counts[3]
        ),
        "mean_support_vector_count": float(
            np.mean(
                support_vector_counts
            )
        ),
        "mean_support_vector_ratio": float(
            np.mean(
                support_vector_ratios
            )
        ),
        "selected": False,
    }

    return estimator, row


def calculate_damage_bin_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    """Calculate zero/low/medium/high test diagnostics."""
    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    bins = {
        "zero": (
            truth <= 1.0e-12
        ),
        "low": (
            (truth > 1.0e-12)
            & (truth <= 0.10)
        ),
        "medium": (
            (truth > 0.10)
            & (truth <= 0.20)
        ),
        "high": (
            truth > 0.20
        ),
    }

    rows: list[
        dict[str, Any]
    ] = []

    for bin_name, mask in (
        bins.items()
    ):
        if not np.any(mask):
            raise ValueError(
                f"Empty damage bin: "
                f"{bin_name}"
            )

        y_true = truth[mask]
        y_pred = prediction[mask]

        error = (
            y_pred
            - y_true
        )

        rows.append(
            {
                "damage_bin": (
                    bin_name
                ),
                "n_entries": int(
                    np.sum(mask)
                ),
                "mae": float(
                    np.mean(
                        np.abs(error)
                    )
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            error ** 2
                        )
                    )
                ),
                "bias": float(
                    np.mean(error)
                ),
                "mean_true": float(
                    np.mean(
                        y_true
                    )
                ),
                "mean_prediction": float(
                    np.mean(
                        y_pred
                    )
                ),
                "underestimation_ratio": (
                    float(
                        np.mean(
                            y_pred
                            < y_true
                        )
                    )
                    if bin_name
                    != "zero"
                    else np.nan
                ),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_processed/"
            "sss_fast_revision/"
            "descriptor_sets/"
            "signal_derived_with_ground/"
            "debug_plus_3000_"
            "signal_derived_with_ground_"
            "features.npz"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "svr_local_search_78d"
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

    x_train = np.asarray(
        arrays["F_train"],
        dtype=np.float64,
    )

    x_val = np.asarray(
        arrays["F_val"],
        dtype=np.float64,
    )

    x_test = np.asarray(
        arrays["F_test"],
        dtype=np.float64,
    )

    y_train = np.asarray(
        arrays["y_train"],
        dtype=np.float64,
    )

    y_val = np.asarray(
        arrays["y_val"],
        dtype=np.float64,
    )

    y_test = np.asarray(
        arrays["y_test"],
        dtype=np.float64,
    )

    if (
        x_train.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Expected 78 features, "
            f"found "
            f"{x_train.shape[1]}."
        )

    print(
        "===== 78D RBF-SVR LOCAL SEARCH ====="
    )

    print(
        "Input:",
        args.input,
    )

    print(
        "Train/val/test:",
        x_train.shape[0],
        x_val.shape[0],
        x_test.shape[0],
    )

    print(
        "Feature count:",
        x_train.shape[1],
    )

    print(
        "Candidate count:",
        (
            len(C_VALUES)
            * len(GAMMA_VALUES)
            * len(EPSILON_VALUES)
        ),
    )

    print()

    candidate_rows: list[
        dict[str, Any]
    ] = []

    best_model = None
    best_row = None

    anchor_row = None

    for C in C_VALUES:
        for gamma in GAMMA_VALUES:
            for epsilon in (
                EPSILON_VALUES
            ):
                (
                    estimator,
                    row,
                ) = fit_candidate(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    C=C,
                    gamma=gamma,
                    epsilon=epsilon,
                )

                candidate_rows.append(
                    row
                )

                print(
                    f"{row['candidate']}: "
                    f"val_mse="
                    f"{row['val_mse']:.10f}, "
                    f"val_mae="
                    f"{row['val_mae']:.10f}, "
                    f"SV_ratio="
                    f"{row['mean_support_vector_ratio']:.4f}"
                )

                if (
                    C
                    == CURRENT_ANCHOR[
                        "C"
                    ]
                    and gamma
                    == CURRENT_ANCHOR[
                        "gamma"
                    ]
                    and epsilon
                    == CURRENT_ANCHOR[
                        "epsilon"
                    ]
                ):
                    anchor_row = (
                        row.copy()
                    )

                if best_row is None:
                    best_model = (
                        estimator
                    )
                    best_row = (
                        row.copy()
                    )
                else:
                    candidate_key = (
                        row["val_mse"],
                        row["candidate"],
                    )

                    best_key = (
                        best_row[
                            "val_mse"
                        ],
                        best_row[
                            "candidate"
                        ],
                    )

                    if (
                        candidate_key
                        < best_key
                    ):
                        best_model = (
                            estimator
                        )
                        best_row = (
                            row.copy()
                        )

    if (
        best_model is None
        or best_row is None
    ):
        raise RuntimeError(
            "No SVR model selected."
        )

    if anchor_row is None:
        raise RuntimeError(
            "Current C=100, gamma=0.01, "
            "epsilon=0.03 anchor was not evaluated."
        )

    anchor_reproduction_passed = (
        np.isclose(
            float(
                anchor_row[
                    "val_mse"
                ]
            ),
            float(
                CURRENT_ANCHOR[
                    "val_mse"
                ]
            ),
            rtol=0.0,
            atol=ANCHOR_TOLERANCE,
        )
    )

    if not anchor_reproduction_passed:
        raise RuntimeError(
            "STOP: current SVR validation anchor "
            "was not reproduced. "
            f"Actual={anchor_row['val_mse']:.12f}, "
            f"Expected="
            f"{CURRENT_ANCHOR['val_mse']:.12f}"
        )

    for row in (
        candidate_rows
    ):
        row["selected"] = (
            row["candidate"]
            == best_row[
                "candidate"
            ]
        )

    # Test evaluation occurs only after validation selection.
    #
    # 到这里才首次使用test。
    test_prediction = (
        clip_damage_prediction(
            best_model.predict(
                x_test
            )
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prediction,
    )

    damage_bin_rows = (
        calculate_damage_bin_metrics(
            truth=y_test,
            prediction=(
                test_prediction
            ),
        )
    )

    candidate_frame = (
        pd.DataFrame(
            candidate_rows
        )
        .sort_values(
            [
                "val_mse",
                "candidate",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    bin_frame = pd.DataFrame(
        damage_bin_rows
    )

    bin_order = {
        "zero": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    bin_frame[
        "_order"
    ] = bin_frame[
        "damage_bin"
    ].map(
        bin_order
    )

    bin_frame = (
        bin_frame.sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )

    candidate_path = (
        args.output_root
        / "svr_local_search_candidates.csv"
    )

    bin_path = (
        args.output_root
        / "svr_local_search_damage_bins.csv"
    )

    report_path = (
        args.output_root
        / "svr_local_search_report.json"
    )

    prediction_path = (
        args.output_root
        / "selected_svr_test_predictions.npz"
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    bin_frame.to_csv(
        bin_path,
        index=False,
    )

    np.savez_compressed(
        prediction_path,
        y_test=y_test,
        y_prediction=(
            test_prediction
        ),
        selected_candidate=(
            np.asarray(
                best_row[
                    "candidate"
                ]
            )
        ),
        selected_C=np.asarray(
            best_row["C"],
            dtype=np.float64,
        ),
        selected_gamma=(
            np.asarray(
                str(
                    best_row[
                        "gamma"
                    ]
                )
            )
        ),
        selected_epsilon=np.asarray(
            best_row[
                "epsilon"
            ],
            dtype=np.float64,
        ),
        test_idx=arrays.get(
            "test_idx",
            np.arange(
                y_test.shape[0]
            ),
        ),
        case_id_test=arrays.get(
            "case_id_test",
            np.arange(
                y_test.shape[0]
            ),
        ),
    )

    C_at_lower_boundary = (
        float(
            best_row["C"]
        )
        == min(C_VALUES)
    )

    C_at_upper_boundary = (
        float(
            best_row["C"]
        )
        == max(C_VALUES)
    )

    numeric_gammas = [
        value
        for value
        in GAMMA_VALUES
        if not isinstance(
            value,
            str,
        )
    ]

    gamma_is_numeric = (
        not isinstance(
            best_row[
                "gamma"
            ],
            str,
        )
    )

    gamma_at_numeric_lower_boundary = (
        gamma_is_numeric
        and float(
            best_row[
                "gamma"
            ]
        )
        == min(
            numeric_gammas
        )
    )

    gamma_at_numeric_upper_boundary = (
        gamma_is_numeric
        and float(
            best_row[
                "gamma"
            ]
        )
        == max(
            numeric_gammas
        )
    )

    epsilon_at_lower_boundary = (
        float(
            best_row[
                "epsilon"
            ]
        )
        == min(
            EPSILON_VALUES
        )
    )

    epsilon_at_upper_boundary = (
        float(
            best_row[
                "epsilon"
            ]
        )
        == max(
            EPSILON_VALUES
        )
    )

    report = {
        "input_dataset": str(
            args.input
        ),
        "feature_count": (
            EXPECTED_FEATURE_COUNT
        ),
        "protocol": {
            "selection_criterion": (
                "clipped validation MSE"
            ),
            "prediction_clip": [
                0.0,
                0.5,
            ],
            "test_evaluations": 1,
        },
        "search_space": {
            "C": C_VALUES,
            "gamma": [
                str(value)
                for value
                in GAMMA_VALUES
            ],
            "epsilon": (
                EPSILON_VALUES
            ),
        },
        "anchor_reproduction_passed": (
            bool(
                anchor_reproduction_passed
            )
        ),
        "anchor": (
            anchor_row
        ),
        "selected_candidate": (
            best_row[
                "candidate"
            ]
        ),
        "selected_validation": {
            "C": best_row["C"],
            "gamma": best_row[
                "gamma"
            ],
            "epsilon": best_row[
                "epsilon"
            ],
            "val_mse": best_row[
                "val_mse"
            ],
            "val_mae": best_row[
                "val_mae"
            ],
            "mean_support_vector_ratio": (
                best_row[
                    "mean_support_vector_ratio"
                ]
            ),
        },
        "test_metrics": (
            test_metrics
        ),
        "boundary_checks": {
            "C_at_lower_boundary": (
                C_at_lower_boundary
            ),
            "C_at_upper_boundary": (
                C_at_upper_boundary
            ),
            "gamma_at_numeric_lower_boundary": (
                gamma_at_numeric_lower_boundary
            ),
            "gamma_at_numeric_upper_boundary": (
                gamma_at_numeric_upper_boundary
            ),
            "epsilon_at_lower_boundary": (
                epsilon_at_lower_boundary
            ),
            "epsilon_at_upper_boundary": (
                epsilon_at_upper_boundary
            ),
        },
        "output_files": {
            "candidate_csv": str(
                candidate_path
            ),
            "damage_bin_csv": str(
                bin_path
            ),
            "prediction_npz": str(
                prediction_path
            ),
        },
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

    print()
    print(
        "===== ANCHOR CHECK ====="
    )

    print(
        "Anchor reproduced:",
        anchor_reproduction_passed,
    )

    print(
        "Anchor val MSE:",
        f"{anchor_row['val_mse']:.12f}",
    )

    print()
    print(
        "===== SELECTED SVR ====="
    )

    for key in [
        "candidate",
        "C",
        "gamma",
        "epsilon",
        "val_mse",
        "val_mae",
        "mean_support_vector_count",
        "mean_support_vector_ratio",
    ]:
        print(
            f"{key}: "
            f"{best_row[key]}"
        )

    print()
    print(
        "===== SELECTED TEST RESULTS ====="
    )

    for key, value in (
        test_metrics.items()
    ):
        print(
            f"{key}: "
            f"{value:.10f}"
        )

    print()
    print(
        "===== DAMAGE-BIN RESULTS ====="
    )

    print(
        bin_frame.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "===== BOUNDARY CHECK ====="
    )

    print(
        "C at lower boundary:",
        C_at_lower_boundary,
    )

    print(
        "C at upper boundary:",
        C_at_upper_boundary,
    )

    print(
        "Gamma at numeric lower boundary:",
        gamma_at_numeric_lower_boundary,
    )

    print(
        "Gamma at numeric upper boundary:",
        gamma_at_numeric_upper_boundary,
    )

    print(
        "Epsilon at lower boundary:",
        epsilon_at_lower_boundary,
    )

    print(
        "Epsilon at upper boundary:",
        epsilon_at_upper_boundary,
    )

    print()
    print(
        "===== TOP 10 VALIDATION CANDIDATES ====="
    )

    print(
        candidate_frame.head(
            10
        )[
            [
                "candidate",
                "C",
                "gamma_display",
                "epsilon",
                "val_mse",
                "val_mae",
                "mean_support_vector_ratio",
                "fit_seconds",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "CHECK PASSED: 78D SVR local "
        "search completed."
    )

    print(
        "Candidates:",
        candidate_path,
    )

    print(
        "Damage bins:",
        bin_path,
    )

    print(
        "Report:",
        report_path,
    )

    print(
        "Predictions:",
        prediction_path,
    )


if __name__ == "__main__":
    main()
