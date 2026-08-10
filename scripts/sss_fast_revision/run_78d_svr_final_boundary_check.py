"""
Final validation-only C-gamma boundary check for the
78D signal-derived-with-ground RBF-SVR.

This is the FINAL fixed-split SVR tuning stage.

IMPORTANT
---------
- F_test and y_test are never accessed.
- Candidate selection uses clipped validation MSE only.
- No further fixed-split grid expansion should be performed
  after this script, irrespective of the result.

78维RBF-SVR固定划分最终边界检查。
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


EXPECTED_FEATURE_COUNT = 78

C_VALUES = [
    300.0,
    500.0,
    750.0,
    1000.0,
]

GAMMA_VALUES = [
    0.0003,
    0.0005,
    0.00075,
    0.001,
    0.0015,
]

EPSILON_VALUES = [
    0.02,
    0.03,
    0.04,
]


ANCHOR = {
    "C": 300.0,
    "gamma": 0.001,
    "epsilon": 0.03,
    "val_mse": 0.002408918560974952,
}

ANCHOR_TOLERANCE = 5.0e-11


def load_train_validation_only(
    path: Path,
) -> dict[str, np.ndarray]:
    """
    Load training and validation arrays only.

    测试集数组不会被访问。
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    required = {
        "F_train",
        "F_val",
        "y_train",
        "y_val",
        "feature_names",
    }

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        missing = sorted(
            required - set(data.files)
        )

        if missing:
            raise KeyError(
                f"Missing arrays: {missing}"
            )

        arrays = {
            key: np.asarray(data[key])
            for key in required
        }

    for key in [
        "F_train",
        "F_val",
        "y_train",
        "y_val",
    ]:
        arrays[key] = np.asarray(
            arrays[key],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(arrays[key])
        ):
            raise FloatingPointError(
                f"{key} contains non-finite values."
            )

    return arrays


def clip_prediction(
    prediction: np.ndarray,
) -> np.ndarray:
    """Clip predicted damage to [0, 0.5]."""
    return np.clip(
        np.asarray(
            prediction,
            dtype=np.float64,
        ),
        0.0,
        0.5,
    )


def mse(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate MSE."""
    error = (
        np.asarray(prediction)
        - np.asarray(truth)
    )

    return float(
        np.mean(
            error ** 2
        )
    )


def mae(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate MAE."""
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction)
                - np.asarray(truth)
            )
        )
    )


def candidate_name(
    C: float,
    gamma: float,
    epsilon: float,
) -> str:
    """Create stable candidate name."""
    gamma_label = (
        f"{gamma:g}"
        .replace(".", "p")
    )

    return (
        f"rbf_svr_C_{C:g}"
        f"_gamma_{gamma_label}"
        f"_eps_{epsilon:g}"
    )


def fit_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
) -> tuple[
    MultiOutputRegressor,
    dict[str, Any],
]:
    """Fit one candidate and evaluate validation only."""
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

    prediction = clip_prediction(
        estimator.predict(
            x_val
        )
    )

    support_counts = [
        int(
            len(model.support_)
        )
        for model
        in estimator.estimators_
    ]

    row = {
        "candidate": candidate_name(
            C=C,
            gamma=gamma,
            epsilon=epsilon,
        ),
        "C": float(C),
        "gamma": float(gamma),
        "epsilon": float(epsilon),
        "val_mse": mse(
            y_val,
            prediction,
        ),
        "val_mae": mae(
            y_val,
            prediction,
        ),
        "fit_seconds": fit_seconds,
        "support_vectors_story_1": (
            support_counts[0]
        ),
        "support_vectors_story_2": (
            support_counts[1]
        ),
        "support_vectors_story_3": (
            support_counts[2]
        ),
        "support_vectors_story_4": (
            support_counts[3]
        ),
        "mean_support_vector_count": float(
            np.mean(
                support_counts
            )
        ),
        "mean_support_vector_ratio": float(
            np.mean(
                support_counts
            )
            / x_train.shape[0]
        ),
        "selected": False,
    }

    return estimator, row


def best_by_parameter(
    frame: pd.DataFrame,
    parameter: str,
) -> pd.DataFrame:
    """Return best validation row for each parameter value."""
    rows = []

    for _, group in frame.groupby(
        parameter,
        sort=True,
    ):
        best = (
            group.sort_values(
                [
                    "val_mse",
                    "candidate",
                ]
            )
            .iloc[0]
        )

        rows.append(
            best.to_dict()
        )

    return pd.DataFrame(
        rows
    )


def validation_damage_bins(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    """Validation-only damage-bin diagnostics."""
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

    rows = []

    for name, mask in bins.items():
        if not np.any(mask):
            raise ValueError(
                f"Empty bin: {name}"
            )

        y_true = truth[mask]
        y_pred = prediction[mask]

        error = (
            y_pred - y_true
        )

        rows.append(
            {
                "damage_bin": name,
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
                    np.nan
                    if name == "zero"
                    else float(
                        np.mean(
                            y_pred
                            < y_true
                        )
                    )
                ),
            }
        )

    order = {
        "zero": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    frame = pd.DataFrame(
        rows
    )

    frame["_order"] = (
        frame["damage_bin"]
        .map(order)
    )

    return (
        frame.sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )


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
            "svr_final_boundary_78d"
        ),
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = (
        load_train_validation_only(
            args.input
        )
    )

    x_train = arrays[
        "F_train"
    ]

    x_val = arrays[
        "F_val"
    ]

    y_train = arrays[
        "y_train"
    ]

    y_val = arrays[
        "y_val"
    ]

    if (
        x_train.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Expected 78 features, "
            f"found {x_train.shape[1]}."
        )

    if (
        y_train.ndim != 2
        or y_train.shape[1] != 4
    ):
        raise ValueError(
            "Expected four-storey targets."
        )

    total_candidates = (
        len(C_VALUES)
        * len(GAMMA_VALUES)
        * len(EPSILON_VALUES)
    )

    print(
        "===== 78D SVR FINAL "
        "VALIDATION-ONLY BOUNDARY CHECK ====="
    )

    print(
        "Train/validation:",
        x_train.shape[0],
        x_val.shape[0],
    )

    print(
        "Feature count:",
        x_train.shape[1],
    )

    print(
        "Candidate count:",
        total_candidates,
    )

    print(
        "Test arrays accessed: False"
    )

    print()

    rows: list[
        dict[str, Any]
    ] = []

    best_model = None
    best_row = None
    anchor_row = None

    for C in C_VALUES:
        for gamma in GAMMA_VALUES:
            for epsilon in EPSILON_VALUES:

                (
                    model,
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

                rows.append(
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
                    C == ANCHOR["C"]
                    and gamma
                    == ANCHOR["gamma"]
                    and epsilon
                    == ANCHOR["epsilon"]
                ):
                    anchor_row = (
                        row.copy()
                    )

                if best_row is None:
                    best_row = (
                        row.copy()
                    )
                    best_model = model
                else:
                    current_key = (
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
                        current_key
                        < best_key
                    ):
                        best_row = (
                            row.copy()
                        )
                        best_model = model

    if (
        best_row is None
        or best_model is None
    ):
        raise RuntimeError(
            "No model selected."
        )

    if anchor_row is None:
        raise RuntimeError(
            "Anchor configuration missing."
        )

    anchor_passed = bool(
        np.isclose(
            float(
                anchor_row[
                    "val_mse"
                ]
            ),
            float(
                ANCHOR[
                    "val_mse"
                ]
            ),
            rtol=0.0,
            atol=ANCHOR_TOLERANCE,
        )
    )

    if not anchor_passed:
        raise RuntimeError(
            "STOP: previous validation "
            "anchor was not reproduced. "
            f"actual="
            f"{anchor_row['val_mse']:.12f}, "
            f"expected="
            f"{ANCHOR['val_mse']:.12f}"
        )

    for row in rows:
        row["selected"] = (
            row["candidate"]
            == best_row[
                "candidate"
            ]
        )

    frame = (
        pd.DataFrame(rows)
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

    by_C = best_by_parameter(
        frame,
        "C",
    )

    by_gamma = (
        best_by_parameter(
            frame,
            "gamma",
        )
    )

    by_epsilon = (
        best_by_parameter(
            frame,
            "epsilon",
        )
    )

    selected_val_prediction = (
        clip_prediction(
            best_model.predict(
                x_val
            )
        )
    )

    bin_frame = (
        validation_damage_bins(
            truth=y_val,
            prediction=(
                selected_val_prediction
            ),
        )
    )

    selected_C = float(
        best_row["C"]
    )

    selected_gamma = float(
        best_row[
            "gamma"
        ]
    )

    selected_epsilon = float(
        best_row[
            "epsilon"
        ]
    )

    C_at_upper_boundary = (
        selected_C
        == max(C_VALUES)
    )

    C_at_lower_boundary = (
        selected_C
        == min(C_VALUES)
    )

    gamma_at_upper_boundary = (
        selected_gamma
        == max(GAMMA_VALUES)
    )

    gamma_at_lower_boundary = (
        selected_gamma
        == min(GAMMA_VALUES)
    )

    epsilon_at_upper_boundary = (
        selected_epsilon
        == max(EPSILON_VALUES)
    )

    epsilon_at_lower_boundary = (
        selected_epsilon
        == min(EPSILON_VALUES)
    )

    # Determine how much the upper C boundary improves
    # over the best interior high-C solution.
    #
    # 对比C=1000与500/750中的最佳配置。
    high_interior_C_values = [
        500.0,
        750.0,
    ]

    interior_rows = (
        by_C.loc[
            by_C["C"].isin(
                high_interior_C_values
            )
        ]
    )

    if interior_rows.empty:
        raise RuntimeError(
            "Missing interior C diagnostic rows."
        )

    best_high_interior = (
        interior_rows.sort_values(
            "val_mse"
        )
        .iloc[0]
    )

    best_high_interior_mse = float(
        best_high_interior[
            "val_mse"
        ]
    )

    best_1000_row = (
        by_C.loc[
            np.isclose(
                by_C["C"],
                1000.0,
            )
        ]
    )

    if best_1000_row.empty:
        raise RuntimeError(
            "C=1000 diagnostic missing."
        )

    mse_C1000 = float(
        best_1000_row[
            "val_mse"
        ].iloc[0]
    )

    C1000_improvement_vs_best_interior_percent = (
        (
            best_high_interior_mse
            - mse_C1000
        )
        / best_high_interior_mse
        * 100.0
    )

    # FINAL stop rule.
    #
    # 无论结果如何，本轮后均停止固定划分SVR调参。
    if not C_at_upper_boundary:
        stop_decision = (
            "STOP_INTERNAL_C_OPTIMUM"
        )
        fixed_split_tuning_complete = True

    elif (
        C1000_improvement_vs_best_interior_percent
        < 1.0
    ):
        stop_decision = (
            "STOP_UPPER_C_PLATEAU"
        )
        fixed_split_tuning_complete = True

    elif (
        C1000_improvement_vs_best_interior_percent
        <= 2.0
    ):
        stop_decision = (
            "STOP_UPPER_C_SMALL_GAIN"
        )
        fixed_split_tuning_complete = True

    else:
        stop_decision = (
            "STOP_FIXED_SPLIT_TUNING_"
            "FLAG_C_TREND_FOR_REPEATED_SPLIT"
        )
        fixed_split_tuning_complete = True

    candidate_path = (
        args.output_root
        / "svr_final_boundary_candidates.csv"
    )

    by_C_path = (
        args.output_root
        / "svr_final_best_by_C.csv"
    )

    by_gamma_path = (
        args.output_root
        / "svr_final_best_by_gamma.csv"
    )

    by_epsilon_path = (
        args.output_root
        / "svr_final_best_by_epsilon.csv"
    )

    bin_path = (
        args.output_root
        / "svr_final_validation_damage_bins.csv"
    )

    report_path = (
        args.output_root
        / "svr_final_boundary_report.json"
    )

    frame.to_csv(
        candidate_path,
        index=False,
    )

    by_C.to_csv(
        by_C_path,
        index=False,
    )

    by_gamma.to_csv(
        by_gamma_path,
        index=False,
    )

    by_epsilon.to_csv(
        by_epsilon_path,
        index=False,
    )

    bin_frame.to_csv(
        bin_path,
        index=False,
    )

    report = {
        "feature_set": (
            "signal_derived_with_ground"
        ),
        "feature_count": (
            EXPECTED_FEATURE_COUNT
        ),
        "test_arrays_accessed": False,
        "selection_criterion": (
            "clipped validation MSE"
        ),
        "search_space": {
            "C": C_VALUES,
            "gamma": GAMMA_VALUES,
            "epsilon": EPSILON_VALUES,
            "candidate_count": (
                total_candidates
            ),
        },
        "anchor_reproduction_passed": (
            anchor_passed
        ),
        "anchor": anchor_row,
        "selected_candidate": (
            best_row
        ),
        "boundary_checks": {
            "C_at_lower_boundary": (
                C_at_lower_boundary
            ),
            "C_at_upper_boundary": (
                C_at_upper_boundary
            ),
            "gamma_at_lower_boundary": (
                gamma_at_lower_boundary
            ),
            "gamma_at_upper_boundary": (
                gamma_at_upper_boundary
            ),
            "epsilon_at_lower_boundary": (
                epsilon_at_lower_boundary
            ),
            "epsilon_at_upper_boundary": (
                epsilon_at_upper_boundary
            ),
        },
        "upper_C_diagnostic": {
            "best_interior_C": float(
                best_high_interior[
                    "C"
                ]
            ),
            "best_interior_val_mse": (
                best_high_interior_mse
            ),
            "C1000_val_mse": (
                mse_C1000
            ),
            "C1000_improvement_vs_best_interior_percent": (
                C1000_improvement_vs_best_interior_percent
            ),
        },
        "stop_decision": (
            stop_decision
        ),
        "fixed_split_tuning_complete": (
            fixed_split_tuning_complete
        ),
        "next_stage": (
            "92D SVR comparison and "
            "10-seed repeated-split validation"
        ),
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
        anchor_passed,
    )

    print(
        "Anchor val MSE:",
        f"{anchor_row['val_mse']:.12f}",
    )

    print()
    print(
        "===== FINAL VALIDATION SELECTION ====="
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
        "Gamma at lower boundary:",
        gamma_at_lower_boundary,
    )

    print(
        "Gamma at upper boundary:",
        gamma_at_upper_boundary,
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
        "===== BEST BY C ====="
    )

    print(
        by_C[
            [
                "C",
                "gamma",
                "epsilon",
                "val_mse",
                "val_mae",
                "mean_support_vector_ratio",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== BEST BY GAMMA ====="
    )

    print(
        by_gamma[
            [
                "gamma",
                "C",
                "epsilon",
                "val_mse",
                "val_mae",
                "mean_support_vector_ratio",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== BEST BY EPSILON ====="
    )

    print(
        by_epsilon[
            [
                "epsilon",
                "C",
                "gamma",
                "val_mse",
                "val_mae",
                "mean_support_vector_ratio",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== SELECTED VALIDATION "
        "DAMAGE-BIN RESULTS ====="
    )

    print(
        bin_frame.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== UPPER-C STOP RULE ====="
    )

    print(
        "Best interior C:",
        float(
            best_high_interior[
                "C"
            ]
        ),
    )

    print(
        "Best interior val MSE:",
        f"{best_high_interior_mse:.10f}",
    )

    print(
        "C=1000 val MSE:",
        f"{mse_C1000:.10f}",
    )

    print(
        "C=1000 improvement vs "
        "best interior (%):",
        (
            f"{C1000_improvement_vs_best_interior_percent:.6f}"
        ),
    )

    print(
        "Decision:",
        stop_decision,
    )

    print(
        "Fixed-split tuning complete:",
        fixed_split_tuning_complete,
    )

    print()
    print(
        "===== TOP 10 VALIDATION "
        "CANDIDATES ====="
    )

    print(
        frame.head(
            10
        )[
            [
                "candidate",
                "C",
                "gamma",
                "epsilon",
                "val_mse",
                "val_mae",
                "mean_support_vector_ratio",
                "fit_seconds",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "CHECK PASSED: final validation-only "
        "SVR boundary check completed."
    )

    print(
        "Test arrays accessed: False"
    )

    print(
        "Fixed-split tuning complete: True"
    )

    print(
        "Next stage: 92D SVR comparison "
        "and repeated-split validation."
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
