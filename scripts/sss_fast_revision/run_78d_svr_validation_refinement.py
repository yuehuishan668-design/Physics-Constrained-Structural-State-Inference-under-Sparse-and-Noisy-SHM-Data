"""
Final validation-only RBF-SVR hyperparameter refinement for the
78-dimensional signal-derived-with-ground descriptor set.

IMPORTANT
---------
This script intentionally does NOT read F_test or y_test.

Model selection is based exclusively on clipped validation MSE.
The purpose is to close the remaining lower-gamma boundary issue
without further use of the fixed test partition.

78维RBF-SVR最终validation-only超参数细化。
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
    50.0,
    100.0,
    200.0,
    300.0,
]

GAMMA_VALUES = [
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.01,
]

EPSILON_VALUES = [
    0.02,
    0.03,
    0.04,
    0.05,
]


# Previously verified validation optimum.
#
# 上一轮已经验证的validation锚点。
ANCHOR = {
    "C": 100.0,
    "gamma": 0.003,
    "epsilon": 0.03,
    "val_mse": 0.002480928318839576,
}

ANCHOR_TOLERANCE = 5.0e-11


def load_train_validation_only(
    path: Path,
) -> dict[str, np.ndarray]:
    """
    Load only training and validation arrays.

    F_test and y_test are deliberately never accessed.
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
                f"Missing required arrays: {missing}"
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

    if (
        arrays["F_train"].shape[0]
        != arrays["y_train"].shape[0]
    ):
        raise ValueError(
            "Training sample count mismatch."
        )

    if (
        arrays["F_val"].shape[0]
        != arrays["y_val"].shape[0]
    ):
        raise ValueError(
            "Validation sample count mismatch."
        )

    return arrays


def clip_prediction(
    prediction: np.ndarray,
) -> np.ndarray:
    """Clip damage predictions to the physical interval."""
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
    """Mean squared error."""
    error = (
        np.asarray(
            prediction,
            dtype=np.float64,
        )
        - np.asarray(
            truth,
            dtype=np.float64,
        )
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
    """Mean absolute error."""
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction)
                - np.asarray(truth)
            )
        )
    )


def build_candidate_name(
    C: float,
    gamma: float,
    epsilon: float,
) -> str:
    """Create a stable candidate identifier."""
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
    """Fit one candidate and evaluate validation data only."""
    model = MultiOutputRegressor(
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

    model.fit(
        x_train,
        y_train,
    )

    fit_seconds = float(
        time.perf_counter()
        - start
    )

    prediction = clip_prediction(
        model.predict(
            x_val
        )
    )

    support_vector_counts = [
        int(
            len(
                estimator.support_
            )
        )
        for estimator
        in model.estimators_
    ]

    row = {
        "candidate": (
            build_candidate_name(
                C=C,
                gamma=gamma,
                epsilon=epsilon,
            )
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
                support_vector_counts
            )
            / x_train.shape[0]
        ),
        "selected": False,
    }

    return model, row


def damage_bin_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    """
    Compute validation-only damage-bin diagnostics.

    These metrics are descriptive only and are NOT used
    for hyperparameter selection.
    """
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
                f"Empty validation damage bin: {name}"
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
                    np.mean(y_true)
                ),
                "mean_prediction": float(
                    np.mean(y_pred)
                ),
                "underestimation_ratio": (
                    np.nan
                    if name == "zero"
                    else float(
                        np.mean(
                            y_pred < y_true
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


def best_row_for_parameter(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Return validation-optimal row for every unique parameter value."""
    rows = []

    for value, group in frame.groupby(
        column,
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
            "svr_validation_refinement_78d"
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

    print(
        "===== 78D SVR VALIDATION-ONLY "
        "REFINEMENT ====="
    )

    print(
        "Input:",
        args.input,
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
        "Test arrays accessed: False"
    )

    total_candidates = (
        len(C_VALUES)
        * len(GAMMA_VALUES)
        * len(EPSILON_VALUES)
    )

    print(
        "Candidate count:",
        total_candidates,
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

                model, row = (
                    fit_candidate(
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        C=C,
                        gamma=gamma,
                        epsilon=epsilon,
                    )
                )

                rows.append(row)

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
            "No candidate selected."
        )

    if anchor_row is None:
        raise RuntimeError(
            "Anchor configuration was "
            "not evaluated."
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
            "STOP: validation anchor "
            "was not reproduced. "
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

    # Validation diagnostics only.
    selected_val_prediction = (
        clip_prediction(
            best_model.predict(
                x_val
            )
        )
    )

    bin_frame = (
        damage_bin_metrics(
            truth=y_val,
            prediction=(
                selected_val_prediction
            ),
        )
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

    by_C = (
        best_row_for_parameter(
            frame,
            "C",
        )
    )

    by_gamma = (
        best_row_for_parameter(
            frame,
            "gamma",
        )
    )

    by_epsilon = (
        best_row_for_parameter(
            frame,
            "epsilon",
        )
    )

    best_gamma = float(
        best_row["gamma"]
    )

    gamma_at_lower_boundary = (
        best_gamma
        == min(
            GAMMA_VALUES
        )
    )

    gamma_at_upper_boundary = (
        best_gamma
        == max(
            GAMMA_VALUES
        )
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

    epsilon_at_lower_boundary = (
        float(
            best_row[
                "epsilon"
            ]
        )
        == min(EPSILON_VALUES)
    )

    epsilon_at_upper_boundary = (
        float(
            best_row[
                "epsilon"
            ]
        )
        == max(EPSILON_VALUES)
    )

    # Quantify whether the new lower gamma boundary
    # provides a meaningful improvement over gamma=0.001.
    best_gamma_0005 = (
        by_gamma.loc[
            np.isclose(
                by_gamma["gamma"],
                0.0005,
            )
        ]
    )

    best_gamma_001 = (
        by_gamma.loc[
            np.isclose(
                by_gamma["gamma"],
                0.001,
            )
        ]
    )

    if (
        best_gamma_0005.empty
        or best_gamma_001.empty
    ):
        raise RuntimeError(
            "Required gamma diagnostic rows "
            "are missing."
        )

    mse_0005 = float(
        best_gamma_0005[
            "val_mse"
        ].iloc[0]
    )

    mse_001 = float(
        best_gamma_001[
            "val_mse"
        ].iloc[0]
    )

    lower_gamma_improvement_vs_001_percent = (
        (
            mse_001
            - mse_0005
        )
        / mse_001
        * 100.0
    )

    if not gamma_at_lower_boundary:
        stop_decision = (
            "STOP_SEARCH_INTERNAL_GAMMA_OPTIMUM"
        )
        additional_lower_gamma_search_needed = False

    elif (
        lower_gamma_improvement_vs_001_percent
        < 2.0
    ):
        stop_decision = (
            "STOP_SEARCH_LOWER_BOUNDARY_PLATEAU"
        )
        additional_lower_gamma_search_needed = False

    else:
        stop_decision = (
            "CONSIDER_ONE_FINAL_LOWER_GAMMA_EXTENSION"
        )
        additional_lower_gamma_search_needed = True

    candidate_path = (
        args.output_root
        / "svr_validation_refinement_candidates.csv"
    )

    bin_path = (
        args.output_root
        / "svr_validation_selected_damage_bins.csv"
    )

    by_C_path = (
        args.output_root
        / "svr_validation_best_by_C.csv"
    )

    by_gamma_path = (
        args.output_root
        / "svr_validation_best_by_gamma.csv"
    )

    by_epsilon_path = (
        args.output_root
        / "svr_validation_best_by_epsilon.csv"
    )

    report_path = (
        args.output_root
        / "svr_validation_refinement_report.json"
    )

    frame.to_csv(
        candidate_path,
        index=False,
    )

    bin_frame.to_csv(
        bin_path,
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

    report = {
        "input_dataset": str(
            args.input
        ),
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
        "lower_gamma_diagnostic": {
            "best_val_mse_gamma_0.0005": (
                mse_0005
            ),
            "best_val_mse_gamma_0.001": (
                mse_001
            ),
            "gamma_0.0005_improvement_vs_0.001_percent": (
                lower_gamma_improvement_vs_001_percent
            ),
        },
        "stop_decision": (
            stop_decision
        ),
        "additional_lower_gamma_search_needed": (
            additional_lower_gamma_search_needed
        ),
        "output_files": {
            "candidates": str(
                candidate_path
            ),
            "validation_damage_bins": str(
                bin_path
            ),
            "best_by_C": str(
                by_C_path
            ),
            "best_by_gamma": str(
                by_gamma_path
            ),
            "best_by_epsilon": str(
                by_epsilon_path
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
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
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
            float_format=lambda value: (
                f"{value:.10f}"
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
            float_format=lambda value: (
                f"{value:.10f}"
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
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "===== LOWER-GAMMA STOP RULE ====="
    )

    print(
        "Best gamma=0.0005 val MSE:",
        f"{mse_0005:.10f}",
    )

    print(
        "Best gamma=0.001 val MSE:",
        f"{mse_001:.10f}",
    )

    print(
        "0.0005 improvement vs 0.001 (%):",
        (
            f"{lower_gamma_improvement_vs_001_percent:.6f}"
        ),
    )

    print(
        "Decision:",
        stop_decision,
    )

    print(
        "Additional lower-gamma search needed:",
        additional_lower_gamma_search_needed,
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
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "CHECK PASSED: validation-only "
        "SVR refinement completed."
    )

    print(
        "Test arrays accessed: False"
    )

    print(
        "Candidates:",
        candidate_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
