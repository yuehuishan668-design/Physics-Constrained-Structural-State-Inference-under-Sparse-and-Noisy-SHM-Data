"""
Fixed-split multi-model comparison on the 78-dimensional
signal-derived-with-ground descriptor set.

Models
------
1. Ridge
2. ElasticNet
3. Random Forest
4. RBF-SVR
5. HistGradientBoosting

Scientific protocol
-------------------
- Frozen 2100 / 450 / 450 split.
- Standardised 78D descriptors.
- Candidate selection using clipped validation MSE only.
- Test set evaluated once for the selected candidate of each model family.
- Predictions clipped to [0, 0.5].
- Detailed zero/low/medium/high damage diagnostics.

78维主描述符集多模型比较。
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn

from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
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

RANDOM_STATE = 42

RIDGE_ALPHAS = [
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


# Preserve the main AES ElasticNet grid for continuity.
#
# 保留旧主实验ElasticNet搜索空间，便于方法连续比较。
ELASTICNET_ALPHAS = [
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
]

ELASTICNET_L1_RATIOS = [
    0.1,
    0.3,
    0.5,
    0.7,
]


# Preserve the main AES random-forest grid.
#
# 保留旧主实验Random Forest搜索空间。
RF_DEPTHS = [
    2,
    3,
    4,
    None,
]

RF_MIN_SAMPLES_LEAF = [
    1,
    3,
    5,
]

RF_N_ESTIMATORS = 300


# New nonlinear SVR search space.
#
# 新增RBF-SVR候选。
SVR_C_VALUES = [
    0.1,
    1.0,
    10.0,
    100.0,
]

SVR_GAMMAS: list[str | float] = [
    "scale",
    0.01,
    0.1,
]

SVR_EPSILONS = [
    0.01,
    0.03,
]


# New histogram-gradient-boosting search space.
#
# 新增HistGradientBoosting候选。
HGB_LEARNING_RATES = [
    0.03,
    0.05,
    0.1,
]

HGB_MAX_LEAF_NODES = [
    7,
    15,
    31,
]

HGB_L2_REGULARIZATIONS = [
    0.0,
    0.1,
]

HGB_MAX_ITER = 300
HGB_MIN_SAMPLES_LEAF = 20


def root_mean_squared_error(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate RMSE."""
    return float(
        np.sqrt(
            mean_squared_error(
                truth,
                prediction,
            )
        )
    )


def calculate_damage_bin_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Compute zero/low/medium/high damage diagnostics.

    Damage bins:
    zero   : y <= 1e-12
    low    : 1e-12 < y <= 0.10
    medium : 0.10 < y <= 0.20
    high   : y > 0.20
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

    rows: list[dict[str, Any]] = []

    for bin_name, mask in bins.items():
        count = int(
            np.sum(mask)
        )

        if count == 0:
            raise ValueError(
                f"Damage bin '{bin_name}' is empty."
            )

        bin_truth = truth[mask]
        bin_prediction = prediction[mask]

        error = (
            bin_prediction
            - bin_truth
        )

        if bin_name == "zero":
            underestimation = np.nan
        else:
            underestimation = float(
                np.mean(
                    bin_prediction
                    < bin_truth
                )
            )

        rows.append(
            {
                "model": model_name,
                "damage_bin": bin_name,
                "n_entries": count,
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
                    np.mean(bin_truth)
                ),
                "mean_prediction": float(
                    np.mean(
                        bin_prediction
                    )
                ),
                "underestimation_ratio": (
                    underestimation
                ),
            }
        )

    return rows


def make_ridge_candidates() -> list[
    tuple[str, dict[str, Any], Callable[[], Any]]
]:
    """Build Ridge candidates."""
    candidates = []

    for alpha in RIDGE_ALPHAS:
        parameters = {
            "alpha": float(alpha),
        }

        label = (
            f"ridge_alpha_{alpha:g}"
        )

        candidates.append(
            (
                label,
                parameters,
                lambda alpha=alpha: Ridge(
                    alpha=alpha,
                    fit_intercept=True,
                    solver="auto",
                ),
            )
        )

    return candidates


def make_elasticnet_candidates() -> list[
    tuple[str, dict[str, Any], Callable[[], Any]]
]:
    """Build ElasticNet candidates."""
    candidates = []

    for alpha, l1_ratio in itertools.product(
        ELASTICNET_ALPHAS,
        ELASTICNET_L1_RATIOS,
    ):
        parameters = {
            "alpha": float(alpha),
            "l1_ratio": float(
                l1_ratio
            ),
            "max_iter": 20000,
        }

        label = (
            f"elasticnet_alpha_{alpha:g}"
            f"_l1_{l1_ratio:g}"
        )

        candidates.append(
            (
                label,
                parameters,
                lambda alpha=alpha,l1_ratio=l1_ratio: ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=True,
                    max_iter=20000,
                    tol=1.0e-6,
                    selection="cyclic",
                ),
            )
        )

    return candidates


def make_random_forest_candidates() -> list[
    tuple[str, dict[str, Any], Callable[[], Any]]
]:
    """Build Random Forest candidates."""
    candidates = []

    for depth, leaf in itertools.product(
        RF_DEPTHS,
        RF_MIN_SAMPLES_LEAF,
    ):
        parameters = {
            "n_estimators": (
                RF_N_ESTIMATORS
            ),
            "max_depth": depth,
            "min_samples_leaf": (
                leaf
            ),
            "random_state": (
                RANDOM_STATE
            ),
            "n_jobs": 1,
        }

        depth_label = (
            "none"
            if depth is None
            else str(depth)
        )

        label = (
            f"random_forest_n_"
            f"{RF_N_ESTIMATORS}"
            f"_depth_{depth_label}"
            f"_leaf_{leaf}"
        )

        candidates.append(
            (
                label,
                parameters,
                lambda depth=depth,leaf=leaf: RandomForestRegressor(
                    n_estimators=(
                        RF_N_ESTIMATORS
                    ),
                    max_depth=depth,
                    min_samples_leaf=(
                        leaf
                    ),
                    random_state=(
                        RANDOM_STATE
                    ),
                    n_jobs=1,
                ),
            )
        )

    return candidates


def make_svr_candidates() -> list[
    tuple[str, dict[str, Any], Callable[[], Any]]
]:
    """Build independent four-output RBF-SVR candidates."""
    candidates = []

    for C, gamma, epsilon in itertools.product(
        SVR_C_VALUES,
        SVR_GAMMAS,
        SVR_EPSILONS,
    ):
        parameters = {
            "kernel": "rbf",
            "C": float(C),
            "gamma": gamma,
            "epsilon": float(
                epsilon
            ),
        }

        gamma_label = str(
            gamma
        ).replace(".", "p")

        label = (
            f"rbf_svr_C_{C:g}"
            f"_gamma_{gamma_label}"
            f"_eps_{epsilon:g}"
        )

        candidates.append(
            (
                label,
                parameters,
                lambda C=C,gamma=gamma,epsilon=epsilon: MultiOutputRegressor(
                    SVR(
                        kernel="rbf",
                        C=C,
                        gamma=gamma,
                        epsilon=epsilon,
                        tol=1.0e-4,
                        cache_size=1024,
                    ),
                    n_jobs=1,
                ),
            )
        )

    return candidates


def make_hist_gradient_boosting_candidates() -> list[
    tuple[str, dict[str, Any], Callable[[], Any]]
]:
    """Build independent four-output HistGradientBoosting candidates."""
    candidates = []

    for (
        learning_rate,
        max_leaf_nodes,
        l2_regularization,
    ) in itertools.product(
        HGB_LEARNING_RATES,
        HGB_MAX_LEAF_NODES,
        HGB_L2_REGULARIZATIONS,
    ):
        parameters = {
            "learning_rate": float(
                learning_rate
            ),
            "max_leaf_nodes": int(
                max_leaf_nodes
            ),
            "l2_regularization": float(
                l2_regularization
            ),
            "max_iter": (
                HGB_MAX_ITER
            ),
            "min_samples_leaf": (
                HGB_MIN_SAMPLES_LEAF
            ),
            "early_stopping": False,
            "random_state": (
                RANDOM_STATE
            ),
        }

        label = (
            "hist_gradient_boosting"
            f"_lr_{learning_rate:g}"
            f"_leaf_{max_leaf_nodes}"
            f"_l2_{l2_regularization:g}"
        )

        candidates.append(
            (
                label,
                parameters,
                lambda learning_rate=learning_rate,max_leaf_nodes=max_leaf_nodes,l2_regularization=l2_regularization,: MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        learning_rate=(
                            learning_rate
                        ),
                        max_leaf_nodes=(
                            max_leaf_nodes
                        ),
                        l2_regularization=(
                            l2_regularization
                        ),
                        max_iter=(
                            HGB_MAX_ITER
                        ),
                        min_samples_leaf=(
                            HGB_MIN_SAMPLES_LEAF
                        ),
                        early_stopping=False,
                        random_state=(
                            RANDOM_STATE
                        ),
                    ),
                    n_jobs=1,
                ),
            )
        )

    return candidates


MODEL_CANDIDATE_BUILDERS = {
    "ridge": make_ridge_candidates,
    "elasticnet": (
        make_elasticnet_candidates
    ),
    "random_forest": (
        make_random_forest_candidates
    ),
    "rbf_svr": make_svr_candidates,
    "hist_gradient_boosting": (
        make_hist_gradient_boosting_candidates
    ),
}


def train_model_family(
    model_name: str,
    candidate_definitions: list[
        tuple[
            str,
            dict[str, Any],
            Callable[[], Any],
        ]
    ],
    arrays: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
]:
    """
    Select one candidate using validation MSE, then evaluate test once.

    测试集只对验证集选中的候选模型进行一次评价。
    """
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

    candidate_rows: list[
        dict[str, Any]
    ] = []

    best_model = None
    best_label = None
    best_parameters = None
    best_val_mse = float("inf")
    best_val_mae = float("inf")

    family_start = time.perf_counter()

    for (
        candidate_label,
        parameters,
        estimator_factory,
    ) in candidate_definitions:

        estimator = (
            estimator_factory()
        )

        fit_start = time.perf_counter()

        with warnings.catch_warnings(
            record=True
        ) as recorded_warnings:
            warnings.simplefilter(
                "always"
            )

            estimator.fit(
                x_train,
                y_train,
            )

        fit_seconds = float(
            time.perf_counter()
            - fit_start
        )

        convergence_warning_count = sum(
            issubclass(
                warning.category,
                ConvergenceWarning,
            )
            for warning
            in recorded_warnings
        )

        val_prediction = (
            clip_damage_prediction(
                estimator.predict(
                    x_val
                )
            )
        )

        val_mse = (
            mean_squared_error(
                y_val,
                val_prediction,
            )
        )

        val_mae = (
            mean_absolute_error(
                y_val,
                val_prediction,
            )
        )

        row = {
            "model": model_name,
            "candidate": (
                candidate_label
            ),
            "parameters_json": json.dumps(
                parameters,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "val_mse": float(
                val_mse
            ),
            "val_mae": float(
                val_mae
            ),
            "fit_seconds": (
                fit_seconds
            ),
            "convergence_warning_count": int(
                convergence_warning_count
            ),
            "selected": False,
        }

        candidate_rows.append(
            row
        )

        selection_key = (
            float(val_mse),
            candidate_label,
        )

        current_best_key = (
            float(best_val_mse),
            (
                best_label
                if best_label
                is not None
                else ""
            ),
        )

        if (
            best_model is None
            or selection_key
            < current_best_key
        ):
            best_model = estimator
            best_label = (
                candidate_label
            )
            best_parameters = (
                parameters
            )
            best_val_mse = float(
                val_mse
            )
            best_val_mae = float(
                val_mae
            )

    if best_model is None:
        raise RuntimeError(
            f"No model selected for "
            f"{model_name}."
        )

    for row in candidate_rows:
        row["selected"] = (
            row["candidate"]
            == best_label
        )

    # Test is touched only here, after validation selection.
    #
    # 只有完成验证集选择后才评价测试集。
    test_prediction = (
        clip_damage_prediction(
            best_model.predict(
                x_test
            )
        )
    )

    metrics = calculate_metrics(
        y_test,
        test_prediction,
    )

    total_seconds = float(
        time.perf_counter()
        - family_start
    )

    summary = {
        "model": model_name,
        "selected_candidate": (
            best_label
        ),
        "selected_parameters_json": (
            json.dumps(
                best_parameters,
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
        "n_candidates": int(
            len(
                candidate_definitions
            )
        ),
        "val_mse": (
            best_val_mse
        ),
        "val_mae": (
            best_val_mae
        ),
        **metrics,
        "family_total_seconds": (
            total_seconds
        ),
    }

    return (
        summary,
        candidate_rows,
        test_prediction,
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
            "multimodel_78d"
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

    if (
        arrays["F_train"].shape[1]
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Expected 78 features, "
            f"found "
            f"{arrays['F_train'].shape[1]}."
        )

    if arrays["y_train"].ndim != 2:
        raise ValueError(
            "Expected a 2D multi-storey "
            "target matrix."
        )

    if arrays["y_train"].shape[1] != 4:
        raise ValueError(
            "Expected four storey targets, "
            f"found "
            f"{arrays['y_train'].shape[1]}."
        )

    print(
        "===== 78D MULTI-MODEL "
        "COMPARISON ====="
    )

    print(
        "scikit-learn:",
        sklearn.__version__,
    )

    print(
        "Input:",
        args.input,
    )

    print(
        "Train/val/test:",
        arrays["F_train"].shape[0],
        arrays["F_val"].shape[0],
        arrays["F_test"].shape[0],
    )

    print(
        "Features:",
        arrays["F_train"].shape[1],
    )

    print()

    summaries: list[
        dict[str, Any]
    ] = []

    all_candidate_rows: list[
        dict[str, Any]
    ] = []

    all_bin_rows: list[
        dict[str, Any]
    ] = []

    for (
        model_name,
        candidate_builder,
    ) in MODEL_CANDIDATE_BUILDERS.items():

        candidate_definitions = (
            candidate_builder()
        )

        print(
            f"Training {model_name}: "
            f"{len(candidate_definitions)} "
            "candidates ..."
        )

        (
            summary,
            candidate_rows,
            test_prediction,
        ) = train_model_family(
            model_name=(
                model_name
            ),
            candidate_definitions=(
                candidate_definitions
            ),
            arrays=arrays,
        )

        summaries.append(
            summary
        )

        all_candidate_rows.extend(
            candidate_rows
        )

        bin_rows = (
            calculate_damage_bin_metrics(
                truth=arrays[
                    "y_test"
                ],
                prediction=(
                    test_prediction
                ),
                model_name=(
                    model_name
                ),
            )
        )

        all_bin_rows.extend(
            bin_rows
        )

        np.savez_compressed(
            args.output_root
            / (
                f"{model_name}_"
                "test_predictions.npz"
            ),
            y_test=arrays[
                "y_test"
            ],
            y_prediction=(
                test_prediction
            ),
            selected_candidate=(
                np.asarray(
                    summary[
                        "selected_candidate"
                    ]
                )
            ),
            selected_parameters_json=(
                np.asarray(
                    summary[
                        "selected_parameters_json"
                    ]
                )
            ),
            test_idx=arrays.get(
                "test_idx",
                np.arange(
                    arrays[
                        "y_test"
                    ].shape[0]
                ),
            ),
            case_id_test=arrays.get(
                "case_id_test",
                np.arange(
                    arrays[
                        "y_test"
                    ].shape[0]
                ),
            ),
        )

        print(
            "  selected:",
            summary[
                "selected_candidate"
            ],
        )

        print(
            "  val_mse:",
            f"{summary['val_mse']:.10f}",
        )

        print(
            "  test_mae:",
            f"{summary['test_mae']:.10f}",
        )

        print(
            "  high_damage_mae:",
            f"{summary['high_damage_mae']:.10f}",
        )

        print(
            "  high_damage_underestimation:",
            (
                f"{summary['high_damage_underestimation_ratio']:.10f}"
            ),
        )

        print()

    summary_frame = pd.DataFrame(
        summaries
    )

    candidate_frame = pd.DataFrame(
        all_candidate_rows
    )

    bin_frame = pd.DataFrame(
        all_bin_rows
    )

    # Sanity-check the already verified Ridge baseline.
    #
    # Ridge必须复现上一阶段78D固定划分结果。
    ridge_row = summary_frame.loc[
        summary_frame["model"]
        == "ridge"
    ].iloc[0]

    expected_ridge_mae = (
        0.0450937333
    )

    expected_ridge_alpha = (
        3.0e-4
    )

    ridge_parameters = json.loads(
        ridge_row[
            "selected_parameters_json"
        ]
    )

    actual_ridge_alpha = float(
        ridge_parameters["alpha"]
    )

    ridge_reproduction_passed = (
        np.isclose(
            float(
                ridge_row[
                    "test_mae"
                ]
            ),
            expected_ridge_mae,
            rtol=0.0,
            atol=5.0e-11,
        )
        and np.isclose(
            actual_ridge_alpha,
            expected_ridge_alpha,
            rtol=0.0,
            atol=1.0e-15,
        )
    )

    if not ridge_reproduction_passed:
        raise RuntimeError(
            "STOP: the verified 78D Ridge "
            "baseline was not reproduced. "
            f"alpha={actual_ridge_alpha}, "
            f"test_mae="
            f"{ridge_row['test_mae']}"
        )

    ridge_mae = float(
        ridge_row["test_mae"]
    )

    ridge_high_mae = float(
        ridge_row[
            "high_damage_mae"
        ]
    )

    ridge_high_bias = float(
        ridge_row[
            "high_damage_bias"
        ]
    )

    ridge_under = float(
        ridge_row[
            "high_damage_underestimation_ratio"
        ]
    )

    summary_frame[
        "test_mae_change_vs_ridge_percent"
    ] = (
        (
            summary_frame["test_mae"]
            - ridge_mae
        )
        / ridge_mae
        * 100.0
    )

    summary_frame[
        "high_damage_mae_change_vs_ridge_percent"
    ] = (
        (
            summary_frame[
                "high_damage_mae"
            ]
            - ridge_high_mae
        )
        / ridge_high_mae
        * 100.0
    )

    summary_frame[
        "high_damage_bias_change_vs_ridge"
    ] = (
        summary_frame[
            "high_damage_bias"
        ]
        - ridge_high_bias
    )

    summary_frame[
        "high_damage_underestimation_change_vs_ridge_percentage_points"
    ] = (
        (
            summary_frame[
                "high_damage_underestimation_ratio"
            ]
            - ridge_under
        )
        * 100.0
    )

    summary_frame = (
        summary_frame.sort_values(
            [
                "test_mae",
                "high_damage_mae",
            ],
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    candidate_frame = (
        candidate_frame.sort_values(
            [
                "model",
                "val_mse",
                "candidate",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    bin_order = {
        "zero": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    bin_frame[
        "_bin_order"
    ] = bin_frame[
        "damage_bin"
    ].map(
        bin_order
    )

    bin_frame = (
        bin_frame.sort_values(
            [
                "model",
                "_bin_order",
            ]
        )
        .drop(
            columns=[
                "_bin_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary_path = (
        args.output_root
        / "multimodel_78d_summary.csv"
    )

    candidate_path = (
        args.output_root
        / "multimodel_78d_candidates.csv"
    )

    bin_path = (
        args.output_root
        / "multimodel_78d_damage_bins.csv"
    )

    report_path = (
        args.output_root
        / "multimodel_78d_report.json"
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    bin_frame.to_csv(
        bin_path,
        index=False,
    )

    best_overall = (
        summary_frame.iloc[0]
    )

    best_high_damage = (
        summary_frame.sort_values(
            "high_damage_mae",
            ascending=True,
        ).iloc[0]
    )

    lowest_underestimation = (
        summary_frame.sort_values(
            "high_damage_underestimation_ratio",
            ascending=True,
        ).iloc[0]
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
        "sklearn_version": (
            sklearn.__version__
        ),
        "protocol": {
            "train_validation_test": [
                int(
                    arrays[
                        "F_train"
                    ].shape[0]
                ),
                int(
                    arrays[
                        "F_val"
                    ].shape[0]
                ),
                int(
                    arrays[
                        "F_test"
                    ].shape[0]
                ),
            ],
            "selection_criterion": (
                "clipped validation MSE"
            ),
            "prediction_clip": [
                0.0,
                0.5,
            ],
            "test_evaluations_per_model_family": 1,
            "random_state_where_applicable": (
                RANDOM_STATE
            ),
        },
        "ridge_reproduction_passed": (
            bool(
                ridge_reproduction_passed
            )
        ),
        "best_overall_test_mae_model": (
            str(
                best_overall[
                    "model"
                ]
            )
        ),
        "best_high_damage_mae_model": (
            str(
                best_high_damage[
                    "model"
                ]
            )
        ),
        "lowest_high_damage_underestimation_model": (
            str(
                lowest_underestimation[
                    "model"
                ]
            )
        ),
        "output_files": {
            "summary": str(
                summary_path
            ),
            "candidates": str(
                candidate_path
            ),
            "damage_bins": str(
                bin_path
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

    print(
        "===== SELECTED MODEL RESULTS ====="
    )

    display_columns = [
        "model",
        "selected_candidate",
        "n_candidates",
        "val_mse",
        "test_mae",
        "test_rmse",
        "damaged_entry_mae",
        "high_damage_mae",
        "high_damage_bias",
        "high_damage_underestimation_ratio",
        "test_mae_change_vs_ridge_percent",
        "high_damage_mae_change_vs_ridge_percent",
        "high_damage_underestimation_change_vs_ridge_percentage_points",
    ]

    print(
        summary_frame[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "===== DAMAGE-BIN METRICS ====="
    )

    print(
        bin_frame[
            [
                "model",
                "damage_bin",
                "n_entries",
                "mae",
                "rmse",
                "bias",
                "mean_true",
                "mean_prediction",
                "underestimation_ratio",
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
        "===== KEY CHECKS ====="
    )

    print(
        "Ridge reproduction passed:",
        ridge_reproduction_passed,
    )

    print(
        "Best overall Test MAE:",
        best_overall["model"],
        f"{best_overall['test_mae']:.10f}",
    )

    print(
        "Best high-damage MAE:",
        best_high_damage["model"],
        f"{best_high_damage['high_damage_mae']:.10f}",
    )

    print(
        "Lowest high-damage "
        "underestimation:",
        lowest_underestimation[
            "model"
        ],
        (
            f"{lowest_underestimation['high_damage_underestimation_ratio']:.10f}"
        ),
    )

    print()
    print(
        "CHECK PASSED: 78D multi-model "
        "comparison completed."
    )

    print(
        "Summary:",
        summary_path,
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


if __name__ == "__main__":
    main()
