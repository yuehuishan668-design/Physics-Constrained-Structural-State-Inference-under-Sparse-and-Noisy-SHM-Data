"""
10-seed paired repeated-split validation:

Standard 78D RBF-SVR
vs
One-sided monotone residual-calibrated 78D RBF-SVR

The calibration protocol was developed in a validation-only pilot
and is frozen before this repeated-split evaluation.

Protocol
--------
For every outer seed:

1. Reuse the exact previously saved train/validation/test split.
2. Reuse the standard RBF-SVR hyperparameters already selected
   for that seed by validation MSE.
3. Inside the 2100-case training partition:
       - generate 5-fold OOF predictions;
       - fit a training-only storey-specific isotonic residual
         correction from:
             residual = truth - OOF prediction
       - correction is non-negative and monotonically increasing.
4. Reproduce the standard validation prediction.
5. Evaluate frozen alpha candidates:
       [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
6. Require:
       MSE       <= baseline * 1.05
       zero MAE  <= baseline * 1.10
       low MAE   <= baseline * 1.10
       high MAE  <= baseline
       |high bias| <= baseline
7. Among eligible alphas:
       a. minimum high-damage underestimation
       b. minimum high-damage MAE
       c. minimum |high-damage bias|
       d. minimum overall validation MSE
8. If no alpha is eligible:
       fallback to frozen standard SVR.
9. Only after validation selection:
       evaluate the calibrated prediction on test.

No test outcome is used for model/calibration selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from scipy.stats import (
    t as student_t,
    ttest_rel,
    wilcoxon,
)

from sklearn.isotonic import (
    IsotonicRegression,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    StandardScaler,
)
from sklearn.svm import SVR


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.sss_fast_revision.run_78d_ridge_svr_repeated_splits import (  # noqa: E402
    load_full_raw_dataset,
)


SEEDS = list(range(10))

EXPECTED_CASE_COUNT = 3000
EXPECTED_FEATURE_COUNT = 78

N_TRAIN = 2100
N_VAL = 450
N_TEST = 450

N_OOF_FOLDS = 5
OOF_RANDOM_STATE = 20260809

HIGH_DAMAGE_THRESHOLD = 0.20

ALPHA_VALUES = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
]

OVERALL_MSE_BUDGET = 1.05
ZERO_MAE_BUDGET = 1.10
LOW_MAE_BUDGET = 1.10

HIGH_MAE_BUDGET = 1.00
HIGH_ABS_BIAS_BUDGET = 1.00


PRIMARY_METRICS = [
    "test_mse",
    "test_mae",
    "test_rmse",
    "damaged_mae",
    "zero_mae",
    "low_mae",
    "medium_mae",
    "high_mae",
    "high_abs_bias",
    "high_underestimation_ratio",
]


CORRECTION_GRID = np.asarray(
    [
        0.000,
        0.025,
        0.050,
        0.075,
        0.100,
        0.125,
        0.150,
        0.175,
        0.200,
        0.225,
        0.250,
        0.300,
        0.350,
        0.400,
    ],
    dtype=np.float64,
)


def load_split(
    split_root: Path,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    path = (
        split_root
        / f"seed_{seed}_split.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        train_idx = np.asarray(
            data["train_idx"],
            dtype=np.int64,
        )

        val_idx = np.asarray(
            data["val_idx"],
            dtype=np.int64,
        )

        test_idx = np.asarray(
            data["test_idx"],
            dtype=np.int64,
        )

    if (
        len(train_idx) != N_TRAIN
        or len(val_idx) != N_VAL
        or len(test_idx) != N_TEST
    ):
        raise RuntimeError(
            f"Seed {seed}: split-size mismatch."
        )

    combined = np.concatenate(
        [
            train_idx,
            val_idx,
            test_idx,
        ]
    )

    if (
        len(np.unique(combined))
        != EXPECTED_CASE_COUNT
    ):
        raise RuntimeError(
            f"Seed {seed}: split overlap "
            "or omission detected."
        )

    if (
        np.min(combined) < 0
        or np.max(combined)
        >= EXPECTED_CASE_COUNT
    ):
        raise RuntimeError(
            f"Seed {seed}: invalid case index."
        )

    return (
        train_idx,
        val_idx,
        test_idx,
    )


def clip_prediction(
    prediction: np.ndarray,
) -> np.ndarray:

    return np.clip(
        np.asarray(
            prediction,
            dtype=np.float64,
        ),
        0.0,
        0.5,
    )


def _fit_one_storey(
    x_train: np.ndarray,
    target_train: np.ndarray,
    x_predict: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
) -> np.ndarray:

    model = SVR(
        kernel="rbf",
        C=C,
        gamma=gamma,
        epsilon=epsilon,
        tol=1e-4,
        cache_size=512,
    )

    model.fit(
        x_train,
        target_train,
    )

    return model.predict(
        x_predict
    )


def fit_multioutput_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
    n_jobs: int,
) -> np.ndarray:

    outputs = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
    )(
        delayed(
            _fit_one_storey
        )(
            x_train=x_train,
            target_train=y_train[
                :,
                storey,
            ],
            x_predict=x_predict,
            C=C,
            gamma=gamma,
            epsilon=epsilon,
        )
        for storey in range(
            y_train.shape[1]
        )
    )

    return clip_prediction(
        np.column_stack(
            outputs
        )
    )


def generate_oof_predictions(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
    n_jobs: int,
) -> np.ndarray:
    """
    Cross-fitted training predictions.

    Every OOF fold fits its own StandardScaler only on the
    corresponding fold-training cases.
    """

    oof = np.full(
        y_train.shape,
        np.nan,
        dtype=np.float64,
    )

    splitter = KFold(
        n_splits=N_OOF_FOLDS,
        shuffle=True,
        random_state=OOF_RANDOM_STATE,
    )

    for (
        fold_number,
        (
            fold_train_idx,
            fold_holdout_idx,
        ),
    ) in enumerate(
        splitter.split(
            np.arange(
                x_train_raw.shape[0]
            )
        ),
        start=1,
    ):

        scaler = StandardScaler()

        x_fold_train = (
            scaler.fit_transform(
                x_train_raw[
                    fold_train_idx
                ]
            )
        )

        x_fold_holdout = (
            scaler.transform(
                x_train_raw[
                    fold_holdout_idx
                ]
            )
        )

        prediction = (
            fit_multioutput_svr(
                x_train=x_fold_train,
                y_train=y_train[
                    fold_train_idx
                ],
                x_predict=x_fold_holdout,
                C=C,
                gamma=gamma,
                epsilon=epsilon,
                n_jobs=n_jobs,
            )
        )

        oof[
            fold_holdout_idx
        ] = prediction

        print(
            f"    OOF fold "
            f"{fold_number}/"
            f"{N_OOF_FOLDS} complete"
        )

    if not np.all(
        np.isfinite(oof)
    ):
        raise RuntimeError(
            "OOF predictions contain "
            "missing/non-finite values."
        )

    return oof


def fit_correction_models(
    y_train: np.ndarray,
    oof_prediction: np.ndarray,
) -> list[
    IsotonicRegression
]:

    models = []

    for storey in range(
        y_train.shape[1]
    ):

        prediction = (
            oof_prediction[
                :,
                storey,
            ]
        )

        residual = (
            y_train[
                :,
                storey,
            ]
            - prediction
        )

        model = IsotonicRegression(
            increasing=True,
            y_min=0.0,
            out_of_bounds="clip",
        )

        model.fit(
            prediction,
            residual,
        )

        models.append(
            model
        )

    return models


def predict_correction(
    models: list[
        IsotonicRegression
    ],
    prediction: np.ndarray,
) -> np.ndarray:

    columns = []

    for storey, model in enumerate(
        models
    ):

        values = model.predict(
            prediction[
                :,
                storey,
            ]
        )

        columns.append(
            np.maximum(
                values,
                0.0,
            )
        )

    correction = np.column_stack(
        columns
    )

    if np.any(
        correction < -1e-12
    ):
        raise RuntimeError(
            "Negative correction detected."
        )

    return correction


def calculate_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    error = (
        prediction - truth
    )

    masks = {
        "zero": (
            truth <= 1e-12
        ),
        "low": (
            (truth > 1e-12)
            & (truth <= 0.10)
        ),
        "medium": (
            (truth > 0.10)
            & (truth <= 0.20)
        ),
        "high": (
            truth > 0.20
        ),
        "damaged": (
            truth > 1e-12
        ),
    }

    for name, mask in (
        masks.items()
    ):
        if not np.any(mask):
            raise RuntimeError(
                f"Empty metric group: {name}"
            )

    high_error = (
        error[
            masks["high"]
        ]
    )

    high_bias = float(
        np.mean(
            high_error
        )
    )

    return {
        "mse": float(
            np.mean(
                error ** 2
            )
        ),
        "mae": float(
            np.mean(
                np.abs(
                    error
                )
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
            np.mean(
                error
            )
        ),
        "damaged_mae": float(
            np.mean(
                np.abs(
                    error[
                        masks[
                            "damaged"
                        ]
                    ]
                )
            )
        ),
        "zero_mae": float(
            np.mean(
                np.abs(
                    error[
                        masks["zero"]
                    ]
                )
            )
        ),
        "low_mae": float(
            np.mean(
                np.abs(
                    error[
                        masks["low"]
                    ]
                )
            )
        ),
        "medium_mae": float(
            np.mean(
                np.abs(
                    error[
                        masks[
                            "medium"
                        ]
                    ]
                )
            )
        ),
        "high_mae": float(
            np.mean(
                np.abs(
                    high_error
                )
            )
        ),
        "high_bias": (
            high_bias
        ),
        "high_abs_bias": (
            abs(
                high_bias
            )
        ),
        "high_underestimation_ratio": float(
            np.mean(
                prediction[
                    masks["high"]
                ]
                < truth[
                    masks["high"]
                ]
            )
        ),
        "high_n": int(
            np.sum(
                masks["high"]
            )
        ),
    }


def confidence_interval(
    values: np.ndarray,
) -> tuple[
    float,
    float,
]:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean_value = float(
        np.mean(values)
    )

    if len(values) < 2:
        return (
            mean_value,
            mean_value,
        )

    se = float(
        np.std(
            values,
            ddof=1,
        )
        / math.sqrt(
            len(values)
        )
    )

    critical = float(
        student_t.ppf(
            0.975,
            df=(
                len(values)
                - 1
            ),
        )
    )

    margin = (
        critical * se
    )

    return (
        mean_value - margin,
        mean_value + margin,
    )


def safe_wilcoxon(
    calibrated_values: np.ndarray,
    standard_values: np.ndarray,
) -> tuple[
    float,
    float,
]:

    try:

        result = wilcoxon(
            calibrated_values,
            standard_values,
            alternative="two-sided",
        )

        return (
            float(
                result.statistic
            ),
            float(
                result.pvalue
            ),
        )

    except ValueError:

        return (
            np.nan,
            np.nan,
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
        "--split-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "repeated_split_78d_ridge_svr/"
            "splits"
        ),
    )

    parser.add_argument(
        "--standard-metrics",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "repeated_split_78d_ridge_svr/"
            "repeated_split_seed_metrics.csv"
        ),
    )

    parser.add_argument(
        "--standard-prediction-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "repeated_split_78d_ridge_svr/"
            "predictions"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "repeated_split_asymmetric_calibration_78d"
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_root = (
        args.output_root
        / "predictions"
    )

    prediction_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        x_full_raw,
        y_full,
        loading_metadata,
    ) = load_full_raw_dataset(
        args.input
    )

    if (
        x_full_raw.shape
        != (
            EXPECTED_CASE_COUNT,
            EXPECTED_FEATURE_COUNT,
        )
    ):
        raise RuntimeError(
            "Unexpected complete raw "
            "descriptor shape."
        )

    if (
        y_full.shape
        != (
            EXPECTED_CASE_COUNT,
            4,
        )
    ):
        raise RuntimeError(
            "Unexpected target shape."
        )

    if not (
        args.standard_metrics
        .is_file()
    ):
        raise FileNotFoundError(
            args.standard_metrics
        )

    standard_all = pd.read_csv(
        args.standard_metrics
    )

    standard_frame = (
        standard_all.loc[
            standard_all[
                "model"
            ]
            == "rbf_svr"
        ]
        .sort_values(
            "seed"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(standard_frame)
        != len(SEEDS)
    ):
        raise RuntimeError(
            "Expected ten frozen "
            "standard-SVR seed rows."
        )

    if (
        standard_frame[
            "seed"
        ]
        .tolist()
        != SEEDS
    ):
        raise RuntimeError(
            "Frozen standard SVR seeds "
            "must equal 0..9."
        )

    print(
        "===== 10-SEED STANDARD VS "
        "ONE-SIDED CALIBRATED 78D SVR ====="
    )

    print(
        "Raw reconstruction mode:",
        loading_metadata[
            "reconstruction_mode"
        ],
    )

    print(
        "OOF folds:",
        N_OOF_FOLDS,
    )

    print(
        "OOF random state:",
        OOF_RANDOM_STATE,
    )

    print(
        "Alpha candidates:",
        ALPHA_VALUES,
    )

    print(
        "Exact previous split files reused:",
        True,
    )

    print(
        "Frozen standard SVR "
        "hyperparameters reused:",
        True,
    )

    print()

    selection_rows = []
    candidate_rows = []
    oof_rows = []
    correction_curve_rows = []
    paired_test_rows = []

    for seed in SEEDS:

        print(
            "================================"
        )

        print(
            f"SEED {seed}"
        )

        print(
            "================================"
        )

        (
            train_idx,
            val_idx,
            test_idx,
        ) = load_split(
            args.split_root,
            seed,
        )

        x_train_raw = (
            x_full_raw[
                train_idx
            ]
        )

        x_val_raw = (
            x_full_raw[
                val_idx
            ]
        )

        y_train = (
            y_full[
                train_idx
            ]
        )

        y_val = (
            y_full[
                val_idx
            ]
        )

        y_test = (
            y_full[
                test_idx
            ]
        )

        standard_row = (
            standard_frame.loc[
                standard_frame[
                    "seed"
                ]
                == seed
            ]
            .iloc[0]
        )

        C = float(
            standard_row[
                "selected_C"
            ]
        )

        gamma = float(
            standard_row[
                "selected_gamma"
            ]
        )

        epsilon = float(
            standard_row[
                "selected_epsilon"
            ]
        )

        print(
            "  Frozen base:",
            (
                f"C={C:g}, "
                f"gamma={gamma:g}, "
                f"epsilon={epsilon:g}"
            ),
        )

        # ---------------------------------------
        # Full outer-training scaler.
        # ---------------------------------------

        scaler = StandardScaler()

        x_train = (
            scaler.fit_transform(
                x_train_raw
            )
        )

        x_val = (
            scaler.transform(
                x_val_raw
            )
        )

        # ---------------------------------------
        # Standard validation prediction.
        # No test access yet.
        # ---------------------------------------

        standard_val_prediction = (
            fit_multioutput_svr(
                x_train=x_train,
                y_train=y_train,
                x_predict=x_val,
                C=C,
                gamma=gamma,
                epsilon=epsilon,
                n_jobs=args.n_jobs,
            )
        )

        standard_val_metrics = (
            calculate_metrics(
                y_val,
                standard_val_prediction,
            )
        )

        stored_val_mse = float(
            standard_row[
                "val_mse"
            ]
        )

        if not np.isclose(
            standard_val_metrics[
                "mse"
            ],
            stored_val_mse,
            rtol=0.0,
            atol=5e-10,
        ):
            raise RuntimeError(
                f"STOP seed {seed}: "
                "standard validation MSE "
                "was not reproduced.\n"
                f"actual="
                f"{standard_val_metrics['mse']:.12f}\n"
                f"stored="
                f"{stored_val_mse:.12f}"
            )

        print(
            "  Standard validation MSE "
            "reproduced:",
            f"{standard_val_metrics['mse']:.10f}",
        )

        # ---------------------------------------
        # Training-only cross-fitted prediction.
        # ---------------------------------------

        print(
            "  Generating OOF predictions ..."
        )

        oof_prediction = (
            generate_oof_predictions(
                x_train_raw=x_train_raw,
                y_train=y_train,
                C=C,
                gamma=gamma,
                epsilon=epsilon,
                n_jobs=args.n_jobs,
            )
        )

        # ---------------------------------------
        # OOF diagnostics.
        # ---------------------------------------

        for storey in range(4):

            truth = (
                y_train[
                    :,
                    storey,
                ]
            )

            prediction = (
                oof_prediction[
                    :,
                    storey,
                ]
            )

            residual = (
                truth - prediction
            )

            high = (
                truth
                > HIGH_DAMAGE_THRESHOLD
            )

            oof_rows.append(
                {
                    "seed": seed,
                    "storey": (
                        storey + 1
                    ),
                    "oof_mae": float(
                        np.mean(
                            np.abs(
                                residual
                            )
                        )
                    ),
                    "oof_mean_residual": float(
                        np.mean(
                            residual
                        )
                    ),
                    "oof_high_mean_residual": float(
                        np.mean(
                            residual[
                                high
                            ]
                        )
                    ),
                    "oof_high_underestimation_ratio": float(
                        np.mean(
                            prediction[
                                high
                            ]
                            < truth[
                                high
                            ]
                        )
                    ),
                    "n_high_damage": int(
                        np.sum(
                            high
                        )
                    ),
                }
            )

        # ---------------------------------------
        # Training-only correction functions.
        # ---------------------------------------

        correction_models = (
            fit_correction_models(
                y_train=y_train,
                oof_prediction=(
                    oof_prediction
                ),
            )
        )

        for (
            storey,
            model,
        ) in enumerate(
            correction_models,
            start=1,
        ):

            correction_values = np.maximum(
                model.predict(
                    CORRECTION_GRID
                ),
                0.0,
            )

            for (
                base_prediction,
                correction,
            ) in zip(
                CORRECTION_GRID,
                correction_values,
            ):

                correction_curve_rows.append(
                    {
                        "seed": seed,
                        "storey": (
                            storey
                        ),
                        "base_prediction": float(
                            base_prediction
                        ),
                        "learned_upward_correction": float(
                            correction
                        ),
                    }
                )

        val_correction = (
            predict_correction(
                models=(
                    correction_models
                ),
                prediction=(
                    standard_val_prediction
                ),
            )
        )

        # ---------------------------------------
        # Frozen validation constraints.
        # ---------------------------------------

        mse_limit = (
            standard_val_metrics[
                "mse"
            ]
            * OVERALL_MSE_BUDGET
        )

        zero_limit = (
            standard_val_metrics[
                "zero_mae"
            ]
            * ZERO_MAE_BUDGET
        )

        low_limit = (
            standard_val_metrics[
                "low_mae"
            ]
            * LOW_MAE_BUDGET
        )

        high_mae_limit = (
            standard_val_metrics[
                "high_mae"
            ]
            * HIGH_MAE_BUDGET
        )

        high_bias_limit = (
            standard_val_metrics[
                "high_abs_bias"
            ]
            * HIGH_ABS_BIAS_BUDGET
        )

        seed_candidate_rows = []

        for alpha in ALPHA_VALUES:

            calibrated_val_prediction = (
                clip_prediction(
                    standard_val_prediction
                    + alpha
                    * val_correction
                )
            )

            metrics = (
                calculate_metrics(
                    y_val,
                    calibrated_val_prediction,
                )
            )

            eligible = bool(
                metrics["mse"]
                <= mse_limit
                and metrics[
                    "zero_mae"
                ]
                <= zero_limit
                and metrics[
                    "low_mae"
                ]
                <= low_limit
                and metrics[
                    "high_mae"
                ]
                <= high_mae_limit
                and metrics[
                    "high_abs_bias"
                ]
                <= high_bias_limit
            )

            seed_candidate_rows.append(
                {
                    "seed": seed,
                    "alpha": float(
                        alpha
                    ),
                    "val_mse": (
                        metrics["mse"]
                    ),
                    "val_mae": (
                        metrics["mae"]
                    ),
                    "zero_mae": (
                        metrics[
                            "zero_mae"
                        ]
                    ),
                    "low_mae": (
                        metrics[
                            "low_mae"
                        ]
                    ),
                    "medium_mae": (
                        metrics[
                            "medium_mae"
                        ]
                    ),
                    "damaged_mae": (
                        metrics[
                            "damaged_mae"
                        ]
                    ),
                    "high_mae": (
                        metrics[
                            "high_mae"
                        ]
                    ),
                    "high_bias": (
                        metrics[
                            "high_bias"
                        ]
                    ),
                    "high_abs_bias": (
                        metrics[
                            "high_abs_bias"
                        ]
                    ),
                    "high_underestimation_ratio": (
                        metrics[
                            "high_underestimation_ratio"
                        ]
                    ),
                    "eligible": (
                        eligible
                    ),
                    "selected": False,
                }
            )

        seed_candidates = pd.DataFrame(
            seed_candidate_rows
        )

        eligible_frame = (
            seed_candidates.loc[
                seed_candidates[
                    "eligible"
                ]
            ]
            .sort_values(
                [
                    "high_underestimation_ratio",
                    "high_mae",
                    "high_abs_bias",
                    "val_mse",
                    "alpha",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        eligible_count = int(
            len(
                eligible_frame
            )
        )

        if eligible_frame.empty:

            selection_mode = (
                "fallback_standard"
            )

            selected_alpha = 0.0

            selected_val_metrics = (
                standard_val_metrics
            )

        else:

            selection_mode = (
                "calibrated"
            )

            selected = (
                eligible_frame.iloc[
                    0
                ]
            )

            selected_alpha = float(
                selected[
                    "alpha"
                ]
            )

            selected_val_metrics = {
                "mse": float(
                    selected[
                        "val_mse"
                    ]
                ),
                "mae": float(
                    selected[
                        "val_mae"
                    ]
                ),
                "zero_mae": float(
                    selected[
                        "zero_mae"
                    ]
                ),
                "low_mae": float(
                    selected[
                        "low_mae"
                    ]
                ),
                "medium_mae": float(
                    selected[
                        "medium_mae"
                    ]
                ),
                "damaged_mae": float(
                    selected[
                        "damaged_mae"
                    ]
                ),
                "high_mae": float(
                    selected[
                        "high_mae"
                    ]
                ),
                "high_bias": float(
                    selected[
                        "high_bias"
                    ]
                ),
                "high_abs_bias": float(
                    selected[
                        "high_abs_bias"
                    ]
                ),
                "high_underestimation_ratio": float(
                    selected[
                        "high_underestimation_ratio"
                    ]
                ),
            }

            seed_candidates.loc[
                np.isclose(
                    seed_candidates[
                        "alpha"
                    ],
                    selected_alpha,
                ),
                "selected",
            ] = True

        candidate_rows.extend(
            seed_candidates.to_dict(
                orient="records"
            )
        )

        selection_rows.append(
            {
                "seed": seed,
                "selection_mode": (
                    selection_mode
                ),
                "eligible_alpha_count": (
                    eligible_count
                ),
                "selected_alpha": (
                    selected_alpha
                ),
                "base_C": C,
                "base_gamma": gamma,
                "base_epsilon": (
                    epsilon
                ),
                "baseline_val_mse": (
                    standard_val_metrics[
                        "mse"
                    ]
                ),
                "selected_val_mse": (
                    selected_val_metrics[
                        "mse"
                    ]
                ),
                "baseline_val_low_mae": (
                    standard_val_metrics[
                        "low_mae"
                    ]
                ),
                "selected_val_low_mae": (
                    selected_val_metrics[
                        "low_mae"
                    ]
                ),
                "baseline_val_high_mae": (
                    standard_val_metrics[
                        "high_mae"
                    ]
                ),
                "selected_val_high_mae": (
                    selected_val_metrics[
                        "high_mae"
                    ]
                ),
                "baseline_val_high_abs_bias": (
                    standard_val_metrics[
                        "high_abs_bias"
                    ]
                ),
                "selected_val_high_abs_bias": (
                    selected_val_metrics[
                        "high_abs_bias"
                    ]
                ),
                "baseline_val_high_underestimation_ratio": (
                    standard_val_metrics[
                        "high_underestimation_ratio"
                    ]
                ),
                "selected_val_high_underestimation_ratio": (
                    selected_val_metrics[
                        "high_underestimation_ratio"
                    ]
                ),
            }
        )

        print(
            "  Eligible alphas:",
            eligible_count,
        )

        print(
            "  Selection mode:",
            selection_mode,
        )

        print(
            "  Selected alpha:",
            f"{selected_alpha:.2f}",
        )

        print(
            "  Validation high underestimation:",
            (
                f"{standard_val_metrics[
                    'high_underestimation_ratio'
                ]:.6f}"
                " -> "
                f"{selected_val_metrics[
                    'high_underestimation_ratio'
                ]:.6f}"
            ),
        )

        # ---------------------------------------
        # TEST ACCESS BEGINS ONLY HERE.
        # Reuse frozen standard test predictions.
        # ---------------------------------------

        standard_prediction_path = (
            args.standard_prediction_root
            / f"seed_{seed}_predictions.npz"
        )

        if not (
            standard_prediction_path
            .is_file()
        ):
            raise FileNotFoundError(
                standard_prediction_path
            )

        with np.load(
            standard_prediction_path,
            allow_pickle=False,
        ) as data:

            frozen_test_idx = np.asarray(
                data[
                    "test_idx"
                ],
                dtype=np.int64,
            )

            frozen_y_test = np.asarray(
                data[
                    "y_test"
                ],
                dtype=np.float64,
            )

            standard_test_prediction = np.asarray(
                data[
                    "svr_prediction"
                ],
                dtype=np.float64,
            )

        if not np.array_equal(
            frozen_test_idx,
            test_idx,
        ):
            raise RuntimeError(
                f"Seed {seed}: standard "
                "prediction test indices "
                "do not match split."
            )

        if not np.array_equal(
            frozen_y_test,
            y_test,
        ):
            raise RuntimeError(
                f"Seed {seed}: frozen "
                "test targets do not match."
            )

        standard_test_metrics = (
            calculate_metrics(
                y_test,
                standard_test_prediction,
            )
        )

        stored_test_mae = float(
            standard_row[
                "test_mae"
            ]
        )

        if not np.isclose(
            standard_test_metrics[
                "mae"
            ],
            stored_test_mae,
            rtol=0.0,
            atol=5e-10,
        ):
            raise RuntimeError(
                f"Seed {seed}: "
                "standard test MAE lock failed."
            )

        if (
            selection_mode
            == "fallback_standard"
        ):

            calibrated_test_prediction = (
                standard_test_prediction.copy()
            )

        else:

            test_correction = (
                predict_correction(
                    models=(
                        correction_models
                    ),
                    prediction=(
                        standard_test_prediction
                    ),
                )
            )

            calibrated_test_prediction = (
                clip_prediction(
                    standard_test_prediction
                    + selected_alpha
                    * test_correction
                )
            )

        calibrated_test_metrics = (
            calculate_metrics(
                y_test,
                calibrated_test_prediction,
            )
        )

        row = {
            "seed": seed,
            "selection_mode": (
                selection_mode
            ),
            "selected_alpha": (
                selected_alpha
            ),
            "eligible_alpha_count": (
                eligible_count
            ),
        }

        metric_mapping = {
            "test_mse": "mse",
            "test_mae": "mae",
            "test_rmse": "rmse",
            "damaged_mae": (
                "damaged_mae"
            ),
            "zero_mae": (
                "zero_mae"
            ),
            "low_mae": (
                "low_mae"
            ),
            "medium_mae": (
                "medium_mae"
            ),
            "high_mae": (
                "high_mae"
            ),
            "high_abs_bias": (
                "high_abs_bias"
            ),
            "high_underestimation_ratio": (
                "high_underestimation_ratio"
            ),
        }

        for (
            output_name,
            metric_name,
        ) in metric_mapping.items():

            standard_value = float(
                standard_test_metrics[
                    metric_name
                ]
            )

            calibrated_value = float(
                calibrated_test_metrics[
                    metric_name
                ]
            )

            row[
                f"standard_{output_name}"
            ] = standard_value

            row[
                f"calibrated_{output_name}"
            ] = calibrated_value

            row[
                f"difference_calibrated_minus_standard_{output_name}"
            ] = (
                calibrated_value
                - standard_value
            )

            if standard_value > 0.0:

                row[
                    f"relative_improvement_percent_{output_name}"
                ] = (
                    (
                        standard_value
                        - calibrated_value
                    )
                    / standard_value
                    * 100.0
                )

        paired_test_rows.append(
            row
        )

        np.savez_compressed(
            prediction_root
            / (
                f"seed_{seed}_"
                "calibrated_predictions.npz"
            ),
            seed=np.asarray(
                seed
            ),
            test_idx=test_idx,
            y_test=y_test,
            standard_prediction=(
                standard_test_prediction
            ),
            calibrated_prediction=(
                calibrated_test_prediction
            ),
            selected_alpha=np.asarray(
                selected_alpha
            ),
        )

        print(
            "  Standard test MAE:",
            f"{standard_test_metrics['mae']:.10f}",
        )

        print(
            "  Calibrated test MAE:",
            f"{calibrated_test_metrics['mae']:.10f}",
        )

        print(
            "  Standard high MAE:",
            f"{standard_test_metrics['high_mae']:.10f}",
        )

        print(
            "  Calibrated high MAE:",
            f"{calibrated_test_metrics['high_mae']:.10f}",
        )

        print(
            "  Standard high underestimation:",
            (
                f"{standard_test_metrics[
                    'high_underestimation_ratio'
                ]:.10f}"
            ),
        )

        print(
            "  Calibrated high underestimation:",
            (
                f"{calibrated_test_metrics[
                    'high_underestimation_ratio'
                ]:.10f}"
            ),
        )

        print()

    # --------------------------------------------------
    # Dataframes
    # --------------------------------------------------

    selection_frame = (
        pd.DataFrame(
            selection_rows
        )
        .sort_values(
            "seed"
        )
        .reset_index(
            drop=True
        )
    )

    candidate_frame = pd.DataFrame(
        candidate_rows
    )

    oof_frame = pd.DataFrame(
        oof_rows
    )

    correction_curve_frame = (
        pd.DataFrame(
            correction_curve_rows
        )
    )

    paired_frame = (
        pd.DataFrame(
            paired_test_rows
        )
        .sort_values(
            "seed"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------
    # Paired statistics
    # --------------------------------------------------

    statistics_rows = []

    for metric in PRIMARY_METRICS:

        standard_values = (
            paired_frame[
                f"standard_{metric}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        calibrated_values = (
            paired_frame[
                f"calibrated_{metric}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        difference = (
            calibrated_values
            - standard_values
        )

        (
            ci_low,
            ci_high,
        ) = confidence_interval(
            difference
        )

        relative_improvement = (
            (
                standard_values
                - calibrated_values
            )
            / standard_values
            * 100.0
        )

        t_result = ttest_rel(
            calibrated_values,
            standard_values,
        )

        (
            wilcoxon_stat,
            wilcoxon_p,
        ) = safe_wilcoxon(
            calibrated_values,
            standard_values,
        )

        statistics_rows.append(
            {
                "metric": metric,
                "difference_definition": (
                    "calibrated_minus_standard"
                ),
                "n_seeds": len(
                    SEEDS
                ),
                "standard_mean": float(
                    np.mean(
                        standard_values
                    )
                ),
                "calibrated_mean": float(
                    np.mean(
                        calibrated_values
                    )
                ),
                "mean_difference": float(
                    np.mean(
                        difference
                    )
                ),
                "std_difference": float(
                    np.std(
                        difference,
                        ddof=1,
                    )
                ),
                "ci95_difference_low": (
                    ci_low
                ),
                "ci95_difference_high": (
                    ci_high
                ),
                "calibrated_wins": int(
                    np.sum(
                        calibrated_values
                        < standard_values
                    )
                ),
                "standard_wins": int(
                    np.sum(
                        standard_values
                        < calibrated_values
                    )
                ),
                "ties": int(
                    np.sum(
                        np.isclose(
                            calibrated_values,
                            standard_values,
                        )
                    )
                ),
                "mean_relative_improvement_percent": float(
                    np.mean(
                        relative_improvement
                    )
                ),
                "paired_t_pvalue": float(
                    t_result.pvalue
                ),
                "wilcoxon_statistic": (
                    wilcoxon_stat
                ),
                "wilcoxon_pvalue": (
                    wilcoxon_p
                ),
            }
        )

    statistics_frame = (
        pd.DataFrame(
            statistics_rows
        )
    )

    # --------------------------------------------------
    # Alpha frequency
    # --------------------------------------------------

    calibrated_selection = (
        selection_frame.loc[
            selection_frame[
                "selection_mode"
            ]
            == "calibrated"
        ]
    )

    if calibrated_selection.empty:

        alpha_frequency = pd.DataFrame(
            columns=[
                "selected_alpha",
                "count",
            ]
        )

    else:

        alpha_frequency = (
            calibrated_selection[
                "selected_alpha"
            ]
            .value_counts()
            .sort_index()
            .rename_axis(
                "selected_alpha"
            )
            .reset_index(
                name="count"
            )
        )

    fallback_count = int(
        np.sum(
            selection_frame[
                "selection_mode"
            ]
            == "fallback_standard"
        )
    )

    # --------------------------------------------------
    # OOF aggregate diagnostics
    # --------------------------------------------------

    oof_aggregate = (
        oof_frame.groupby(
            "storey",
            as_index=False,
        )
        .agg(
            oof_mae_mean=(
                "oof_mae",
                "mean",
            ),
            oof_high_mean_residual_mean=(
                "oof_high_mean_residual",
                "mean",
            ),
            oof_high_underestimation_mean=(
                "oof_high_underestimation_ratio",
                "mean",
            ),
        )
    )

    # --------------------------------------------------
    # Predeclared method-level decision
    # --------------------------------------------------

    def stat_row(
        metric: str,
    ) -> pd.Series:

        return (
            statistics_frame.loc[
                statistics_frame[
                    "metric"
                ]
                == metric
            ]
            .iloc[0]
        )

    overall = stat_row(
        "test_mae"
    )

    low = stat_row(
        "low_mae"
    )

    high_mae = stat_row(
        "high_mae"
    )

    high_bias = stat_row(
        "high_abs_bias"
    )

    high_under = stat_row(
        "high_underestimation_ratio"
    )

    overall_test_mae_change_percent = (
        (
            float(
                overall[
                    "calibrated_mean"
                ]
            )
            - float(
                overall[
                    "standard_mean"
                ]
            )
        )
        / float(
            overall[
                "standard_mean"
            ]
        )
        * 100.0
    )

    low_mae_change_percent = (
        (
            float(
                low[
                    "calibrated_mean"
                ]
            )
            - float(
                low[
                    "standard_mean"
                ]
            )
        )
        / float(
            low[
                "standard_mean"
            ]
        )
        * 100.0
    )

    high_mae_improvement_percent = (
        (
            float(
                high_mae[
                    "standard_mean"
                ]
            )
            - float(
                high_mae[
                    "calibrated_mean"
                ]
            )
        )
        / float(
            high_mae[
                "standard_mean"
            ]
        )
        * 100.0
    )

    high_bias_improvement_percent = (
        (
            float(
                high_bias[
                    "standard_mean"
                ]
            )
            - float(
                high_bias[
                    "calibrated_mean"
                ]
            )
        )
        / float(
            high_bias[
                "standard_mean"
            ]
        )
        * 100.0
    )

    high_under_change_points = (
        float(
            high_under[
                "mean_difference"
            ]
        )
        * 100.0
    )

    high_mae_wins = int(
        high_mae[
            "calibrated_wins"
        ]
    )

    high_bias_wins = int(
        high_bias[
            "calibrated_wins"
        ]
    )

    high_under_wins = int(
        high_under[
            "calibrated_wins"
        ]
    )

    # Formal-extension criteria were declared
    # before running this repeated-split experiment.
    if (
        high_under_change_points
        <= -7.5
        and high_under_wins >= 8
        and high_bias_improvement_percent
        >= 20.0
        and high_bias_wins >= 8
        and high_mae_improvement_percent
        >= 7.5
        and high_mae_wins >= 8
        and overall_test_mae_change_percent
        <= 5.0
        and low_mae_change_percent
        <= 10.0
        and fallback_count <= 2
    ):

        method_decision = (
            "ASYMMETRIC_CALIBRATION_ROBUST_"
            "FORMAL_EXTENSION"
        )

    elif (
        high_under_change_points
        <= -5.0
        and high_under_wins >= 7
        and high_bias_improvement_percent
        >= 15.0
        and high_bias_wins >= 7
        and high_mae_improvement_percent
        >= 5.0
        and high_mae_wins >= 7
        and overall_test_mae_change_percent
        <= 5.0
        and low_mae_change_percent
        <= 10.0
        and fallback_count <= 3
    ):

        method_decision = (
            "ASYMMETRIC_CALIBRATION_ROBUST_"
            "DIRECTIONAL_EFFECT_BELOW_"
            "FORMAL_THRESHOLD"
        )

    else:

        method_decision = (
            "ASYMMETRIC_CALIBRATION_"
            "NOT_ROBUST_ENOUGH"
        )

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    selection_path = (
        args.output_root
        / "calibration_selected_by_seed.csv"
    )

    candidate_path = (
        args.output_root
        / "calibration_validation_candidates.csv"
    )

    oof_path = (
        args.output_root
        / "calibration_oof_residuals.csv"
    )

    oof_aggregate_path = (
        args.output_root
        / "calibration_oof_residual_summary.csv"
    )

    curve_path = (
        args.output_root
        / "calibration_correction_curves.csv"
    )

    paired_path = (
        args.output_root
        / "calibration_paired_test_results.csv"
    )

    statistics_path = (
        args.output_root
        / "calibration_paired_statistics.csv"
    )

    alpha_path = (
        args.output_root
        / "calibration_alpha_frequency.csv"
    )

    report_path = (
        args.output_root
        / "calibration_repeated_split_report.json"
    )

    selection_frame.to_csv(
        selection_path,
        index=False,
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    oof_frame.to_csv(
        oof_path,
        index=False,
    )

    oof_aggregate.to_csv(
        oof_aggregate_path,
        index=False,
    )

    correction_curve_frame.to_csv(
        curve_path,
        index=False,
    )

    paired_frame.to_csv(
        paired_path,
        index=False,
    )

    statistics_frame.to_csv(
        statistics_path,
        index=False,
    )

    alpha_frequency.to_csv(
        alpha_path,
        index=False,
    )

    report = {
        "experiment": (
            "10_seed_standard_vs_"
            "one_sided_residual_calibrated_"
            "78D_RBF_SVR"
        ),
        "descriptor_set": (
            "signal_derived_with_ground"
        ),
        "seeds": SEEDS,
        "exact_outer_splits_reused": (
            True
        ),
        "base_hyperparameters": (
            "reused from each seed's frozen "
            "standard-SVR validation selection"
        ),
        "oof": {
            "fold_count": (
                N_OOF_FOLDS
            ),
            "shuffle": True,
            "random_state": (
                OOF_RANDOM_STATE
            ),
            "per_fold_scaling": (
                "fold-training only"
            ),
        },
        "calibration": {
            "type": (
                "storey-specific non-negative "
                "monotone isotonic residual correction"
            ),
            "residual": (
                "truth_minus_OOF_prediction"
            ),
            "direction": (
                "upward_only"
            ),
            "alpha_candidates": (
                ALPHA_VALUES
            ),
        },
        "validation_constraints": {
            "overall_mse_multiplier": (
                OVERALL_MSE_BUDGET
            ),
            "zero_mae_multiplier": (
                ZERO_MAE_BUDGET
            ),
            "low_mae_multiplier": (
                LOW_MAE_BUDGET
            ),
            "high_mae_multiplier": (
                HIGH_MAE_BUDGET
            ),
            "high_abs_bias_multiplier": (
                HIGH_ABS_BIAS_BUDGET
            ),
        },
        "selection_rule": (
            "minimum high-damage underestimation; "
            "then high MAE; then |high bias|; "
            "then validation MSE"
        ),
        "fallback_rule": (
            "use frozen standard SVR if "
            "no alpha is validation-eligible"
        ),
        "fallback_count": (
            fallback_count
        ),
        "formal_success_criteria": {
            "high_underestimation_change_pp_max": (
                -7.5
            ),
            "high_underestimation_wins_min": (
                8
            ),
            "high_abs_bias_improvement_percent_min": (
                20.0
            ),
            "high_abs_bias_wins_min": (
                8
            ),
            "high_mae_improvement_percent_min": (
                7.5
            ),
            "high_mae_wins_min": (
                8
            ),
            "overall_test_mae_change_percent_max": (
                5.0
            ),
            "low_mae_change_percent_max": (
                10.0
            ),
            "fallback_count_max": (
                2
            ),
        },
        "method_decision": (
            method_decision
        ),
        "overall_test_mae_change_percent": float(
            overall_test_mae_change_percent
        ),
        "low_mae_change_percent": float(
            low_mae_change_percent
        ),
        "high_damage_mae_improvement_percent": float(
            high_mae_improvement_percent
        ),
        "high_damage_abs_bias_improvement_percent": float(
            high_bias_improvement_percent
        ),
        "high_damage_underestimation_change_percentage_points": float(
            high_under_change_points
        ),
        "high_mae_wins": (
            high_mae_wins
        ),
        "high_abs_bias_wins": (
            high_bias_wins
        ),
        "high_underestimation_wins": (
            high_under_wins
        ),
        "statistical_note": (
            "The ten repeated holdout partitions "
            "reuse the same 3000 simulated cases. "
            "Effect sizes, paired direction and "
            "confidence intervals are primary; "
            "p-values are supportive."
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

    # --------------------------------------------------
    # Console
    # --------------------------------------------------

    print()
    print(
        "===== SEED-BY-SEED CALIBRATION SELECTION ====="
    )

    print(
        selection_frame[
            [
                "seed",
                "selection_mode",
                "eligible_alpha_count",
                "selected_alpha",
                "base_C",
                "base_gamma",
                "base_epsilon",
                "baseline_val_mse",
                "selected_val_mse",
                "baseline_val_low_mae",
                "selected_val_low_mae",
                "baseline_val_high_mae",
                "selected_val_high_mae",
                "baseline_val_high_abs_bias",
                "selected_val_high_abs_bias",
                "baseline_val_high_underestimation_ratio",
                "selected_val_high_underestimation_ratio",
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
        "===== SEED-BY-SEED TEST RESULTS ====="
    )

    print(
        paired_frame[
            [
                "seed",
                "selection_mode",
                "selected_alpha",
                "standard_test_mae",
                "calibrated_test_mae",
                "standard_low_mae",
                "calibrated_low_mae",
                "standard_high_mae",
                "calibrated_high_mae",
                "standard_high_abs_bias",
                "calibrated_high_abs_bias",
                "standard_high_underestimation_ratio",
                "calibrated_high_underestimation_ratio",
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
        "===== PAIRED CALIBRATION STATISTICS ====="
    )

    print(
        statistics_frame[
            [
                "metric",
                "standard_mean",
                "calibrated_mean",
                "mean_difference",
                "ci95_difference_low",
                "ci95_difference_high",
                "calibrated_wins",
                "standard_wins",
                "ties",
                "mean_relative_improvement_percent",
                "paired_t_pvalue",
                "wilcoxon_pvalue",
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
        "===== ALPHA FREQUENCY ====="
    )

    if alpha_frequency.empty:

        print(
            "No calibrated alpha selected."
        )

    else:

        print(
            alpha_frequency.to_string(
                index=False
            )
        )

    print()
    print(
        "===== OOF HIGH-DAMAGE RESIDUAL SUMMARY ====="
    )

    print(
        oof_aggregate.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== METHOD-LEVEL DECISION ====="
    )

    print(
        "Fallback count:",
        fallback_count,
    )

    print(
        "Overall test MAE change (%):",
        f"{overall_test_mae_change_percent:.6f}",
    )

    print(
        "Low-damage MAE change (%):",
        f"{low_mae_change_percent:.6f}",
    )

    print(
        "High-damage MAE improvement (%):",
        f"{high_mae_improvement_percent:.6f}",
    )

    print(
        "High-damage |bias| improvement (%):",
        f"{high_bias_improvement_percent:.6f}",
    )

    print(
        "High-damage underestimation "
        "change (percentage points):",
        f"{high_under_change_points:.6f}",
    )

    print(
        "High-MAE calibrated wins:",
        high_mae_wins,
        "/ 10",
    )

    print(
        "High-|bias| calibrated wins:",
        high_bias_wins,
        "/ 10",
    )

    print(
        "Underestimation calibrated wins:",
        high_under_wins,
        "/ 10",
    )

    print(
        "Decision:",
        method_decision,
    )

    print()
    print(
        "===== INTEGRITY CHECKS ====="
    )

    print(
        "Case count:",
        x_full_raw.shape[0],
    )

    print(
        "Feature count:",
        x_full_raw.shape[1],
    )

    print(
        "Seeds completed:",
        len(
            selection_frame
        ),
    )

    print(
        "Exact previous outer splits reused:",
        True,
    )

    print(
        "Frozen standard hyperparameters reused:",
        True,
    )

    print(
        "Training-only OOF calibration:",
        True,
    )

    print(
        "OOF folds:",
        N_OOF_FOLDS,
    )

    print(
        "Per-OOF-fold training-only scaling:",
        True,
    )

    print(
        "Correction non-negative:",
        True,
    )

    print(
        "Correction monotone:",
        True,
    )

    print(
        "Alpha candidates frozen:",
        True,
    )

    print(
        "Validation constraints frozen:",
        True,
    )

    print(
        "Test accessed only after "
        "alpha selection:",
        True,
    )

    print()
    print(
        "CHECK PASSED: 10-seed "
        "one-sided calibration validation "
        "completed."
    )

    print(
        "Selection:",
        selection_path,
    )

    print(
        "Paired statistics:",
        statistics_path,
    )

    print(
        "Correction curves:",
        curve_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
