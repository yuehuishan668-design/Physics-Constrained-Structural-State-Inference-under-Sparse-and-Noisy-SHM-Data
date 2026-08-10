"""
Validation-only pilot for one-sided monotone residual calibration.

Base estimator
--------------
78D signal-derived-with-ground RBF-SVR

Frozen base configuration
-------------------------
C       = 1000
gamma   = 0.00075
epsilon = 0.03

The base model is the previously locked fixed-split validation optimum.

Scientific question
-------------------
Can a training-derived, prediction-dependent, upward-only residual
correction reduce the residual high-damage underestimation tendency
without materially degrading overall / zero / low-damage accuracy?

Calibration design
------------------
1. Generate 5-fold out-of-fold predictions on TRAINING DATA ONLY.
2. For each storey independently:
       residual = true - OOF_prediction
3. Fit a monotone non-negative isotonic mapping:
       correction = g_storey(prediction)
4. Apply to validation predictions:
       corrected = clip(
           base_prediction
           + alpha * correction,
           0,
           0.5
       )

The correction curve is:
- learned from training OOF residuals only;
- constrained to be non-negative;
- constrained to increase monotonically with predicted severity.

Only alpha is selected using validation.

IMPORTANT
---------
F_test and y_test are never accessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


EXPECTED_FEATURE_COUNT = 78

BASE_C = 1000.0
BASE_GAMMA = 0.00075
BASE_EPSILON = 0.03

BASELINE_VAL_MSE_ANCHOR = (
    0.002256262230781473
)

ANCHOR_TOLERANCE = 5e-10

N_OOF_FOLDS = 5
OOF_RANDOM_STATE = 20260809

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

# Prevent a trivial reduction of underestimation
# by simply overpredicting severe damage.
HIGH_MAE_BUDGET = 1.00
HIGH_ABS_BIAS_BUDGET = 1.00

HIGH_DAMAGE_THRESHOLD = 0.20


def _find_vector(
    data: Any,
    exact_names: list[str],
    keyword: str,
) -> tuple[str, np.ndarray] | None:

    for key in exact_names:

        if key not in data.files:
            continue

        array = np.asarray(
            data[key]
        )

        if (
            array.ndim == 1
            and array.shape[0]
            == EXPECTED_FEATURE_COUNT
            and np.issubdtype(
                array.dtype,
                np.number,
            )
        ):
            return (
                key,
                np.asarray(
                    array,
                    dtype=np.float64,
                ),
            )

    possible = []

    for key in data.files:

        if keyword not in key.lower():
            continue

        try:
            array = np.asarray(
                data[key]
            )
        except Exception:
            continue

        if (
            array.ndim == 1
            and array.shape[0]
            == EXPECTED_FEATURE_COUNT
            and np.issubdtype(
                array.dtype,
                np.number,
            )
        ):
            possible.append(
                (
                    key,
                    np.asarray(
                        array,
                        dtype=np.float64,
                    ),
                )
            )

    if len(possible) == 1:
        return possible[0]

    return None


def load_train_validation_only(
    path: Path,
) -> dict[str, Any]:
    """
    Load standardised train/validation arrays and recover raw
    train/validation descriptors without accessing test arrays.
    """

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    required = {
        "F_train",
        "F_val",
        "y_train",
        "y_val",
    }

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        missing = sorted(
            required
            - set(data.files)
        )

        if missing:
            raise RuntimeError(
                f"Missing arrays: {missing}"
            )

        F_train = np.asarray(
            data["F_train"],
            dtype=np.float64,
        )

        F_val = np.asarray(
            data["F_val"],
            dtype=np.float64,
        )

        y_train = np.asarray(
            data["y_train"],
            dtype=np.float64,
        )

        y_val = np.asarray(
            data["y_val"],
            dtype=np.float64,
        )

        raw_keys = {
            "F_train_raw",
            "F_val_raw",
        }

        metadata: dict[str, Any] = {}

        if raw_keys.issubset(
            set(data.files)
        ):

            F_train_raw = np.asarray(
                data["F_train_raw"],
                dtype=np.float64,
            )

            F_val_raw = np.asarray(
                data["F_val_raw"],
                dtype=np.float64,
            )

            metadata[
                "raw_reconstruction_mode"
            ] = "explicit_raw_arrays"

        else:

            mean_result = _find_vector(
                data,
                exact_names=[
                    "feature_mean",
                    "feature_means",
                    "F_mean",
                    "train_feature_mean",
                    "mean",
                ],
                keyword="mean",
            )

            std_result = _find_vector(
                data,
                exact_names=[
                    "feature_std",
                    "feature_stds",
                    "F_std",
                    "train_feature_std",
                    "std",
                ],
                keyword="std",
            )

            if (
                mean_result is None
                or std_result is None
            ):
                raise RuntimeError(
                    "Could not recover raw training/"
                    "validation descriptors without "
                    "accessing test arrays.\n"
                    f"Available keys: {sorted(data.files)}"
                )

            (
                mean_key,
                feature_mean,
            ) = mean_result

            (
                std_key,
                feature_std,
            ) = std_result

            F_train_raw = (
                F_train
                * feature_std
                + feature_mean
            )

            F_val_raw = (
                F_val
                * feature_std
                + feature_mean
            )

            metadata[
                "raw_reconstruction_mode"
            ] = (
                "inverse_standardisation"
            )

            metadata[
                "mean_key"
            ] = mean_key

            metadata[
                "std_key"
            ] = std_key

    arrays = {
        "F_train": F_train,
        "F_val": F_val,
        "F_train_raw": F_train_raw,
        "F_val_raw": F_val_raw,
        "y_train": y_train,
        "y_val": y_val,
        "metadata": metadata,
    }

    for key in [
        "F_train",
        "F_val",
        "F_train_raw",
        "F_val_raw",
        "y_train",
        "y_val",
    ]:

        if not np.all(
            np.isfinite(
                arrays[key]
            )
        ):
            raise RuntimeError(
                f"{key} contains non-finite values."
            )

    if (
        F_train.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            f"Expected 78 features, got "
            f"{F_train.shape[1]}."
        )

    if (
        y_train.ndim != 2
        or y_train.shape[1] != 4
    ):
        raise RuntimeError(
            "Expected four-storey target matrix."
        )

    return arrays


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


def build_svr() -> SVR:

    return SVR(
        kernel="rbf",
        C=BASE_C,
        gamma=BASE_GAMMA,
        epsilon=BASE_EPSILON,
        tol=1e-4,
        cache_size=1024,
    )


def fit_base_multioutput(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
) -> np.ndarray:

    predictions = []

    for storey in range(
        y_train.shape[1]
    ):

        model = build_svr()

        model.fit(
            x_train,
            y_train[
                :,
                storey,
            ],
        )

        predictions.append(
            model.predict(
                x_predict
            )
        )

    return clip_prediction(
        np.column_stack(
            predictions
        )
    )


def generate_oof_predictions(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """
    Training-only cross-fitted predictions.

    Each OOF fold fits its own StandardScaler using only the
    corresponding fold-training cases.
    """

    n_cases = (
        x_train_raw.shape[0]
    )

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
                n_cases
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

        fold_prediction = (
            fit_base_multioutput(
                x_train=x_fold_train,
                y_train=y_train[
                    fold_train_idx
                ],
                x_predict=x_fold_holdout,
            )
        )

        oof[
            fold_holdout_idx
        ] = fold_prediction

        print(
            f"OOF fold "
            f"{fold_number}/"
            f"{N_OOF_FOLDS} complete."
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

        x = (
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
            - x
        )

        model = IsotonicRegression(
            increasing=True,
            y_min=0.0,
            out_of_bounds="clip",
        )

        model.fit(
            x,
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

    correction_columns = []

    for storey, model in enumerate(
        models
    ):

        correction = model.predict(
            prediction[
                :,
                storey,
            ]
        )

        correction_columns.append(
            np.maximum(
                correction,
                0.0,
            )
        )

    return np.column_stack(
        correction_columns
    )


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
        prediction
        - truth
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

        if not np.any(
            mask
        ):
            raise RuntimeError(
                f"Empty group: {name}"
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
        "val_mse": float(
            np.mean(
                error ** 2
            )
        ),
        "val_mae": float(
            np.mean(
                np.abs(error)
            )
        ),
        "val_rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
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
        "high_mean_true": float(
            np.mean(
                truth[
                    masks["high"]
                ]
            )
        ),
        "high_mean_prediction": float(
            np.mean(
                prediction[
                    masks["high"]
                ]
            )
        ),
    }


def relative_change_percent(
    value: float,
    baseline: float,
) -> float:

    return float(
        (
            value
            - baseline
        )
        / baseline
        * 100.0
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
            "asymmetric_residual_calibration_validation_pilot"
        ),
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = load_train_validation_only(
        args.input
    )

    F_train = arrays[
        "F_train"
    ]

    F_val = arrays[
        "F_val"
    ]

    F_train_raw = arrays[
        "F_train_raw"
    ]

    y_train = arrays[
        "y_train"
    ]

    y_val = arrays[
        "y_val"
    ]

    print(
        "===== 78D ONE-SIDED "
        "RESIDUAL-CALIBRATION PILOT ====="
    )

    print(
        "Train shape:",
        F_train.shape,
    )

    print(
        "Validation shape:",
        F_val.shape,
    )

    print(
        "Test arrays accessed:",
        False,
    )

    print(
        "Base SVR:",
        (
            f"C={BASE_C:g}, "
            f"gamma={BASE_GAMMA:g}, "
            f"epsilon={BASE_EPSILON:g}"
        ),
    )

    print(
        "OOF folds:",
        N_OOF_FOLDS,
    )

    print(
        "Alpha candidates:",
        ALPHA_VALUES,
    )

    print()

    # --------------------------------------------------
    # Frozen base model on original standardised split
    # --------------------------------------------------

    base_val_prediction = (
        fit_base_multioutput(
            x_train=F_train,
            y_train=y_train,
            x_predict=F_val,
        )
    )

    baseline_metrics = (
        calculate_metrics(
            y_val,
            base_val_prediction,
        )
    )

    anchor_passed = bool(
        np.isclose(
            baseline_metrics[
                "val_mse"
            ],
            BASELINE_VAL_MSE_ANCHOR,
            atol=ANCHOR_TOLERANCE,
            rtol=0.0,
        )
    )

    if not anchor_passed:
        raise RuntimeError(
            "STOP: frozen base SVR "
            "anchor not reproduced.\n"
            f"actual="
            f"{baseline_metrics['val_mse']:.12f}\n"
            f"expected="
            f"{BASELINE_VAL_MSE_ANCHOR:.12f}"
        )

    # --------------------------------------------------
    # TRAIN-ONLY OOF residual model
    # --------------------------------------------------

    print(
        "Generating training-only "
        "OOF predictions ..."
    )

    oof_prediction = (
        generate_oof_predictions(
            x_train_raw=F_train_raw,
            y_train=y_train,
        )
    )

    oof_residual = (
        y_train
        - oof_prediction
    )

    calibration_models = (
        fit_correction_models(
            y_train=y_train,
            oof_prediction=(
                oof_prediction
            ),
        )
    )

    val_correction = (
        predict_correction(
            models=(
                calibration_models
            ),
            prediction=(
                base_val_prediction
            ),
        )
    )

    # --------------------------------------------------
    # OOF diagnostics
    # --------------------------------------------------

    oof_rows = []

    for storey in range(4):

        truth = (
            y_train[
                :,
                storey,
            ]
        )

        pred = (
            oof_prediction[
                :,
                storey,
            ]
        )

        residual = (
            truth
            - pred
        )

        high = (
            truth
            > HIGH_DAMAGE_THRESHOLD
        )

        oof_rows.append(
            {
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
                        pred[
                            high
                        ]
                        < truth[
                            high
                        ]
                    )
                ),
            }
        )

    oof_frame = pd.DataFrame(
        oof_rows
    )

    # --------------------------------------------------
    # Correction curve diagnostics
    # --------------------------------------------------

    prediction_grid = np.asarray(
        [
            0.00,
            0.025,
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.175,
            0.20,
            0.225,
            0.25,
            0.30,
            0.35,
            0.40,
        ],
        dtype=np.float64,
    )

    correction_rows = []

    for storey, model in enumerate(
        calibration_models,
        start=1,
    ):

        corrections = (
            model.predict(
                prediction_grid
            )
        )

        corrections = np.maximum(
            corrections,
            0.0,
        )

        for (
            predicted_damage,
            correction,
        ) in zip(
            prediction_grid,
            corrections,
        ):

            correction_rows.append(
                {
                    "storey": (
                        storey
                    ),
                    "base_prediction": float(
                        predicted_damage
                    ),
                    "learned_upward_correction": float(
                        correction
                    ),
                }
            )

    correction_frame = (
        pd.DataFrame(
            correction_rows
        )
    )

    # --------------------------------------------------
    # Validation candidate evaluation
    # --------------------------------------------------

    rows = []

    for alpha in ALPHA_VALUES:

        corrected_prediction = (
            clip_prediction(
                base_val_prediction
                + alpha
                * val_correction
            )
        )

        metrics = (
            calculate_metrics(
                y_val,
                corrected_prediction,
            )
        )

        eligible = bool(
            metrics[
                "val_mse"
            ]
            <= (
                baseline_metrics[
                    "val_mse"
                ]
                * OVERALL_MSE_BUDGET
            )
            and metrics[
                "zero_mae"
            ]
            <= (
                baseline_metrics[
                    "zero_mae"
                ]
                * ZERO_MAE_BUDGET
            )
            and metrics[
                "low_mae"
            ]
            <= (
                baseline_metrics[
                    "low_mae"
                ]
                * LOW_MAE_BUDGET
            )
            and metrics[
                "high_mae"
            ]
            <= (
                baseline_metrics[
                    "high_mae"
                ]
                * HIGH_MAE_BUDGET
            )
            and metrics[
                "high_abs_bias"
            ]
            <= (
                baseline_metrics[
                    "high_abs_bias"
                ]
                * HIGH_ABS_BIAS_BUDGET
            )
        )

        rows.append(
            {
                "alpha": float(
                    alpha
                ),
                **metrics,
                "eligible": (
                    eligible
                ),
                "selected": False,
            }
        )

    candidate_frame = (
        pd.DataFrame(
            rows
        )
    )

    eligible_frame = (
        candidate_frame.loc[
            candidate_frame[
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

    if eligible_frame.empty:

        decision = (
            "NO_ACCEPTABLE_ASYMMETRIC_"
            "CALIBRATION"
        )

        selected = None

    else:

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

        candidate_frame.loc[
            np.isclose(
                candidate_frame[
                    "alpha"
                ],
                selected_alpha,
            ),
            "selected",
        ] = True

        decision = (
            "ASYMMETRIC_CALIBRATION_SELECTED"
        )

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    comparison_rows = [
        {
            "configuration": (
                "standard_svr"
            ),
            "alpha": 0.0,
            **baseline_metrics,
        }
    ]

    if selected is not None:

        comparison_rows.append(
            {
                "configuration": (
                    "one_sided_residual_calibration"
                ),
                "alpha": float(
                    selected[
                        "alpha"
                    ]
                ),
                **{
                    key: float(
                        selected[
                            key
                        ]
                    )
                    for key in (
                        baseline_metrics.keys()
                    )
                },
            }
        )

    comparison_frame = (
        pd.DataFrame(
            comparison_rows
        )
    )

    # --------------------------------------------------
    # Decision metrics
    # --------------------------------------------------

    if selected is None:

        under_change_points = (
            np.nan
        )

        high_mae_improvement = (
            np.nan
        )

        high_bias_improvement = (
            np.nan
        )

        overall_mse_change = (
            np.nan
        )

        zero_change = (
            np.nan
        )

        low_change = (
            np.nan
        )

        next_stage = (
            "DO_NOT_PROCEED_TO_REPEATED_SPLIT"
        )

    else:

        under_change_points = (
            (
                float(
                    selected[
                        "high_underestimation_ratio"
                    ]
                )
                - baseline_metrics[
                    "high_underestimation_ratio"
                ]
            )
            * 100.0
        )

        high_mae_improvement = (
            -relative_change_percent(
                float(
                    selected[
                        "high_mae"
                    ]
                ),
                baseline_metrics[
                    "high_mae"
                ],
            )
        )

        high_bias_improvement = (
            -relative_change_percent(
                float(
                    selected[
                        "high_abs_bias"
                    ]
                ),
                baseline_metrics[
                    "high_abs_bias"
                ],
            )
        )

        overall_mse_change = (
            relative_change_percent(
                float(
                    selected[
                        "val_mse"
                    ]
                ),
                baseline_metrics[
                    "val_mse"
                ],
            )
        )

        zero_change = (
            relative_change_percent(
                float(
                    selected[
                        "zero_mae"
                    ]
                ),
                baseline_metrics[
                    "zero_mae"
                ],
            )
        )

        low_change = (
            relative_change_percent(
                float(
                    selected[
                        "low_mae"
                    ]
                ),
                baseline_metrics[
                    "low_mae"
                ],
            )
        )

        if (
            under_change_points
            <= -5.0
            and high_mae_improvement
            >= 0.0
            and high_bias_improvement
            >= 0.0
            and overall_mse_change
            <= 5.0
        ):

            next_stage = (
                "PROCEED_TO_10_SEED_"
                "ASYMMETRIC_CALIBRATION"
            )

        elif (
            under_change_points
            <= -2.0
            and high_mae_improvement
            >= 0.0
            and high_bias_improvement
            >= 0.0
            and overall_mse_change
            <= 5.0
        ):

            next_stage = (
                "MODEST_DIRECTIONAL_EFFECT_"
                "PROCEED_CAUTIOUSLY"
            )

        else:

            next_stage = (
                "ASYMMETRIC_CALIBRATION_EFFECT_WEAK"
            )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    oof_path = (
        args.output_root
        / "oof_residual_summary.csv"
    )

    correction_path = (
        args.output_root
        / "learned_correction_curve.csv"
    )

    candidate_path = (
        args.output_root
        / "asymmetric_calibration_candidates.csv"
    )

    comparison_path = (
        args.output_root
        / "asymmetric_calibration_comparison.csv"
    )

    report_path = (
        args.output_root
        / "asymmetric_calibration_report.json"
    )

    oof_frame.to_csv(
        oof_path,
        index=False,
    )

    correction_frame.to_csv(
        correction_path,
        index=False,
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    comparison_frame.to_csv(
        comparison_path,
        index=False,
    )

    report = {
        "experiment": (
            "78D_one_sided_monotone_"
            "residual_calibration_validation_pilot"
        ),
        "test_arrays_accessed": False,
        "base_model": {
            "C": BASE_C,
            "gamma": BASE_GAMMA,
            "epsilon": (
                BASE_EPSILON
            ),
        },
        "base_anchor_reproduced": (
            anchor_passed
        ),
        "oof": {
            "folds": (
                N_OOF_FOLDS
            ),
            "shuffle": True,
            "random_state": (
                OOF_RANDOM_STATE
            ),
            "per_fold_scaling": (
                "training-fold only"
            ),
        },
        "calibration": {
            "type": (
                "storey-specific "
                "non-negative monotone "
                "isotonic residual correction"
            ),
            "residual_definition": (
                "truth_minus_OOF_prediction"
            ),
            "correction_direction": (
                "upward_only"
            ),
            "alpha_candidates": (
                ALPHA_VALUES
            ),
        },
        "eligibility_constraints": {
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
            "min high-damage underestimation, "
            "then high MAE, |high bias|, "
            "overall validation MSE"
        ),
        "decision": (
            decision
        ),
        "next_stage": (
            next_stage
        ),
        "selected_alpha": (
            None
            if selected is None
            else float(
                selected[
                    "alpha"
                ]
            )
        ),
        "underestimation_change_percentage_points": (
            None
            if np.isnan(
                under_change_points
            )
            else float(
                under_change_points
            )
        ),
        "high_mae_improvement_percent": (
            None
            if np.isnan(
                high_mae_improvement
            )
            else float(
                high_mae_improvement
            )
        ),
        "high_abs_bias_improvement_percent": (
            None
            if np.isnan(
                high_bias_improvement
            )
            else float(
                high_bias_improvement
            )
        ),
        "overall_mse_change_percent": (
            None
            if np.isnan(
                overall_mse_change
            )
            else float(
                overall_mse_change
            )
        ),
        "zero_mae_change_percent": (
            None
            if np.isnan(
                zero_change
            )
            else float(
                zero_change
            )
        ),
        "low_mae_change_percent": (
            None
            if np.isnan(
                low_change
            )
            else float(
                low_change
            )
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
        "===== BASELINE ANCHOR ====="
    )

    print(
        "Anchor reproduced:",
        anchor_passed,
    )

    print(
        "Baseline val MSE:",
        f"{baseline_metrics['val_mse']:.12f}",
    )

    print(
        "Baseline high MAE:",
        f"{baseline_metrics['high_mae']:.10f}",
    )

    print(
        "Baseline high bias:",
        f"{baseline_metrics['high_bias']:.10f}",
    )

    print(
        "Baseline high underestimation:",
        (
            f"{baseline_metrics[
                'high_underestimation_ratio'
            ]:.10f}"
        ),
    )

    print()
    print(
        "===== TRAINING-ONLY OOF "
        "RESIDUAL SUMMARY ====="
    )

    print(
        oof_frame.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== LEARNED CORRECTION "
        "CURVE ====="
    )

    print(
        correction_frame.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== CALIBRATION CANDIDATES ====="
    )

    print(
        candidate_frame[
            [
                "alpha",
                "val_mse",
                "val_mae",
                "zero_mae",
                "low_mae",
                "medium_mae",
                "high_mae",
                "high_bias",
                "high_abs_bias",
                "high_underestimation_ratio",
                "eligible",
                "selected",
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
        "===== SELECTED ASYMMETRIC "
        "COMPARISON ====="
    )

    print(
        comparison_frame.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== ASYMMETRIC EFFECT ====="
    )

    print(
        "Decision:",
        decision,
    )

    print(
        "Next stage:",
        next_stage,
    )

    if selected is not None:

        print(
            "Selected alpha:",
            f"{float(selected['alpha']):.6f}",
        )

        print(
            "High-damage underestimation "
            "change (percentage points):",
            f"{under_change_points:.6f}",
        )

        print(
            "High-damage MAE improvement (%):",
            f"{high_mae_improvement:.6f}",
        )

        print(
            "High-damage |bias| improvement (%):",
            f"{high_bias_improvement:.6f}",
        )

        print(
            "Overall validation MSE change (%):",
            f"{overall_mse_change:.6f}",
        )

        print(
            "Zero MAE change (%):",
            f"{zero_change:.6f}",
        )

        print(
            "Low MAE change (%):",
            f"{low_change:.6f}",
        )

    print()
    print(
        "===== INTEGRITY CHECKS ====="
    )

    print(
        "Test arrays accessed:",
        False,
    )

    print(
        "Baseline anchor reproduced:",
        anchor_passed,
    )

    print(
        "OOF predictions training-only:",
        True,
    )

    print(
        "Per-OOF-fold scaling:",
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
        "Storey-specific correction:",
        True,
    )

    print(
        "Eligibility constraints predeclared:",
        True,
    )

    print()
    print(
        "CHECK PASSED: validation-only "
        "one-sided residual-calibration "
        "pilot completed."
    )

    print(
        "Correction curve:",
        correction_path,
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
