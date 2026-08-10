"""
Controlled matched measurement-noise robustness experiment.

Primary estimator
-----------------
Clean-trained 78D RBF-SVR.

Reliability-oriented extension
------------------------------
The same clean-trained SVR plus a training-only, one-sided monotone
residual calibration.

Experimental protocol
---------------------
1. StandardScaler is fitted ONLY on clean training descriptors.
2. SVR hyperparameters are selected ONLY on clean validation data.
3. One-sided calibration curves are fitted ONLY from clean-training
   out-of-fold predictions.
4. Calibration alpha is selected ONLY on clean validation data.
5. All model components are then frozen.
6. The exact same frozen models are evaluated on matched:
      0%, 5%, 10%, 20%
   structural-response measurement-noise conditions.
7. Non-zero noise levels have five deterministic realizations.
8. No noisy condition is used for fitting, tuning, or adaptation.

Ground/base-input descriptors remain based on the clean ground signal,
consistent with the matched-noise dataset construction protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


CLIP_LOW = 0.0
CLIP_HIGH = 0.5

DAMAGE_ZERO_TOL = 1e-12

SVR_C_GRID = [
    100.0,
    300.0,
    500.0,
    750.0,
    1000.0,
]

SVR_GAMMA_GRID = [
    0.0005,
    0.00075,
    0.001,
    0.003,
    0.01,
]

SVR_EPSILON_GRID = [
    0.02,
    0.03,
    0.04,
]

ALPHA_GRID = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
]

OOF_FOLDS = 5
OOF_RANDOM_STATE = 20260809

EXPECTED_CASES = 3000
EXPECTED_FEATURES = 78
EXPECTED_STORIES = 4


def clip_prediction(
    prediction: np.ndarray,
) -> np.ndarray:

    return np.clip(
        np.asarray(
            prediction,
            dtype=np.float64,
        ),
        CLIP_LOW,
        CLIP_HIGH,
    )


def damage_masks(
    y_true: np.ndarray,
) -> dict[str, np.ndarray]:

    y = np.asarray(
        y_true,
        dtype=np.float64,
    )

    zero = (
        y
        <= DAMAGE_ZERO_TOL
    )

    low = (
        (y > DAMAGE_ZERO_TOL)
        & (y <= 0.10)
    )

    medium = (
        (y > 0.10)
        & (y <= 0.20)
    )

    high = (
        y > 0.20
    )

    damaged = (
        y > DAMAGE_ZERO_TOL
    )

    return {
        "zero": zero,
        "low": low,
        "medium": medium,
        "high": high,
        "damaged": damaged,
    }


def masked_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:

    if not np.any(
        mask
    ):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                y_pred[mask]
                - y_true[mask]
            )
        )
    )


def masked_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:

    if not np.any(
        mask
    ):
        return float("nan")

    return float(
        np.mean(
            y_pred[mask]
            - y_true[mask]
        )
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = clip_prediction(
        y_pred
    )

    error = (
        y_pred
        - y_true
    )

    masks = damage_masks(
        y_true
    )

    high_bias = masked_bias(
        y_true,
        y_pred,
        masks["high"],
    )

    if np.any(
        masks["high"]
    ):

        high_underestimation = float(
            np.mean(
                y_pred[
                    masks["high"]
                ]
                <
                y_true[
                    masks["high"]
                ]
            )
        )

    else:

        high_underestimation = (
            float("nan")
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
        "damaged_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks["damaged"],
            )
        ),
        "zero_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks["zero"],
            )
        ),
        "low_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks["low"],
            )
        ),
        "medium_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks["medium"],
            )
        ),
        "high_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks["high"],
            )
        ),
        "high_bias": (
            high_bias
        ),
        "high_abs_bias": float(
            abs(
                high_bias
            )
        ),
        "high_underestimation": (
            high_underestimation
        ),
        "n_zero": int(
            np.sum(
                masks["zero"]
            )
        ),
        "n_low": int(
            np.sum(
                masks["low"]
            )
        ),
        "n_medium": int(
            np.sum(
                masks["medium"]
            )
        ),
        "n_high": int(
            np.sum(
                masks["high"]
            )
        ),
    }


def fit_story_svrs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    gamma: float,
    epsilon: float,
) -> list[SVR]:

    models: list[SVR] = []

    for story in range(
        EXPECTED_STORIES
    ):

        model = SVR(
            kernel="rbf",
            C=C,
            gamma=gamma,
            epsilon=epsilon,
        )

        model.fit(
            X,
            y[:, story],
        )

        models.append(
            model
        )

    return models


def predict_story_svrs(
    models: list[SVR],
    X: np.ndarray,
) -> np.ndarray:

    columns = []

    for model in models:

        columns.append(
            model.predict(
                X
            )
        )

    prediction = np.column_stack(
        columns
    )

    return clip_prediction(
        prediction
    )


def load_condition(
    path: Path,
) -> dict[str, np.ndarray]:

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        result = {
            key: np.asarray(
                data[key]
            )
            for key
            in data.files
        }

    required = {
        "F_78_raw",
        "feature_names_78",
        "y_damage",
        "case_id",
        "noise_level",
        "replicate",
        "train_idx",
        "val_idx",
        "test_idx",
    }

    missing = sorted(
        required
        - set(
            result.keys()
        )
    )

    if missing:

        raise RuntimeError(
            f"{path} missing keys: "
            f"{missing}"
        )

    return result


def resolve_manifest_file(
    value: str,
    project_root: Path,
) -> Path:

    path = Path(
        value
    )

    if not path.is_absolute():

        path = (
            project_root
            / path
        )

    return path.resolve()


def validate_condition_against_clean(
    condition: dict[str, np.ndarray],
    clean: dict[str, np.ndarray],
    path: Path,
) -> None:

    if (
        condition["F_78_raw"].shape
        != (
            EXPECTED_CASES,
            EXPECTED_FEATURES,
        )
    ):

        raise RuntimeError(
            f"Unexpected 78D shape in {path}: "
            f"{condition['F_78_raw'].shape}"
        )

    exact_keys = [
        "feature_names_78",
        "y_damage",
        "case_id",
        "train_idx",
        "val_idx",
        "test_idx",
    ]

    for key in exact_keys:

        if not np.array_equal(
            condition[key],
            clean[key],
        ):

            raise RuntimeError(
                f"Condition integrity failure "
                f"for key={key}: {path}"
            )


def calibration_prediction(
    base_prediction: np.ndarray,
    calibrators: list[
        IsotonicRegression
    ],
    alpha: float,
) -> np.ndarray:

    correction = np.zeros_like(
        base_prediction,
        dtype=np.float64,
    )

    for story in range(
        EXPECTED_STORIES
    ):

        correction[
            :,
            story,
        ] = calibrators[
            story
        ].predict(
            base_prediction[
                :,
                story,
            ]
        )

    corrected = (
        base_prediction
        + alpha
        * correction
    )

    return clip_prediction(
        corrected
    )


def percentage_change(
    value: float,
    baseline: float,
) -> float:

    if (
        not np.isfinite(
            value
        )
        or not np.isfinite(
            baseline
        )
        or abs(
            baseline
        )
        < 1e-15
    ):

        return float("nan")

    return float(
        (
            value
            - baseline
        )
        / baseline
        * 100.0
    )


def relative_improvement(
    baseline: float,
    improved: float,
) -> float:

    if (
        not np.isfinite(
            baseline
        )
        or not np.isfinite(
            improved
        )
        or abs(
            baseline
        )
        < 1e-15
    ):

        return float("nan")

    return float(
        (
            baseline
            - improved
        )
        / baseline
        * 100.0
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--matched-root",
        type=Path,
        default=Path(
            "data_processed/"
            "sss_fast_revision/"
            "matched_noise_78d"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "matched_noise_robustness_"
            "clean_trained"
        ),
    )

    args = parser.parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    matched_root = (
        args.matched_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        matched_root
        / "matched_noise_manifest.csv"
    )

    if not manifest_path.is_file():

        raise FileNotFoundError(
            manifest_path
        )

    manifest = pd.read_csv(
        manifest_path
    )

    if len(
        manifest
    ) != 16:

        raise RuntimeError(
            "Expected exactly 16 "
            "matched-noise conditions."
        )

    clean_rows = manifest[
        np.isclose(
            manifest["noise_level"],
            0.0,
        )
    ]

    if len(
        clean_rows
    ) != 1:

        raise RuntimeError(
            "Expected one clean condition."
        )

    clean_path = resolve_manifest_file(
        str(
            clean_rows.iloc[0][
                "file"
            ]
        ),
        project_root,
    )

    clean = load_condition(
        clean_path
    )

    F_clean = np.asarray(
        clean[
            "F_78_raw"
        ],
        dtype=np.float64,
    )

    y_all = np.asarray(
        clean[
            "y_damage"
        ],
        dtype=np.float64,
    )

    case_id_all = np.asarray(
        clean[
            "case_id"
        ],
        dtype=np.int64,
    )

    train_idx = np.asarray(
        clean[
            "train_idx"
        ],
        dtype=np.int64,
    )

    val_idx = np.asarray(
        clean[
            "val_idx"
        ],
        dtype=np.int64,
    )

    test_idx = np.asarray(
        clean[
            "test_idx"
        ],
        dtype=np.int64,
    )

    feature_names = [
        str(
            value
        )
        for value
        in clean[
            "feature_names_78"
        ].tolist()
    ]

    if (
        len(
            feature_names
        )
        != EXPECTED_FEATURES
    ):

        raise RuntimeError(
            "Expected 78 feature names."
        )

    if (
        len(
            train_idx
        )
        != 2100
        or len(
            val_idx
        )
        != 450
        or len(
            test_idx
        )
        != 450
    ):

        raise RuntimeError(
            "Unexpected fixed split sizes."
        )

    print(
        "=" * 80
    )

    print(
        "CLEAN-TRAINED MATCHED "
        "NOISE ROBUSTNESS"
    )

    print(
        "=" * 80
    )

    print(
        "Clean descriptor:",
        clean_path,
    )

    print(
        "Train / val / test:",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    print(
        "Features:",
        len(feature_names),
    )

    print(
        "SVR candidates:",
        (
            len(
                SVR_C_GRID
            )
            * len(
                SVR_GAMMA_GRID
            )
            * len(
                SVR_EPSILON_GRID
            )
        ),
    )

    print(
        "Calibration alphas:",
        ALPHA_GRID,
    )

    print()

    # ==================================================
    # Clean split.
    # ==================================================

    X_train_raw = (
        F_clean[
            train_idx
        ]
    )

    X_val_raw = (
        F_clean[
            val_idx
        ]
    )

    X_test_clean_raw = (
        F_clean[
            test_idx
        ]
    )

    y_train = (
        y_all[
            train_idx
        ]
    )

    y_val = (
        y_all[
            val_idx
        ]
    )

    y_test = (
        y_all[
            test_idx
        ]
    )

    test_case_id = (
        case_id_all[
            test_idx
        ]
    )

    # ==================================================
    # Clean-only global scaler.
    # ==================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train_raw
    )

    X_val = scaler.transform(
        X_val_raw
    )

    X_test_clean = scaler.transform(
        X_test_clean_raw
    )

    # ==================================================
    # Clean-validation SVR selection.
    # ==================================================

    print(
        "=" * 80
    )

    print(
        "CLEAN VALIDATION SVR SEARCH"
    )

    print(
        "=" * 80
    )

    grid_rows = []

    best = None

    search_start = (
        time.perf_counter()
    )

    candidate_number = 0

    for C in SVR_C_GRID:

        for gamma in (
            SVR_GAMMA_GRID
        ):

            for epsilon in (
                SVR_EPSILON_GRID
            ):

                candidate_number += 1

                models = fit_story_svrs(
                    X_train,
                    y_train,
                    C=C,
                    gamma=gamma,
                    epsilon=epsilon,
                )

                val_prediction = (
                    predict_story_svrs(
                        models,
                        X_val,
                    )
                )

                metrics = (
                    compute_metrics(
                        y_val,
                        val_prediction,
                    )
                )

                row = {
                    "candidate": (
                        candidate_number
                    ),
                    "C": C,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    **metrics,
                }

                grid_rows.append(
                    row
                )

                key = (
                    metrics["mse"],
                    C,
                    gamma,
                    epsilon,
                )

                if (
                    best is None
                    or key
                    < best["key"]
                ):

                    best = {
                        "key": key,
                        "C": C,
                        "gamma": gamma,
                        "epsilon": (
                            epsilon
                        ),
                        "metrics": (
                            metrics
                        ),
                    }

    search_elapsed = float(
        time.perf_counter()
        - search_start
    )

    if best is None:

        raise RuntimeError(
            "SVR search failed."
        )

    grid_frame = pd.DataFrame(
        grid_rows
    ).sort_values(
        [
            "mse",
            "C",
            "gamma",
            "epsilon",
        ],
        ascending=True,
    )

    grid_path = (
        output_root
        / "clean_svr_grid_results.csv"
    )

    grid_frame.to_csv(
        grid_path,
        index=False,
    )

    selected_C = float(
        best["C"]
    )

    selected_gamma = float(
        best["gamma"]
    )

    selected_epsilon = float(
        best["epsilon"]
    )

    print(
        "Selected C:",
        selected_C,
    )

    print(
        "Selected gamma:",
        selected_gamma,
    )

    print(
        "Selected epsilon:",
        selected_epsilon,
    )

    print(
        "Clean validation MSE:",
        best["metrics"]["mse"],
    )

    print(
        "Clean validation MAE:",
        best["metrics"]["mae"],
    )

    print(
        "Clean validation high MAE:",
        best["metrics"]["high_mae"],
    )

    print(
        "Grid search seconds:",
        f"{search_elapsed:.3f}",
    )

    print()

    # ==================================================
    # Final standard model:
    # fit ONLY on clean training.
    # ==================================================

    standard_models = fit_story_svrs(
        X_train,
        y_train,
        C=selected_C,
        gamma=selected_gamma,
        epsilon=selected_epsilon,
    )

    standard_val_prediction = (
        predict_story_svrs(
            standard_models,
            X_val,
        )
    )

    standard_val_metrics = (
        compute_metrics(
            y_val,
            standard_val_prediction,
        )
    )

    # ==================================================
    # Clean-training OOF predictions.
    # ==================================================

    print(
        "=" * 80
    )

    print(
        "CLEAN TRAINING OOF CALIBRATION"
    )

    print(
        "=" * 80
    )

    kfold = KFold(
        n_splits=OOF_FOLDS,
        shuffle=True,
        random_state=(
            OOF_RANDOM_STATE
        ),
    )

    oof_prediction = np.empty_like(
        y_train,
        dtype=np.float64,
    )

    for fold_number, (
        fold_train,
        fold_valid,
    ) in enumerate(
        kfold.split(
            X_train_raw
        ),
        start=1,
    ):

        fold_scaler = (
            StandardScaler()
        )

        X_fold_train = (
            fold_scaler
            .fit_transform(
                X_train_raw[
                    fold_train
                ]
            )
        )

        X_fold_valid = (
            fold_scaler
            .transform(
                X_train_raw[
                    fold_valid
                ]
            )
        )

        fold_models = (
            fit_story_svrs(
                X_fold_train,
                y_train[
                    fold_train
                ],
                C=selected_C,
                gamma=(
                    selected_gamma
                ),
                epsilon=(
                    selected_epsilon
                ),
            )
        )

        oof_prediction[
            fold_valid
        ] = (
            predict_story_svrs(
                fold_models,
                X_fold_valid,
            )
        )

        print(
            f"OOF fold "
            f"{fold_number}/"
            f"{OOF_FOLDS} complete."
        )

    # ==================================================
    # Fit one-sided isotonic correction curves.
    # ==================================================

    calibrators: list[
        IsotonicRegression
    ] = []

    oof_story_diagnostics = []

    for story in range(
        EXPECTED_STORIES
    ):

        residual = (
            y_train[
                :,
                story
            ]
            - oof_prediction[
                :,
                story
            ]
        )

        calibrator = (
            IsotonicRegression(
                increasing=True,
                y_min=0.0,
                out_of_bounds="clip",
            )
        )

        calibrator.fit(
            oof_prediction[
                :,
                story
            ],
            residual,
        )

        calibrators.append(
            calibrator
        )

        high_mask = (
            y_train[
                :,
                story
            ]
            > 0.20
        )

        if np.any(
            high_mask
        ):

            high_residual_mean = float(
                np.mean(
                    residual[
                        high_mask
                    ]
                )
            )

            high_under = float(
                np.mean(
                    oof_prediction[
                        high_mask,
                        story,
                    ]
                    <
                    y_train[
                        high_mask,
                        story,
                    ]
                )
            )

        else:

            high_residual_mean = (
                float("nan")
            )

            high_under = (
                float("nan")
            )

        oof_story_diagnostics.append(
            {
                "story": (
                    story + 1
                ),
                "high_residual_mean": (
                    high_residual_mean
                ),
                "high_underestimation": (
                    high_under
                ),
            }
        )

    print()
    print(
        "OOF high-damage diagnostics:"
    )

    for row in (
        oof_story_diagnostics
    ):

        print(
            "Story",
            row["story"],
            "high residual mean =",
            row[
                "high_residual_mean"
            ],
            "high underestimation =",
            row[
                "high_underestimation"
            ],
        )

    # ==================================================
    # Clean-validation alpha selection.
    # ==================================================

    print()
    print(
        "=" * 80
    )

    print(
        "CLEAN VALIDATION "
        "CALIBRATION ALPHA SEARCH"
    )

    print(
        "=" * 80
    )

    alpha_rows = []

    eligible_rows = []

    baseline = (
        standard_val_metrics
    )

    for alpha in ALPHA_GRID:

        prediction = (
            calibration_prediction(
                standard_val_prediction,
                calibrators,
                alpha,
            )
        )

        metrics = compute_metrics(
            y_val,
            prediction,
        )

        mse_change = (
            percentage_change(
                metrics["mse"],
                baseline["mse"],
            )
        )

        zero_change = (
            percentage_change(
                metrics["zero_mae"],
                baseline["zero_mae"],
            )
        )

        low_change = (
            percentage_change(
                metrics["low_mae"],
                baseline["low_mae"],
            )
        )

        mse_ok = (
            metrics["mse"]
            <= baseline["mse"]
            * 1.05
        )

        zero_ok = (
            metrics["zero_mae"]
            <= baseline["zero_mae"]
            * 1.10
        )

        low_ok = (
            metrics["low_mae"]
            <= baseline["low_mae"]
            * 1.10
        )

        high_mae_ok = (
            metrics["high_mae"]
            <= baseline["high_mae"]
            + 1e-15
        )

        high_bias_ok = (
            metrics[
                "high_abs_bias"
            ]
            <= baseline[
                "high_abs_bias"
            ]
            + 1e-15
        )

        eligible = bool(
            mse_ok
            and zero_ok
            and low_ok
            and high_mae_ok
            and high_bias_ok
        )

        row = {
            "alpha": alpha,
            **metrics,
            "mse_change_percent": (
                mse_change
            ),
            "zero_mae_change_percent": (
                zero_change
            ),
            "low_mae_change_percent": (
                low_change
            ),
            "constraint_mse": (
                mse_ok
            ),
            "constraint_zero": (
                zero_ok
            ),
            "constraint_low": (
                low_ok
            ),
            "constraint_high_mae": (
                high_mae_ok
            ),
            "constraint_high_abs_bias": (
                high_bias_ok
            ),
            "eligible": (
                eligible
            ),
        }

        alpha_rows.append(
            row
        )

        if eligible:
            eligible_rows.append(
                row
            )

    alpha_frame = (
        pd.DataFrame(
            alpha_rows
        )
    )

    alpha_path = (
        output_root
        / "clean_calibration_alpha_results.csv"
    )

    alpha_frame.to_csv(
        alpha_path,
        index=False,
    )

    fallback_to_standard = False

    if eligible_rows:

        eligible_rows = sorted(
            eligible_rows,
            key=lambda row: (
                row[
                    "high_underestimation"
                ],
                row[
                    "high_mae"
                ],
                row[
                    "high_abs_bias"
                ],
                row[
                    "mse"
                ],
                row[
                    "alpha"
                ],
            ),
        )

        selected_alpha = float(
            eligible_rows[0][
                "alpha"
            ]
        )

    else:

        fallback_to_standard = True
        selected_alpha = 0.0

    print(
        "Eligible alpha count:",
        len(
            eligible_rows
        ),
    )

    print(
        "Selected alpha:",
        selected_alpha,
    )

    print(
        "Fallback to standard:",
        fallback_to_standard,
    )

    if not fallback_to_standard:

        selected_val_prediction = (
            calibration_prediction(
                standard_val_prediction,
                calibrators,
                selected_alpha,
            )
        )

        calibrated_val_metrics = (
            compute_metrics(
                y_val,
                selected_val_prediction,
            )
        )

    else:

        calibrated_val_metrics = (
            standard_val_metrics.copy()
        )

    print()
    print(
        "Validation standard high MAE:",
        standard_val_metrics[
            "high_mae"
        ],
    )

    print(
        "Validation calibrated high MAE:",
        calibrated_val_metrics[
            "high_mae"
        ],
    )

    print(
        "Validation standard high |bias|:",
        standard_val_metrics[
            "high_abs_bias"
        ],
    )

    print(
        "Validation calibrated high |bias|:",
        calibrated_val_metrics[
            "high_abs_bias"
        ],
    )

    print(
        "Validation standard underestimation:",
        standard_val_metrics[
            "high_underestimation"
        ],
    )

    print(
        "Validation calibrated underestimation:",
        calibrated_val_metrics[
            "high_underestimation"
        ],
    )

    # ==================================================
    # Freeze everything.
    # From this point onward: TEST ONLY.
    # ==================================================

    print()
    print(
        "=" * 80
    )

    print(
        "MODEL FREEZE"
    )

    print(
        "=" * 80
    )

    print(
        "Scaler: clean train only"
    )

    print(
        "SVR: clean train only"
    )

    print(
        "SVR hyperparameters: "
        "clean validation selected"
    )

    print(
        "Isotonic curves: "
        "clean-training OOF only"
    )

    print(
        "Calibration alpha: "
        "clean validation selected"
    )

    print(
        "No further fitting or tuning "
        "will occur."
    )

    print()

    # ==================================================
    # Evaluate all matched test conditions.
    # ==================================================

    condition_metric_rows = []

    standard_predictions = []
    calibrated_predictions = []

    condition_noise_levels = []
    condition_replicates = []
    condition_files = []

    ordered_manifest = (
        manifest.sort_values(
            [
                "noise_level",
                "replicate",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    for condition_number, row in (
        ordered_manifest.iterrows()
    ):

        condition_path = (
            resolve_manifest_file(
                str(
                    row["file"]
                ),
                project_root,
            )
        )

        condition = (
            load_condition(
                condition_path
            )
        )

        validate_condition_against_clean(
            condition,
            clean,
            condition_path,
        )

        noise_level = float(
            condition[
                "noise_level"
            ]
        )

        replicate = int(
            condition[
                "replicate"
            ]
        )

        F_condition = np.asarray(
            condition[
                "F_78_raw"
            ],
            dtype=np.float64,
        )

        # SAME CLEAN-TRAIN SCALER.
        X_condition_test = (
            scaler.transform(
                F_condition[
                    test_idx
                ]
            )
        )

        standard_prediction = (
            predict_story_svrs(
                standard_models,
                X_condition_test,
            )
        )

        if fallback_to_standard:

            calibrated_prediction = (
                standard_prediction.copy()
            )

        else:

            calibrated_prediction = (
                calibration_prediction(
                    standard_prediction,
                    calibrators,
                    selected_alpha,
                )
            )

        standard_metrics = (
            compute_metrics(
                y_test,
                standard_prediction,
            )
        )

        calibrated_metrics = (
            compute_metrics(
                y_test,
                calibrated_prediction,
            )
        )

        for (
            method,
            metrics,
        ) in [
            (
                "standard_svr",
                standard_metrics,
            ),
            (
                "calibrated_svr",
                calibrated_metrics,
            ),
        ]:

            condition_metric_rows.append(
                {
                    "condition": (
                        condition_number
                    ),
                    "noise_level": (
                        noise_level
                    ),
                    "noise_percent": (
                        int(
                            round(
                                noise_level
                                * 100
                            )
                        )
                    ),
                    "replicate": (
                        replicate
                    ),
                    "method": (
                        method
                    ),
                    "file": str(
                        condition_path
                    ),
                    **metrics,
                }
            )

        standard_predictions.append(
            standard_prediction
        )

        calibrated_predictions.append(
            calibrated_prediction
        )

        condition_noise_levels.append(
            noise_level
        )

        condition_replicates.append(
            replicate
        )

        condition_files.append(
            str(
                condition_path
            )
        )

        print(
            "Test condition:",
            f"noise={noise_level:.2f}",
            f"rep={replicate}",
            "| standard MAE =",
            f"{standard_metrics['mae']:.6f}",
            "| calibrated MAE =",
            f"{calibrated_metrics['mae']:.6f}",
            "| standard high MAE =",
            f"{standard_metrics['high_mae']:.6f}",
            "| calibrated high MAE =",
            f"{calibrated_metrics['high_mae']:.6f}",
        )

    condition_metrics = (
        pd.DataFrame(
            condition_metric_rows
        )
    )

    condition_metrics_path = (
        output_root
        / "condition_metrics.csv"
    )

    condition_metrics.to_csv(
        condition_metrics_path,
        index=False,
    )

    # ==================================================
    # Noise-level replicate summary.
    # ==================================================

    summary_metrics = [
        "mse",
        "mae",
        "rmse",
        "damaged_mae",
        "zero_mae",
        "low_mae",
        "medium_mae",
        "high_mae",
        "high_abs_bias",
        "high_underestimation",
    ]

    summary_rows = []

    for (
        method,
        noise_level,
    ), group in (
        condition_metrics.groupby(
            [
                "method",
                "noise_level",
            ]
        )
    ):

        row = {
            "method": method,
            "noise_level": float(
                noise_level
            ),
            "noise_percent": int(
                round(
                    float(
                        noise_level
                    )
                    * 100
                )
            ),
            "n_replicates": int(
                len(
                    group
                )
            ),
        }

        for metric in (
            summary_metrics
        ):

            values = np.asarray(
                group[
                    metric
                ],
                dtype=np.float64,
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            row[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
                if len(
                    values
                ) > 1
                else 0.0
            )

        summary_rows.append(
            row
        )

    noise_summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "method",
                "noise_level",
            ]
        )
    )

    noise_summary_path = (
        output_root
        / "noise_level_summary.csv"
    )

    noise_summary.to_csv(
        noise_summary_path,
        index=False,
    )

    # ==================================================
    # Degradation relative to clean test.
    # ==================================================

    degradation_rows = []

    for method in [
        "standard_svr",
        "calibrated_svr",
    ]:

        method_frame = (
            condition_metrics[
                condition_metrics[
                    "method"
                ]
                == method
            ]
        )

        clean_metric_row = (
            method_frame[
                np.isclose(
                    method_frame[
                        "noise_level"
                    ],
                    0.0,
                )
            ]
            .iloc[0]
        )

        for _, row in (
            method_frame.iterrows()
        ):

            result = {
                "method": (
                    method
                ),
                "noise_level": float(
                    row[
                        "noise_level"
                    ]
                ),
                "noise_percent": int(
                    row[
                        "noise_percent"
                    ]
                ),
                "replicate": int(
                    row[
                        "replicate"
                    ]
                ),
            }

            for metric in [
                "mse",
                "mae",
                "rmse",
                "damaged_mae",
                "zero_mae",
                "low_mae",
                "medium_mae",
                "high_mae",
                "high_abs_bias",
            ]:

                result[
                    f"{metric}_change_percent"
                ] = (
                    percentage_change(
                        float(
                            row[
                                metric
                            ]
                        ),
                        float(
                            clean_metric_row[
                                metric
                            ]
                        ),
                    )
                )

            result[
                "high_underestimation_change_pp"
            ] = float(
                (
                    row[
                        "high_underestimation"
                    ]
                    - clean_metric_row[
                        "high_underestimation"
                    ]
                )
                * 100.0
            )

            degradation_rows.append(
                result
            )

    degradation = (
        pd.DataFrame(
            degradation_rows
        )
        .sort_values(
            [
                "method",
                "noise_level",
                "replicate",
            ]
        )
    )

    degradation_path = (
        output_root
        / "degradation_vs_clean.csv"
    )

    degradation.to_csv(
        degradation_path,
        index=False,
    )

    # ==================================================
    # Calibration benefit at each condition.
    # ==================================================

    benefit_rows = []

    condition_keys = (
        ordered_manifest[
            [
                "noise_level",
                "replicate",
            ]
        ]
        .drop_duplicates()
    )

    for _, key_row in (
        condition_keys.iterrows()
    ):

        noise_level = float(
            key_row[
                "noise_level"
            ]
        )

        replicate = int(
            key_row[
                "replicate"
            ]
        )

        subset = condition_metrics[
            np.isclose(
                condition_metrics[
                    "noise_level"
                ],
                noise_level,
            )
            & (
                condition_metrics[
                    "replicate"
                ]
                == replicate
            )
        ]

        standard_row = (
            subset[
                subset[
                    "method"
                ]
                == "standard_svr"
            ]
            .iloc[0]
        )

        calibrated_row = (
            subset[
                subset[
                    "method"
                ]
                == "calibrated_svr"
            ]
            .iloc[0]
        )

        result = {
            "noise_level": (
                noise_level
            ),
            "noise_percent": int(
                round(
                    noise_level
                    * 100
                )
            ),
            "replicate": (
                replicate
            ),
        }

        for metric in [
            "mse",
            "mae",
            "rmse",
            "damaged_mae",
            "zero_mae",
            "low_mae",
            "medium_mae",
            "high_mae",
            "high_abs_bias",
        ]:

            standard_value = float(
                standard_row[
                    metric
                ]
            )

            calibrated_value = float(
                calibrated_row[
                    metric
                ]
            )

            result[
                f"{metric}_cal_minus_standard"
            ] = (
                calibrated_value
                - standard_value
            )

            result[
                f"{metric}_relative_improvement_percent"
            ] = (
                relative_improvement(
                    standard_value,
                    calibrated_value,
                )
            )

        result[
            "high_underestimation_cal_minus_standard_pp"
        ] = float(
            (
                calibrated_row[
                    "high_underestimation"
                ]
                - standard_row[
                    "high_underestimation"
                ]
            )
            * 100.0
        )

        benefit_rows.append(
            result
        )

    benefit = (
        pd.DataFrame(
            benefit_rows
        )
        .sort_values(
            [
                "noise_level",
                "replicate",
            ]
        )
    )

    benefit_path = (
        output_root
        / "calibration_benefit.csv"
    )

    benefit.to_csv(
        benefit_path,
        index=False,
    )

    # ==================================================
    # Calibration benefit summary by noise level.
    # ==================================================

    benefit_summary_rows = []

    for (
        noise_level,
        group,
    ) in benefit.groupby(
        "noise_level"
    ):

        result = {
            "noise_level": float(
                noise_level
            ),
            "noise_percent": int(
                round(
                    float(
                        noise_level
                    )
                    * 100
                )
            ),
            "n_replicates": int(
                len(
                    group
                )
            ),
        }

        numeric_columns = [
            column
            for column
            in benefit.columns
            if (
                column
                not in {
                    "noise_level",
                    "noise_percent",
                    "replicate",
                }
            )
        ]

        for column in (
            numeric_columns
        ):

            values = np.asarray(
                group[
                    column
                ],
                dtype=np.float64,
            )

            result[
                f"{column}_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            result[
                f"{column}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
                if len(
                    values
                ) > 1
                else 0.0
            )

        benefit_summary_rows.append(
            result
        )

    benefit_summary = (
        pd.DataFrame(
            benefit_summary_rows
        )
        .sort_values(
            "noise_level"
        )
    )

    benefit_summary_path = (
        output_root
        / "calibration_benefit_summary.csv"
    )

    benefit_summary.to_csv(
        benefit_summary_path,
        index=False,
    )

    # ==================================================
    # Save test predictions for later paired diagnostics.
    # ==================================================

    predictions_path = (
        output_root
        / "matched_noise_test_predictions.npz"
    )

    np.savez_compressed(
        predictions_path,
        standard_prediction=np.stack(
            standard_predictions,
            axis=0,
        ),
        calibrated_prediction=np.stack(
            calibrated_predictions,
            axis=0,
        ),
        y_test=y_test,
        test_case_id=test_case_id,
        noise_level=np.asarray(
            condition_noise_levels,
            dtype=np.float64,
        ),
        replicate=np.asarray(
            condition_replicates,
            dtype=np.int64,
        ),
        condition_file=np.asarray(
            condition_files
        ),
        feature_names=np.asarray(
            feature_names
        ),
        scaler_mean=np.asarray(
            scaler.mean_,
            dtype=np.float64,
        ),
        scaler_scale=np.asarray(
            scaler.scale_,
            dtype=np.float64,
        ),
        selected_C=np.asarray(
            selected_C
        ),
        selected_gamma=np.asarray(
            selected_gamma
        ),
        selected_epsilon=np.asarray(
            selected_epsilon
        ),
        selected_alpha=np.asarray(
            selected_alpha
        ),
    )

    # ==================================================
    # Console summary.
    # ==================================================

    print()
    print(
        "=" * 80
    )

    print(
        "NOISE-LEVEL SUMMARY"
    )

    print(
        "=" * 80
    )

    compact_columns = [
        "method",
        "noise_percent",
        "n_replicates",
        "mae_mean",
        "mae_std",
        "high_mae_mean",
        "high_mae_std",
        "high_abs_bias_mean",
        "high_abs_bias_std",
        "high_underestimation_mean",
        "high_underestimation_std",
    ]

    print(
        noise_summary[
            compact_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "CALIBRATION BENEFIT SUMMARY"
    )

    print(
        "=" * 80
    )

    benefit_compact = [
        "noise_percent",
        "n_replicates",
        "mae_relative_improvement_percent_mean",
        "high_mae_relative_improvement_percent_mean",
        "high_abs_bias_relative_improvement_percent_mean",
        "high_underestimation_cal_minus_standard_pp_mean",
    ]

    print(
        benefit_summary[
            benefit_compact
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    # ==================================================
    # Basic integrity / final check.
    # ==================================================

    expected_metric_rows = (
        16
        * 2
    )

    condition_rows_ok = bool(
        len(
            condition_metrics
        )
        == expected_metric_rows
    )

    prediction_shape_ok = bool(
        np.stack(
            standard_predictions,
            axis=0,
        ).shape
        == (
            16,
            450,
            4,
        )
        and np.stack(
            calibrated_predictions,
            axis=0,
        ).shape
        == (
            16,
            450,
            4,
        )
    )

    all_finite = bool(
        np.isfinite(
            condition_metrics[
                [
                    "mse",
                    "mae",
                    "rmse",
                    "damaged_mae",
                    "zero_mae",
                    "low_mae",
                    "medium_mae",
                    "high_mae",
                    "high_abs_bias",
                    "high_underestimation",
                ]
            ].to_numpy(
                dtype=np.float64
            )
        ).all()
    )

    overall_passed = bool(
        condition_rows_ok
        and prediction_shape_ok
        and all_finite
    )

    # ==================================================
    # Report.
    # ==================================================

    report = {
        "experiment": (
            "clean_trained_matched_"
            "measurement_noise_robustness"
        ),
        "training_distribution": (
            "clean 78D descriptors only"
        ),
        "validation_distribution": (
            "clean 78D descriptors only"
        ),
        "test_distribution": (
            "matched 0/5/10/20% "
            "structural-response "
            "measurement noise"
        ),
        "ground_signal_perturbed": (
            False
        ),
        "scaler": (
            "StandardScaler fitted on "
            "clean training only"
        ),
        "prediction_clip": [
            CLIP_LOW,
            CLIP_HIGH,
        ],
        "svr_grid": {
            "C": SVR_C_GRID,
            "gamma": (
                SVR_GAMMA_GRID
            ),
            "epsilon": (
                SVR_EPSILON_GRID
            ),
            "selection_metric": (
                "clean validation MSE"
            ),
        },
        "selected_svr": {
            "C": selected_C,
            "gamma": (
                selected_gamma
            ),
            "epsilon": (
                selected_epsilon
            ),
            "clean_validation_metrics": (
                standard_val_metrics
            ),
        },
        "calibration": {
            "type": (
                "training-only one-sided "
                "monotone residual "
                "calibration"
            ),
            "oof_folds": (
                OOF_FOLDS
            ),
            "oof_random_state": (
                OOF_RANDOM_STATE
            ),
            "alpha_grid": (
                ALPHA_GRID
            ),
            "selected_alpha": (
                selected_alpha
            ),
            "fallback_to_standard": (
                fallback_to_standard
            ),
            "clean_validation_metrics": (
                calibrated_val_metrics
            ),
            "oof_story_diagnostics": (
                oof_story_diagnostics
            ),
        },
        "n_test_conditions": 16,
        "n_test_cases_per_condition": (
            450
        ),
        "condition_metric_rows_ok": (
            condition_rows_ok
        ),
        "prediction_shape_ok": (
            prediction_shape_ok
        ),
        "all_metrics_finite": (
            all_finite
        ),
        "overall_passed": (
            overall_passed
        ),
        "outputs": {
            "svr_grid": str(
                grid_path
            ),
            "alpha_grid": str(
                alpha_path
            ),
            "condition_metrics": str(
                condition_metrics_path
            ),
            "noise_summary": str(
                noise_summary_path
            ),
            "degradation_vs_clean": str(
                degradation_path
            ),
            "calibration_benefit": str(
                benefit_path
            ),
            "calibration_benefit_summary": str(
                benefit_summary_path
            ),
            "test_predictions": str(
                predictions_path
            ),
        },
    }

    report_path = (
        output_root
        / "matched_noise_robustness_report.json"
    )

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
        "=" * 80
    )

    print(
        "FINAL ROBUSTNESS CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Selected SVR:",
        {
            "C": selected_C,
            "gamma": (
                selected_gamma
            ),
            "epsilon": (
                selected_epsilon
            ),
        },
    )

    print(
        "Selected alpha:",
        selected_alpha,
    )

    print(
        "Calibration fallback:",
        fallback_to_standard,
    )

    print(
        "Metric rows correct:",
        condition_rows_ok,
    )

    print(
        "Prediction shape correct:",
        prediction_shape_ok,
    )

    print(
        "All metrics finite:",
        all_finite,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    if overall_passed:

        print(
            "CHECK PASSED: clean-trained "
            "matched-noise robustness "
            "evaluation completed."
        )

        print(
            "Noisy test conditions were "
            "used for inference only."
        )

    else:

        print(
            "CHECK FAILED: inspect outputs "
            "before interpreting results."
        )

    print()

    print(
        "Noise summary:",
        noise_summary_path,
    )

    print(
        "Calibration summary:",
        benefit_summary_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
