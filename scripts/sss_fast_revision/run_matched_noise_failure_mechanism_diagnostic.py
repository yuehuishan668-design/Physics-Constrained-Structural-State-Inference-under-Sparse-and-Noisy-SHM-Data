"""
Matched-noise failure-mechanism diagnostic.

This script performs NO training and NO tuning.

It reads the already-frozen:
1. matched clean/noisy 78D descriptor datasets;
2. clean-trained standard/calibrated SVR predictions.

Questions addressed
-------------------
A. Does measurement noise induce signed-bias reversal?
B. Does reduced underestimation merely reflect increased overestimation?
C. Does prediction clipping/saturation contribute to high-noise failure?
D. Which 78D descriptors undergo the largest paired distribution shifts?
E. Is descriptor shift associated with case-level error degradation?

No model component is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CLIP_LOW = 0.0
CLIP_HIGH = 0.5
DIRECTION_TOL = 0.0

EXPECTED_CONDITIONS = 16
EXPECTED_TEST_CASES = 450
EXPECTED_STORIES = 4
EXPECTED_FEATURES = 78


def severity_masks(
    y: np.ndarray,
) -> dict[str, np.ndarray]:

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    return {
        "all": np.ones_like(
            y,
            dtype=bool,
        ),
        "zero": (
            y <= 1e-12
        ),
        "low": (
            (y > 1e-12)
            & (y <= 0.10)
        ),
        "medium": (
            (y > 0.10)
            & (y <= 0.20)
        ),
        "high": (
            y > 0.20
        ),
        "damaged": (
            y > 1e-12
        ),
    }


def directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:

    truth = np.asarray(
        y_true[mask],
        dtype=np.float64,
    )

    pred = np.asarray(
        y_pred[mask],
        dtype=np.float64,
    )

    if truth.size == 0:

        return {
            "n_entries": 0,
            "signed_bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "underestimation_ratio": float("nan"),
            "overestimation_ratio": float("nan"),
            "equal_ratio": float("nan"),
        }

    error = (
        pred - truth
    )

    under = (
        error < -DIRECTION_TOL
    )

    over = (
        error > DIRECTION_TOL
    )

    equal = (
        ~(under | over)
    )

    return {
        "n_entries": int(
            truth.size
        ),
        "signed_bias": float(
            np.mean(
                error
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
        "underestimation_ratio": float(
            np.mean(
                under
            )
        ),
        "overestimation_ratio": float(
            np.mean(
                over
            )
        ),
        "equal_ratio": float(
            np.mean(
                equal
            )
        ),
    }


def prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:

    pred = np.asarray(
        y_pred[mask],
        dtype=np.float64,
    )

    if pred.size == 0:

        return {
            "n_entries": 0,
            "prediction_mean": float("nan"),
            "prediction_std": float("nan"),
            "prediction_q05": float("nan"),
            "prediction_q25": float("nan"),
            "prediction_q50": float("nan"),
            "prediction_q75": float("nan"),
            "prediction_q95": float("nan"),
            "clip_low_ratio": float("nan"),
            "clip_high_ratio": float("nan"),
        }

    return {
        "n_entries": int(
            pred.size
        ),
        "prediction_mean": float(
            np.mean(
                pred
            )
        ),
        "prediction_std": float(
            np.std(
                pred
            )
        ),
        "prediction_q05": float(
            np.quantile(
                pred,
                0.05,
            )
        ),
        "prediction_q25": float(
            np.quantile(
                pred,
                0.25,
            )
        ),
        "prediction_q50": float(
            np.quantile(
                pred,
                0.50,
            )
        ),
        "prediction_q75": float(
            np.quantile(
                pred,
                0.75,
            )
        ),
        "prediction_q95": float(
            np.quantile(
                pred,
                0.95,
            )
        ),
        "clip_low_ratio": float(
            np.mean(
                pred <= CLIP_LOW
            )
        ),
        "clip_high_ratio": float(
            np.mean(
                pred >= CLIP_HIGH
            )
        ),
    }


def resolve_path(
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


def load_descriptor_condition(
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
            for key in data.files
        }

    required = {
        "F_78_raw",
        "feature_names_78",
        "case_id",
        "y_damage",
        "test_idx",
        "noise_level",
        "replicate",
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


def safe_pearson(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if (
        x.size < 3
        or np.std(x) < 1e-15
        or np.std(y) < 1e-15
    ):

        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if (
        x.size < 3
        or np.std(x) < 1e-15
        or np.std(y) < 1e-15
    ):

        return (
            float("nan"),
            float("nan"),
        )

    result = spearmanr(
        x,
        y,
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "matched_noise_robustness_clean_trained"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "matched_noise_failure_mechanism"
        ),
    )

    args = parser.parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    robustness_root = (
        args.robustness_root
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

    predictions_path = (
        robustness_root
        / "matched_noise_test_predictions.npz"
    )

    if not predictions_path.is_file():

        raise FileNotFoundError(
            predictions_path
        )

    print(
        "=" * 80
    )

    print(
        "MATCHED-NOISE FAILURE-MECHANISM DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    print(
        "Predictions:",
        predictions_path,
    )

    print(
        "No training or tuning will occur."
    )

    print()

    # ==================================================
    # Load frozen predictions.
    # ==================================================

    with np.load(
        predictions_path,
        allow_pickle=False,
    ) as data:

        standard_prediction = np.asarray(
            data[
                "standard_prediction"
            ],
            dtype=np.float64,
        )

        calibrated_prediction = np.asarray(
            data[
                "calibrated_prediction"
            ],
            dtype=np.float64,
        )

        y_test = np.asarray(
            data[
                "y_test"
            ],
            dtype=np.float64,
        )

        test_case_id = np.asarray(
            data[
                "test_case_id"
            ],
            dtype=np.int64,
        )

        noise_levels = np.asarray(
            data[
                "noise_level"
            ],
            dtype=np.float64,
        )

        replicates = np.asarray(
            data[
                "replicate"
            ],
            dtype=np.int64,
        )

        condition_files = [
            str(value)
            for value in data[
                "condition_file"
            ].tolist()
        ]

        feature_names = [
            str(value)
            for value in data[
                "feature_names"
            ].tolist()
        ]

        scaler_mean = np.asarray(
            data[
                "scaler_mean"
            ],
            dtype=np.float64,
        )

        scaler_scale = np.asarray(
            data[
                "scaler_scale"
            ],
            dtype=np.float64,
        )

    expected_prediction_shape = (
        EXPECTED_CONDITIONS,
        EXPECTED_TEST_CASES,
        EXPECTED_STORIES,
    )

    if (
        standard_prediction.shape
        != expected_prediction_shape
        or calibrated_prediction.shape
        != expected_prediction_shape
    ):

        raise RuntimeError(
            "Unexpected prediction shape."
        )

    if (
        y_test.shape
        != (
            EXPECTED_TEST_CASES,
            EXPECTED_STORIES,
        )
    ):

        raise RuntimeError(
            "Unexpected y_test shape."
        )

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
        scaler_mean.shape
        != (
            EXPECTED_FEATURES,
        )
        or scaler_scale.shape
        != (
            EXPECTED_FEATURES,
        )
    ):

        raise RuntimeError(
            "Unexpected scaler shape."
        )

    if np.any(
        scaler_scale <= 0.0
    ):

        raise RuntimeError(
            "Non-positive scaler scale."
        )

    clean_indices = np.where(
        np.isclose(
            noise_levels,
            0.0,
        )
    )[0]

    if len(
        clean_indices
    ) != 1:

        raise RuntimeError(
            "Expected one clean condition."
        )

    clean_condition_index = int(
        clean_indices[0]
    )

    masks = severity_masks(
        y_test
    )

    # ==================================================
    # A/B. Signed bias + directional error.
    # ==================================================

    directional_rows = []

    methods = {
        "standard_svr": (
            standard_prediction
        ),
        "calibrated_svr": (
            calibrated_prediction
        ),
    }

    for method, predictions in (
        methods.items()
    ):

        for condition_index in range(
            EXPECTED_CONDITIONS
        ):

            noise_level = float(
                noise_levels[
                    condition_index
                ]
            )

            replicate = int(
                replicates[
                    condition_index
                ]
            )

            prediction = (
                predictions[
                    condition_index
                ]
            )

            for (
                severity,
                mask,
            ) in masks.items():

                metrics = (
                    directional_metrics(
                        y_test,
                        prediction,
                        mask,
                    )
                )

                directional_rows.append(
                    {
                        "method": (
                            method
                        ),
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
                        "severity": (
                            severity
                        ),
                        **metrics,
                    }
                )

    directional_frame = pd.DataFrame(
        directional_rows
    )

    directional_path = (
        output_root
        / "severity_directional_metrics.csv"
    )

    directional_frame.to_csv(
        directional_path,
        index=False,
    )

    # Replicate summary.
    directional_summary_rows = []

    summary_metric_names = [
        "signed_bias",
        "mae",
        "rmse",
        "underestimation_ratio",
        "overestimation_ratio",
        "equal_ratio",
    ]

    for (
        method,
        noise_level,
        severity,
    ), group in (
        directional_frame.groupby(
            [
                "method",
                "noise_level",
                "severity",
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
            "severity": severity,
            "n_replicates": int(
                len(
                    group
                )
            ),
        }

        for metric in (
            summary_metric_names
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
                if len(values) > 1
                else 0.0
            )

        directional_summary_rows.append(
            row
        )

    directional_summary = (
        pd.DataFrame(
            directional_summary_rows
        )
        .sort_values(
            [
                "method",
                "severity",
                "noise_level",
            ]
        )
    )

    directional_summary_path = (
        output_root
        / "severity_directional_summary.csv"
    )

    directional_summary.to_csv(
        directional_summary_path,
        index=False,
    )

    # ==================================================
    # C. Prediction distribution + clipping.
    # ==================================================

    distribution_rows = []

    for method, predictions in (
        methods.items()
    ):

        for condition_index in range(
            EXPECTED_CONDITIONS
        ):

            noise_level = float(
                noise_levels[
                    condition_index
                ]
            )

            replicate = int(
                replicates[
                    condition_index
                ]
            )

            prediction = (
                predictions[
                    condition_index
                ]
            )

            for (
                severity,
                mask,
            ) in masks.items():

                metrics = (
                    prediction_distribution(
                        y_test,
                        prediction,
                        mask,
                    )
                )

                distribution_rows.append(
                    {
                        "method": (
                            method
                        ),
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
                        "severity": (
                            severity
                        ),
                        **metrics,
                    }
                )

    distribution_frame = pd.DataFrame(
        distribution_rows
    )

    distribution_path = (
        output_root
        / "prediction_distribution_metrics.csv"
    )

    distribution_frame.to_csv(
        distribution_path,
        index=False,
    )

    distribution_summary_rows = []

    distribution_metrics = [
        "prediction_mean",
        "prediction_std",
        "prediction_q05",
        "prediction_q25",
        "prediction_q50",
        "prediction_q75",
        "prediction_q95",
        "clip_low_ratio",
        "clip_high_ratio",
    ]

    for (
        method,
        noise_level,
        severity,
    ), group in (
        distribution_frame.groupby(
            [
                "method",
                "noise_level",
                "severity",
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
            "severity": severity,
            "n_replicates": int(
                len(
                    group
                )
            ),
        }

        for metric in (
            distribution_metrics
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
                if len(values) > 1
                else 0.0
            )

        distribution_summary_rows.append(
            row
        )

    distribution_summary = (
        pd.DataFrame(
            distribution_summary_rows
        )
        .sort_values(
            [
                "method",
                "severity",
                "noise_level",
            ]
        )
    )

    distribution_summary_path = (
        output_root
        / "prediction_distribution_summary.csv"
    )

    distribution_summary.to_csv(
        distribution_summary_path,
        index=False,
    )

    # ==================================================
    # D. Load matched descriptors and calculate paired
    #    clean-normalized descriptor shifts.
    # ==================================================

    clean_condition_path = (
        resolve_path(
            condition_files[
                clean_condition_index
            ],
            project_root,
        )
    )

    clean_condition = (
        load_descriptor_condition(
            clean_condition_path
        )
    )

    clean_test_idx = np.asarray(
        clean_condition[
            "test_idx"
        ],
        dtype=np.int64,
    )

    clean_case_id = np.asarray(
        clean_condition[
            "case_id"
        ],
        dtype=np.int64,
    )[
        clean_test_idx
    ]

    if not np.array_equal(
        clean_case_id,
        test_case_id,
    ):

        raise RuntimeError(
            "Clean test case IDs do not "
            "match saved predictions."
        )

    clean_F = np.asarray(
        clean_condition[
            "F_78_raw"
        ],
        dtype=np.float64,
    )[
        clean_test_idx
    ]

    clean_names = [
        str(value)
        for value in clean_condition[
            "feature_names_78"
        ].tolist()
    ]

    if clean_names != feature_names:

        raise RuntimeError(
            "Feature-name mismatch."
        )

    descriptor_shift_rows = []

    case_shift_rows = []

    clean_standard_case_mae = np.mean(
        np.abs(
            standard_prediction[
                clean_condition_index
            ]
            - y_test
        ),
        axis=1,
    )

    clean_calibrated_case_mae = np.mean(
        np.abs(
            calibrated_prediction[
                clean_condition_index
            ]
            - y_test
        ),
        axis=1,
    )

    for condition_index in range(
        EXPECTED_CONDITIONS
    ):

        noise_level = float(
            noise_levels[
                condition_index
            ]
        )

        replicate = int(
            replicates[
                condition_index
            ]
        )

        condition_path = (
            resolve_path(
                condition_files[
                    condition_index
                ],
                project_root,
            )
        )

        condition = (
            load_descriptor_condition(
                condition_path
            )
        )

        test_idx = np.asarray(
            condition[
                "test_idx"
            ],
            dtype=np.int64,
        )

        condition_case_id = np.asarray(
            condition[
                "case_id"
            ],
            dtype=np.int64,
        )[
            test_idx
        ]

        if not np.array_equal(
            condition_case_id,
            test_case_id,
        ):

            raise RuntimeError(
                "Matched test case identity "
                f"failed: {condition_path}"
            )

        F_condition = np.asarray(
            condition[
                "F_78_raw"
            ],
            dtype=np.float64,
        )[
            test_idx
        ]

        delta_raw = (
            F_condition
            - clean_F
        )

        # Since all conditions use the same clean-train scaler:
        #
        # z_noisy - z_clean
        # = (F_noisy - F_clean) / clean_train_scale
        delta_z = (
            delta_raw
            / scaler_scale.reshape(
                1,
                -1,
            )
        )

        for feature_index, feature_name in enumerate(
            feature_names
        ):

            values = (
                delta_z[
                    :,
                    feature_index,
                ]
            )

            abs_values = np.abs(
                values
            )

            descriptor_shift_rows.append(
                {
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
                    "feature_index": (
                        feature_index
                    ),
                    "feature_name": (
                        feature_name
                    ),
                    "signed_mean_delta_z": float(
                        np.mean(
                            values
                        )
                    ),
                    "mean_abs_delta_z": float(
                        np.mean(
                            abs_values
                        )
                    ),
                    "median_abs_delta_z": float(
                        np.median(
                            abs_values
                        )
                    ),
                    "rms_delta_z": float(
                        np.sqrt(
                            np.mean(
                                values ** 2
                            )
                        )
                    ),
                    "p95_abs_delta_z": float(
                        np.quantile(
                            abs_values,
                            0.95,
                        )
                    ),
                    "fraction_abs_delta_z_gt_1": float(
                        np.mean(
                            abs_values > 1.0
                        )
                    ),
                    "fraction_abs_delta_z_gt_2": float(
                        np.mean(
                            abs_values > 2.0
                        )
                    ),
                }
            )

        case_shift_l2 = np.sqrt(
            np.mean(
                delta_z ** 2,
                axis=1,
            )
        )

        case_shift_l1 = np.mean(
            np.abs(
                delta_z
            ),
            axis=1,
        )

        standard_case_mae = np.mean(
            np.abs(
                standard_prediction[
                    condition_index
                ]
                - y_test
            ),
            axis=1,
        )

        calibrated_case_mae = np.mean(
            np.abs(
                calibrated_prediction[
                    condition_index
                ]
                - y_test
            ),
            axis=1,
        )

        for row_index in range(
            EXPECTED_TEST_CASES
        ):

            case_shift_rows.append(
                {
                    "case_id": int(
                        test_case_id[
                            row_index
                        ]
                    ),
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
                    "descriptor_shift_l2": float(
                        case_shift_l2[
                            row_index
                        ]
                    ),
                    "descriptor_shift_l1": float(
                        case_shift_l1[
                            row_index
                        ]
                    ),
                    "standard_case_mae": float(
                        standard_case_mae[
                            row_index
                        ]
                    ),
                    "standard_case_mae_change": float(
                        standard_case_mae[
                            row_index
                        ]
                        - clean_standard_case_mae[
                            row_index
                        ]
                    ),
                    "calibrated_case_mae": float(
                        calibrated_case_mae[
                            row_index
                        ]
                    ),
                    "calibrated_case_mae_change": float(
                        calibrated_case_mae[
                            row_index
                        ]
                        - clean_calibrated_case_mae[
                            row_index
                        ]
                    ),
                }
            )

    descriptor_shift = pd.DataFrame(
        descriptor_shift_rows
    )

    descriptor_shift_path = (
        output_root
        / "descriptor_shift_by_condition.csv"
    )

    descriptor_shift.to_csv(
        descriptor_shift_path,
        index=False,
    )

    # Aggregate descriptor shift across replicates.
    descriptor_summary_rows = []

    shift_metrics = [
        "signed_mean_delta_z",
        "mean_abs_delta_z",
        "median_abs_delta_z",
        "rms_delta_z",
        "p95_abs_delta_z",
        "fraction_abs_delta_z_gt_1",
        "fraction_abs_delta_z_gt_2",
    ]

    for (
        noise_level,
        feature_index,
        feature_name,
    ), group in (
        descriptor_shift.groupby(
            [
                "noise_level",
                "feature_index",
                "feature_name",
            ]
        )
    ):

        row = {
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
            "feature_index": int(
                feature_index
            ),
            "feature_name": (
                feature_name
            ),
            "n_replicates": int(
                len(
                    group
                )
            ),
        }

        for metric in (
            shift_metrics
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
                if len(values) > 1
                else 0.0
            )

        descriptor_summary_rows.append(
            row
        )

    descriptor_summary = pd.DataFrame(
        descriptor_summary_rows
    )

    descriptor_summary[
        "rank_by_mean_abs_shift"
    ] = (
        descriptor_summary.groupby(
            "noise_level"
        )[
            "mean_abs_delta_z_mean"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    descriptor_summary = (
        descriptor_summary.sort_values(
            [
                "noise_level",
                "rank_by_mean_abs_shift",
            ]
        )
    )

    descriptor_summary_path = (
        output_root
        / "descriptor_shift_summary.csv"
    )

    descriptor_summary.to_csv(
        descriptor_summary_path,
        index=False,
    )

    case_shift = pd.DataFrame(
        case_shift_rows
    )

    case_shift_path = (
        output_root
        / "case_level_descriptor_shift_and_error.csv"
    )

    case_shift.to_csv(
        case_shift_path,
        index=False,
    )

    # ==================================================
    # E. Association between descriptor displacement
    #    and error degradation.
    # ==================================================

    association_rows = []

    noisy_case_shift = (
        case_shift[
            case_shift[
                "noise_level"
            ] > 0.0
        ]
    )

    for (
        noise_level,
        replicate,
    ), group in (
        noisy_case_shift.groupby(
            [
                "noise_level",
                "replicate",
            ]
        )
    ):

        x = np.asarray(
            group[
                "descriptor_shift_l2"
            ],
            dtype=np.float64,
        )

        for method in [
            "standard",
            "calibrated",
        ]:

            y = np.asarray(
                group[
                    f"{method}_case_mae_change"
                ],
                dtype=np.float64,
            )

            pearson = safe_pearson(
                x,
                y,
            )

            (
                spearman,
                spearman_p,
            ) = safe_spearman(
                x,
                y,
            )

            association_rows.append(
                {
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
                    "replicate": int(
                        replicate
                    ),
                    "method": (
                        method
                    ),
                    "pearson_r": (
                        pearson
                    ),
                    "spearman_rho": (
                        spearman
                    ),
                    "spearman_p": (
                        spearman_p
                    ),
                }
            )

    association = pd.DataFrame(
        association_rows
    )

    association_path = (
        output_root
        / "descriptor_shift_error_association.csv"
    )

    association.to_csv(
        association_path,
        index=False,
    )

    association_summary_rows = []

    for (
        noise_level,
        method,
    ), group in (
        association.groupby(
            [
                "noise_level",
                "method",
            ]
        )
    ):

        association_summary_rows.append(
            {
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
                "method": method,
                "n_replicates": int(
                    len(
                        group
                    )
                ),
                "pearson_r_mean": float(
                    np.mean(
                        group[
                            "pearson_r"
                        ]
                    )
                ),
                "pearson_r_std": float(
                    np.std(
                        group[
                            "pearson_r"
                        ],
                        ddof=1,
                    )
                ),
                "spearman_rho_mean": float(
                    np.mean(
                        group[
                            "spearman_rho"
                        ]
                    )
                ),
                "spearman_rho_std": float(
                    np.std(
                        group[
                            "spearman_rho"
                        ],
                        ddof=1,
                    )
                ),
            }
        )

    association_summary = (
        pd.DataFrame(
            association_summary_rows
        )
        .sort_values(
            [
                "method",
                "noise_level",
            ]
        )
    )

    association_summary_path = (
        output_root
        / "descriptor_shift_error_association_summary.csv"
    )

    association_summary.to_csv(
        association_summary_path,
        index=False,
    )

    # ==================================================
    # Explicit bias-reversal diagnostic.
    # ==================================================

    high_directional = (
        directional_summary[
            directional_summary[
                "severity"
            ]
            == "high"
        ]
        .copy()
    )

    bias_reversal_rows = []

    for method in [
        "standard_svr",
        "calibrated_svr",
    ]:

        subset = (
            high_directional[
                high_directional[
                    "method"
                ]
                == method
            ]
            .sort_values(
                "noise_level"
            )
        )

        clean_bias = float(
            subset[
                np.isclose(
                    subset[
                        "noise_level"
                    ],
                    0.0,
                )
            ][
                "signed_bias_mean"
            ].iloc[0]
        )

        for _, row in (
            subset.iterrows()
        ):

            bias = float(
                row[
                    "signed_bias_mean"
                ]
            )

            if bias < 0.0:

                sign = "negative"

            elif bias > 0.0:

                sign = "positive"

            else:

                sign = "zero"

            bias_reversal_rows.append(
                {
                    "method": method,
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
                    "high_signed_bias": (
                        bias
                    ),
                    "bias_sign": sign,
                    "clean_bias": (
                        clean_bias
                    ),
                    "sign_differs_from_clean": bool(
                        np.sign(
                            bias
                        )
                        != np.sign(
                            clean_bias
                        )
                    ),
                }
            )

    bias_reversal = pd.DataFrame(
        bias_reversal_rows
    )

    bias_reversal_path = (
        output_root
        / "high_damage_bias_reversal.csv"
    )

    bias_reversal.to_csv(
        bias_reversal_path,
        index=False,
    )

    # ==================================================
    # Console summaries.
    # ==================================================

    print()
    print(
        "=" * 80
    )

    print(
        "HIGH-DAMAGE DIRECTIONAL SUMMARY"
    )

    print(
        "=" * 80
    )

    high_print = (
        directional_summary[
            directional_summary[
                "severity"
            ]
            == "high"
        ][
            [
                "method",
                "noise_percent",
                "n_replicates",
                "signed_bias_mean",
                "signed_bias_std",
                "mae_mean",
                "underestimation_ratio_mean",
                "overestimation_ratio_mean",
                "equal_ratio_mean",
            ]
        ]
        .sort_values(
            [
                "method",
                "noise_percent",
            ]
        )
    )

    print(
        high_print.to_string(
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
        "HIGH-DAMAGE BIAS REVERSAL"
    )

    print(
        "=" * 80
    )

    print(
        bias_reversal.to_string(
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
        "PREDICTION SATURATION SUMMARY"
    )

    print(
        "=" * 80
    )

    saturation_print = (
        distribution_summary[
            distribution_summary[
                "severity"
            ]
            .isin(
                [
                    "all",
                    "zero",
                    "low",
                    "high",
                ]
            )
        ][
            [
                "method",
                "severity",
                "noise_percent",
                "prediction_mean_mean",
                "prediction_q50_mean",
                "prediction_q95_mean",
                "clip_low_ratio_mean",
                "clip_high_ratio_mean",
            ]
        ]
        .sort_values(
            [
                "method",
                "severity",
                "noise_percent",
            ]
        )
    )

    print(
        saturation_print.to_string(
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
        "TOP 10 SHIFTED DESCRIPTORS BY NOISE LEVEL"
    )

    print(
        "=" * 80
    )

    for noise_level in [
        0.05,
        0.10,
        0.20,
    ]:

        subset = (
            descriptor_summary[
                np.isclose(
                    descriptor_summary[
                        "noise_level"
                    ],
                    noise_level,
                )
            ]
            .sort_values(
                "rank_by_mean_abs_shift"
            )
            .head(
                10
            )
        )

        print()
        print(
            f"Noise = "
            f"{int(noise_level * 100)}%"
        )

        print(
            subset[
                [
                    "rank_by_mean_abs_shift",
                    "feature_index",
                    "feature_name",
                    "signed_mean_delta_z_mean",
                    "mean_abs_delta_z_mean",
                    "rms_delta_z_mean",
                    "p95_abs_delta_z_mean",
                    "fraction_abs_delta_z_gt_1_mean",
                    "fraction_abs_delta_z_gt_2_mean",
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
        "=" * 80
    )

    print(
        "DESCRIPTOR SHIFT / ERROR ASSOCIATION"
    )

    print(
        "=" * 80
    )

    print(
        association_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    # ==================================================
    # Integrity checks.
    # ==================================================

    clean_descriptor_shift = (
        descriptor_summary[
            np.isclose(
                descriptor_summary[
                    "noise_level"
                ],
                0.0,
            )
        ]
    )

    clean_shift_exact = bool(
        (
            clean_descriptor_shift[
                "mean_abs_delta_z_mean"
            ]
            == 0.0
        ).all()
    )

    directional_finite = bool(
        np.isfinite(
            directional_frame[
                [
                    "signed_bias",
                    "mae",
                    "rmse",
                    "underestimation_ratio",
                    "overestimation_ratio",
                    "equal_ratio",
                ]
            ].to_numpy(
                dtype=np.float64,
            )
        ).all()
    )

    descriptor_finite = bool(
        np.isfinite(
            descriptor_shift[
                [
                    "signed_mean_delta_z",
                    "mean_abs_delta_z",
                    "median_abs_delta_z",
                    "rms_delta_z",
                    "p95_abs_delta_z",
                    "fraction_abs_delta_z_gt_1",
                    "fraction_abs_delta_z_gt_2",
                ]
            ].to_numpy(
                dtype=np.float64,
            )
        ).all()
    )

    overall_passed = bool(
        clean_shift_exact
        and directional_finite
        and descriptor_finite
    )

    report = {
        "experiment": (
            "matched_noise_failure_"
            "mechanism_diagnostic"
        ),
        "training_or_tuning_performed": (
            False
        ),
        "questions": [
            "signed bias reversal",
            "under/overestimation transition",
            "prediction clipping/saturation",
            "paired 78D descriptor distribution shift",
            "descriptor-shift/error association",
        ],
        "clean_descriptor_shift_exact_zero": (
            clean_shift_exact
        ),
        "directional_metrics_finite": (
            directional_finite
        ),
        "descriptor_metrics_finite": (
            descriptor_finite
        ),
        "overall_passed": (
            overall_passed
        ),
        "outputs": {
            "directional_metrics": str(
                directional_path
            ),
            "directional_summary": str(
                directional_summary_path
            ),
            "prediction_distribution": str(
                distribution_path
            ),
            "prediction_distribution_summary": str(
                distribution_summary_path
            ),
            "descriptor_shift": str(
                descriptor_shift_path
            ),
            "descriptor_shift_summary": str(
                descriptor_summary_path
            ),
            "case_shift": str(
                case_shift_path
            ),
            "association": str(
                association_path
            ),
            "association_summary": str(
                association_summary_path
            ),
            "bias_reversal": str(
                bias_reversal_path
            ),
        },
    }

    report_path = (
        output_root
        / "matched_noise_failure_mechanism_report.json"
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
        "FINAL DIAGNOSTIC CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Training/tuning performed:",
        False,
    )

    print(
        "Clean descriptor shift "
        "exactly zero:",
        clean_shift_exact,
    )

    print(
        "Directional metrics finite:",
        directional_finite,
    )

    print(
        "Descriptor metrics finite:",
        descriptor_finite,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    if overall_passed:

        print(
            "CHECK PASSED: failure-mechanism "
            "diagnostic completed without "
            "modifying the frozen models."
        )

    else:

        print(
            "CHECK FAILED: inspect diagnostics "
            "before scientific interpretation."
        )

    print()

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
