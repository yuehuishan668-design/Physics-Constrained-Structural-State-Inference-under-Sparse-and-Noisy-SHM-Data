"""
Exhaustive dependency-aware structural sensor-layout RBF-SVR experiment.

Purpose
-------
Evaluate all 15 non-empty structural sensor layouts using the
dependency-aware clean descriptor datasets.

Protocol
--------
For every layout:

    layout-specific physically available descriptors
        ↓
    identical frozen train / validation / test cases
        ↓
    StandardScaler fitted on layout training data only
        ↓
    same frozen 75-candidate RBF-SVR grid
        ↓
    minimum validation MSE selection
        ↓
    refit on training data only
        ↓
    evaluate test once

No calibration is used in this experiment.

The full four-sensor layout {1,2,3,4} is evaluated first and must
reproduce the previously frozen clean-trained 78D Standard SVR
experiment before any reduced-sensor result is accepted.

Outputs
-------
- complete validation-grid results
- layout-level test metrics
- layout ranking
- sensor-count aggregate
- placement spread
- marginal sensor-value analysis
- all test predictions
- full experiment report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ============================================================
# Frozen experimental protocol
# ============================================================

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

CLIP_LOW = 0.0
CLIP_HIGH = 0.5

DAMAGE_ZERO_TOL = 1e-12

EXPECTED_LAYOUTS = 15
EXPECTED_CASES = 3000
EXPECTED_STORIES = 4

ANCHOR_ATOL = 1e-12
ANCHOR_RTOL = 1e-12


# ============================================================
# Utility functions
# ============================================================

def resolve_path(
    value: str | Path,
    project_root: Path,
) -> Path:

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():

        path = (
            project_root
            / path
        )

    return path.resolve()


def layout_tuple_from_tag(
    tag: str,
) -> tuple[int, ...]:

    return tuple(
        int(character)
        for character in str(tag)
    )


def layout_tag(
    layout: tuple[int, ...],
) -> str:

    return "".join(
        str(sensor)
        for sensor in layout
    )


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

    return {
        "zero": (
            y <= DAMAGE_ZERO_TOL
        ),
        "low": (
            (y > DAMAGE_ZERO_TOL)
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
            y > DAMAGE_ZERO_TOL
        ),
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
        masks[
            "high"
        ],
    )

    if np.any(
        masks[
            "high"
        ]
    ):

        high_underestimation = float(
            np.mean(
                y_pred[
                    masks[
                        "high"
                    ]
                ]
                <
                y_true[
                    masks[
                        "high"
                    ]
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
                masks[
                    "damaged"
                ],
            )
        ),
        "zero_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks[
                    "zero"
                ],
            )
        ),
        "low_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks[
                    "low"
                ],
            )
        ),
        "medium_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks[
                    "medium"
                ],
            )
        ),
        "high_mae": (
            masked_mae(
                y_true,
                y_pred,
                masks[
                    "high"
                ],
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
                masks[
                    "zero"
                ]
            )
        ),
        "n_low": int(
            np.sum(
                masks[
                    "low"
                ]
            )
        ),
        "n_medium": int(
            np.sum(
                masks[
                    "medium"
                ]
            )
        ),
        "n_high": int(
            np.sum(
                masks[
                    "high"
                ]
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
            y[
                :,
                story,
            ],
        )

        models.append(
            model
        )

    return models


def predict_story_svrs(
    models: list[SVR],
    X: np.ndarray,
) -> np.ndarray:

    prediction = np.column_stack(
        [
            model.predict(
                X
            )
            for model
            in models
        ]
    )

    return clip_prediction(
        prediction
    )


def load_layout_dataset(
    path: Path,
) -> dict[str, np.ndarray]:

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        result = {
            key: np.asarray(
                data[
                    key
                ]
            )
            for key
            in data.files
        }

    required = {
        "F_raw",
        "feature_names",
        "feature_indices_in_full_78d",
        "sensor_keep_1based",
        "sensor_count",
        "y_damage",
        "case_id",
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


def validate_against_reference(
    dataset: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    path: Path,
) -> None:

    keys = [
        "y_damage",
        "case_id",
        "train_idx",
        "val_idx",
        "test_idx",
    ]

    for key in keys:

        if not np.array_equal(
            dataset[
                key
            ],
            reference[
                key
            ],
        ):

            raise RuntimeError(
                "Layout case/split "
                f"integrity failure: "
                f"{path}, key={key}"
            )


def close_enough(
    actual: float,
    expected: float,
) -> bool:

    return bool(
        np.isclose(
            actual,
            expected,
            atol=ANCHOR_ATOL,
            rtol=ANCHOR_RTOL,
        )
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layout-root",
        type=Path,
        default=Path(
            "data_processed/"
            "sss_fast_revision/"
            "sensor_layouts_dependency_aware"
        ),
    )

    parser.add_argument(
        "--clean-reference-root",
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
            "exhaustive_sensor_layout_svr"
        ),
    )

    args = parser.parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    layout_root = (
        args.layout_root
        .expanduser()
        .resolve()
    )

    clean_reference_root = (
        args.clean_reference_root
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
        layout_root
        / "sensor_layout_manifest.csv"
    )

    if not manifest_path.is_file():

        raise FileNotFoundError(
            manifest_path
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "layout_tag": str,
            "sensor_layout": str,
        },
    )

    if len(
        manifest
    ) != EXPECTED_LAYOUTS:

        raise RuntimeError(
            "Expected exactly 15 layouts."
        )

    if not bool(
        manifest[
            "dimension_ok"
        ].all()
    ):

        raise RuntimeError(
            "Layout manifest contains "
            "dimension failure."
        )

    # ========================================================
    # Load exact clean-model anchor from previous experiment.
    # ========================================================

    reference_report_path = (
        clean_reference_root
        / "matched_noise_robustness_report.json"
    )

    reference_metrics_path = (
        clean_reference_root
        / "condition_metrics.csv"
    )

    if not reference_report_path.is_file():

        raise FileNotFoundError(
            reference_report_path
        )

    if not reference_metrics_path.is_file():

        raise FileNotFoundError(
            reference_metrics_path
        )

    with reference_report_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        reference_report = (
            json.load(
                file
            )
        )

    reference_selected = (
        reference_report[
            "selected_svr"
        ]
    )

    reference_metrics = pd.read_csv(
        reference_metrics_path
    )

    clean_reference_rows = (
        reference_metrics[
            (
                reference_metrics[
                    "method"
                ]
                == "standard_svr"
            )
            & np.isclose(
                reference_metrics[
                    "noise_level"
                ],
                0.0,
            )
            & (
                reference_metrics[
                    "replicate"
                ]
                == 0
            )
        ]
    )

    if len(
        clean_reference_rows
    ) != 1:

        raise RuntimeError(
            "Could not uniquely resolve "
            "clean Standard SVR anchor."
        )

    clean_reference_row = (
        clean_reference_rows
        .iloc[0]
    )

    reference_C = float(
        reference_selected[
            "C"
        ]
    )

    reference_gamma = float(
        reference_selected[
            "gamma"
        ]
    )

    reference_epsilon = float(
        reference_selected[
            "epsilon"
        ]
    )

    # ========================================================
    # Put full layout first.
    # ========================================================

    manifest[
        "_full_first"
    ] = (
        manifest[
            "layout_tag"
        ]
        != "1234"
    ).astype(
        int
    )

    manifest = (
        manifest.sort_values(
            [
                "_full_first",
                "sensor_count",
                "layout_tag",
            ]
        )
        .drop(
            columns=[
                "_full_first",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    full_manifest_row = (
        manifest[
            manifest[
                "layout_tag"
            ]
            == "1234"
        ]
    )

    if len(
        full_manifest_row
    ) != 1:

        raise RuntimeError(
            "Full layout 1234 missing."
        )

    full_path = resolve_path(
        str(
            full_manifest_row.iloc[
                0
            ][
                "file"
            ]
        ),
        project_root,
    )

    full_dataset = (
        load_layout_dataset(
            full_path
        )
    )

    y_all_reference = np.asarray(
        full_dataset[
            "y_damage"
        ],
        dtype=np.float64,
    )

    case_id_reference = np.asarray(
        full_dataset[
            "case_id"
        ],
        dtype=np.int64,
    )

    train_idx_reference = np.asarray(
        full_dataset[
            "train_idx"
        ],
        dtype=np.int64,
    )

    val_idx_reference = np.asarray(
        full_dataset[
            "val_idx"
        ],
        dtype=np.int64,
    )

    test_idx_reference = np.asarray(
        full_dataset[
            "test_idx"
        ],
        dtype=np.int64,
    )

    if (
        len(
            train_idx_reference
        )
        != 2100
        or len(
            val_idx_reference
        )
        != 450
        or len(
            test_idx_reference
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
        "EXHAUSTIVE SENSOR-LAYOUT "
        "RBF-SVR EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        "Layouts:",
        len(
            manifest
        ),
    )

    print(
        "SVR candidates per layout:",
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
        "Train / val / test:",
        len(
            train_idx_reference
        ),
        len(
            val_idx_reference
        ),
        len(
            test_idx_reference
        ),
    )

    print(
        "Calibration used:",
        False,
    )

    print(
        "Reference full-layout SVR:",
        {
            "C": reference_C,
            "gamma": reference_gamma,
            "epsilon": (
                reference_epsilon
            ),
        },
    )

    print()

    # ========================================================
    # Evaluate each layout.
    # ========================================================

    grid_rows: list[
        dict[str, Any]
    ] = []

    layout_rows: list[
        dict[str, Any]
    ] = []

    prediction_arrays = []
    prediction_layout_tags = []

    full_model_anchor_passed = False

    for (
        layout_run_index,
        manifest_row,
    ) in manifest.iterrows():

        tag = str(
            manifest_row[
                "layout_tag"
            ]
        )

        layout = (
            layout_tuple_from_tag(
                tag
            )
        )

        dataset_path = resolve_path(
            str(
                manifest_row[
                    "file"
                ]
            ),
            project_root,
        )

        dataset = (
            load_layout_dataset(
                dataset_path
            )
        )

        validate_against_reference(
            dataset,
            full_dataset,
            dataset_path,
        )

        F_raw = np.asarray(
            dataset[
                "F_raw"
            ],
            dtype=np.float64,
        )

        feature_names = [
            str(value)
            for value
            in dataset[
                "feature_names"
            ].tolist()
        ]

        feature_indices = np.asarray(
            dataset[
                "feature_indices_in_full_78d"
            ],
            dtype=np.int64,
        )

        sensor_keep = tuple(
            int(value)
            for value
            in dataset[
                "sensor_keep_1based"
            ].reshape(
                -1
            )
        )

        sensor_count = int(
            dataset[
                "sensor_count"
            ]
        )

        if (
            sensor_keep
            != layout
        ):

            raise RuntimeError(
                "Manifest/dataset layout "
                f"mismatch for {tag}."
            )

        if (
            sensor_count
            != len(
                layout
            )
        ):

            raise RuntimeError(
                "Sensor-count mismatch "
                f"for {tag}."
            )

        expected_feature_count = int(
            manifest_row[
                "feature_count"
            ]
        )

        if (
            F_raw.shape
            != (
                EXPECTED_CASES,
                expected_feature_count,
            )
        ):

            raise RuntimeError(
                f"Layout {tag} feature "
                "shape mismatch."
            )

        if (
            len(
                feature_names
            )
            != expected_feature_count
            or len(
                feature_indices
            )
            != expected_feature_count
        ):

            raise RuntimeError(
                f"Layout {tag} feature "
                "metadata mismatch."
            )

        y_all = np.asarray(
            dataset[
                "y_damage"
            ],
            dtype=np.float64,
        )

        train_idx = np.asarray(
            dataset[
                "train_idx"
            ],
            dtype=np.int64,
        )

        val_idx = np.asarray(
            dataset[
                "val_idx"
            ],
            dtype=np.int64,
        )

        test_idx = np.asarray(
            dataset[
                "test_idx"
            ],
            dtype=np.int64,
        )

        X_train_raw = (
            F_raw[
                train_idx
            ]
        )

        X_val_raw = (
            F_raw[
                val_idx
            ]
        )

        X_test_raw = (
            F_raw[
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

        # ----------------------------------------------------
        # Layout-specific train-only scaler.
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train = (
            scaler.fit_transform(
                X_train_raw
            )
        )

        X_val = (
            scaler.transform(
                X_val_raw
            )
        )

        X_test = (
            scaler.transform(
                X_test_raw
            )
        )

        print(
            "=" * 80
        )

        print(
            f"LAYOUT {tag} "
            f"({layout_run_index + 1}/15)"
        )

        print(
            "=" * 80
        )

        print(
            "Sensors:",
            layout,
        )

        print(
            "Sensor count:",
            sensor_count,
        )

        print(
            "Feature count:",
            expected_feature_count,
        )

        # ----------------------------------------------------
        # Frozen 75-candidate validation search.
        # ----------------------------------------------------

        best = None

        candidate_number = 0

        for C in SVR_C_GRID:

            for gamma in (
                SVR_GAMMA_GRID
            ):

                for epsilon in (
                    SVR_EPSILON_GRID
                ):

                    candidate_number += 1

                    models = (
                        fit_story_svrs(
                            X_train,
                            y_train,
                            C=C,
                            gamma=gamma,
                            epsilon=(
                                epsilon
                            ),
                        )
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

                    grid_rows.append(
                        {
                            "layout_tag": (
                                tag
                            ),
                            "sensor_layout": (
                                ",".join(
                                    str(sensor)
                                    for sensor
                                    in layout
                                )
                            ),
                            "sensor_count": (
                                sensor_count
                            ),
                            "feature_count": (
                                expected_feature_count
                            ),
                            "candidate": (
                                candidate_number
                            ),
                            "C": C,
                            "gamma": (
                                gamma
                            ),
                            "epsilon": (
                                epsilon
                            ),
                            **{
                                f"val_{key}": value
                                for (
                                    key,
                                    value,
                                )
                                in metrics.items()
                            },
                        }
                    )

                    selection_key = (
                        metrics[
                            "mse"
                        ],
                        C,
                        gamma,
                        epsilon,
                    )

                    if (
                        best is None
                        or selection_key
                        < best[
                            "selection_key"
                        ]
                    ):

                        best = {
                            "selection_key": (
                                selection_key
                            ),
                            "C": C,
                            "gamma": gamma,
                            "epsilon": (
                                epsilon
                            ),
                            "validation_metrics": (
                                metrics
                            ),
                        }

        if best is None:

            raise RuntimeError(
                f"SVR search failed "
                f"for layout {tag}."
            )

        selected_C = float(
            best[
                "C"
            ]
        )

        selected_gamma = float(
            best[
                "gamma"
            ]
        )

        selected_epsilon = float(
            best[
                "epsilon"
            ]
        )

        # ----------------------------------------------------
        # Refit selected estimator on TRAIN ONLY,
        # matching the frozen clean benchmark.
        # ----------------------------------------------------

        final_models = (
            fit_story_svrs(
                X_train,
                y_train,
                C=selected_C,
                gamma=(
                    selected_gamma
                ),
                epsilon=(
                    selected_epsilon
                ),
            )
        )

        test_prediction = (
            predict_story_svrs(
                final_models,
                X_test,
            )
        )

        test_metrics = (
            compute_metrics(
                y_test,
                test_prediction,
            )
        )

        prediction_arrays.append(
            test_prediction
        )

        prediction_layout_tags.append(
            tag
        )

        print(
            "Selected:",
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
            "Validation MSE:",
            best[
                "validation_metrics"
            ][
                "mse"
            ],
        )

        print(
            "Test MAE:",
            test_metrics[
                "mae"
            ],
        )

        print(
            "Test high MAE:",
            test_metrics[
                "high_mae"
            ],
        )

        print(
            "Test high bias:",
            test_metrics[
                "high_bias"
            ],
        )

        print(
            "Test high underestimation:",
            test_metrics[
                "high_underestimation"
            ],
        )

        # ----------------------------------------------------
        # Full-layout exact model anchor.
        # ----------------------------------------------------

        if tag == "1234":

            parameter_anchor = bool(
                selected_C
                == reference_C
                and selected_gamma
                == reference_gamma
                and selected_epsilon
                == reference_epsilon
            )

            anchor_metric_names = [
                "mse",
                "mae",
                "rmse",
                "damaged_mae",
                "zero_mae",
                "low_mae",
                "medium_mae",
                "high_mae",
                "high_bias",
                "high_abs_bias",
                "high_underestimation",
            ]

            metric_anchor_checks = {}

            for metric in (
                anchor_metric_names
            ):

                actual = float(
                    test_metrics[
                        metric
                    ]
                )

                expected = float(
                    clean_reference_row[
                        metric
                    ]
                )

                metric_anchor_checks[
                    metric
                ] = {
                    "actual": (
                        actual
                    ),
                    "expected": (
                        expected
                    ),
                    "abs_error": float(
                        abs(
                            actual
                            - expected
                        )
                    ),
                    "passed": (
                        close_enough(
                            actual,
                            expected,
                        )
                    ),
                }

            metric_anchor = bool(
                all(
                    value[
                        "passed"
                    ]
                    for value
                    in (
                        metric_anchor_checks
                        .values()
                    )
                )
            )

            full_model_anchor_passed = bool(
                parameter_anchor
                and metric_anchor
            )

            print()
            print(
                "-" * 80
            )

            print(
                "FULL-LAYOUT MODEL ANCHOR"
            )

            print(
                "-" * 80
            )

            print(
                "Hyperparameters exact:",
                parameter_anchor,
            )

            for (
                metric,
                check,
            ) in (
                metric_anchor_checks
                .items()
            ):

                print(
                    f"{metric}:",
                    "actual=",
                    check[
                        "actual"
                    ],
                    "reference=",
                    check[
                        "expected"
                    ],
                    "abs_error=",
                    check[
                        "abs_error"
                    ],
                    "passed=",
                    check[
                        "passed"
                    ],
                )

            print(
                "FULL MODEL ANCHOR PASSED:",
                full_model_anchor_passed,
            )

            if not full_model_anchor_passed:

                raise RuntimeError(
                    "STOP: full-layout model "
                    "anchor failed. Reduced "
                    "sensor layouts must not "
                    "be interpreted."
                )

        layout_rows.append(
            {
                "layout_tag": (
                    tag
                ),
                "sensor_layout": (
                    ",".join(
                        str(sensor)
                        for sensor
                        in layout
                    )
                ),
                "sensor_count": (
                    sensor_count
                ),
                "feature_count": (
                    expected_feature_count
                ),
                "feature_fraction_of_full": float(
                    expected_feature_count
                    / 78.0
                ),
                "selected_C": (
                    selected_C
                ),
                "selected_gamma": (
                    selected_gamma
                ),
                "selected_epsilon": (
                    selected_epsilon
                ),
                **{
                    f"val_{key}": value
                    for (
                        key,
                        value,
                    )
                    in best[
                        "validation_metrics"
                    ].items()
                },
                **{
                    f"test_{key}": value
                    for (
                        key,
                        value,
                    )
                    in test_metrics.items()
                },
                "dataset_file": str(
                    dataset_path
                ),
            }
        )

        print()

    # ========================================================
    # Save complete candidate table.
    # ========================================================

    grid_frame = pd.DataFrame(
        grid_rows
    )

    grid_path = (
        output_root
        / "all_layout_validation_grid_results.csv"
    )

    grid_frame.to_csv(
        grid_path,
        index=False,
    )

    # ========================================================
    # Layout-level results and ranking.
    # ========================================================

    layout_results = (
        pd.DataFrame(
            layout_rows
        )
    )

    full_result = (
        layout_results[
            layout_results[
                "layout_tag"
            ]
            == "1234"
        ]
        .iloc[0]
    )

    full_mae = float(
        full_result[
            "test_mae"
        ]
    )

    full_high_mae = float(
        full_result[
            "test_high_mae"
        ]
    )

    layout_results[
        "test_mae_ratio_to_full"
    ] = (
        layout_results[
            "test_mae"
        ]
        / full_mae
    )

    layout_results[
        "test_mae_excess_vs_full_percent"
    ] = (
        (
            layout_results[
                "test_mae"
            ]
            - full_mae
        )
        / full_mae
        * 100.0
    )

    layout_results[
        "test_high_mae_ratio_to_full"
    ] = (
        layout_results[
            "test_high_mae"
        ]
        / full_high_mae
    )

    layout_results[
        "test_high_mae_excess_vs_full_percent"
    ] = (
        (
            layout_results[
                "test_high_mae"
            ]
            - full_high_mae
        )
        / full_high_mae
        * 100.0
    )

    layout_results[
        "test_mae_rank"
    ] = (
        layout_results[
            "test_mae"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    layout_results[
        "test_high_mae_rank"
    ] = (
        layout_results[
            "test_high_mae"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    layout_results_path = (
        output_root
        / "sensor_layout_results.csv"
    )

    layout_results.to_csv(
        layout_results_path,
        index=False,
    )

    ranking = (
        layout_results.sort_values(
            [
                "test_mae",
                "sensor_count",
                "layout_tag",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ranking_path = (
        output_root
        / "sensor_layout_ranking.csv"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    # ========================================================
    # Sensor-count aggregation.
    # ========================================================

    count_rows = []

    for (
        sensor_count,
        group,
    ) in (
        layout_results.groupby(
            "sensor_count"
        )
    ):

        group = (
            group.copy()
        )

        best_mae_row = (
            group.loc[
                group[
                    "test_mae"
                ]
                .idxmin()
            ]
        )

        worst_mae_row = (
            group.loc[
                group[
                    "test_mae"
                ]
                .idxmax()
            ]
        )

        best_high_row = (
            group.loc[
                group[
                    "test_high_mae"
                ]
                .idxmin()
            ]
        )

        worst_high_row = (
            group.loc[
                group[
                    "test_high_mae"
                ]
                .idxmax()
            ]
        )

        mae_values = np.asarray(
            group[
                "test_mae"
            ],
            dtype=np.float64,
        )

        high_values = np.asarray(
            group[
                "test_high_mae"
            ],
            dtype=np.float64,
        )

        feature_values = np.asarray(
            group[
                "feature_count"
            ],
            dtype=np.float64,
        )

        count_rows.append(
            {
                "sensor_count": int(
                    sensor_count
                ),
                "n_layouts": int(
                    len(
                        group
                    )
                ),
                "feature_count_mean": float(
                    np.mean(
                        feature_values
                    )
                ),
                "feature_count_min": int(
                    np.min(
                        feature_values
                    )
                ),
                "feature_count_max": int(
                    np.max(
                        feature_values
                    )
                ),
                "test_mae_mean": float(
                    np.mean(
                        mae_values
                    )
                ),
                "test_mae_std": float(
                    np.std(
                        mae_values,
                        ddof=1,
                    )
                    if len(
                        mae_values
                    ) > 1
                    else 0.0
                ),
                "test_mae_min": float(
                    np.min(
                        mae_values
                    )
                ),
                "test_mae_max": float(
                    np.max(
                        mae_values
                    )
                ),
                "test_mae_range": float(
                    np.max(
                        mae_values
                    )
                    - np.min(
                        mae_values
                    )
                ),
                "best_mae_layout": str(
                    best_mae_row[
                        "layout_tag"
                    ]
                ),
                "worst_mae_layout": str(
                    worst_mae_row[
                        "layout_tag"
                    ]
                ),
                "best_mae_ratio_to_full": float(
                    best_mae_row[
                        "test_mae"
                    ]
                    / full_mae
                ),
                "best_mae_excess_vs_full_percent": float(
                    (
                        best_mae_row[
                            "test_mae"
                        ]
                        - full_mae
                    )
                    / full_mae
                    * 100.0
                ),
                "test_high_mae_mean": float(
                    np.mean(
                        high_values
                    )
                ),
                "test_high_mae_std": float(
                    np.std(
                        high_values,
                        ddof=1,
                    )
                    if len(
                        high_values
                    ) > 1
                    else 0.0
                ),
                "test_high_mae_min": float(
                    np.min(
                        high_values
                    )
                ),
                "test_high_mae_max": float(
                    np.max(
                        high_values
                    )
                ),
                "test_high_mae_range": float(
                    np.max(
                        high_values
                    )
                    - np.min(
                        high_values
                    )
                ),
                "best_high_mae_layout": str(
                    best_high_row[
                        "layout_tag"
                    ]
                ),
                "worst_high_mae_layout": str(
                    worst_high_row[
                        "layout_tag"
                    ]
                ),
            }
        )

    count_summary = (
        pd.DataFrame(
            count_rows
        )
        .sort_values(
            "sensor_count"
        )
    )

    count_summary_path = (
        output_root
        / "sensor_count_summary.csv"
    )

    count_summary.to_csv(
        count_summary_path,
        index=False,
    )

    # ========================================================
    # Placement spread.
    # ========================================================

    placement_rows = []

    for _, row in (
        count_summary.iterrows()
    ):

        minimum = float(
            row[
                "test_mae_min"
            ]
        )

        maximum = float(
            row[
                "test_mae_max"
            ]
        )

        range_value = float(
            row[
                "test_mae_range"
            ]
        )

        placement_rows.append(
            {
                "sensor_count": int(
                    row[
                        "sensor_count"
                    ]
                ),
                "n_layouts": int(
                    row[
                        "n_layouts"
                    ]
                ),
                "best_layout": str(
                    row[
                        "best_mae_layout"
                    ]
                ),
                "worst_layout": str(
                    row[
                        "worst_mae_layout"
                    ]
                ),
                "best_test_mae": (
                    minimum
                ),
                "worst_test_mae": (
                    maximum
                ),
                "absolute_placement_spread": (
                    range_value
                ),
                "relative_placement_spread_percent": float(
                    (
                        range_value
                        / minimum
                        * 100.0
                    )
                    if minimum
                    > 1e-15
                    else float(
                        "nan"
                    )
                ),
                "high_mae_spread": float(
                    row[
                        "test_high_mae_range"
                    ]
                ),
            }
        )

    placement_spread = (
        pd.DataFrame(
            placement_rows
        )
    )

    placement_spread_path = (
        output_root
        / "sensor_placement_spread.csv"
    )

    placement_spread.to_csv(
        placement_spread_path,
        index=False,
    )

    # ========================================================
    # Marginal sensor value.
    #
    # E(S) - E(S U {j})
    #
    # Positive = adding sensor j improves performance.
    # ========================================================

    result_lookup = {
        layout_tuple_from_tag(
            str(
                row[
                    "layout_tag"
                ]
            )
        ): row
        for _, row
        in layout_results.iterrows()
    }

    marginal_rows = []

    for (
        base_layout,
        base_row,
    ) in result_lookup.items():

        base_set = set(
            base_layout
        )

        if len(
            base_layout
        ) >= 4:

            continue

        for added_sensor in (
            1,
            2,
            3,
            4,
        ):

            if (
                added_sensor
                in base_set
            ):

                continue

            augmented_layout = tuple(
                sorted(
                    (
                        base_set
                        | {
                            added_sensor,
                        }
                    )
                )
            )

            if (
                augmented_layout
                not in result_lookup
            ):

                raise RuntimeError(
                    "Missing augmented layout: "
                    f"{augmented_layout}"
                )

            augmented_row = (
                result_lookup[
                    augmented_layout
                ]
            )

            base_mae = float(
                base_row[
                    "test_mae"
                ]
            )

            augmented_mae = float(
                augmented_row[
                    "test_mae"
                ]
            )

            base_high_mae = float(
                base_row[
                    "test_high_mae"
                ]
            )

            augmented_high_mae = float(
                augmented_row[
                    "test_high_mae"
                ]
            )

            mae_improvement = (
                base_mae
                - augmented_mae
            )

            high_mae_improvement = (
                base_high_mae
                - augmented_high_mae
            )

            marginal_rows.append(
                {
                    "base_layout": (
                        layout_tag(
                            base_layout
                        )
                    ),
                    "base_sensor_count": (
                        len(
                            base_layout
                        )
                    ),
                    "added_sensor": (
                        added_sensor
                    ),
                    "augmented_layout": (
                        layout_tag(
                            augmented_layout
                        )
                    ),
                    "base_feature_count": int(
                        base_row[
                            "feature_count"
                        ]
                    ),
                    "augmented_feature_count": int(
                        augmented_row[
                            "feature_count"
                        ]
                    ),
                    "feature_gain": int(
                        augmented_row[
                            "feature_count"
                        ]
                        - base_row[
                            "feature_count"
                        ]
                    ),
                    "base_test_mae": (
                        base_mae
                    ),
                    "augmented_test_mae": (
                        augmented_mae
                    ),
                    "mae_improvement": (
                        mae_improvement
                    ),
                    "mae_relative_improvement_percent": float(
                        mae_improvement
                        / base_mae
                        * 100.0
                    ),
                    "base_high_mae": (
                        base_high_mae
                    ),
                    "augmented_high_mae": (
                        augmented_high_mae
                    ),
                    "high_mae_improvement": (
                        high_mae_improvement
                    ),
                    "high_mae_relative_improvement_percent": float(
                        high_mae_improvement
                        / base_high_mae
                        * 100.0
                    ),
                }
            )

    marginal = (
        pd.DataFrame(
            marginal_rows
        )
        .sort_values(
            [
                "base_sensor_count",
                "base_layout",
                "added_sensor",
            ]
        )
    )

    marginal_path = (
        output_root
        / "marginal_sensor_value.csv"
    )

    marginal.to_csv(
        marginal_path,
        index=False,
    )

    marginal_summary_rows = []

    for (
        added_sensor,
        group,
    ) in marginal.groupby(
        "added_sensor"
    ):

        values = np.asarray(
            group[
                "mae_relative_improvement_percent"
            ],
            dtype=np.float64,
        )

        high_values = np.asarray(
            group[
                "high_mae_relative_improvement_percent"
            ],
            dtype=np.float64,
        )

        marginal_summary_rows.append(
            {
                "added_sensor": int(
                    added_sensor
                ),
                "n_base_layouts": int(
                    len(
                        group
                    )
                ),
                "mae_relative_improvement_mean_percent": float(
                    np.mean(
                        values
                    )
                ),
                "mae_relative_improvement_median_percent": float(
                    np.median(
                        values
                    )
                ),
                "mae_relative_improvement_min_percent": float(
                    np.min(
                        values
                    )
                ),
                "mae_relative_improvement_max_percent": float(
                    np.max(
                        values
                    )
                ),
                "mae_positive_improvement_fraction": float(
                    np.mean(
                        values > 0.0
                    )
                ),
                "high_mae_relative_improvement_mean_percent": float(
                    np.mean(
                        high_values
                    )
                ),
                "high_mae_positive_improvement_fraction": float(
                    np.mean(
                        high_values > 0.0
                    )
                ),
            }
        )

    marginal_summary = (
        pd.DataFrame(
            marginal_summary_rows
        )
        .sort_values(
            "added_sensor"
        )
    )

    marginal_summary_path = (
        output_root
        / "marginal_sensor_value_summary.csv"
    )

    marginal_summary.to_csv(
        marginal_summary_path,
        index=False,
    )

    # ========================================================
    # Save predictions for future paired diagnostics.
    # ========================================================

    predictions = np.stack(
        prediction_arrays,
        axis=0,
    )

    if (
        predictions.shape
        != (
            EXPECTED_LAYOUTS,
            450,
            EXPECTED_STORIES,
        )
    ):

        raise RuntimeError(
            "Unexpected stacked "
            "prediction shape."
        )

    predictions_path = (
        output_root
        / "sensor_layout_test_predictions.npz"
    )

    np.savez_compressed(
        predictions_path,
        prediction=(
            predictions
        ),
        layout_tag=np.asarray(
            prediction_layout_tags
        ),
        y_test=(
            y_all_reference[
                test_idx_reference
            ]
        ),
        test_case_id=(
            case_id_reference[
                test_idx_reference
            ]
        ),
    )

    # ========================================================
    # Experiment integrity.
    # ========================================================

    expected_grid_rows = (
        EXPECTED_LAYOUTS
        * len(
            SVR_C_GRID
        )
        * len(
            SVR_GAMMA_GRID
        )
        * len(
            SVR_EPSILON_GRID
        )
    )

    grid_row_count_ok = bool(
        len(
            grid_frame
        )
        == expected_grid_rows
    )

    layout_row_count_ok = bool(
        len(
            layout_results
        )
        == EXPECTED_LAYOUTS
    )

    metrics_finite = bool(
        np.isfinite(
            layout_results[
                [
                    "test_mse",
                    "test_mae",
                    "test_rmse",
                    "test_damaged_mae",
                    "test_zero_mae",
                    "test_low_mae",
                    "test_medium_mae",
                    "test_high_mae",
                    "test_high_bias",
                    "test_high_abs_bias",
                    "test_high_underestimation",
                ]
            ].to_numpy(
                dtype=np.float64,
            )
        ).all()
    )

    marginal_edge_count_ok = bool(
        len(
            marginal
        )
        == 28
    )

    overall_passed = bool(
        full_model_anchor_passed
        and grid_row_count_ok
        and layout_row_count_ok
        and metrics_finite
        and marginal_edge_count_ok
    )

    # ========================================================
    # Console summaries.
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "LAYOUT TEST RESULTS"
    )

    print(
        "=" * 80
    )

    compact_columns = [
        "layout_tag",
        "sensor_count",
        "feature_count",
        "selected_C",
        "selected_gamma",
        "selected_epsilon",
        "test_mae",
        "test_high_mae",
        "test_high_bias",
        "test_high_underestimation",
        "test_mae_excess_vs_full_percent",
    ]

    print(
        layout_results[
            compact_columns
        ]
        .sort_values(
            [
                "sensor_count",
                "test_mae",
            ]
        )
        .to_string(
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
        "OVERALL LAYOUT RANKING"
    )

    print(
        "=" * 80
    )

    print(
        ranking[
            [
                "test_mae_rank",
                "layout_tag",
                "sensor_count",
                "feature_count",
                "test_mae",
                "test_high_mae",
                "test_mae_excess_vs_full_percent",
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
        "SENSOR-COUNT SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        count_summary.to_string(
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
        "PLACEMENT SPREAD"
    )

    print(
        "=" * 80
    )

    print(
        placement_spread.to_string(
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
        "MARGINAL SENSOR VALUE SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        marginal_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    # ========================================================
    # Save report.
    # ========================================================

    report = {
        "experiment": (
            "exhaustive_dependency_aware_"
            "sensor_layout_rbf_svr"
        ),
        "n_layouts": (
            EXPECTED_LAYOUTS
        ),
        "structural_sensor_layouts": [
            str(
                tag
            )
            for tag in (
                layout_results[
                    "layout_tag"
                ]
                .tolist()
            )
        ],
        "ground_sensor_assumed_available": (
            True
        ),
        "zero_masking_used": (
            False
        ),
        "descriptor_redefinition_used": (
            False
        ),
        "calibration_used": (
            False
        ),
        "split": {
            "train": 2100,
            "validation": 450,
            "test": 450,
        },
        "svr_grid": {
            "C": (
                SVR_C_GRID
            ),
            "gamma": (
                SVR_GAMMA_GRID
            ),
            "epsilon": (
                SVR_EPSILON_GRID
            ),
            "candidate_count": int(
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
            "selection_metric": (
                "validation MSE"
            ),
        },
        "full_layout_model_anchor": {
            "reference_C": (
                reference_C
            ),
            "reference_gamma": (
                reference_gamma
            ),
            "reference_epsilon": (
                reference_epsilon
            ),
            "passed": (
                full_model_anchor_passed
            ),
        },
        "grid_row_count_expected": (
            expected_grid_rows
        ),
        "grid_row_count_actual": int(
            len(
                grid_frame
            )
        ),
        "grid_row_count_ok": (
            grid_row_count_ok
        ),
        "layout_row_count_ok": (
            layout_row_count_ok
        ),
        "metrics_finite": (
            metrics_finite
        ),
        "marginal_edge_count_expected": (
            28
        ),
        "marginal_edge_count_actual": int(
            len(
                marginal
            )
        ),
        "marginal_edge_count_ok": (
            marginal_edge_count_ok
        ),
        "overall_passed": (
            overall_passed
        ),
        "outputs": {
            "validation_grid": str(
                grid_path
            ),
            "layout_results": str(
                layout_results_path
            ),
            "ranking": str(
                ranking_path
            ),
            "sensor_count_summary": str(
                count_summary_path
            ),
            "placement_spread": str(
                placement_spread_path
            ),
            "marginal_sensor_value": str(
                marginal_path
            ),
            "marginal_sensor_value_summary": str(
                marginal_summary_path
            ),
            "test_predictions": str(
                predictions_path
            ),
        },
    }

    report_path = (
        output_root
        / "exhaustive_sensor_layout_svr_report.json"
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
        "FINAL SENSOR-LAYOUT CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Full-layout model anchor passed:",
        full_model_anchor_passed,
    )

    print(
        "Grid rows:",
        len(
            grid_frame
        ),
        "/",
        expected_grid_rows,
    )

    print(
        "Grid row count correct:",
        grid_row_count_ok,
    )

    print(
        "Layout result count correct:",
        layout_row_count_ok,
    )

    print(
        "All metrics finite:",
        metrics_finite,
    )

    print(
        "Marginal subset edges:",
        len(
            marginal
        ),
        "/ 28",
    )

    print(
        "Marginal edge count correct:",
        marginal_edge_count_ok,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    if overall_passed:

        print(
            "CHECK PASSED: exhaustive "
            "dependency-aware sensor-layout "
            "SVR experiment completed."
        )

        print(
            "Sensor-count, placement, and "
            "marginal-value results may "
            "now be interpreted."
        )

    else:

        print(
            "CHECK FAILED: do not "
            "scientifically interpret "
            "sensor-layout results."
        )

    print()

    print(
        "Layout results:",
        layout_results_path,
    )

    print(
        "Sensor-count summary:",
        count_summary_path,
    )

    print(
        "Placement spread:",
        placement_spread_path,
    )

    print(
        "Marginal summary:",
        marginal_summary_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
