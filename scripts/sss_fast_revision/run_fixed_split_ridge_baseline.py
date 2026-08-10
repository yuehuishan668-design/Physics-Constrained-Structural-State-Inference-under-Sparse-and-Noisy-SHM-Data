"""
Reproduce the frozen fixed-split Ridge baseline and evaluate the
new canonical descriptor sets.

Protocol
--------
1. Use the frozen 2100/450/450 train/validation/test split.
2. Train Ridge on the standardised descriptor matrices.
3. Select alpha using clipped validation MSE.
4. Clip validation and test predictions to [0, 0.5].
5. Evaluate overall, damaged-entry and high-damage performance.

复现AES固定划分Ridge基线，并评估SSS修订版的四套描述符。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DESCRIPTOR_DATASETS = {
    "oracle_full": (
        "oracle_full/"
        "debug_plus_3000_oracle_full_features.npz"
    ),
    "legacy_no_meta": (
        "legacy_no_meta/"
        "debug_plus_3000_legacy_no_meta_features.npz"
    ),
    "signal_derived_with_ground": (
        "signal_derived_with_ground/"
        "debug_plus_3000_signal_derived_with_ground_features.npz"
    ),
    "structural_response_only": (
        "structural_response_only/"
        "debug_plus_3000_structural_response_only_features.npz"
    ),
}


# Frozen fixed-split hyperparameter grid.
#
# 原固定划分实验使用的Ridge正则化参数候选。
RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    300.0,
]


# Published four-decimal values used only as a reproduction check.
#
# 仅用于验证新数据构建流程是否复现旧结果。
CANONICAL_DISPLAY_VALUES = {
    "oracle_full": {
        "test_mae": 0.0393,
        "test_rmse": 0.0585,
        "damaged_entry_mae": 0.0634,
    },
    "legacy_no_meta": {
        "test_mae": 0.0425,
        "test_rmse": 0.0617,
        "damaged_entry_mae": 0.0697,
    },
}


def clip_damage_prediction(
    prediction: np.ndarray,
) -> np.ndarray:
    """Clip damage predictions to the physical interval [0, 0.5]."""
    return np.clip(
        np.asarray(prediction, dtype=np.float64),
        0.0,
        0.5,
    )


def mean_absolute_error(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate MAE over all supplied entries."""
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction)
                - np.asarray(truth)
            )
        )
    )


def mean_squared_error(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate MSE over all supplied entries."""
    error = (
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(truth, dtype=np.float64)
    )

    return float(np.mean(error ** 2))


def root_mean_squared_error(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate RMSE over all supplied entries."""
    return float(
        np.sqrt(
            mean_squared_error(
                truth,
                prediction,
            )
        )
    )


def masked_mae(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Calculate MAE over a Boolean subset."""
    mask = np.asarray(mask, dtype=bool)

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                prediction[mask]
                - truth[mask]
            )
        )
    )


def masked_bias(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Calculate mean signed error, prediction minus truth.

    负值表示系统性低估。
    """
    mask = np.asarray(mask, dtype=bool)

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            prediction[mask]
            - truth[mask]
        )
    )


def underestimation_ratio(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Calculate the proportion of selected entries with prediction < truth.

    Exact equality is not counted as underestimation.
    """
    mask = np.asarray(mask, dtype=bool)

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            prediction[mask]
            < truth[mask]
        )
    )


def calculate_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Calculate aggregate and damage-stratified metrics."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if truth.shape != prediction.shape:
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"truth={truth.shape}, prediction={prediction.shape}"
        )

    damaged_mask = truth > 1.0e-12
    high_mask = truth > 0.20
    zero_mask = truth <= 1.0e-12

    return {
        "test_mae": mean_absolute_error(
            truth,
            prediction,
        ),
        "test_rmse": root_mean_squared_error(
            truth,
            prediction,
        ),
        "test_bias": float(
            np.mean(
                prediction - truth
            )
        ),
        "damaged_entry_mae": masked_mae(
            truth,
            prediction,
            damaged_mask,
        ),
        "zero_damage_mae": masked_mae(
            truth,
            prediction,
            zero_mask,
        ),
        "high_damage_mae": masked_mae(
            truth,
            prediction,
            high_mask,
        ),
        "high_damage_bias": masked_bias(
            truth,
            prediction,
            high_mask,
        ),
        "high_damage_underestimation_ratio": (
            underestimation_ratio(
                truth,
                prediction,
                high_mask,
            )
        ),
    }


def load_dataset(
    path: Path,
) -> dict[str, np.ndarray]:
    """Load and validate one descriptor NPZ."""
    required = {
        "F_train",
        "F_val",
        "F_test",
        "y_train",
        "y_val",
        "y_test",
        "feature_names",
    }

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as loaded:
        arrays = {
            key: loaded[key]
            for key in loaded.files
        }

    missing = sorted(
        required - set(arrays)
    )

    if missing:
        raise KeyError(
            f"{path} is missing keys: {missing}"
        )

    for key in [
        "F_train",
        "F_val",
        "F_test",
        "y_train",
        "y_val",
        "y_test",
    ]:
        arrays[key] = np.asarray(
            arrays[key],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(arrays[key])
        ):
            raise FloatingPointError(
                f"{path}: {key} contains NaN or infinity."
            )

    if arrays["F_train"].shape[0] != arrays["y_train"].shape[0]:
        raise ValueError(
            f"{path}: train sample count mismatch."
        )

    if arrays["F_val"].shape[0] != arrays["y_val"].shape[0]:
        raise ValueError(
            f"{path}: validation sample count mismatch."
        )

    if arrays["F_test"].shape[0] != arrays["y_test"].shape[0]:
        raise ValueError(
            f"{path}: test sample count mismatch."
        )

    return arrays


def confirm_shared_partitions(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    candidate_name: str,
) -> None:
    """
    Ensure that all descriptor sets use identical labels and partitions.

    保证四套描述符使用完全相同的目标和数据划分。
    """
    keys = [
        "y_train",
        "y_val",
        "y_test",
        "train_idx",
        "val_idx",
        "test_idx",
        "case_id_train",
        "case_id_val",
        "case_id_test",
    ]

    for key in keys:
        if key not in reference or key not in candidate:
            continue

        if not np.array_equal(
            reference[key],
            candidate[key],
        ):
            raise ValueError(
                f"{candidate_name}: shared array differs: {key}"
            )


def train_one_descriptor_set(
    descriptor_set: str,
    arrays: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
]:
    """Select Ridge alpha on validation MSE and evaluate test data."""
    x_train = arrays["F_train"]
    x_val = arrays["F_val"]
    x_test = arrays["F_test"]

    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]

    candidate_rows: list[dict[str, Any]] = []
    candidate_models: dict[float, Ridge] = {}

    for alpha in RIDGE_ALPHAS:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="auto",
        )

        model.fit(
            x_train,
            y_train,
        )

        val_prediction = clip_damage_prediction(
            model.predict(x_val)
        )

        val_mse = mean_squared_error(
            y_val,
            val_prediction,
        )

        val_mae = mean_absolute_error(
            y_val,
            val_prediction,
        )

        candidate_rows.append(
            {
                "descriptor_set": descriptor_set,
                "n_features": int(
                    x_train.shape[1]
                ),
                "alpha": float(alpha),
                "val_mse": val_mse,
                "val_mae": val_mae,
                "selected": False,
            }
        )

        candidate_models[float(alpha)] = model

    selected_row = min(
        candidate_rows,
        key=lambda row: (
            row["val_mse"],
            row["alpha"],
        ),
    )

    selected_alpha = float(
        selected_row["alpha"]
    )

    for row in candidate_rows:
        row["selected"] = (
            float(row["alpha"])
            == selected_alpha
        )

    selected_model = candidate_models[
        selected_alpha
    ]

    test_prediction = clip_damage_prediction(
        selected_model.predict(x_test)
    )

    metrics = calculate_metrics(
        y_test,
        test_prediction,
    )

    summary = {
        "descriptor_set": descriptor_set,
        "n_features": int(
            x_train.shape[1]
        ),
        "selected_alpha": selected_alpha,
        "val_mse": float(
            selected_row["val_mse"]
        ),
        "val_mae": float(
            selected_row["val_mae"]
        ),
        **metrics,
    }

    return (
        summary,
        candidate_rows,
        test_prediction,
    )


def matches_four_decimal_display(
    actual: float,
    expected: float,
) -> bool:
    """
    Compare values as published to four decimal places.

    按论文表格的四位小数精度核对。
    """
    return (
        f"{actual:.4f}"
        == f"{expected:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data_processed/sss_fast_revision/"
            "descriptor_sets"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/sss_fast_revision/"
            "fixed_split_ridge"
        ),
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []

    reference_arrays: dict[str, np.ndarray] | None = None

    for descriptor_set, relative_path in DESCRIPTOR_DATASETS.items():
        dataset_path = (
            args.dataset_root
            / relative_path
        )

        arrays = load_dataset(
            dataset_path
        )

        if reference_arrays is None:
            reference_arrays = arrays
        else:
            confirm_shared_partitions(
                reference=reference_arrays,
                candidate=arrays,
                candidate_name=descriptor_set,
            )

        (
            summary,
            candidates,
            prediction,
        ) = train_one_descriptor_set(
            descriptor_set=descriptor_set,
            arrays=arrays,
        )

        summaries.append(summary)
        all_candidates.extend(candidates)

        prediction_path = (
            args.output_root
            / f"{descriptor_set}_test_predictions.npz"
        )

        np.savez_compressed(
            prediction_path,
            y_test=arrays["y_test"],
            y_prediction=prediction,
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
            selected_alpha=np.asarray(
                summary["selected_alpha"]
            ),
            descriptor_set=np.asarray(
                descriptor_set
            ),
        )

    summary_frame = pd.DataFrame(
        summaries
    )

    oracle_mae = float(
        summary_frame.loc[
            summary_frame["descriptor_set"]
            == "oracle_full",
            "test_mae",
        ].iloc[0]
    )

    summary_frame[
        "mae_increase_vs_oracle_percent"
    ] = (
        (
            summary_frame["test_mae"]
            - oracle_mae
        )
        / oracle_mae
        * 100.0
    )

    summary_frame = summary_frame.sort_values(
        "test_mae",
        ascending=True,
    ).reset_index(drop=True)

    candidate_frame = pd.DataFrame(
        all_candidates
    ).sort_values(
        [
            "descriptor_set",
            "alpha",
        ]
    )

    summary_path = (
        args.output_root
        / "fixed_split_ridge_summary.csv"
    )

    candidate_path = (
        args.output_root
        / "fixed_split_ridge_candidates.csv"
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    reproduction_checks: dict[str, Any] = {}
    reproduction_passed = True

    indexed_summary = summary_frame.set_index(
        "descriptor_set"
    )

    for descriptor_set, expected_metrics in (
        CANONICAL_DISPLAY_VALUES.items()
    ):
        actual_row = indexed_summary.loc[
            descriptor_set
        ]

        metric_checks = {}

        for metric_name, expected_value in (
            expected_metrics.items()
        ):
            actual_value = float(
                actual_row[metric_name]
            )

            passed = (
                matches_four_decimal_display(
                    actual=actual_value,
                    expected=expected_value,
                )
            )

            metric_checks[metric_name] = {
                "actual_full_precision": actual_value,
                "actual_four_decimals": (
                    f"{actual_value:.4f}"
                ),
                "expected_four_decimals": (
                    f"{expected_value:.4f}"
                ),
                "passed": passed,
            }

            reproduction_passed = (
                reproduction_passed
                and passed
            )

        reproduction_checks[
            descriptor_set
        ] = metric_checks

    report = {
        "protocol": {
            "estimator": "Ridge",
            "alpha_candidates": RIDGE_ALPHAS,
            "selection_criterion": (
                "clipped validation MSE"
            ),
            "prediction_clip": [
                0.0,
                0.5,
            ],
            "fit_data": "training split only",
            "test_evaluations_per_configuration": 1,
        },
        "reproduction_passed": reproduction_passed,
        "reproduction_checks": reproduction_checks,
        "summary_csv": str(summary_path),
        "candidate_csv": str(candidate_path),
    }

    report_path = (
        args.output_root
        / "fixed_split_ridge_reproduction_report.json"
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

    print("===== FIXED-SPLIT RIDGE RESULTS =====")
    print(
        summary_frame[
            [
                "descriptor_set",
                "n_features",
                "selected_alpha",
                "test_mae",
                "test_rmse",
                "damaged_entry_mae",
                "high_damage_mae",
                "high_damage_bias",
                "high_damage_underestimation_ratio",
                "mae_increase_vs_oracle_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print("===== REPRODUCTION CHECK =====")

    for descriptor_set, metric_checks in (
        reproduction_checks.items()
    ):
        print()
        print(descriptor_set)

        for metric_name, check in (
            metric_checks.items()
        ):
            print(
                f"  {metric_name:24s} "
                f"actual={check['actual_four_decimals']} "
                f"expected={check['expected_four_decimals']} "
                f"passed={check['passed']}"
            )

    print()
    print(
        "REPRODUCTION PASSED:",
        reproduction_passed,
    )
    print("Summary:", summary_path)
    print("Candidates:", candidate_path)
    print("Report:", report_path)

    if not reproduction_passed:
        raise SystemExit(
            "STOP: the canonical 92- or 86-dimensional "
            "fixed-split result was not reproduced."
        )


if __name__ == "__main__":
    main()
