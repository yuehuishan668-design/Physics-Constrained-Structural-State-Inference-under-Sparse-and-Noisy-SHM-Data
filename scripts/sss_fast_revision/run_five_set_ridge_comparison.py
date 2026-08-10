"""
Unified fixed-split Ridge comparison across five descriptor sets.

Descriptor sets
---------------
1. oracle_full
2. legacy_no_meta
3. measured_ground_augmented
4. signal_derived_with_ground
5. structural_response_only

All models use:
- identical frozen train/validation/test partitions;
- identical standardised feature protocol;
- identical Ridge alpha search grid;
- clipped validation/test predictions in [0, 0.5];
- validation MSE for hyperparameter selection.

五套描述符统一Ridge对照实验。
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


from scripts.sss_fast_revision.run_fixed_split_ridge_baseline import (  # noqa: E402
    calculate_metrics,
    clip_damage_prediction,
    confirm_shared_partitions,
    load_dataset,
    mean_absolute_error,
    mean_squared_error,
)


DATASETS = {
    "oracle_full": (
        "oracle_full/"
        "debug_plus_3000_oracle_full_features.npz"
    ),
    "legacy_no_meta": (
        "legacy_no_meta/"
        "debug_plus_3000_legacy_no_meta_features.npz"
    ),
    "measured_ground_augmented": (
        "measured_ground_augmented/"
        "debug_plus_3000_"
        "measured_ground_augmented_features.npz"
    ),
    "signal_derived_with_ground": (
        "signal_derived_with_ground/"
        "debug_plus_3000_"
        "signal_derived_with_ground_features.npz"
    ),
    "structural_response_only": (
        "structural_response_only/"
        "debug_plus_3000_"
        "structural_response_only_features.npz"
    ),
}


EXPECTED_FEATURE_COUNTS = {
    "oracle_full": 92,
    "legacy_no_meta": 86,
    "measured_ground_augmented": 86,
    "signal_derived_with_ground": 78,
    "structural_response_only": 59,
}


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


def train_one_set(
    descriptor_set: str,
    arrays: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
]:
    """Train all Ridge candidates and select using validation MSE."""
    x_train = arrays["F_train"]
    x_val = arrays["F_val"]
    x_test = arrays["F_test"]

    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]

    expected_count = EXPECTED_FEATURE_COUNTS[
        descriptor_set
    ]

    if x_train.shape[1] != expected_count:
        raise ValueError(
            f"{descriptor_set}: expected "
            f"{expected_count} features, found "
            f"{x_train.shape[1]}."
        )

    candidate_rows: list[dict[str, Any]] = []
    models: dict[float, Ridge] = {}

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
                "n_features": expected_count,
                "alpha": float(alpha),
                "val_mse": val_mse,
                "val_mae": val_mae,
                "selected": False,
            }
        )

        models[float(alpha)] = model

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

    selected_model = models[selected_alpha]

    test_prediction = clip_damage_prediction(
        selected_model.predict(x_test)
    )

    metrics = calculate_metrics(
        y_test,
        test_prediction,
    )

    summary = {
        "descriptor_set": descriptor_set,
        "n_features": expected_count,
        "selected_alpha": selected_alpha,
        "alpha_at_lower_boundary": (
            selected_alpha == min(RIDGE_ALPHAS)
        ),
        "alpha_at_upper_boundary": (
            selected_alpha == max(RIDGE_ALPHAS)
        ),
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


def add_relative_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add relative performance differences to key reference sets."""
    frame = frame.copy()

    indexed = frame.set_index(
        "descriptor_set"
    )

    reference_names = [
        "oracle_full",
        "legacy_no_meta",
        "signal_derived_with_ground",
        "measured_ground_augmented",
    ]

    for reference_name in reference_names:
        if reference_name not in indexed.index:
            raise KeyError(
                f"Missing reference set: {reference_name}"
            )

        reference_mae = float(
            indexed.loc[
                reference_name,
                "test_mae",
            ]
        )

        column_name = (
            f"test_mae_change_vs_"
            f"{reference_name}_percent"
        )

        frame[column_name] = (
            (
                frame["test_mae"]
                - reference_mae
            )
            / reference_mae
            * 100.0
        )

    return frame


def build_pairwise_comparison(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build scientifically relevant pairwise comparisons.

    Positive MAE improvement means the first configuration is better.
    """
    indexed = frame.set_index(
        "descriptor_set"
    )

    comparisons = [
        (
            "measured_ground_augmented",
            "signal_derived_with_ground",
            "Does measured-ground normalisation improve the 78D set?",
        ),
        (
            "measured_ground_augmented",
            "legacy_no_meta",
            "Can measured-ground features recover the legacy 86D result?",
        ),
        (
            "measured_ground_augmented",
            "oracle_full",
            "How far is the measured-ground set from the oracle upper bound?",
        ),
        (
            "signal_derived_with_ground",
            "structural_response_only",
            "What is the value of measured ground-input information?",
        ),
    ]

    rows = []

    metrics = [
        "test_mae",
        "test_rmse",
        "damaged_entry_mae",
        "high_damage_mae",
        "high_damage_bias",
        "high_damage_underestimation_ratio",
    ]

    for first, second, question in comparisons:
        first_row = indexed.loc[first]
        second_row = indexed.loc[second]

        result = {
            "configuration_a": first,
            "configuration_b": second,
            "scientific_question": question,
        }

        for metric in metrics:
            a = float(first_row[metric])
            b = float(second_row[metric])

            result[f"{metric}_a"] = a
            result[f"{metric}_b"] = b
            result[f"{metric}_a_minus_b"] = (
                a - b
            )

            if b != 0.0:
                result[
                    f"{metric}_a_minus_b_percent"
                ] = (
                    (a - b)
                    / abs(b)
                    * 100.0
                )
            else:
                result[
                    f"{metric}_a_minus_b_percent"
                ] = float("nan")

        rows.append(result)

    return pd.DataFrame(rows)


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
            "five_set_ridge_comparison"
        ),
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    reference_arrays: dict[
        str,
        np.ndarray,
    ] | None = None

    for descriptor_set, relative_path in (
        DATASETS.items()
    ):
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
            candidate_rows,
            prediction,
        ) = train_one_set(
            descriptor_set=descriptor_set,
            arrays=arrays,
        )

        summaries.append(summary)
        candidates.extend(candidate_rows)

        np.savez_compressed(
            args.output_root
            / f"{descriptor_set}_test_predictions.npz",
            y_test=arrays["y_test"],
            y_prediction=prediction,
            descriptor_set=np.asarray(
                descriptor_set
            ),
            selected_alpha=np.asarray(
                summary["selected_alpha"],
                dtype=np.float64,
            ),
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
        )

    summary_frame = pd.DataFrame(
        summaries
    )

    summary_frame = add_relative_metrics(
        summary_frame
    )

    summary_frame = summary_frame.sort_values(
        "test_mae",
        ascending=True,
    ).reset_index(drop=True)

    candidate_frame = pd.DataFrame(
        candidates
    ).sort_values(
        [
            "descriptor_set",
            "alpha",
        ]
    ).reset_index(drop=True)

    pairwise_frame = (
        build_pairwise_comparison(
            summary_frame
        )
    )

    summary_path = (
        args.output_root
        / "five_set_ridge_summary.csv"
    )

    candidate_path = (
        args.output_root
        / "five_set_ridge_candidates.csv"
    )

    pairwise_path = (
        args.output_root
        / "five_set_pairwise_comparison.csv"
    )

    report_path = (
        args.output_root
        / "five_set_ridge_report.json"
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    pairwise_frame.to_csv(
        pairwise_path,
        index=False,
    )

    measured = summary_frame.loc[
        summary_frame["descriptor_set"]
        == "measured_ground_augmented"
    ].iloc[0]

    signal = summary_frame.loc[
        summary_frame["descriptor_set"]
        == "signal_derived_with_ground"
    ].iloc[0]

    legacy = summary_frame.loc[
        summary_frame["descriptor_set"]
        == "legacy_no_meta"
    ].iloc[0]

    oracle = summary_frame.loc[
        summary_frame["descriptor_set"]
        == "oracle_full"
    ].iloc[0]

    measured_vs_78_percent = (
        (
            float(measured["test_mae"])
            - float(signal["test_mae"])
        )
        / float(signal["test_mae"])
        * 100.0
    )

    measured_vs_legacy_percent = (
        (
            float(measured["test_mae"])
            - float(legacy["test_mae"])
        )
        / float(legacy["test_mae"])
        * 100.0
    )

    measured_vs_oracle_percent = (
        (
            float(measured["test_mae"])
            - float(oracle["test_mae"])
        )
        / float(oracle["test_mae"])
        * 100.0
    )

    if measured_vs_78_percent < -2.0:
        augmentation_interpretation = (
            "meaningful_improvement"
        )
    elif measured_vs_78_percent < -0.5:
        augmentation_interpretation = (
            "modest_improvement"
        )
    elif measured_vs_78_percent <= 0.5:
        augmentation_interpretation = (
            "negligible_change"
        )
    else:
        augmentation_interpretation = (
            "performance_degradation"
        )

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
            "dataset_split": (
                "frozen 2100/450/450"
            ),
        },
        "feature_counts": (
            EXPECTED_FEATURE_COUNTS
        ),
        "measured_ground_augmented_analysis": {
            "test_mae_change_vs_78_percent": (
                measured_vs_78_percent
            ),
            "test_mae_change_vs_legacy_percent": (
                measured_vs_legacy_percent
            ),
            "test_mae_change_vs_oracle_percent": (
                measured_vs_oracle_percent
            ),
            "augmentation_interpretation": (
                augmentation_interpretation
            ),
        },
        "lower_boundary_sets": (
            summary_frame.loc[
                summary_frame[
                    "alpha_at_lower_boundary"
                ],
                "descriptor_set",
            ].tolist()
        ),
        "upper_boundary_sets": (
            summary_frame.loc[
                summary_frame[
                    "alpha_at_upper_boundary"
                ],
                "descriptor_set",
            ].tolist()
        ),
        "summary_csv": str(
            summary_path
        ),
        "candidate_csv": str(
            candidate_path
        ),
        "pairwise_csv": str(
            pairwise_path
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

    display_columns = [
        "descriptor_set",
        "n_features",
        "selected_alpha",
        "val_mse",
        "test_mae",
        "test_rmse",
        "damaged_entry_mae",
        "high_damage_mae",
        "high_damage_bias",
        "high_damage_underestimation_ratio",
        "test_mae_change_vs_oracle_full_percent",
    ]

    print(
        "===== FIVE-SET RIDGE RESULTS ====="
    )

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
        "===== MEASURED-GROUND AUGMENTATION ====="
    )

    print(
        "Measured-ground vs 78D Test MAE change (%):",
        f"{measured_vs_78_percent:.6f}",
    )

    print(
        "Measured-ground vs legacy 86D Test MAE change (%):",
        f"{measured_vs_legacy_percent:.6f}",
    )

    print(
        "Measured-ground vs oracle Test MAE change (%):",
        f"{measured_vs_oracle_percent:.6f}",
    )

    print(
        "Interpretation:",
        augmentation_interpretation,
    )

    print()
    print(
        "===== PAIRWISE TEST-MAE COMPARISON ====="
    )

    print(
        pairwise_frame[
            [
                "configuration_a",
                "configuration_b",
                "test_mae_a",
                "test_mae_b",
                "test_mae_a_minus_b",
                "test_mae_a_minus_b_percent",
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
        "Lower-boundary alpha selections:",
        report["lower_boundary_sets"],
    )

    print(
        "Upper-boundary alpha selections:",
        report["upper_boundary_sets"],
    )

    print()
    print(
        "CHECK PASSED: five-set Ridge comparison completed."
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Pairwise:",
        pairwise_path,
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
