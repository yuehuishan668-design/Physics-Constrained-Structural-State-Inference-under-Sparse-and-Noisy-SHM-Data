"""
Paired case-level bootstrap statistical closure for the exhaustive
dependency-aware sensor-layout experiment.

NO training.
NO hyperparameter tuning.
NO descriptor modification.
NO new test data.

Statistical unit
----------------
The bootstrap resamples the 450 complete structural cases, not individual
story outputs. Each selected case therefore retains its four correlated
storey-level targets/predictions.

Pairing
-------
Every sensor layout uses exactly the same bootstrap-resampled cases in
each replicate. Layout contrasts are therefore paired.

Interpretation
--------------
The resulting intervals quantify case-resampling uncertainty conditional
on the current simulated 450-case fixed test set and the already-frozen
models. They are not independent-structure or physical-experiment
confidence intervals.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260810
CI_LEVEL = 0.95

HIGH_DAMAGE_THRESHOLD = 0.20

EXPECTED_LAYOUTS = 15
EXPECTED_CASES = 450
EXPECTED_STORIES = 4

ANCHOR_ATOL = 1e-12
ANCHOR_RTOL = 1e-12


def percentile_ci(
    values: np.ndarray,
    level: float = CI_LEVEL,
) -> tuple[float, float]:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    alpha = 1.0 - level

    lower = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


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


def tag_to_tuple(
    tag: str,
) -> tuple[int, ...]:

    return tuple(
        int(character)
        for character in str(tag)
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sensor-result-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "exhaustive_sensor_layout_svr"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "sensor_layout_paired_bootstrap_closure"
        ),
    )

    args = parser.parse_args()

    sensor_result_root = (
        args.sensor_result_root
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
        sensor_result_root
        / "sensor_layout_test_predictions.npz"
    )

    layout_results_path = (
        sensor_result_root
        / "sensor_layout_results.csv"
    )

    if not predictions_path.is_file():

        raise FileNotFoundError(
            predictions_path
        )

    if not layout_results_path.is_file():

        raise FileNotFoundError(
            layout_results_path
        )

    # ========================================================
    # Load frozen predictions/results.
    # ========================================================

    with np.load(
        predictions_path,
        allow_pickle=False,
    ) as data:

        prediction = np.asarray(
            data[
                "prediction"
            ],
            dtype=np.float64,
        )

        prediction_tags = [
            str(value)
            for value in data[
                "layout_tag"
            ].tolist()
        ]

        y_test = np.asarray(
            data[
                "y_test"
            ],
            dtype=np.float64,
        )

        test_case_id = np.asarray(
            data[
                "test_case_id"
            ]
        )

    layout_results = pd.read_csv(
        layout_results_path,
        dtype={
            "layout_tag": str,
            "sensor_layout": str,
        },
    )

    if (
        prediction.shape
        != (
            EXPECTED_LAYOUTS,
            EXPECTED_CASES,
            EXPECTED_STORIES,
        )
    ):

        raise RuntimeError(
            "Unexpected prediction shape: "
            f"{prediction.shape}"
        )

    if (
        y_test.shape
        != (
            EXPECTED_CASES,
            EXPECTED_STORIES,
        )
    ):

        raise RuntimeError(
            "Unexpected y_test shape."
        )

    if (
        len(
            prediction_tags
        )
        != EXPECTED_LAYOUTS
        or len(
            set(
                prediction_tags
            )
        )
        != EXPECTED_LAYOUTS
    ):

        raise RuntimeError(
            "Prediction layout tags invalid."
        )

    if len(
        layout_results
    ) != EXPECTED_LAYOUTS:

        raise RuntimeError(
            "Expected 15 layout-result rows."
        )

    result_tags = set(
        layout_results[
            "layout_tag"
        ].tolist()
    )

    if (
        set(
            prediction_tags
        )
        != result_tags
    ):

        raise RuntimeError(
            "Prediction/result layout-tag mismatch."
        )

    # --------------------------------------------------------
    # Reorder result table to exactly match prediction array.
    # --------------------------------------------------------

    result_lookup = (
        layout_results
        .set_index(
            "layout_tag"
        )
    )

    layout_results_ordered = (
        result_lookup.loc[
            prediction_tags
        ]
        .reset_index()
    )

    sensor_counts = np.asarray(
        layout_results_ordered[
            "sensor_count"
        ],
        dtype=np.int64,
    )

    feature_counts = np.asarray(
        layout_results_ordered[
            "feature_count"
        ],
        dtype=np.int64,
    )

    # ========================================================
    # Precompute case-wise sufficient statistics.
    #
    # This permits whole-case bootstrap while preserving
    # all four storey outcomes within a case.
    # ========================================================

    error = (
        prediction
        - y_test[
            None,
            :,
            :,
        ]
    )

    abs_error = np.abs(
        error
    )

    # Overall MAE numerator per layout/case.
    case_abs_error_sum = np.sum(
        abs_error,
        axis=2,
    )

    # High-damage mask is common to every layout.
    high_mask = (
        y_test
        > HIGH_DAMAGE_THRESHOLD
    )

    high_count_per_case = np.sum(
        high_mask,
        axis=1,
    ).astype(
        np.float64
    )

    high_abs_error_sum = np.sum(
        abs_error
        * high_mask[
            None,
            :,
            :,
        ],
        axis=2,
    )

    high_signed_error_sum = np.sum(
        error
        * high_mask[
            None,
            :,
            :,
        ],
        axis=2,
    )

    high_under_count = np.sum(
        (
            error < 0.0
        )
        & high_mask[
            None,
            :,
            :,
        ],
        axis=2,
    ).astype(
        np.float64
    )

    # ========================================================
    # Point-estimate reconstruction lock.
    # ========================================================

    full_case_weights = np.ones(
        EXPECTED_CASES,
        dtype=np.float64,
    )

    all_denominator = float(
        EXPECTED_CASES
        * EXPECTED_STORIES
    )

    high_denominator = float(
        np.dot(
            high_count_per_case,
            full_case_weights,
        )
    )

    if high_denominator <= 0:

        raise RuntimeError(
            "No high-damage entries."
        )

    point_mae = (
        case_abs_error_sum
        @ full_case_weights
        / all_denominator
    )

    point_high_mae = (
        high_abs_error_sum
        @ full_case_weights
        / high_denominator
    )

    point_high_bias = (
        high_signed_error_sum
        @ full_case_weights
        / high_denominator
    )

    point_high_under = (
        high_under_count
        @ full_case_weights
        / high_denominator
    )

    anchor_rows = []

    anchor_passed = True

    for layout_index, tag in enumerate(
        prediction_tags
    ):

        expected_row = (
            layout_results_ordered
            .iloc[
                layout_index
            ]
        )

        checks = {
            "test_mae": (
                point_mae[
                    layout_index
                ],
                float(
                    expected_row[
                        "test_mae"
                    ]
                ),
            ),
            "test_high_mae": (
                point_high_mae[
                    layout_index
                ],
                float(
                    expected_row[
                        "test_high_mae"
                    ]
                ),
            ),
            "test_high_bias": (
                point_high_bias[
                    layout_index
                ],
                float(
                    expected_row[
                        "test_high_bias"
                    ]
                ),
            ),
            "test_high_underestimation": (
                point_high_under[
                    layout_index
                ],
                float(
                    expected_row[
                        "test_high_underestimation"
                    ]
                ),
            ),
        }

        for metric, (
            actual,
            expected,
        ) in checks.items():

            passed = close_enough(
                float(
                    actual
                ),
                float(
                    expected
                ),
            )

            anchor_passed = bool(
                anchor_passed
                and passed
            )

            anchor_rows.append(
                {
                    "layout_tag": tag,
                    "metric": metric,
                    "actual": float(
                        actual
                    ),
                    "expected": float(
                        expected
                    ),
                    "abs_error": float(
                        abs(
                            actual
                            - expected
                        )
                    ),
                    "passed": (
                        passed
                    ),
                }
            )

    anchor_frame = pd.DataFrame(
        anchor_rows
    )

    anchor_path = (
        output_root
        / "bootstrap_input_anchor.csv"
    )

    anchor_frame.to_csv(
        anchor_path,
        index=False,
    )

    if not anchor_passed:

        raise RuntimeError(
            "STOP: bootstrap prediction "
            "reconstruction anchor failed."
        )

    # ========================================================
    # Paired whole-case bootstrap.
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "PAIRED SENSOR-LAYOUT BOOTSTRAP CLOSURE"
    )

    print(
        "=" * 80
    )

    print(
        "Layouts:",
        EXPECTED_LAYOUTS,
    )

    print(
        "Test cases:",
        EXPECTED_CASES,
    )

    print(
        "Stories retained within each case:",
        EXPECTED_STORIES,
    )

    print(
        "Bootstrap replicates:",
        N_BOOTSTRAP,
    )

    print(
        "Bootstrap seed:",
        BOOTSTRAP_SEED,
    )

    print(
        "CI:",
        f"{CI_LEVEL * 100:.1f}% percentile",
    )

    print(
        "Input reconstruction anchor:",
        anchor_passed,
    )

    print(
        "Training/tuning performed:",
        False,
    )

    print()

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    boot_mae = np.empty(
        (
            N_BOOTSTRAP,
            EXPECTED_LAYOUTS,
        ),
        dtype=np.float64,
    )

    boot_high_mae = np.empty_like(
        boot_mae
    )

    boot_high_bias = np.empty_like(
        boot_mae
    )

    boot_high_under = np.empty_like(
        boot_mae
    )

    for bootstrap_index in range(
        N_BOOTSTRAP
    ):

        sampled_indices = rng.integers(
            0,
            EXPECTED_CASES,
            size=EXPECTED_CASES,
        )

        weights = np.bincount(
            sampled_indices,
            minlength=EXPECTED_CASES,
        ).astype(
            np.float64
        )

        denominator_all = float(
            np.sum(
                weights
            )
            * EXPECTED_STORIES
        )

        denominator_high = float(
            np.dot(
                high_count_per_case,
                weights,
            )
        )

        if denominator_high <= 0.0:

            raise RuntimeError(
                "Bootstrap replicate contained "
                "no high-damage entries."
            )

        boot_mae[
            bootstrap_index
        ] = (
            case_abs_error_sum
            @ weights
            / denominator_all
        )

        boot_high_mae[
            bootstrap_index
        ] = (
            high_abs_error_sum
            @ weights
            / denominator_high
        )

        boot_high_bias[
            bootstrap_index
        ] = (
            high_signed_error_sum
            @ weights
            / denominator_high
        )

        boot_high_under[
            bootstrap_index
        ] = (
            high_under_count
            @ weights
            / denominator_high
        )

    # ========================================================
    # Layout-level bootstrap intervals.
    # ========================================================

    layout_summary_rows = []

    metric_arrays = {
        "mae": (
            point_mae,
            boot_mae,
        ),
        "high_mae": (
            point_high_mae,
            boot_high_mae,
        ),
        "high_bias": (
            point_high_bias,
            boot_high_bias,
        ),
        "high_underestimation": (
            point_high_under,
            boot_high_under,
        ),
    }

    for layout_index, tag in enumerate(
        prediction_tags
    ):

        row = {
            "layout_tag": tag,
            "sensor_count": int(
                sensor_counts[
                    layout_index
                ]
            ),
            "feature_count": int(
                feature_counts[
                    layout_index
                ]
            ),
        }

        for metric, (
            point_values,
            bootstrap_values,
        ) in metric_arrays.items():

            values = (
                bootstrap_values[
                    :,
                    layout_index,
                ]
            )

            lower, upper = (
                percentile_ci(
                    values
                )
            )

            row[
                f"{metric}_point"
            ] = float(
                point_values[
                    layout_index
                ]
            )

            row[
                f"{metric}_bootstrap_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            row[
                f"{metric}_bootstrap_sd"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            row[
                f"{metric}_ci_lower"
            ] = (
                lower
            )

            row[
                f"{metric}_ci_upper"
            ] = (
                upper
            )

        layout_summary_rows.append(
            row
        )

    layout_summary = pd.DataFrame(
        layout_summary_rows
    )

    layout_summary_path = (
        output_root
        / "layout_bootstrap_ci.csv"
    )

    layout_summary.to_csv(
        layout_summary_path,
        index=False,
    )

    # ========================================================
    # Sensor-count aggregate.
    #
    # Layouts are exhaustive, not random.
    # Bootstrap uncertainty comes from cases only.
    # ========================================================

    unique_sensor_counts = [
        1,
        2,
        3,
        4,
    ]

    count_boot_mae = {}
    count_boot_high_mae = {}

    count_point_mae = {}
    count_point_high_mae = {}

    count_rows = []

    for sensor_count in (
        unique_sensor_counts
    ):

        indices = np.where(
            sensor_counts
            == sensor_count
        )[0]

        if len(
            indices
        ) == 0:

            raise RuntimeError(
                f"No layouts for "
                f"sensor_count={sensor_count}"
            )

        mae_values = np.mean(
            boot_mae[
                :,
                indices,
            ],
            axis=1,
        )

        high_mae_values = np.mean(
            boot_high_mae[
                :,
                indices,
            ],
            axis=1,
        )

        count_boot_mae[
            sensor_count
        ] = mae_values

        count_boot_high_mae[
            sensor_count
        ] = high_mae_values

        point_count_mae = float(
            np.mean(
                point_mae[
                    indices
                ]
            )
        )

        point_count_high_mae = float(
            np.mean(
                point_high_mae[
                    indices
                ]
            )
        )

        count_point_mae[
            sensor_count
        ] = point_count_mae

        count_point_high_mae[
            sensor_count
        ] = point_count_high_mae

        mae_lower, mae_upper = (
            percentile_ci(
                mae_values
            )
        )

        high_lower, high_upper = (
            percentile_ci(
                high_mae_values
            )
        )

        count_rows.append(
            {
                "sensor_count": (
                    sensor_count
                ),
                "n_layouts": int(
                    len(
                        indices
                    )
                ),
                "mae_point": (
                    point_count_mae
                ),
                "mae_bootstrap_mean": float(
                    np.mean(
                        mae_values
                    )
                ),
                "mae_bootstrap_sd": float(
                    np.std(
                        mae_values,
                        ddof=1,
                    )
                ),
                "mae_ci_lower": (
                    mae_lower
                ),
                "mae_ci_upper": (
                    mae_upper
                ),
                "high_mae_point": (
                    point_count_high_mae
                ),
                "high_mae_bootstrap_mean": float(
                    np.mean(
                        high_mae_values
                    )
                ),
                "high_mae_bootstrap_sd": float(
                    np.std(
                        high_mae_values,
                        ddof=1,
                    )
                ),
                "high_mae_ci_lower": (
                    high_lower
                ),
                "high_mae_ci_upper": (
                    high_upper
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
        / "sensor_count_bootstrap_ci.csv"
    )

    count_summary.to_csv(
        count_summary_path,
        index=False,
    )

    # ========================================================
    # Consecutive sensor-count effects.
    #
    # Positive improvement means k+1 sensors have lower error.
    # ========================================================

    count_contrast_rows = []

    for lower_count, upper_count in [
        (1, 2),
        (2, 3),
        (3, 4),
    ]:

        mae_improvement = (
            count_boot_mae[
                lower_count
            ]
            - count_boot_mae[
                upper_count
            ]
        )

        high_mae_improvement = (
            count_boot_high_mae[
                lower_count
            ]
            - count_boot_high_mae[
                upper_count
            ]
        )

        mae_point = (
            count_point_mae[
                lower_count
            ]
            - count_point_mae[
                upper_count
            ]
        )

        high_point = (
            count_point_high_mae[
                lower_count
            ]
            - count_point_high_mae[
                upper_count
            ]
        )

        mae_lower, mae_upper = (
            percentile_ci(
                mae_improvement
            )
        )

        high_lower, high_upper = (
            percentile_ci(
                high_mae_improvement
            )
        )

        count_contrast_rows.append(
            {
                "from_sensor_count": (
                    lower_count
                ),
                "to_sensor_count": (
                    upper_count
                ),
                "mae_improvement_point": float(
                    mae_point
                ),
                "mae_improvement_percent_point": float(
                    mae_point
                    / count_point_mae[
                        lower_count
                    ]
                    * 100.0
                ),
                "mae_ci_lower": (
                    mae_lower
                ),
                "mae_ci_upper": (
                    mae_upper
                ),
                "mae_bootstrap_positive_fraction": float(
                    np.mean(
                        mae_improvement
                        > 0.0
                    )
                ),
                "high_mae_improvement_point": float(
                    high_point
                ),
                "high_mae_improvement_percent_point": float(
                    high_point
                    / count_point_high_mae[
                        lower_count
                    ]
                    * 100.0
                ),
                "high_mae_ci_lower": (
                    high_lower
                ),
                "high_mae_ci_upper": (
                    high_upper
                ),
                "high_mae_bootstrap_positive_fraction": float(
                    np.mean(
                        high_mae_improvement
                        > 0.0
                    )
                ),
            }
        )

    count_contrasts = pd.DataFrame(
        count_contrast_rows
    )

    count_contrasts_path = (
        output_root
        / "sensor_count_step_contrasts.csv"
    )

    count_contrasts.to_csv(
        count_contrasts_path,
        index=False,
    )

    # ========================================================
    # Exhaustive within-count placement contrasts.
    #
    # This avoids selecting only favorable layout pairs.
    #
    # A-minus-B < 0 means A has lower error.
    # prob_A_lower estimates bootstrap ranking stability.
    # ========================================================

    pairwise_rows = []

    for sensor_count in [
        1,
        2,
        3,
    ]:

        indices = np.where(
            sensor_counts
            == sensor_count
        )[0]

        for index_a, index_b in combinations(
            indices,
            2,
        ):

            tag_a = (
                prediction_tags[
                    index_a
                ]
            )

            tag_b = (
                prediction_tags[
                    index_b
                ]
            )

            mae_difference = (
                boot_mae[
                    :,
                    index_a,
                ]
                - boot_mae[
                    :,
                    index_b,
                ]
            )

            high_difference = (
                boot_high_mae[
                    :,
                    index_a,
                ]
                - boot_high_mae[
                    :,
                    index_b,
                ]
            )

            mae_point_difference = float(
                point_mae[
                    index_a
                ]
                - point_mae[
                    index_b
                ]
            )

            high_point_difference = float(
                point_high_mae[
                    index_a
                ]
                - point_high_mae[
                    index_b
                ]
            )

            mae_lower, mae_upper = (
                percentile_ci(
                    mae_difference
                )
            )

            high_lower, high_upper = (
                percentile_ci(
                    high_difference
                )
            )

            pairwise_rows.append(
                {
                    "sensor_count": (
                        sensor_count
                    ),
                    "layout_A": (
                        tag_a
                    ),
                    "layout_B": (
                        tag_b
                    ),
                    "feature_count_A": int(
                        feature_counts[
                            index_a
                        ]
                    ),
                    "feature_count_B": int(
                        feature_counts[
                            index_b
                        ]
                    ),
                    "same_feature_count": bool(
                        feature_counts[
                            index_a
                        ]
                        == feature_counts[
                            index_b
                        ]
                    ),
                    "mae_A_minus_B_point": (
                        mae_point_difference
                    ),
                    "mae_A_minus_B_ci_lower": (
                        mae_lower
                    ),
                    "mae_A_minus_B_ci_upper": (
                        mae_upper
                    ),
                    "mae_prob_A_lower": float(
                        np.mean(
                            mae_difference
                            < 0.0
                        )
                    ),
                    "high_mae_A_minus_B_point": (
                        high_point_difference
                    ),
                    "high_mae_A_minus_B_ci_lower": (
                        high_lower
                    ),
                    "high_mae_A_minus_B_ci_upper": (
                        high_upper
                    ),
                    "high_mae_prob_A_lower": float(
                        np.mean(
                            high_difference
                            < 0.0
                        )
                    ),
                }
            )

    pairwise = pd.DataFrame(
        pairwise_rows
    )

    pairwise_path = (
        output_root
        / "within_count_pairwise_layout_contrasts.csv"
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    # ========================================================
    # Bootstrap stability of "best layout" within each
    # sensor-count budget.
    # ========================================================

    best_stability_rows = []

    for sensor_count in [
        1,
        2,
        3,
    ]:

        indices = np.where(
            sensor_counts
            == sensor_count
        )[0]

        mae_local_best = np.argmin(
            boot_mae[
                :,
                indices,
            ],
            axis=1,
        )

        high_local_best = np.argmin(
            boot_high_mae[
                :,
                indices,
            ],
            axis=1,
        )

        for local_index, global_index in enumerate(
            indices
        ):

            best_stability_rows.append(
                {
                    "sensor_count": (
                        sensor_count
                    ),
                    "layout_tag": (
                        prediction_tags[
                            global_index
                        ]
                    ),
                    "feature_count": int(
                        feature_counts[
                            global_index
                        ]
                    ),
                    "mae_best_fraction": float(
                        np.mean(
                            mae_local_best
                            == local_index
                        )
                    ),
                    "high_mae_best_fraction": float(
                        np.mean(
                            high_local_best
                            == local_index
                        )
                    ),
                }
            )

    best_stability = (
        pd.DataFrame(
            best_stability_rows
        )
        .sort_values(
            [
                "sensor_count",
                "mae_best_fraction",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    best_stability_path = (
        output_root
        / "best_layout_bootstrap_stability.csv"
    )

    best_stability.to_csv(
        best_stability_path,
        index=False,
    )

    # ========================================================
    # Exhaustive 28 marginal sensor-addition edges.
    #
    # Positive = adding one sensor improves the metric.
    # ========================================================

    tag_to_index = {
        tag: index
        for index, tag
        in enumerate(
            prediction_tags
        )
    }

    layout_sets = {
        tag: set(
            tag_to_tuple(
                tag
            )
        )
        for tag
        in prediction_tags
    }

    marginal_rows = []

    for base_tag in prediction_tags:

        base_set = (
            layout_sets[
                base_tag
            ]
        )

        if len(
            base_set
        ) >= 4:

            continue

        for added_sensor in [
            1,
            2,
            3,
            4,
        ]:

            if (
                added_sensor
                in base_set
            ):

                continue

            augmented_set = (
                base_set
                | {
                    added_sensor,
                }
            )

            augmented_tag = "".join(
                str(sensor)
                for sensor in sorted(
                    augmented_set
                )
            )

            if (
                augmented_tag
                not in tag_to_index
            ):

                raise RuntimeError(
                    "Missing augmented layout "
                    f"{augmented_tag}"
                )

            base_index = (
                tag_to_index[
                    base_tag
                ]
            )

            augmented_index = (
                tag_to_index[
                    augmented_tag
                ]
            )

            mae_improvement = (
                boot_mae[
                    :,
                    base_index,
                ]
                - boot_mae[
                    :,
                    augmented_index,
                ]
            )

            high_improvement = (
                boot_high_mae[
                    :,
                    base_index,
                ]
                - boot_high_mae[
                    :,
                    augmented_index,
                ]
            )

            point_mae_improvement = float(
                point_mae[
                    base_index
                ]
                - point_mae[
                    augmented_index
                ]
            )

            point_high_improvement = float(
                point_high_mae[
                    base_index
                ]
                - point_high_mae[
                    augmented_index
                ]
            )

            mae_lower, mae_upper = (
                percentile_ci(
                    mae_improvement
                )
            )

            high_lower, high_upper = (
                percentile_ci(
                    high_improvement
                )
            )

            marginal_rows.append(
                {
                    "base_layout": (
                        base_tag
                    ),
                    "base_sensor_count": int(
                        len(
                            base_set
                        )
                    ),
                    "added_sensor": (
                        added_sensor
                    ),
                    "augmented_layout": (
                        augmented_tag
                    ),
                    "base_feature_count": int(
                        feature_counts[
                            base_index
                        ]
                    ),
                    "augmented_feature_count": int(
                        feature_counts[
                            augmented_index
                        ]
                    ),
                    "feature_gain": int(
                        feature_counts[
                            augmented_index
                        ]
                        - feature_counts[
                            base_index
                        ]
                    ),
                    "mae_improvement_point": (
                        point_mae_improvement
                    ),
                    "mae_relative_improvement_percent_point": float(
                        point_mae_improvement
                        / point_mae[
                            base_index
                        ]
                        * 100.0
                    ),
                    "mae_ci_lower": (
                        mae_lower
                    ),
                    "mae_ci_upper": (
                        mae_upper
                    ),
                    "mae_ci_entirely_positive": bool(
                        mae_lower
                        > 0.0
                    ),
                    "mae_bootstrap_positive_fraction": float(
                        np.mean(
                            mae_improvement
                            > 0.0
                        )
                    ),
                    "high_mae_improvement_point": (
                        point_high_improvement
                    ),
                    "high_mae_relative_improvement_percent_point": float(
                        point_high_improvement
                        / point_high_mae[
                            base_index
                        ]
                        * 100.0
                    ),
                    "high_mae_ci_lower": (
                        high_lower
                    ),
                    "high_mae_ci_upper": (
                        high_upper
                    ),
                    "high_mae_ci_entirely_positive": bool(
                        high_lower
                        > 0.0
                    ),
                    "high_mae_bootstrap_positive_fraction": float(
                        np.mean(
                            high_improvement
                            > 0.0
                        )
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

    if len(
        marginal
    ) != 28:

        raise RuntimeError(
            "Expected 28 marginal "
            f"edges, got {len(marginal)}."
        )

    marginal_path = (
        output_root
        / "marginal_sensor_edge_bootstrap_ci.csv"
    )

    marginal.to_csv(
        marginal_path,
        index=False,
    )

    # ========================================================
    # Marginal closure summary.
    # ========================================================

    marginal_summary_rows = []

    for added_sensor, group in (
        marginal.groupby(
            "added_sensor"
        )
    ):

        marginal_summary_rows.append(
            {
                "added_sensor": int(
                    added_sensor
                ),
                "n_edges": int(
                    len(
                        group
                    )
                ),
                "mae_point_positive_fraction": float(
                    np.mean(
                        group[
                            "mae_improvement_point"
                        ]
                        > 0.0
                    )
                ),
                "mae_ci_positive_fraction": float(
                    np.mean(
                        group[
                            "mae_ci_entirely_positive"
                        ]
                    )
                ),
                "high_mae_point_positive_fraction": float(
                    np.mean(
                        group[
                            "high_mae_improvement_point"
                        ]
                        > 0.0
                    )
                ),
                "high_mae_ci_positive_fraction": float(
                    np.mean(
                        group[
                            "high_mae_ci_entirely_positive"
                        ]
                    )
                ),
            }
        )

    marginal_summary = pd.DataFrame(
        marginal_summary_rows
    )

    marginal_summary_path = (
        output_root
        / "marginal_sensor_bootstrap_summary.csv"
    )

    marginal_summary.to_csv(
        marginal_summary_path,
        index=False,
    )

    # ========================================================
    # Save bootstrap arrays for reproducibility.
    # ========================================================

    bootstrap_arrays_path = (
        output_root
        / "sensor_layout_bootstrap_arrays.npz"
    )

    np.savez_compressed(
        bootstrap_arrays_path,
        layout_tag=np.asarray(
            prediction_tags
        ),
        test_case_id=(
            test_case_id
        ),
        boot_mae=(
            boot_mae
        ),
        boot_high_mae=(
            boot_high_mae
        ),
        boot_high_bias=(
            boot_high_bias
        ),
        boot_high_underestimation=(
            boot_high_under
        ),
        n_bootstrap=np.asarray(
            N_BOOTSTRAP,
            dtype=np.int64,
        ),
        bootstrap_seed=np.asarray(
            BOOTSTRAP_SEED,
            dtype=np.int64,
        ),
    )

    # ========================================================
    # Final integrity/statistical closure.
    # ========================================================

    arrays_finite = bool(
        np.isfinite(
            boot_mae
        ).all()
        and np.isfinite(
            boot_high_mae
        ).all()
        and np.isfinite(
            boot_high_bias
        ).all()
        and np.isfinite(
            boot_high_under
        ).all()
    )

    all_count_mae_steps_positive = bool(
        (
            count_contrasts[
                "mae_improvement_point"
            ]
            > 0.0
        ).all()
    )

    all_count_high_steps_positive = bool(
        (
            count_contrasts[
                "high_mae_improvement_point"
            ]
            > 0.0
        ).all()
    )

    count_mae_ci_all_positive = bool(
        (
            count_contrasts[
                "mae_ci_lower"
            ]
            > 0.0
        ).all()
    )

    count_high_ci_all_positive = bool(
        (
            count_contrasts[
                "high_mae_ci_lower"
            ]
            > 0.0
        ).all()
    )

    marginal_mae_point_positive_count = int(
        np.sum(
            marginal[
                "mae_improvement_point"
            ]
            > 0.0
        )
    )

    marginal_mae_ci_positive_count = int(
        np.sum(
            marginal[
                "mae_ci_entirely_positive"
            ]
        )
    )

    marginal_high_point_positive_count = int(
        np.sum(
            marginal[
                "high_mae_improvement_point"
            ]
            > 0.0
        )
    )

    marginal_high_ci_positive_count = int(
        np.sum(
            marginal[
                "high_mae_ci_entirely_positive"
            ]
        )
    )

    overall_passed = bool(
        anchor_passed
        and arrays_finite
        and len(
            pairwise
        )
        == 27
        and len(
            marginal
        )
        == 28
    )

    # ========================================================
    # Console output.
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "SENSOR-COUNT BOOTSTRAP CI"
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
        "CONSECUTIVE SENSOR-COUNT EFFECTS"
    )

    print(
        "=" * 80
    )

    print(
        count_contrasts.to_string(
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
        "WITHIN-COUNT PLACEMENT CONTRASTS"
    )

    print(
        "=" * 80
    )

    print(
        pairwise.to_string(
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
        "BEST-LAYOUT BOOTSTRAP STABILITY"
    )

    print(
        "=" * 80
    )

    print(
        best_stability.to_string(
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
        "MARGINAL SENSOR EDGE BOOTSTRAP"
    )

    print(
        "=" * 80
    )

    print(
        marginal.to_string(
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
        "MARGINAL SENSOR SUMMARY"
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
    # Report.
    # ========================================================

    report = {
        "experiment": (
            "paired_case_bootstrap_"
            "sensor_layout_statistical_closure"
        ),
        "training_or_tuning_performed": (
            False
        ),
        "bootstrap_unit": (
            "complete structural case "
            "with all four storey outputs"
        ),
        "paired_across_layouts": (
            True
        ),
        "n_test_cases": (
            EXPECTED_CASES
        ),
        "n_layouts": (
            EXPECTED_LAYOUTS
        ),
        "n_bootstrap": (
            N_BOOTSTRAP
        ),
        "bootstrap_seed": (
            BOOTSTRAP_SEED
        ),
        "ci_level": (
            CI_LEVEL
        ),
        "ci_method": (
            "percentile bootstrap"
        ),
        "scope_limitation": (
            "case-resampling uncertainty "
            "conditional on the current "
            "simulated fixed test set; "
            "not independent physical-"
            "structure uncertainty"
        ),
        "input_anchor_passed": (
            anchor_passed
        ),
        "bootstrap_arrays_finite": (
            arrays_finite
        ),
        "within_count_pair_count": int(
            len(
                pairwise
            )
        ),
        "marginal_edge_count": int(
            len(
                marginal
            )
        ),
        "all_sensor_count_mae_steps_point_positive": (
            all_count_mae_steps_positive
        ),
        "all_sensor_count_high_mae_steps_point_positive": (
            all_count_high_steps_positive
        ),
        "all_sensor_count_mae_step_ci_positive": (
            count_mae_ci_all_positive
        ),
        "all_sensor_count_high_mae_step_ci_positive": (
            count_high_ci_all_positive
        ),
        "marginal_mae_point_positive_count": (
            marginal_mae_point_positive_count
        ),
        "marginal_mae_ci_positive_count": (
            marginal_mae_ci_positive_count
        ),
        "marginal_high_mae_point_positive_count": (
            marginal_high_point_positive_count
        ),
        "marginal_high_mae_ci_positive_count": (
            marginal_high_ci_positive_count
        ),
        "overall_passed": (
            overall_passed
        ),
        "outputs": {
            "anchor": str(
                anchor_path
            ),
            "layout_ci": str(
                layout_summary_path
            ),
            "sensor_count_ci": str(
                count_summary_path
            ),
            "sensor_count_contrasts": str(
                count_contrasts_path
            ),
            "placement_pairwise": str(
                pairwise_path
            ),
            "best_layout_stability": str(
                best_stability_path
            ),
            "marginal_edges": str(
                marginal_path
            ),
            "marginal_summary": str(
                marginal_summary_path
            ),
            "bootstrap_arrays": str(
                bootstrap_arrays_path
            ),
        },
    }

    report_path = (
        output_root
        / "sensor_layout_paired_bootstrap_report.json"
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
        "FINAL STATISTICAL CLOSURE CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Training/tuning performed:",
        False,
    )

    print(
        "Prediction reconstruction anchor:",
        anchor_passed,
    )

    print(
        "Bootstrap arrays finite:",
        arrays_finite,
    )

    print(
        "Within-count layout pairs:",
        len(
            pairwise
        ),
        "/ 27",
    )

    print(
        "Marginal subset edges:",
        len(
            marginal
        ),
        "/ 28",
    )

    print(
        "All count MAE steps point-positive:",
        all_count_mae_steps_positive,
    )

    print(
        "All count MAE step CIs > 0:",
        count_mae_ci_all_positive,
    )

    print(
        "All count high-MAE steps point-positive:",
        all_count_high_steps_positive,
    )

    print(
        "All count high-MAE step CIs > 0:",
        count_high_ci_all_positive,
    )

    print(
        "Marginal MAE point-positive edges:",
        marginal_mae_point_positive_count,
        "/ 28",
    )

    print(
        "Marginal MAE CI-positive edges:",
        marginal_mae_ci_positive_count,
        "/ 28",
    )

    print(
        "Marginal high-MAE point-positive edges:",
        marginal_high_point_positive_count,
        "/ 28",
    )

    print(
        "Marginal high-MAE CI-positive edges:",
        marginal_high_ci_positive_count,
        "/ 28",
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    if overall_passed:

        print(
            "CHECK PASSED: paired case-level "
            "sensor-layout statistical closure "
            "completed."
        )

        print(
            "No model or descriptor parameter "
            "was changed."
        )

    else:

        print(
            "CHECK FAILED: inspect bootstrap "
            "outputs before interpretation."
        )

    print()

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
