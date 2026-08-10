"""
Validation-only pilot for severity-weighted RBF-SVR.

Main descriptor representation:
    78D signal_derived_with_ground

Scientific question
-------------------
Can storey-specific high-damage sample weighting reduce the residual
severity-dependent underestimation of the deployable 78D RBF-SVR
without materially degrading overall / zero / low-damage accuracy?

IMPORTANT
---------
- Only F_train, F_val, y_train, y_val are accessed.
- F_test and y_test are NEVER accessed.
- This is a protocol-development pilot.
- No test performance is reported.
- After this pilot, the weighting protocol will be frozen before
  repeated-split evaluation.

Weight definition
-----------------
For each storey-specific SVR:

    y <= 0.20 : raw weight = 1
    y >  0.20 : raw weight = lambda

Weights are normalised to mean 1 within each storey.

Baseline:
    lambda = 1

Damage-aware selection:
    1. overall validation MSE <= baseline * 1.05
    2. zero-damage MAE       <= baseline * 1.10
    3. low-damage MAE        <= baseline * 1.10

Among eligible weighted candidates:
    1. minimise high-damage MAE
    2. minimise |high-damage bias|
    3. minimise high-damage underestimation ratio
    4. minimise overall validation MSE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.svm import SVR


EXPECTED_FEATURE_COUNT = 78

HIGH_DAMAGE_THRESHOLD = 0.20

WEIGHT_MULTIPLIERS = [
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]

C_VALUES = [
    100.0,
    300.0,
    500.0,
    750.0,
    1000.0,
]

GAMMA_VALUES = [
    0.00075,
    0.001,
    0.003,
]

EPSILON_VALUES = [
    0.02,
    0.03,
    0.04,
]

OVERALL_MSE_BUDGET = 1.05
ZERO_MAE_BUDGET = 1.10
LOW_MAE_BUDGET = 1.10

BASELINE_ANCHOR = {
    "C": 1000.0,
    "gamma": 0.00075,
    "epsilon": 0.03,
    "val_mse": 0.002256262230781473,
}

ANCHOR_TOLERANCE = 5.0e-11


def load_train_validation_only(
    path: Path,
) -> dict[str, np.ndarray]:

    if not path.is_file():
        raise FileNotFoundError(path)

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
            required - set(data.files)
        )

        if missing:
            raise KeyError(
                f"Missing arrays: {missing}"
            )

        arrays = {
            key: np.asarray(
                data[key],
                dtype=np.float64,
            )
            for key in required
        }

    for key, array in arrays.items():
        if not np.all(
            np.isfinite(array)
        ):
            raise FloatingPointError(
                f"{key} contains non-finite values."
            )

    if (
        arrays["F_train"].shape[1]
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Expected 78 descriptors, found "
            f"{arrays['F_train'].shape[1]}."
        )

    if (
        arrays["y_train"].ndim != 2
        or arrays["y_train"].shape[1] != 4
    ):
        raise ValueError(
            "Expected four-storey targets."
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


def make_storey_weights(
    target: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """
    Storey-specific severity weights normalised to mean 1.
    """

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    raw = np.where(
        target > HIGH_DAMAGE_THRESHOLD,
        multiplier,
        1.0,
    ).astype(
        np.float64
    )

    mean_weight = float(
        np.mean(raw)
    )

    if mean_weight <= 0.0:
        raise RuntimeError(
            "Invalid sample-weight mean."
        )

    return (
        raw / mean_weight
    )


def fit_weighted_multioutput_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
    weight_multiplier: float,
) -> tuple[
    np.ndarray,
    list[int],
]:

    predictions = []

    support_counts = []

    for storey in range(
        y_train.shape[1]
    ):

        weights = (
            make_storey_weights(
                y_train[
                    :,
                    storey,
                ],
                multiplier=(
                    weight_multiplier
                ),
            )
        )

        model = SVR(
            kernel="rbf",
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            tol=1e-4,
            cache_size=1024,
        )

        model.fit(
            x_train,
            y_train[
                :,
                storey,
            ],
            sample_weight=weights,
        )

        prediction = (
            model.predict(
                x_val
            )
        )

        predictions.append(
            prediction
        )

        support_counts.append(
            int(
                len(
                    model.support_
                )
            )
        )

    matrix = np.column_stack(
        predictions
    )

    return (
        clip_prediction(
            matrix
        ),
        support_counts,
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
        prediction - truth
    )

    zero = (
        truth <= 1e-12
    )

    low = (
        (truth > 1e-12)
        & (truth <= 0.10)
    )

    medium = (
        (truth > 0.10)
        & (truth <= 0.20)
    )

    high = (
        truth > 0.20
    )

    damaged = (
        truth > 1e-12
    )

    for name, mask in {
        "zero": zero,
        "low": low,
        "medium": medium,
        "high": high,
        "damaged": damaged,
    }.items():
        if not np.any(mask):
            raise RuntimeError(
                f"Empty validation group: {name}"
            )

    high_bias = float(
        np.mean(
            error[high]
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
        "val_bias": float(
            np.mean(error)
        ),
        "damaged_mae": float(
            np.mean(
                np.abs(
                    error[damaged]
                )
            )
        ),
        "zero_mae": float(
            np.mean(
                np.abs(
                    error[zero]
                )
            )
        ),
        "low_mae": float(
            np.mean(
                np.abs(
                    error[low]
                )
            )
        ),
        "medium_mae": float(
            np.mean(
                np.abs(
                    error[medium]
                )
            )
        ),
        "high_mae": float(
            np.mean(
                np.abs(
                    error[high]
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
                prediction[high]
                < truth[high]
            )
        ),
        "high_mean_true": float(
            np.mean(
                truth[high]
            )
        ),
        "high_mean_prediction": float(
            np.mean(
                prediction[high]
            )
        ),
    }


def candidate_name(
    C: float,
    gamma: float,
    epsilon: float,
    multiplier: float,
) -> str:

    gamma_label = (
        f"{gamma:g}"
        .replace(
            ".",
            "p",
        )
    )

    weight_label = (
        f"{multiplier:g}"
        .replace(
            ".",
            "p",
        )
    )

    return (
        f"weighted_svr_w_{weight_label}"
        f"_C_{C:g}"
        f"_gamma_{gamma_label}"
        f"_eps_{epsilon:g}"
    )


def relative_change_percent(
    value: float,
    reference: float,
) -> float:

    return float(
        (
            value - reference
        )
        / reference
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
            "damage_weighted_svr_validation_pilot"
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

    total_candidates = (
        len(
            WEIGHT_MULTIPLIERS
        )
        * len(
            C_VALUES
        )
        * len(
            GAMMA_VALUES
        )
        * len(
            EPSILON_VALUES
        )
    )

    print(
        "===== 78D DAMAGE-WEIGHTED "
        "SVR VALIDATION PILOT ====="
    )

    print(
        "Train shape:",
        x_train.shape,
    )

    print(
        "Validation shape:",
        x_val.shape,
    )

    print(
        "Test arrays accessed: False"
    )

    print(
        "High-damage threshold:",
        HIGH_DAMAGE_THRESHOLD,
    )

    print(
        "Weight multipliers:",
        WEIGHT_MULTIPLIERS,
    )

    print(
        "Candidates:",
        total_candidates,
    )

    print()

    # -----------------------------------
    # Training severity prevalence
    # -----------------------------------

    prevalence_rows = []

    for storey in range(
        y_train.shape[1]
    ):

        high_mask = (
            y_train[
                :,
                storey,
            ]
            > HIGH_DAMAGE_THRESHOLD
        )

        prevalence_rows.append(
            {
                "storey": (
                    storey + 1
                ),
                "n_train": int(
                    len(
                        high_mask
                    )
                ),
                "n_high_damage": int(
                    np.sum(
                        high_mask
                    )
                ),
                "high_damage_fraction": float(
                    np.mean(
                        high_mask
                    )
                ),
            }
        )

    prevalence_frame = (
        pd.DataFrame(
            prevalence_rows
        )
    )

    print(
        "===== TRAIN HIGH-DAMAGE "
        "PREVALENCE ====="
    )

    print(
        prevalence_frame.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()

    # -----------------------------------
    # Candidate search
    # -----------------------------------

    rows = []

    counter = 0

    for multiplier in (
        WEIGHT_MULTIPLIERS
    ):
        for C in C_VALUES:
            for gamma in (
                GAMMA_VALUES
            ):
                for epsilon in (
                    EPSILON_VALUES
                ):

                    counter += 1

                    start = (
                        time.perf_counter()
                    )

                    (
                        prediction,
                        support_counts,
                    ) = (
                        fit_weighted_multioutput_svr(
                            x_train=x_train,
                            y_train=y_train,
                            x_val=x_val,
                            C=C,
                            gamma=gamma,
                            epsilon=epsilon,
                            weight_multiplier=(
                                multiplier
                            ),
                        )
                    )

                    elapsed = float(
                        time.perf_counter()
                        - start
                    )

                    metrics = (
                        calculate_metrics(
                            y_val,
                            prediction,
                        )
                    )

                    row = {
                        "candidate": (
                            candidate_name(
                                C=C,
                                gamma=gamma,
                                epsilon=epsilon,
                                multiplier=(
                                    multiplier
                                ),
                            )
                        ),
                        "weight_multiplier": float(
                            multiplier
                        ),
                        "C": float(
                            C
                        ),
                        "gamma": float(
                            gamma
                        ),
                        "epsilon": float(
                            epsilon
                        ),
                        **metrics,
                        "mean_support_vector_count": float(
                            np.mean(
                                support_counts
                            )
                        ),
                        "mean_support_vector_ratio": float(
                            np.mean(
                                support_counts
                            )
                            / x_train.shape[0]
                        ),
                        "fit_seconds": elapsed,
                        "eligible": False,
                        "selected_baseline": False,
                        "selected_weighted": False,
                    }

                    rows.append(
                        row
                    )

                    if (
                        counter % 15
                        == 0
                        or counter
                        == total_candidates
                    ):
                        print(
                            f"Progress: "
                            f"{counter}/"
                            f"{total_candidates}"
                        )

    frame = pd.DataFrame(
        rows
    )

    # -----------------------------------
    # Baseline selection
    # -----------------------------------

    baseline_frame = (
        frame.loc[
            np.isclose(
                frame[
                    "weight_multiplier"
                ],
                1.0,
            )
        ]
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

    baseline = (
        baseline_frame.iloc[
            0
        ]
    )

    baseline_candidate = str(
        baseline[
            "candidate"
        ]
    )

    frame.loc[
        frame[
            "candidate"
        ]
        == baseline_candidate,
        "selected_baseline",
    ] = True

    # -----------------------------------
    # Anchor reproduction
    # -----------------------------------

    anchor_mask = (
        np.isclose(
            frame[
                "weight_multiplier"
            ],
            1.0,
        )
        & np.isclose(
            frame["C"],
            BASELINE_ANCHOR["C"],
        )
        & np.isclose(
            frame["gamma"],
            BASELINE_ANCHOR["gamma"],
        )
        & np.isclose(
            frame["epsilon"],
            BASELINE_ANCHOR["epsilon"],
        )
    )

    anchor_rows = (
        frame.loc[
            anchor_mask
        ]
    )

    if len(
        anchor_rows
    ) != 1:
        raise RuntimeError(
            "Baseline anchor missing."
        )

    anchor_mse = float(
        anchor_rows[
            "val_mse"
        ].iloc[0]
    )

    anchor_passed = bool(
        np.isclose(
            anchor_mse,
            BASELINE_ANCHOR[
                "val_mse"
            ],
            atol=(
                ANCHOR_TOLERANCE
            ),
            rtol=0.0,
        )
    )

    if not anchor_passed:
        raise RuntimeError(
            "STOP: unweighted SVR anchor "
            "was not reproduced.\n"
            f"actual={anchor_mse:.12f}\n"
            f"expected="
            f"{BASELINE_ANCHOR['val_mse']:.12f}"
        )

    # Compact grid should retain
    # the locked baseline optimum.
    expected_baseline_match = (
        np.isclose(
            float(
                baseline["C"]
            ),
            BASELINE_ANCHOR["C"],
        )
        and np.isclose(
            float(
                baseline["gamma"]
            ),
            BASELINE_ANCHOR["gamma"],
        )
        and np.isclose(
            float(
                baseline["epsilon"]
            ),
            BASELINE_ANCHOR["epsilon"],
        )
    )

    if not expected_baseline_match:
        raise RuntimeError(
            "STOP: compact pilot grid does not "
            "reproduce the locked baseline optimum."
        )

    # -----------------------------------
    # Frozen eligibility constraints
    # -----------------------------------

    mse_limit = (
        float(
            baseline[
                "val_mse"
            ]
        )
        * OVERALL_MSE_BUDGET
    )

    zero_mae_limit = (
        float(
            baseline[
                "zero_mae"
            ]
        )
        * ZERO_MAE_BUDGET
    )

    low_mae_limit = (
        float(
            baseline[
                "low_mae"
            ]
        )
        * LOW_MAE_BUDGET
    )

    weighted_mask = (
        frame[
            "weight_multiplier"
        ]
        > 1.0
    )

    eligible_mask = (
        weighted_mask
        & (
            frame[
                "val_mse"
            ]
            <= mse_limit
        )
        & (
            frame[
                "zero_mae"
            ]
            <= zero_mae_limit
        )
        & (
            frame[
                "low_mae"
            ]
            <= low_mae_limit
        )
    )

    frame.loc[
        eligible_mask,
        "eligible",
    ] = True

    eligible = (
        frame.loc[
            eligible_mask
        ]
        .sort_values(
            [
                "high_mae",
                "high_abs_bias",
                "high_underestimation_ratio",
                "val_mse",
                "candidate",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if eligible.empty:
        decision = (
            "NO_ACCEPTABLE_WEIGHTED_CANDIDATE"
        )

        selected_weighted = None

    else:
        selected_weighted = (
            eligible.iloc[
                0
            ]
        )

        selected_name = str(
            selected_weighted[
                "candidate"
            ]
        )

        frame.loc[
            frame[
                "candidate"
            ]
            == selected_name,
            "selected_weighted",
        ] = True

        decision = (
            "WEIGHTED_CANDIDATE_SELECTED"
        )

    # -----------------------------------
    # Best eligible candidate by weight
    # -----------------------------------

    best_by_weight_rows = []

    for multiplier in (
        WEIGHT_MULTIPLIERS
    ):

        group = (
            frame.loc[
                np.isclose(
                    frame[
                        "weight_multiplier"
                    ],
                    multiplier,
                )
            ]
            .copy()
        )

        if np.isclose(
            multiplier,
            1.0,
        ):
            best = (
                group.sort_values(
                    [
                        "val_mse",
                        "candidate",
                    ]
                )
                .iloc[0]
                .copy()
            )

            best[
                "weight_status"
            ] = (
                "baseline"
            )

        else:
            valid = (
                group.loc[
                    group[
                        "eligible"
                    ]
                ]
            )

            if valid.empty:
                best = (
                    group.sort_values(
                        [
                            "high_mae",
                            "val_mse",
                        ]
                    )
                    .iloc[0]
                    .copy()
                )

                best[
                    "weight_status"
                ] = (
                    "no_eligible_candidate"
                )

            else:
                best = (
                    valid.sort_values(
                        [
                            "high_mae",
                            "high_abs_bias",
                            "high_underestimation_ratio",
                            "val_mse",
                        ]
                    )
                    .iloc[0]
                    .copy()
                )

                best[
                    "weight_status"
                ] = (
                    "eligible"
                )

        best_by_weight_rows.append(
            best
        )

    best_by_weight = (
        pd.DataFrame(
            best_by_weight_rows
        )
    )

    # -----------------------------------
    # Comparison table
    # -----------------------------------

    comparison_rows = [
        {
            "configuration": (
                "standard_unweighted_svr"
            ),
            "candidate": (
                baseline_candidate
            ),
            "weight_multiplier": float(
                baseline[
                    "weight_multiplier"
                ]
            ),
            "val_mse": float(
                baseline[
                    "val_mse"
                ]
            ),
            "val_mae": float(
                baseline[
                    "val_mae"
                ]
            ),
            "zero_mae": float(
                baseline[
                    "zero_mae"
                ]
            ),
            "low_mae": float(
                baseline[
                    "low_mae"
                ]
            ),
            "medium_mae": float(
                baseline[
                    "medium_mae"
                ]
            ),
            "high_mae": float(
                baseline[
                    "high_mae"
                ]
            ),
            "high_bias": float(
                baseline[
                    "high_bias"
                ]
            ),
            "high_abs_bias": float(
                baseline[
                    "high_abs_bias"
                ]
            ),
            "high_underestimation_ratio": float(
                baseline[
                    "high_underestimation_ratio"
                ]
            ),
        }
    ]

    if (
        selected_weighted
        is not None
    ):

        comparison_rows.append(
            {
                "configuration": (
                    "damage_weighted_svr"
                ),
                "candidate": str(
                    selected_weighted[
                        "candidate"
                    ]
                ),
                "weight_multiplier": float(
                    selected_weighted[
                        "weight_multiplier"
                    ]
                ),
                "val_mse": float(
                    selected_weighted[
                        "val_mse"
                    ]
                ),
                "val_mae": float(
                    selected_weighted[
                        "val_mae"
                    ]
                ),
                "zero_mae": float(
                    selected_weighted[
                        "zero_mae"
                    ]
                ),
                "low_mae": float(
                    selected_weighted[
                        "low_mae"
                    ]
                ),
                "medium_mae": float(
                    selected_weighted[
                        "medium_mae"
                    ]
                ),
                "high_mae": float(
                    selected_weighted[
                        "high_mae"
                    ]
                ),
                "high_bias": float(
                    selected_weighted[
                        "high_bias"
                    ]
                ),
                "high_abs_bias": float(
                    selected_weighted[
                        "high_abs_bias"
                    ]
                ),
                "high_underestimation_ratio": float(
                    selected_weighted[
                        "high_underestimation_ratio"
                    ]
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    # -----------------------------------
    # Decision diagnostics
    # -----------------------------------

    if (
        selected_weighted
        is None
    ):
        high_mae_improvement = np.nan
        bias_improvement = np.nan
        under_change_points = np.nan
        overall_mse_change = np.nan

        next_stage = (
            "severity_weighting_insufficient_"
            "consider_asymmetric_objective"
        )

    else:
        high_mae_improvement = (
            -relative_change_percent(
                float(
                    selected_weighted[
                        "high_mae"
                    ]
                ),
                float(
                    baseline[
                        "high_mae"
                    ]
                ),
            )
        )

        bias_improvement = (
            -relative_change_percent(
                float(
                    selected_weighted[
                        "high_abs_bias"
                    ]
                ),
                float(
                    baseline[
                        "high_abs_bias"
                    ]
                ),
            )
        )

        under_change_points = (
            (
                float(
                    selected_weighted[
                        "high_underestimation_ratio"
                    ]
                )
                - float(
                    baseline[
                        "high_underestimation_ratio"
                    ]
                )
            )
            * 100.0
        )

        overall_mse_change = (
            relative_change_percent(
                float(
                    selected_weighted[
                        "val_mse"
                    ]
                ),
                float(
                    baseline[
                        "val_mse"
                    ]
                ),
            )
        )

        if (
            high_mae_improvement
            >= 10.0
            and under_change_points
            < 0.0
        ):
            next_stage = (
                "PROCEED_TO_10_SEED_"
                "WEIGHTED_SVR_VALIDATION"
            )

        elif (
            high_mae_improvement
            >= 5.0
            and under_change_points
            < 0.0
        ):
            next_stage = (
                "MODEST_EFFECT_PROCEED_"
                "CAUTIOUSLY_TO_REPEATED_SPLIT"
            )

        else:
            next_stage = (
                "WEIGHTING_EFFECT_WEAK_"
                "CONSIDER_ASYMMETRIC_OBJECTIVE"
            )

    # -----------------------------------
    # Save
    # -----------------------------------

    candidates_path = (
        args.output_root
        / "damage_weighted_validation_candidates.csv"
    )

    best_weight_path = (
        args.output_root
        / "damage_weighted_best_by_weight.csv"
    )

    comparison_path = (
        args.output_root
        / "damage_weighted_baseline_comparison.csv"
    )

    prevalence_path = (
        args.output_root
        / "training_high_damage_prevalence.csv"
    )

    report_path = (
        args.output_root
        / "damage_weighted_validation_report.json"
    )

    frame.to_csv(
        candidates_path,
        index=False,
    )

    best_by_weight.to_csv(
        best_weight_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    prevalence_frame.to_csv(
        prevalence_path,
        index=False,
    )

    report = {
        "experiment": (
            "78D_damage_weighted_RBF_SVR_"
            "validation_only_pilot"
        ),
        "test_arrays_accessed": False,
        "feature_count": (
            EXPECTED_FEATURE_COUNT
        ),
        "weighting": {
            "type": (
                "storey_specific_high_damage_weight"
            ),
            "threshold": (
                HIGH_DAMAGE_THRESHOLD
            ),
            "multipliers": (
                WEIGHT_MULTIPLIERS
            ),
            "normalisation": (
                "per-storey mean weight = 1"
            ),
        },
        "search_space": {
            "C": C_VALUES,
            "gamma": (
                GAMMA_VALUES
            ),
            "epsilon": (
                EPSILON_VALUES
            ),
            "candidate_count": (
                total_candidates
            ),
        },
        "baseline_selection": (
            "minimum clipped validation MSE "
            "among multiplier=1 candidates"
        ),
        "baseline_anchor_reproduced": (
            anchor_passed
        ),
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
        },
        "weighted_selection": (
            "min high-damage MAE, then "
            "min |high bias|, then min "
            "underestimation ratio, then "
            "min overall validation MSE"
        ),
        "decision": (
            decision
        ),
        "next_stage": (
            next_stage
        ),
        "high_damage_mae_improvement_percent": (
            None
            if np.isnan(
                high_mae_improvement
            )
            else float(
                high_mae_improvement
            )
        ),
        "high_damage_abs_bias_improvement_percent": (
            None
            if np.isnan(
                bias_improvement
            )
            else float(
                bias_improvement
            )
        ),
        "high_damage_underestimation_change_percentage_points": (
            None
            if np.isnan(
                under_change_points
            )
            else float(
                under_change_points
            )
        ),
        "overall_validation_mse_change_percent": (
            None
            if np.isnan(
                overall_mse_change
            )
            else float(
                overall_mse_change
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

    # -----------------------------------
    # Console output
    # -----------------------------------

    print()
    print(
        "===== BASELINE ANCHOR ====="
    )

    print(
        "Anchor reproduced:",
        anchor_passed,
    )

    print(
        "Baseline candidate:",
        baseline_candidate,
    )

    print(
        "Baseline val MSE:",
        f"{float(baseline['val_mse']):.12f}",
    )

    print(
        "Baseline high MAE:",
        f"{float(baseline['high_mae']):.10f}",
    )

    print(
        "Baseline high bias:",
        f"{float(baseline['high_bias']):.10f}",
    )

    print(
        "Baseline high underestimation:",
        (
            f"{float(
                baseline[
                    'high_underestimation_ratio'
                ]
            ):.10f}"
        ),
    )

    print()
    print(
        "===== ELIGIBILITY LIMITS ====="
    )

    print(
        "Overall MSE limit:",
        f"{mse_limit:.10f}",
    )

    print(
        "Zero MAE limit:",
        f"{zero_mae_limit:.10f}",
    )

    print(
        "Low MAE limit:",
        f"{low_mae_limit:.10f}",
    )

    print(
        "Eligible weighted candidates:",
        len(
            eligible
        ),
    )

    print()
    print(
        "===== BEST BY WEIGHT ====="
    )

    print(
        best_by_weight[
            [
                "weight_multiplier",
                "weight_status",
                "C",
                "gamma",
                "epsilon",
                "val_mse",
                "zero_mae",
                "low_mae",
                "high_mae",
                "high_bias",
                "high_underestimation_ratio",
                "mean_support_vector_ratio",
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
        "===== SELECTED DAMAGE-AWARE "
        "COMPARISON ====="
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== DAMAGE-AWARE EFFECT ====="
    )

    print(
        "Decision:",
        decision,
    )

    print(
        "Next stage:",
        next_stage,
    )

    if (
        selected_weighted
        is not None
    ):

        print(
            "High-damage MAE improvement (%):",
            f"{high_mae_improvement:.6f}",
        )

        print(
            "High-damage |bias| improvement (%):",
            f"{bias_improvement:.6f}",
        )

        print(
            "High-damage underestimation "
            "change (percentage points):",
            f"{under_change_points:.6f}",
        )

        print(
            "Overall validation MSE change (%):",
            f"{overall_mse_change:.6f}",
        )

    print()
    print(
        "===== TOP 15 ELIGIBLE "
        "WEIGHTED CANDIDATES ====="
    )

    if eligible.empty:
        print(
            "No eligible weighted candidate."
        )
    else:
        print(
            eligible.head(
                15
            )[
                [
                    "candidate",
                    "weight_multiplier",
                    "C",
                    "gamma",
                    "epsilon",
                    "val_mse",
                    "val_mae",
                    "zero_mae",
                    "low_mae",
                    "medium_mae",
                    "high_mae",
                    "high_bias",
                    "high_underestimation_ratio",
                    "mean_support_vector_ratio",
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
        "Storey-specific weighting:",
        True,
    )

    print(
        "Weights mean-normalised:",
        True,
    )

    print(
        "Selection constraints predeclared:",
        True,
    )

    print()
    print(
        "CHECK PASSED: validation-only "
        "damage-weighted SVR pilot completed."
    )

    print(
        "Candidates:",
        candidates_path,
    )

    print(
        "Comparison:",
        comparison_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
