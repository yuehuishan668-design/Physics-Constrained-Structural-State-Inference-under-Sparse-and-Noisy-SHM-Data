"""
10-seed paired robustness validation:

Standard 78D RBF-SVR
vs
Constrained severity-weighted 78D RBF-SVR

The weighting protocol was developed in a validation-only pilot and
is frozen before this repeated-split test.

Frozen damage-aware protocol
----------------------------
High-damage threshold:
    y > 0.20

Storey-specific sample weighting:
    normal entry     -> 1
    high-damage entry -> 1.5

Weights are normalised to mean 1 separately for every storey.

Weighted SVR candidate grid:
    C       = [100, 300, 500, 750, 1000]
    gamma   = [0.00075, 0.001, 0.003]
    epsilon = [0.02, 0.03, 0.04]

Eligibility relative to the already-selected standard SVR
on the SAME seed's validation partition:
    weighted overall MSE <= baseline MSE * 1.05
    weighted zero MAE    <= baseline zero MAE * 1.10
    weighted low MAE     <= baseline low MAE * 1.10

Among eligible weighted candidates:
    1. minimum high-damage MAE
    2. minimum absolute high-damage bias
    3. minimum high-damage underestimation ratio
    4. minimum overall validation MSE

If no weighted candidate is eligible:
    fall back to the frozen standard SVR for that seed.

IMPORTANT
---------
- Exact split files from the previous repeated-split experiment
  are reused.
- The frozen standard-SVR seed results are reused.
- Standard test predictions are reused rather than retrained.
- Weighted test prediction is generated only AFTER validation
  selection for the corresponding seed.
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
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

HIGH_DAMAGE_THRESHOLD = 0.20
WEIGHT_MULTIPLIER = 1.5

OVERALL_MSE_BUDGET = 1.05
ZERO_MAE_BUDGET = 1.10
LOW_MAE_BUDGET = 1.10


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


def make_storey_weights(
    target: np.ndarray,
) -> np.ndarray:

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    raw = np.where(
        target > HIGH_DAMAGE_THRESHOLD,
        WEIGHT_MULTIPLIER,
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


def _fit_one_storey(
    x_train: np.ndarray,
    target_train: np.ndarray,
    x_predict: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
    weighted: bool,
) -> tuple[
    np.ndarray,
    int,
]:

    if weighted:
        sample_weight = (
            make_storey_weights(
                target_train
            )
        )
    else:
        sample_weight = None

    model = SVR(
        kernel="rbf",
        C=C,
        gamma=gamma,
        epsilon=epsilon,
        tol=1e-4,
        cache_size=512,
    )

    if sample_weight is None:
        model.fit(
            x_train,
            target_train,
        )
    else:
        model.fit(
            x_train,
            target_train,
            sample_weight=sample_weight,
        )

    prediction = model.predict(
        x_predict
    )

    return (
        prediction,
        int(
            len(model.support_)
        ),
    )


def fit_multioutput_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
    weighted: bool,
    n_jobs: int,
) -> tuple[
    np.ndarray,
    list[int],
]:

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
            weighted=weighted,
        )
        for storey in range(
            y_train.shape[1]
        )
    )

    predictions = [
        item[0]
        for item in outputs
    ]

    support_counts = [
        item[1]
        for item in outputs
    ]

    return (
        clip_prediction(
            np.column_stack(
                predictions
            )
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
            abs(high_bias)
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

    standard_error = float(
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
        critical
        * standard_error
    )

    return (
        mean_value - margin,
        mean_value + margin,
    )


def safe_wilcoxon(
    weighted_values: np.ndarray,
    standard_values: np.ndarray,
) -> tuple[
    float,
    float,
]:

    try:
        result = wilcoxon(
            weighted_values,
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


def candidate_name(
    C: float,
    gamma: float,
    epsilon: float,
) -> str:

    gamma_label = (
        f"{gamma:g}"
        .replace(
            ".",
            "p",
        )
    )

    return (
        "weighted_svr_w_1p5"
        f"_C_{C:g}"
        f"_gamma_{gamma_label}"
        f"_eps_{epsilon:g}"
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
            "repeated_split_weighted_svr_78d"
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
            "Unexpected full raw feature shape."
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
            "Expected exactly ten frozen "
            "standard SVR rows."
        )

    if (
        standard_frame[
            "seed"
        ]
        .tolist()
        != SEEDS
    ):
        raise RuntimeError(
            "Standard SVR seed rows "
            "must be 0..9 exactly."
        )

    total_weighted_candidates = (
        len(C_VALUES)
        * len(GAMMA_VALUES)
        * len(EPSILON_VALUES)
    )

    print(
        "===== 10-SEED STANDARD VS "
        "DAMAGE-WEIGHTED 78D SVR ====="
    )

    print(
        "Raw reconstruction mode:",
        loading_metadata[
            "reconstruction_mode"
        ],
    )

    print(
        "Weight multiplier:",
        WEIGHT_MULTIPLIER,
    )

    print(
        "High-damage threshold:",
        HIGH_DAMAGE_THRESHOLD,
    )

    print(
        "Weighted candidates / seed:",
        total_weighted_candidates,
    )

    print(
        "Exact previous split files reused:",
        True,
    )

    print(
        "Frozen standard test results reused:",
        True,
    )

    print()

    selected_rows = []
    candidate_rows = []

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

        x_test_raw = (
            x_full_raw[
                test_idx
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

        x_test = (
            scaler.transform(
                x_test_raw
            )
        )

        # ---------------------------------------
        # Reproduce frozen standard validation
        # model for this seed.
        # ---------------------------------------

        standard_row = (
            standard_frame.loc[
                standard_frame[
                    "seed"
                ]
                == seed
            ]
            .iloc[0]
        )

        standard_C = float(
            standard_row[
                "selected_C"
            ]
        )

        standard_gamma = float(
            standard_row[
                "selected_gamma"
            ]
        )

        standard_epsilon = float(
            standard_row[
                "selected_epsilon"
            ]
        )

        (
            standard_val_prediction,
            standard_support_counts,
        ) = fit_multioutput_svr(
            x_train=x_train,
            y_train=y_train,
            x_predict=x_val,
            C=standard_C,
            gamma=standard_gamma,
            epsilon=standard_epsilon,
            weighted=False,
            n_jobs=args.n_jobs,
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
                f"STOP seed {seed}: frozen "
                "standard validation MSE "
                "was not reproduced.\n"
                f"actual="
                f"{standard_val_metrics['mse']:.12f}\n"
                f"stored="
                f"{stored_val_mse:.12f}"
            )

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

        print(
            "  Standard selected:",
            standard_row[
                "selected_candidate"
            ],
        )

        print(
            "  Standard validation MSE reproduced:",
            f"{standard_val_metrics['mse']:.10f}",
        )

        print(
            "  Eligibility limits:",
            (
                f"MSE={mse_limit:.10f}, "
                f"zero={zero_limit:.10f}, "
                f"low={low_limit:.10f}"
            ),
        )

        # ---------------------------------------
        # Search frozen weighted candidate grid.
        # ---------------------------------------

        seed_candidate_rows = []

        counter = 0

        for C in C_VALUES:
            for gamma in GAMMA_VALUES:
                for epsilon in EPSILON_VALUES:

                    counter += 1

                    (
                        prediction,
                        support_counts,
                    ) = fit_multioutput_svr(
                        x_train=x_train,
                        y_train=y_train,
                        x_predict=x_val,
                        C=C,
                        gamma=gamma,
                        epsilon=epsilon,
                        weighted=True,
                        n_jobs=args.n_jobs,
                    )

                    metrics = (
                        calculate_metrics(
                            y_val,
                            prediction,
                        )
                    )

                    eligible = bool(
                        (
                            metrics[
                                "mse"
                            ]
                            <= mse_limit
                        )
                        and (
                            metrics[
                                "zero_mae"
                            ]
                            <= zero_limit
                        )
                        and (
                            metrics[
                                "low_mae"
                            ]
                            <= low_limit
                        )
                    )

                    row = {
                        "seed": seed,
                        "candidate": (
                            candidate_name(
                                C=C,
                                gamma=gamma,
                                epsilon=epsilon,
                            )
                        ),
                        "weight_multiplier": (
                            WEIGHT_MULTIPLIER
                        ),
                        "C": float(C),
                        "gamma": float(
                            gamma
                        ),
                        "epsilon": float(
                            epsilon
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
                        "eligible": (
                            eligible
                        ),
                        "selected": False,
                    }

                    seed_candidate_rows.append(
                        row
                    )

                    if (
                        counter % 15
                        == 0
                        or counter
                        == total_weighted_candidates
                    ):
                        print(
                            "    weighted progress:",
                            f"{counter}/"
                            f"{total_weighted_candidates}",
                        )

        seed_candidates = (
            pd.DataFrame(
                seed_candidate_rows
            )
        )

        eligible_frame = (
            seed_candidates.loc[
                seed_candidates[
                    "eligible"
                ]
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

        eligible_count = int(
            len(
                eligible_frame
            )
        )

        if eligible_frame.empty:
            selection_mode = (
                "fallback_standard"
            )

            selected_weighted = None

            selected_candidate = str(
                standard_row[
                    "selected_candidate"
                ]
            )

            weighted_C = standard_C
            weighted_gamma = (
                standard_gamma
            )
            weighted_epsilon = (
                standard_epsilon
            )

            selected_validation_metrics = (
                standard_val_metrics
            )

            selected_support_ratio = float(
                np.mean(
                    standard_support_counts
                )
                / x_train.shape[0]
            )

        else:
            selection_mode = (
                "weighted_candidate"
            )

            selected_weighted = (
                eligible_frame.iloc[
                    0
                ]
            )

            selected_candidate = str(
                selected_weighted[
                    "candidate"
                ]
            )

            weighted_C = float(
                selected_weighted["C"]
            )

            weighted_gamma = float(
                selected_weighted[
                    "gamma"
                ]
            )

            weighted_epsilon = float(
                selected_weighted[
                    "epsilon"
                ]
            )

            selected_validation_metrics = {
                "mse": float(
                    selected_weighted[
                        "val_mse"
                    ]
                ),
                "mae": float(
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

            selected_support_ratio = float(
                selected_weighted[
                    "mean_support_vector_ratio"
                ]
            )

            seed_candidates.loc[
                seed_candidates[
                    "candidate"
                ]
                == selected_candidate,
                "selected",
            ] = True

        candidate_rows.extend(
            seed_candidates.to_dict(
                orient="records"
            )
        )

        # ---------------------------------------
        # Test only after validation selection.
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
            frozen_test_idx = (
                np.asarray(
                    data[
                        "test_idx"
                    ],
                    dtype=np.int64,
                )
            )

            frozen_y_test = (
                np.asarray(
                    data[
                        "y_test"
                    ],
                    dtype=np.float64,
                )
            )

            standard_test_prediction = (
                np.asarray(
                    data[
                        "svr_prediction"
                    ],
                    dtype=np.float64,
                )
            )

        if not np.array_equal(
            frozen_test_idx,
            test_idx,
        ):
            raise RuntimeError(
                f"Seed {seed}: frozen "
                "standard prediction indices "
                "do not match split file."
            )

        if not np.array_equal(
            frozen_y_test,
            y_test,
        ):
            raise RuntimeError(
                f"Seed {seed}: frozen y_test "
                "does not match canonical data."
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
                f"Seed {seed}: frozen "
                "standard test MAE lock failed."
            )

        if (
            selection_mode
            == "fallback_standard"
        ):
            weighted_test_prediction = (
                standard_test_prediction.copy()
            )

        else:
            (
                weighted_test_prediction,
                _,
            ) = fit_multioutput_svr(
                x_train=x_train,
                y_train=y_train,
                x_predict=x_test,
                C=weighted_C,
                gamma=weighted_gamma,
                epsilon=weighted_epsilon,
                weighted=True,
                n_jobs=args.n_jobs,
            )

        weighted_test_metrics = (
            calculate_metrics(
                y_test,
                weighted_test_prediction,
            )
        )

        selected_rows.append(
            {
                "seed": seed,
                "selection_mode": (
                    selection_mode
                ),
                "eligible_candidate_count": (
                    eligible_count
                ),
                "selected_candidate": (
                    selected_candidate
                ),
                "selected_C": (
                    weighted_C
                ),
                "selected_gamma": (
                    weighted_gamma
                ),
                "selected_epsilon": (
                    weighted_epsilon
                ),
                "standard_C": (
                    standard_C
                ),
                "standard_gamma": (
                    standard_gamma
                ),
                "standard_epsilon": (
                    standard_epsilon
                ),
                "baseline_val_mse": (
                    standard_val_metrics[
                        "mse"
                    ]
                ),
                "selected_val_mse": (
                    selected_validation_metrics[
                        "mse"
                    ]
                ),
                "baseline_val_zero_mae": (
                    standard_val_metrics[
                        "zero_mae"
                    ]
                ),
                "selected_val_zero_mae": (
                    selected_validation_metrics[
                        "zero_mae"
                    ]
                ),
                "baseline_val_low_mae": (
                    standard_val_metrics[
                        "low_mae"
                    ]
                ),
                "selected_val_low_mae": (
                    selected_validation_metrics[
                        "low_mae"
                    ]
                ),
                "baseline_val_high_mae": (
                    standard_val_metrics[
                        "high_mae"
                    ]
                ),
                "selected_val_high_mae": (
                    selected_validation_metrics[
                        "high_mae"
                    ]
                ),
                "baseline_val_high_abs_bias": (
                    standard_val_metrics[
                        "high_abs_bias"
                    ]
                ),
                "selected_val_high_abs_bias": (
                    selected_validation_metrics[
                        "high_abs_bias"
                    ]
                ),
                "baseline_val_high_underestimation_ratio": (
                    standard_val_metrics[
                        "high_underestimation_ratio"
                    ]
                ),
                "selected_val_high_underestimation_ratio": (
                    selected_validation_metrics[
                        "high_underestimation_ratio"
                    ]
                ),
                "selected_support_vector_ratio": (
                    selected_support_ratio
                ),
            }
        )

        paired_test_row = {
            "seed": seed,
            "selection_mode": (
                selection_mode
            ),
            "eligible_candidate_count": (
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
            metric_key,
        ) in metric_mapping.items():

            standard_value = float(
                standard_test_metrics[
                    metric_key
                ]
            )

            weighted_value = float(
                weighted_test_metrics[
                    metric_key
                ]
            )

            paired_test_row[
                f"standard_{output_name}"
            ] = standard_value

            paired_test_row[
                f"weighted_{output_name}"
            ] = weighted_value

            paired_test_row[
                f"difference_weighted_minus_standard_{output_name}"
            ] = (
                weighted_value
                - standard_value
            )

            if standard_value > 0.0:
                paired_test_row[
                    f"relative_improvement_percent_{output_name}"
                ] = (
                    (
                        standard_value
                        - weighted_value
                    )
                    / standard_value
                    * 100.0
                )

        paired_test_rows.append(
            paired_test_row
        )

        np.savez_compressed(
            prediction_root
            / f"seed_{seed}_weighted_predictions.npz",
            seed=np.asarray(seed),
            test_idx=test_idx,
            y_test=y_test,
            standard_prediction=(
                standard_test_prediction
            ),
            weighted_prediction=(
                weighted_test_prediction
            ),
        )

        print(
            "  Eligible weighted candidates:",
            eligible_count,
        )

        print(
            "  Selection mode:",
            selection_mode,
        )

        print(
            "  Selected:",
            selected_candidate,
        )

        print(
            "  Standard test MAE:",
            f"{standard_test_metrics['mae']:.10f}",
        )

        print(
            "  Weighted test MAE:",
            f"{weighted_test_metrics['mae']:.10f}",
        )

        print(
            "  Standard high MAE:",
            f"{standard_test_metrics['high_mae']:.10f}",
        )

        print(
            "  Weighted high MAE:",
            f"{weighted_test_metrics['high_mae']:.10f}",
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
            "  Weighted high underestimation:",
            (
                f"{weighted_test_metrics[
                    'high_underestimation_ratio'
                ]:.10f}"
            ),
        )

        print()

    selected_frame = (
        pd.DataFrame(
            selected_rows
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )

    paired_frame = (
        pd.DataFrame(
            paired_test_rows
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )

    candidate_frame = (
        pd.DataFrame(
            candidate_rows
        )
    )

    # ---------------------------------------
    # Paired aggregate statistics
    # ---------------------------------------

    paired_statistics_rows = []

    for metric in PRIMARY_METRICS:

        standard_values = (
            paired_frame[
                f"standard_{metric}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        weighted_values = (
            paired_frame[
                f"weighted_{metric}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        differences = (
            weighted_values
            - standard_values
        )

        (
            ci_low,
            ci_high,
        ) = confidence_interval(
            differences
        )

        relative_improvement = (
            (
                standard_values
                - weighted_values
            )
            / standard_values
            * 100.0
        )

        t_result = ttest_rel(
            weighted_values,
            standard_values,
        )

        (
            wilcoxon_stat,
            wilcoxon_p,
        ) = safe_wilcoxon(
            weighted_values,
            standard_values,
        )

        paired_statistics_rows.append(
            {
                "metric": metric,
                "difference_definition": (
                    "weighted_minus_standard"
                ),
                "n_seeds": int(
                    len(SEEDS)
                ),
                "standard_mean": float(
                    np.mean(
                        standard_values
                    )
                ),
                "weighted_mean": float(
                    np.mean(
                        weighted_values
                    )
                ),
                "mean_difference": float(
                    np.mean(
                        differences
                    )
                ),
                "std_difference": float(
                    np.std(
                        differences,
                        ddof=1,
                    )
                ),
                "ci95_difference_low": (
                    ci_low
                ),
                "ci95_difference_high": (
                    ci_high
                ),
                "weighted_wins": int(
                    np.sum(
                        weighted_values
                        < standard_values
                    )
                ),
                "standard_wins": int(
                    np.sum(
                        standard_values
                        < weighted_values
                    )
                ),
                "ties": int(
                    np.sum(
                        np.isclose(
                            weighted_values,
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

    paired_statistics = (
        pd.DataFrame(
            paired_statistics_rows
        )
    )

    # ---------------------------------------
    # Hyperparameter and fallback stability
    # ---------------------------------------

    weighted_only = (
        selected_frame.loc[
            selected_frame[
                "selection_mode"
            ]
            == "weighted_candidate"
        ]
    )

    if weighted_only.empty:
        frequency = pd.DataFrame(
            columns=[
                "selected_C",
                "selected_gamma",
                "selected_epsilon",
                "count",
            ]
        )
    else:
        frequency = (
            weighted_only.groupby(
                [
                    "selected_C",
                    "selected_gamma",
                    "selected_epsilon",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                [
                    "count",
                    "selected_C",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    fallback_count = int(
        np.sum(
            selected_frame[
                "selection_mode"
            ]
            == "fallback_standard"
        )
    )

    # ---------------------------------------
    # Predeclared interpretation diagnostics
    # ---------------------------------------

    def stat_row(
        metric: str,
    ) -> pd.Series:
        return (
            paired_statistics.loc[
                paired_statistics[
                    "metric"
                ]
                == metric
            ]
            .iloc[0]
        )

    overall = stat_row(
        "test_mae"
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

    overall_change_percent = (
        -float(
            overall[
                "mean_relative_improvement_percent"
            ]
        )
    )

    high_mae_improvement = float(
        high_mae[
            "mean_relative_improvement_percent"
        ]
    )

    high_bias_improvement = float(
        high_bias[
            "mean_relative_improvement_percent"
        ]
    )

    under_difference_points = (
        float(
            high_under[
                "mean_difference"
            ]
        )
        * 100.0
    )

    high_mae_wins = int(
        high_mae[
            "weighted_wins"
        ]
    )

    high_bias_wins = int(
        high_bias[
            "weighted_wins"
        ]
    )

    under_wins = int(
        high_under[
            "weighted_wins"
        ]
    )

    if (
        high_mae_improvement >= 10.0
        and high_bias_improvement >= 10.0
        and high_mae_wins >= 8
        and high_bias_wins >= 8
        and under_wins >= 7
        and under_difference_points <= -5.0
        and overall_change_percent <= 5.0
        and fallback_count <= 2
    ):
        interpretation_decision = (
            "WEIGHTING_ROBUST_AND_DIRECTIONAL_"
            "UNDERPREDICTION_SUBSTANTIALLY_REDUCED"
        )

    elif (
        high_mae_improvement >= 10.0
        and high_bias_improvement >= 10.0
        and high_mae_wins >= 8
        and high_bias_wins >= 8
        and under_wins >= 6
        and under_difference_points < 0.0
        and overall_change_percent <= 5.0
        and fallback_count <= 3
    ):
        interpretation_decision = (
            "WEIGHTING_ROBUST_FOR_ERROR_MAGNITUDE_"
            "BUT_DIRECTIONAL_BIAS_REMAINS"
        )

    else:
        interpretation_decision = (
            "WEIGHTING_NOT_ROBUST_ENOUGH_"
            "FOR_FORMAL_METHOD_EXTENSION"
        )

    # ---------------------------------------
    # Save
    # ---------------------------------------

    selected_path = (
        args.output_root
        / "weighted_selected_by_seed.csv"
    )

    candidate_path = (
        args.output_root
        / "weighted_validation_candidates.csv"
    )

    paired_path = (
        args.output_root
        / "weighted_paired_test_results.csv"
    )

    statistics_path = (
        args.output_root
        / "weighted_paired_statistics.csv"
    )

    frequency_path = (
        args.output_root
        / "weighted_hyperparameter_frequency.csv"
    )

    report_path = (
        args.output_root
        / "weighted_repeated_split_report.json"
    )

    selected_frame.to_csv(
        selected_path,
        index=False,
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    paired_frame.to_csv(
        paired_path,
        index=False,
    )

    paired_statistics.to_csv(
        statistics_path,
        index=False,
    )

    frequency.to_csv(
        frequency_path,
        index=False,
    )

    report = {
        "experiment": (
            "10_seed_standard_vs_"
            "severity_weighted_78D_RBF_SVR"
        ),
        "descriptor_set": (
            "signal_derived_with_ground"
        ),
        "seeds": SEEDS,
        "split_reuse": True,
        "weight_multiplier": (
            WEIGHT_MULTIPLIER
        ),
        "high_damage_threshold": (
            HIGH_DAMAGE_THRESHOLD
        ),
        "weight_normalisation": (
            "storey-specific mean weight = 1"
        ),
        "weighted_grid": {
            "C": C_VALUES,
            "gamma": GAMMA_VALUES,
            "epsilon": EPSILON_VALUES,
            "candidate_count_per_seed": (
                total_weighted_candidates
            ),
        },
        "eligibility": {
            "overall_mse_budget": (
                OVERALL_MSE_BUDGET
            ),
            "zero_mae_budget": (
                ZERO_MAE_BUDGET
            ),
            "low_mae_budget": (
                LOW_MAE_BUDGET
            ),
        },
        "selection_rule": (
            "Among eligible weighted candidates: "
            "min high MAE, then min |high bias|, "
            "then min high underestimation, "
            "then min validation MSE."
        ),
        "fallback_rule": (
            "If no weighted candidate satisfies "
            "validation constraints, use the frozen "
            "standard SVR for that seed."
        ),
        "fallback_count": (
            fallback_count
        ),
        "interpretation_decision": (
            interpretation_decision
        ),
        "overall_test_mae_change_percent": (
            overall_change_percent
        ),
        "high_damage_mae_improvement_percent": (
            high_mae_improvement
        ),
        "high_damage_abs_bias_improvement_percent": (
            high_bias_improvement
        ),
        "high_damage_underestimation_change_percentage_points": (
            under_difference_points
        ),
        "statistical_note": (
            "Ten repeated paired holdout splits reuse "
            "the same 3000 simulated cases. Effect "
            "sizes, paired direction and intervals "
            "are primary; p-values are supportive."
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

    # ---------------------------------------
    # Console
    # ---------------------------------------

    print()
    print(
        "===== SEED-BY-SEED SELECTION ====="
    )

    print(
        selected_frame[
            [
                "seed",
                "selection_mode",
                "eligible_candidate_count",
                "selected_candidate",
                "selected_C",
                "selected_gamma",
                "selected_epsilon",
                "baseline_val_mse",
                "selected_val_mse",
                "baseline_val_high_mae",
                "selected_val_high_mae",
                "baseline_val_high_underestimation_ratio",
                "selected_val_high_underestimation_ratio",
                "selected_support_vector_ratio",
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
                "standard_test_mae",
                "weighted_test_mae",
                "standard_high_mae",
                "weighted_high_mae",
                "standard_high_abs_bias",
                "weighted_high_abs_bias",
                "standard_high_underestimation_ratio",
                "weighted_high_underestimation_ratio",
                "standard_zero_mae",
                "weighted_zero_mae",
                "standard_low_mae",
                "weighted_low_mae",
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
        "===== PAIRED WEIGHTING STATISTICS ====="
    )

    print(
        paired_statistics[
            [
                "metric",
                "standard_mean",
                "weighted_mean",
                "mean_difference",
                "ci95_difference_low",
                "ci95_difference_high",
                "weighted_wins",
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
        "===== WEIGHTED HYPERPARAMETER "
        "FREQUENCY ====="
    )

    if frequency.empty:
        print(
            "No weighted candidate selected."
        )
    else:
        print(
            frequency.to_string(
                index=False
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
        f"{overall_change_percent:.6f}",
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
        "High-damage underestimation "
        "change (percentage points):",
        f"{under_difference_points:.6f}",
    )

    print(
        "High-MAE weighted wins:",
        high_mae_wins,
        "/ 10",
    )

    print(
        "High-|bias| weighted wins:",
        high_bias_wins,
        "/ 10",
    )

    print(
        "Underestimation weighted wins:",
        under_wins,
        "/ 10",
    )

    print(
        "Decision:",
        interpretation_decision,
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
            selected_frame
        ),
    )

    print(
        "Exact previous split files reused:",
        True,
    )

    print(
        "Frozen standard seed models reused:",
        True,
    )

    print(
        "Per-seed training-only scaling:",
        True,
    )

    print(
        "Weight multiplier frozen:",
        WEIGHT_MULTIPLIER,
    )

    print(
        "Eligibility constraints frozen:",
        True,
    )

    print(
        "Weighted candidate grid frozen:",
        True,
    )

    print(
        "Test evaluated only after "
        "weighted validation selection:",
        True,
    )

    print()
    print(
        "CHECK PASSED: 10-seed "
        "damage-weighted SVR validation "
        "completed."
    )

    print(
        "Selected:",
        selected_path,
    )

    print(
        "Paired statistics:",
        statistics_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
