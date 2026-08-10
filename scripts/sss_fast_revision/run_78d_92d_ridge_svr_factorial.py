"""
Fixed-split 2x2 factorial comparison:

Descriptor information:
- 78D signal_derived_with_ground
- 92D oracle_full / privileged-information reference

Estimator:
- Ridge
- RBF-SVR

Scientific purpose
------------------
Separate:
1. descriptor-information effect;
2. estimator-nonlinearity effect.

Protocol
--------
- Frozen identical train/validation/test partitions.
- Validation MSE selects hyperparameters.
- Predictions clipped to [0, 0.5].
- Test is evaluated only after validation selection.
- Same SVR candidate grid is used for both descriptor sets.
- This is a fixed-split diagnostic; repeated-split validation
  will provide the stronger robustness evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


RIDGE_ALPHAS = [
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    300.0,
]

SVR_C_VALUES = [
    100.0,
    300.0,
    500.0,
    750.0,
    1000.0,
]

SVR_GAMMA_VALUES = [
    0.0005,
    0.00075,
    0.001,
    0.003,
    0.01,
]

SVR_EPSILON_VALUES = [
    0.02,
    0.03,
    0.04,
]

EXPECTED_FEATURE_COUNTS = {
    "signal_derived_with_ground": 78,
    "oracle_full": 92,
}

# Verified locks from previous experiments.
RIDGE_LOCKS = {
    "signal_derived_with_ground": {
        "alpha": 3e-4,
        "test_mae": 0.0450937333,
    },
    "oracle_full": {
        "alpha": 1e-4,
        "test_mae": 0.0392874519,
    },
}

SVR_78D_ANCHOR = {
    "C": 1000.0,
    "gamma": 0.00075,
    "epsilon": 0.03,
    "val_mse": 0.002256262230781473,
}


def load_dataset(
    path: Path,
    expected_features: int,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)

    required = {
        "F_train",
        "F_val",
        "F_test",
        "y_train",
        "y_val",
        "y_test",
    }

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        missing = sorted(
            required - set(data.files)
        )

        if missing:
            raise KeyError(
                f"{path} missing arrays: {missing}"
            )

        arrays = {
            key: np.asarray(data[key])
            for key in data.files
        }

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
                f"{path}: {key} contains "
                "non-finite values."
            )

    if (
        arrays["F_train"].shape[1]
        != expected_features
    ):
        raise ValueError(
            f"{path}: expected "
            f"{expected_features} features, "
            f"found "
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


def confirm_shared_partitions(
    data_78: dict[str, np.ndarray],
    data_92: dict[str, np.ndarray],
) -> None:
    """
    Verify that 78D and 92D use exactly the same targets
    and, where available, the same partition identifiers.
    """
    for key in [
        "y_train",
        "y_val",
        "y_test",
    ]:
        if not np.array_equal(
            data_78[key],
            data_92[key],
        ):
            raise RuntimeError(
                f"STOP: {key} differs between "
                "78D and 92D datasets."
            )

    optional_partition_keys = [
        "train_idx",
        "val_idx",
        "test_idx",
        "case_id_train",
        "case_id_val",
        "case_id_test",
    ]

    for key in optional_partition_keys:
        if (
            key in data_78
            and key in data_92
        ):
            if not np.array_equal(
                data_78[key],
                data_92[key],
            ):
                raise RuntimeError(
                    f"STOP: partition identifier "
                    f"{key} differs."
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


def mse(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    error = (
        np.asarray(prediction)
        - np.asarray(truth)
    )

    return float(
        np.mean(error ** 2)
    )


def mae(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction)
                - np.asarray(truth)
            )
        )
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

    damaged_mask = (
        truth > 1e-12
    )

    zero_mask = (
        truth <= 1e-12
    )

    high_mask = (
        truth > 0.20
    )

    if not np.any(damaged_mask):
        raise ValueError(
            "No damaged entries."
        )

    if not np.any(zero_mask):
        raise ValueError(
            "No zero-damage entries."
        )

    if not np.any(high_mask):
        raise ValueError(
            "No high-damage entries."
        )

    return {
        "test_mae": float(
            np.mean(
                np.abs(error)
            )
        ),
        "test_rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
        "test_bias": float(
            np.mean(error)
        ),
        "damaged_entry_mae": float(
            np.mean(
                np.abs(
                    error[
                        damaged_mask
                    ]
                )
            )
        ),
        "zero_damage_mae": float(
            np.mean(
                np.abs(
                    error[
                        zero_mask
                    ]
                )
            )
        ),
        "high_damage_mae": float(
            np.mean(
                np.abs(
                    error[
                        high_mask
                    ]
                )
            )
        ),
        "high_damage_bias": float(
            np.mean(
                error[
                    high_mask
                ]
            )
        ),
        "high_damage_underestimation_ratio": float(
            np.mean(
                prediction[
                    high_mask
                ]
                < truth[
                    high_mask
                ]
            )
        ),
    }


def calculate_damage_bins(
    descriptor_set: str,
    model_name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:

    bins = {
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
    }

    rows = []

    for bin_name, mask in bins.items():

        if not np.any(mask):
            raise ValueError(
                f"Empty damage bin: "
                f"{bin_name}"
            )

        y_true = truth[mask]
        y_pred = prediction[mask]

        error = (
            y_pred - y_true
        )

        rows.append(
            {
                "descriptor_set": descriptor_set,
                "model": model_name,
                "damage_bin": bin_name,
                "n_entries": int(
                    np.sum(mask)
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
                "mean_true": float(
                    np.mean(
                        y_true
                    )
                ),
                "mean_prediction": float(
                    np.mean(
                        y_pred
                    )
                ),
                "underestimation_ratio": (
                    np.nan
                    if bin_name == "zero"
                    else float(
                        np.mean(
                            y_pred < y_true
                        )
                    )
                ),
            }
        )

    return rows


def select_ridge(
    descriptor_set: str,
    arrays: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
]:

    x_train = arrays["F_train"]
    x_val = arrays["F_val"]
    x_test = arrays["F_test"]

    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]

    candidate_rows = []

    best_model = None
    best_row = None

    for alpha in RIDGE_ALPHAS:

        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
        )

        start = time.perf_counter()

        model.fit(
            x_train,
            y_train,
        )

        fit_seconds = float(
            time.perf_counter()
            - start
        )

        val_prediction = (
            clip_prediction(
                model.predict(
                    x_val
                )
            )
        )

        row = {
            "descriptor_set": descriptor_set,
            "model": "ridge",
            "candidate": (
                f"ridge_alpha_{alpha:g}"
            ),
            "alpha": float(alpha),
            "C": np.nan,
            "gamma": np.nan,
            "epsilon": np.nan,
            "val_mse": mse(
                y_val,
                val_prediction,
            ),
            "val_mae": mae(
                y_val,
                val_prediction,
            ),
            "fit_seconds": fit_seconds,
            "mean_support_vector_ratio": (
                np.nan
            ),
            "selected": False,
        }

        candidate_rows.append(row)

        if best_row is None:
            best_row = row.copy()
            best_model = model

        else:
            current_key = (
                row["val_mse"],
                row["candidate"],
            )

            best_key = (
                best_row["val_mse"],
                best_row["candidate"],
            )

            if current_key < best_key:
                best_row = row.copy()
                best_model = model

    if (
        best_model is None
        or best_row is None
    ):
        raise RuntimeError(
            "Ridge selection failed."
        )

    for row in candidate_rows:
        row["selected"] = (
            row["candidate"]
            == best_row["candidate"]
        )

    # Test accessed only after validation selection.
    test_prediction = (
        clip_prediction(
            best_model.predict(
                x_test
            )
        )
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
        "model": "ridge",
        "selected_candidate": (
            best_row["candidate"]
        ),
        "selected_alpha": float(
            best_row["alpha"]
        ),
        "selected_C": np.nan,
        "selected_gamma": np.nan,
        "selected_epsilon": np.nan,
        "val_mse": float(
            best_row["val_mse"]
        ),
        "val_mae": float(
            best_row["val_mae"]
        ),
        **metrics,
    }

    return (
        summary,
        candidate_rows,
        test_prediction,
    )


def select_svr(
    descriptor_set: str,
    arrays: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
]:

    x_train = arrays["F_train"]
    x_val = arrays["F_val"]
    x_test = arrays["F_test"]

    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]

    candidate_rows = []

    best_model = None
    best_row = None

    for (
        C,
        gamma,
        epsilon,
    ) in itertools.product(
        SVR_C_VALUES,
        SVR_GAMMA_VALUES,
        SVR_EPSILON_VALUES,
    ):

        model = (
            MultiOutputRegressor(
                SVR(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    epsilon=epsilon,
                    tol=1e-4,
                    cache_size=1024,
                ),
                n_jobs=1,
            )
        )

        start = time.perf_counter()

        model.fit(
            x_train,
            y_train,
        )

        fit_seconds = float(
            time.perf_counter()
            - start
        )

        val_prediction = (
            clip_prediction(
                model.predict(
                    x_val
                )
            )
        )

        support_counts = [
            len(
                estimator.support_
            )
            for estimator
            in model.estimators_
        ]

        gamma_label = (
            f"{gamma:g}"
            .replace(".", "p")
        )

        candidate = (
            f"rbf_svr_C_{C:g}"
            f"_gamma_{gamma_label}"
            f"_eps_{epsilon:g}"
        )

        row = {
            "descriptor_set": descriptor_set,
            "model": "rbf_svr",
            "candidate": candidate,
            "alpha": np.nan,
            "C": float(C),
            "gamma": float(gamma),
            "epsilon": float(epsilon),
            "val_mse": mse(
                y_val,
                val_prediction,
            ),
            "val_mae": mae(
                y_val,
                val_prediction,
            ),
            "fit_seconds": fit_seconds,
            "mean_support_vector_ratio": float(
                np.mean(
                    support_counts
                )
                / x_train.shape[0]
            ),
            "selected": False,
        }

        candidate_rows.append(row)

        if best_row is None:
            best_row = row.copy()
            best_model = model

        else:
            current_key = (
                row["val_mse"],
                row["candidate"],
            )

            best_key = (
                best_row["val_mse"],
                best_row["candidate"],
            )

            if current_key < best_key:
                best_row = row.copy()
                best_model = model

    if (
        best_model is None
        or best_row is None
    ):
        raise RuntimeError(
            "SVR selection failed."
        )

    for row in candidate_rows:
        row["selected"] = (
            row["candidate"]
            == best_row["candidate"]
        )

    # Test accessed only after validation selection.
    test_prediction = (
        clip_prediction(
            best_model.predict(
                x_test
            )
        )
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
        "model": "rbf_svr",
        "selected_candidate": (
            best_row["candidate"]
        ),
        "selected_alpha": np.nan,
        "selected_C": float(
            best_row["C"]
        ),
        "selected_gamma": float(
            best_row["gamma"]
        ),
        "selected_epsilon": float(
            best_row["epsilon"]
        ),
        "val_mse": float(
            best_row["val_mse"]
        ),
        "val_mae": float(
            best_row["val_mae"]
        ),
        **metrics,
        "mean_support_vector_ratio": float(
            best_row[
                "mean_support_vector_ratio"
            ]
        ),
    }

    return (
        summary,
        candidate_rows,
        test_prediction,
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
        "--input-78d",
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
        "--input-92d",
        type=Path,
        default=Path(
            "data_processed/"
            "sss_fast_revision/"
            "descriptor_sets/"
            "oracle_full/"
            "debug_plus_3000_"
            "oracle_full_features.npz"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "factorial_78d_92d_ridge_svr"
        ),
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = {
        "signal_derived_with_ground": (
            load_dataset(
                args.input_78d,
                78,
            )
        ),
        "oracle_full": (
            load_dataset(
                args.input_92d,
                92,
            )
        ),
    }

    confirm_shared_partitions(
        datasets[
            "signal_derived_with_ground"
        ],
        datasets[
            "oracle_full"
        ],
    )

    print(
        "===== 78D / 92D × RIDGE / SVR "
        "FACTORIAL COMPARISON ====="
    )

    print(
        "Shared partitions check: PASSED"
    )

    print(
        "78D shape:",
        datasets[
            "signal_derived_with_ground"
        ]["F_train"].shape,
    )

    print(
        "92D shape:",
        datasets[
            "oracle_full"
        ]["F_train"].shape,
    )

    print(
        "SVR candidates per descriptor set:",
        (
            len(SVR_C_VALUES)
            * len(
                SVR_GAMMA_VALUES
            )
            * len(
                SVR_EPSILON_VALUES
            )
        ),
    )

    print()

    summaries = []
    candidate_rows = []
    bin_rows = []

    predictions = {}

    for descriptor_set in [
        "signal_derived_with_ground",
        "oracle_full",
    ]:

        arrays = datasets[
            descriptor_set
        ]

        print(
            f"===== {descriptor_set} ====="
        )

        print(
            "Training Ridge ..."
        )

        (
            ridge_summary,
            ridge_candidates,
            ridge_prediction,
        ) = select_ridge(
            descriptor_set,
            arrays,
        )

        summaries.append(
            ridge_summary
        )

        candidate_rows.extend(
            ridge_candidates
        )

        predictions[
            (
                descriptor_set,
                "ridge",
            )
        ] = ridge_prediction

        bin_rows.extend(
            calculate_damage_bins(
                descriptor_set,
                "ridge",
                arrays["y_test"],
                ridge_prediction,
            )
        )

        print(
            "  Ridge selected:",
            ridge_summary[
                "selected_candidate"
            ],
        )

        print(
            "  Ridge test MAE:",
            f"{ridge_summary['test_mae']:.10f}",
        )

        print(
            "Training RBF-SVR ..."
        )

        (
            svr_summary,
            svr_candidates,
            svr_prediction,
        ) = select_svr(
            descriptor_set,
            arrays,
        )

        summaries.append(
            svr_summary
        )

        candidate_rows.extend(
            svr_candidates
        )

        predictions[
            (
                descriptor_set,
                "rbf_svr",
            )
        ] = svr_prediction

        bin_rows.extend(
            calculate_damage_bins(
                descriptor_set,
                "rbf_svr",
                arrays["y_test"],
                svr_prediction,
            )
        )

        print(
            "  SVR selected:",
            svr_summary[
                "selected_candidate"
            ],
        )

        print(
            "  SVR val MSE:",
            f"{svr_summary['val_mse']:.10f}",
        )

        print(
            "  SVR test MAE:",
            f"{svr_summary['test_mae']:.10f}",
        )

        print()

    summary_frame = (
        pd.DataFrame(
            summaries
        )
    )

    candidate_frame = (
        pd.DataFrame(
            candidate_rows
        )
    )

    bin_frame = (
        pd.DataFrame(
            bin_rows
        )
    )

    # ----------------------------
    # Reproduction integrity checks
    # ----------------------------

    for (
        descriptor_set,
        lock,
    ) in RIDGE_LOCKS.items():

        row = summary_frame.loc[
            (
                summary_frame[
                    "descriptor_set"
                ]
                == descriptor_set
            )
            & (
                summary_frame[
                    "model"
                ]
                == "ridge"
            )
        ].iloc[0]

        if not np.isclose(
            float(
                row[
                    "selected_alpha"
                ]
            ),
            lock["alpha"],
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(
                f"STOP: Ridge alpha lock "
                f"failed for "
                f"{descriptor_set}."
            )

        if not np.isclose(
            float(
                row[
                    "test_mae"
                ]
            ),
            lock["test_mae"],
            rtol=0.0,
            atol=5e-11,
        ):
            raise RuntimeError(
                f"STOP: Ridge MAE lock "
                f"failed for "
                f"{descriptor_set}."
            )

    anchor_mask = (
        (
            candidate_frame[
                "descriptor_set"
            ]
            == "signal_derived_with_ground"
        )
        & (
            candidate_frame[
                "model"
            ]
            == "rbf_svr"
        )
        & np.isclose(
            candidate_frame["C"],
            SVR_78D_ANCHOR["C"],
        )
        & np.isclose(
            candidate_frame["gamma"],
            SVR_78D_ANCHOR[
                "gamma"
            ],
        )
        & np.isclose(
            candidate_frame["epsilon"],
            SVR_78D_ANCHOR[
                "epsilon"
            ],
        )
    )

    anchor_rows = (
        candidate_frame.loc[
            anchor_mask
        ]
    )

    if len(anchor_rows) != 1:
        raise RuntimeError(
            "STOP: 78D SVR anchor "
            "configuration missing."
        )

    actual_anchor_mse = float(
        anchor_rows[
            "val_mse"
        ].iloc[0]
    )

    svr_anchor_passed = bool(
        np.isclose(
            actual_anchor_mse,
            SVR_78D_ANCHOR[
                "val_mse"
            ],
            rtol=0.0,
            atol=5e-11,
        )
    )

    if not svr_anchor_passed:
        raise RuntimeError(
            "STOP: 78D SVR validation "
            "anchor was not reproduced. "
            f"actual="
            f"{actual_anchor_mse:.12f}, "
            f"expected="
            f"{SVR_78D_ANCHOR['val_mse']:.12f}"
        )

    # ----------------------------
    # 2x2 effect decomposition
    # ----------------------------

    indexed = (
        summary_frame.set_index(
            [
                "descriptor_set",
                "model",
            ]
        )
    )

    r78 = indexed.loc[
        (
            "signal_derived_with_ground",
            "ridge",
        )
    ]

    s78 = indexed.loc[
        (
            "signal_derived_with_ground",
            "rbf_svr",
        )
    ]

    r92 = indexed.loc[
        (
            "oracle_full",
            "ridge",
        )
    ]

    s92 = indexed.loc[
        (
            "oracle_full",
            "rbf_svr",
        )
    ]

    effect_rows = [
        {
            "effect": (
                "information_effect_under_ridge"
            ),
            "configuration_a": (
                "oracle_full + ridge"
            ),
            "configuration_b": (
                "78D + ridge"
            ),
            "test_mae_a": float(
                r92["test_mae"]
            ),
            "test_mae_b": float(
                r78["test_mae"]
            ),
            "test_mae_change_percent": (
                relative_change_percent(
                    float(
                        r92["test_mae"]
                    ),
                    float(
                        r78["test_mae"]
                    ),
                )
            ),
            "high_damage_mae_a": float(
                r92[
                    "high_damage_mae"
                ]
            ),
            "high_damage_mae_b": float(
                r78[
                    "high_damage_mae"
                ]
            ),
        },
        {
            "effect": (
                "information_effect_under_svr"
            ),
            "configuration_a": (
                "oracle_full + svr"
            ),
            "configuration_b": (
                "78D + svr"
            ),
            "test_mae_a": float(
                s92["test_mae"]
            ),
            "test_mae_b": float(
                s78["test_mae"]
            ),
            "test_mae_change_percent": (
                relative_change_percent(
                    float(
                        s92["test_mae"]
                    ),
                    float(
                        s78["test_mae"]
                    ),
                )
            ),
            "high_damage_mae_a": float(
                s92[
                    "high_damage_mae"
                ]
            ),
            "high_damage_mae_b": float(
                s78[
                    "high_damage_mae"
                ]
            ),
        },
        {
            "effect": (
                "model_effect_on_78D"
            ),
            "configuration_a": (
                "78D + svr"
            ),
            "configuration_b": (
                "78D + ridge"
            ),
            "test_mae_a": float(
                s78["test_mae"]
            ),
            "test_mae_b": float(
                r78["test_mae"]
            ),
            "test_mae_change_percent": (
                relative_change_percent(
                    float(
                        s78["test_mae"]
                    ),
                    float(
                        r78["test_mae"]
                    ),
                )
            ),
            "high_damage_mae_a": float(
                s78[
                    "high_damage_mae"
                ]
            ),
            "high_damage_mae_b": float(
                r78[
                    "high_damage_mae"
                ]
            ),
        },
        {
            "effect": (
                "model_effect_on_92D"
            ),
            "configuration_a": (
                "oracle_full + svr"
            ),
            "configuration_b": (
                "oracle_full + ridge"
            ),
            "test_mae_a": float(
                s92["test_mae"]
            ),
            "test_mae_b": float(
                r92["test_mae"]
            ),
            "test_mae_change_percent": (
                relative_change_percent(
                    float(
                        s92["test_mae"]
                    ),
                    float(
                        r92["test_mae"]
                    ),
                )
            ),
            "high_damage_mae_a": float(
                s92[
                    "high_damage_mae"
                ]
            ),
            "high_damage_mae_b": float(
                r92[
                    "high_damage_mae"
                ]
            ),
        },
    ]

    effect_frame = pd.DataFrame(
        effect_rows
    )

    ridge_information_effect = (
        float(
            r92["test_mae"]
        )
        - float(
            r78["test_mae"]
        )
    )

    svr_information_effect = (
        float(
            s92["test_mae"]
        )
        - float(
            s78["test_mae"]
        )
    )

    interaction_mae = (
        svr_information_effect
        - ridge_information_effect
    )

    # ----------------------------
    # Save
    # ----------------------------

    summary_path = (
        args.output_root
        / "factorial_summary.csv"
    )

    candidate_path = (
        args.output_root
        / "factorial_candidates.csv"
    )

    effect_path = (
        args.output_root
        / "factorial_effects.csv"
    )

    bin_path = (
        args.output_root
        / "factorial_damage_bins.csv"
    )

    report_path = (
        args.output_root
        / "factorial_report.json"
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    candidate_frame.to_csv(
        candidate_path,
        index=False,
    )

    effect_frame.to_csv(
        effect_path,
        index=False,
    )

    bin_frame.to_csv(
        bin_path,
        index=False,
    )

    for (
        descriptor_set,
        model_name,
    ), prediction in (
        predictions.items()
    ):

        safe_name = (
            descriptor_set
            .replace(
                "signal_derived_with_ground",
                "78d"
            )
            .replace(
                "oracle_full",
                "92d"
            )
        )

        arrays = datasets[
            descriptor_set
        ]

        np.savez_compressed(
            args.output_root
            / (
                f"{safe_name}_"
                f"{model_name}_"
                "test_predictions.npz"
            ),
            y_test=arrays[
                "y_test"
            ],
            y_prediction=prediction,
        )

    report = {
        "experiment": (
            "78D_92D_x_Ridge_SVR"
        ),
        "shared_partitions_passed": (
            True
        ),
        "ridge_locks_passed": (
            True
        ),
        "svr_78d_anchor_passed": (
            svr_anchor_passed
        ),
        "selection_criterion": (
            "clipped validation MSE"
        ),
        "prediction_clip": [
            0.0,
            0.5,
        ],
        "svr_search_space": {
            "C": SVR_C_VALUES,
            "gamma": (
                SVR_GAMMA_VALUES
            ),
            "epsilon": (
                SVR_EPSILON_VALUES
            ),
            "candidates_per_descriptor": (
                len(SVR_C_VALUES)
                * len(
                    SVR_GAMMA_VALUES
                )
                * len(
                    SVR_EPSILON_VALUES
                )
            ),
        },
        "interaction_test_mae": float(
            interaction_mae
        ),
        "interpretation_note": (
            "Fixed-split diagnostic only; "
            "repeated-split validation is "
            "required for robustness claims."
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

    # ----------------------------
    # Print
    # ----------------------------

    print(
        "===== 2x2 SELECTED RESULTS ====="
    )

    display_columns = [
        "descriptor_set",
        "n_features",
        "model",
        "selected_candidate",
        "val_mse",
        "test_mae",
        "test_rmse",
        "damaged_entry_mae",
        "high_damage_mae",
        "high_damage_bias",
        "high_damage_underestimation_ratio",
    ]

    print(
        summary_frame[
            display_columns
        ].sort_values(
            [
                "descriptor_set",
                "model",
            ]
        ).to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== FACTORIAL EFFECTS ====="
    )

    print(
        effect_frame.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== INFORMATION × MODEL "
        "INTERACTION ====="
    )

    print(
        "Ridge information effect "
        "(92D - 78D MAE):",
        f"{ridge_information_effect:.10f}",
    )

    print(
        "SVR information effect "
        "(92D - 78D MAE):",
        f"{svr_information_effect:.10f}",
    )

    print(
        "Difference-in-differences:",
        f"{interaction_mae:.10f}",
    )

    print()
    print(
        "===== SELECTED SVR CANDIDATES ====="
    )

    selected_svr = (
        candidate_frame.loc[
            (
                candidate_frame[
                    "model"
                ]
                == "rbf_svr"
            )
            & (
                candidate_frame[
                    "selected"
                ]
            )
        ]
    )

    print(
        selected_svr[
            [
                "descriptor_set",
                "candidate",
                "C",
                "gamma",
                "epsilon",
                "val_mse",
                "val_mae",
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
        "Shared partitions:",
        True,
    )

    print(
        "Ridge historical locks:",
        True,
    )

    print(
        "78D SVR anchor reproduced:",
        svr_anchor_passed,
    )

    print()
    print(
        "CHECK PASSED: 2x2 factorial "
        "comparison completed."
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Effects:",
        effect_path,
    )

    print(
        "Damage bins:",
        bin_path,
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
