"""
Complete 10-seed paired 2x2 factorial analysis:

Descriptor information
----------------------
78D signal-derived-with-ground
92D oracle_full / privileged-information reference

Estimator
---------
Ridge
RBF-SVR

Already available
-----------------
78D + Ridge : repeated_split_78d_ridge_svr
78D + SVR   : repeated_split_78d_ridge_svr
92D + SVR   : repeated_split_78d_92d_svr

This script trains only the missing cell:
92D + Ridge

It then constructs a complete repeated-split factorial analysis.

Protocol
--------
- Reuse EXACTLY the ten previously saved split files.
- Per-seed 92D StandardScaler fitted on training fold only.
- Ridge hyperparameters selected using clipped validation MSE only.
- Test evaluated only after validation selection.
- Predictions clipped to [0, 0.5].
- Existing 78D/92D SVR results are reused, not retrained.

Interaction definition
----------------------
For every metric:

information_under_ridge = 92D_Ridge - 78D_Ridge
information_under_svr   = 92D_SVR   - 78D_SVR

model_on_78D = 78D_SVR - 78D_Ridge
model_on_92D = 92D_SVR - 92D_Ridge

interaction =
    information_under_svr
    - information_under_ridge

For all error/reliability metrics used here, lower is better.
Therefore:
- negative main effect = improvement;
- negative interaction = privileged-information benefit is
  larger under SVR than under Ridge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_1samp,
    wilcoxon,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.sss_fast_revision.run_78d_92d_svr_repeated_splits import (  # noqa: E402
    SEEDS,
    EXPECTED_78D_FEATURE_COUNT,
    EXPECTED_92D_FEATURE_COUNT,
    recover_raw_features,
    verify_canonical_alignment,
    load_split,
    clip_prediction,
    calculate_metrics,
    confidence_interval,
)


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


PRIMARY_METRICS = [
    "test_mae",
    "test_rmse",
    "damaged_entry_mae",
    "zero_damage_mae",
    "high_damage_mae",
    "high_damage_abs_bias",
    "high_damage_underestimation_ratio",
]


def validation_mse(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    error = (
        np.asarray(
            prediction,
            dtype=np.float64,
        )
        - np.asarray(
            truth,
            dtype=np.float64,
        )
    )

    return float(
        np.mean(
            error ** 2
        )
    )


def validation_mae(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(
                    prediction,
                    dtype=np.float64,
                )
                - np.asarray(
                    truth,
                    dtype=np.float64,
                )
            )
        )
    )


def select_ridge(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[
    Ridge,
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Select Ridge alpha using clipped validation MSE only.
    """

    best_model = None
    best_row = None

    candidate_rows = []

    for alpha in RIDGE_ALPHAS:

        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
        )

        start = (
            time.perf_counter()
        )

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
            "seed": seed,
            "descriptor_set": (
                "oracle_full"
            ),
            "model": "ridge",
            "candidate": (
                f"ridge_alpha_{alpha:g}"
            ),
            "alpha": float(
                alpha
            ),
            "val_mse": (
                validation_mse(
                    y_val,
                    val_prediction,
                )
            ),
            "val_mae": (
                validation_mae(
                    y_val,
                    val_prediction,
                )
            ),
            "fit_seconds": (
                fit_seconds
            ),
            "selected": False,
        }

        candidate_rows.append(
            row
        )

        if best_row is None:
            best_row = (
                row.copy()
            )

            best_model = (
                model
            )

        else:
            current_key = (
                row[
                    "val_mse"
                ],
                row[
                    "candidate"
                ],
            )

            best_key = (
                best_row[
                    "val_mse"
                ],
                best_row[
                    "candidate"
                ],
            )

            if (
                current_key
                < best_key
            ):
                best_row = (
                    row.copy()
                )

                best_model = (
                    model
                )

    if (
        best_model is None
        or best_row is None
    ):
        raise RuntimeError(
            f"Seed {seed}: Ridge "
            "selection failed."
        )

    for row in (
        candidate_rows
    ):
        row[
            "selected"
        ] = (
            row[
                "candidate"
            ]
            == best_row[
                "candidate"
            ]
        )

    return (
        best_model,
        best_row,
        candidate_rows,
    )


def damage_bin_metrics(
    seed: int,
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

    for (
        bin_name,
        mask,
    ) in bins.items():

        if not np.any(
            mask
        ):
            raise RuntimeError(
                f"Seed {seed}: empty "
                f"damage bin {bin_name}."
            )

        y_true = (
            truth[
                mask
            ]
        )

        y_pred = (
            prediction[
                mask
            ]
        )

        error = (
            y_pred
            - y_true
        )

        rows.append(
            {
                "seed": seed,
                "descriptor_set": (
                    "oracle_full"
                ),
                "model": "ridge",
                "damage_bin": (
                    bin_name
                ),
                "n_entries": int(
                    np.sum(
                        mask
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
                    if bin_name
                    == "zero"
                    else float(
                        np.mean(
                            y_pred
                            < y_true
                        )
                    )
                ),
            }
        )

    return rows


def validate_seed_frame(
    frame: pd.DataFrame,
    name: str,
) -> pd.DataFrame:

    required = (
        set(
            PRIMARY_METRICS
        )
        | {
            "seed",
        }
    )

    missing = sorted(
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{name} missing columns: "
            f"{missing}"
        )

    frame = (
        frame.sort_values(
            "seed"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        frame[
            "seed"
        ]
        .tolist()
        != SEEDS
    ):
        raise RuntimeError(
            f"{name} does not contain "
            "seeds 0..9 exactly."
        )

    if len(
        frame
    ) != len(
        SEEDS
    ):
        raise RuntimeError(
            f"{name}: expected "
            f"{len(SEEDS)} rows."
        )

    return frame


def safe_wilcoxon(
    values: np.ndarray,
) -> tuple[
    float,
    float,
]:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    try:
        result = wilcoxon(
            values,
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


def main() -> None:

    parser = (
        argparse.ArgumentParser()
    )

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
        "--reference-78d",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "repeated_split_78d_ridge_svr/"
            "repeated_split_seed_metrics.csv"
        ),
    )

    parser.add_argument(
        "--reference-92d-svr",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "repeated_split_78d_92d_svr/"
            "information_seed_metrics.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "repeated_factorial_78d_92d_ridge_svr"
        ),
    )

    args = (
        parser.parse_args()
    )

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

    # -------------------------------------------------
    # Canonical dataset alignment
    # -------------------------------------------------

    (
        x_78_raw,
        y_78,
        metadata_78,
    ) = recover_raw_features(
        args.input_78d,
        EXPECTED_78D_FEATURE_COUNT,
    )

    (
        x_92_raw,
        y_92,
        metadata_92,
    ) = recover_raw_features(
        args.input_92d,
        EXPECTED_92D_FEATURE_COUNT,
    )

    verify_canonical_alignment(
        y_78=y_78,
        y_92=y_92,
        metadata_78=metadata_78,
        metadata_92=metadata_92,
    )

    # -------------------------------------------------
    # Load previously frozen three cells
    # -------------------------------------------------

    if not args.reference_78d.is_file():
        raise FileNotFoundError(
            args.reference_78d
        )

    if not (
        args.reference_92d_svr
        .is_file()
    ):
        raise FileNotFoundError(
            args.reference_92d_svr
        )

    reference_78 = (
        pd.read_csv(
            args.reference_78d
        )
    )

    r78 = (
        reference_78.loc[
            reference_78[
                "model"
            ]
            == "ridge"
        ]
        .copy()
    )

    s78 = (
        reference_78.loc[
            reference_78[
                "model"
            ]
            == "rbf_svr"
        ]
        .copy()
    )

    s92 = (
        pd.read_csv(
            args.reference_92d_svr
        )
    )

    if (
        "model"
        in s92.columns
    ):
        s92 = (
            s92.loc[
                s92[
                    "model"
                ]
                == "rbf_svr"
            ]
            .copy()
        )

    r78 = validate_seed_frame(
        r78,
        "78D Ridge",
    )

    s78 = validate_seed_frame(
        s78,
        "78D SVR",
    )

    s92 = validate_seed_frame(
        s92,
        "92D SVR",
    )

    print(
        "===== REPEATED 2x2 FACTORIAL "
        "78D/92D × RIDGE/SVR ====="
    )

    print(
        "Canonical 78D/92D alignment: PASSED"
    )

    print(
        "Frozen 78D Ridge rows:",
        len(
            r78
        ),
    )

    print(
        "Frozen 78D SVR rows:",
        len(
            s78
        ),
    )

    print(
        "Frozen 92D SVR rows:",
        len(
            s92
        ),
    )

    print(
        "Training only missing cell: "
        "92D + Ridge"
    )

    print(
        "Ridge candidates / seed:",
        len(
            RIDGE_ALPHAS
        ),
    )

    print()

    # -------------------------------------------------
    # Train missing 92D + Ridge cell
    # -------------------------------------------------

    r92_rows = []
    candidate_rows = []
    damage_rows = []

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
            x_92_raw[
                train_idx
            ]
        )

        x_val_raw = (
            x_92_raw[
                val_idx
            ]
        )

        x_test_raw = (
            x_92_raw[
                test_idx
            ]
        )

        y_train = (
            y_92[
                train_idx
            ]
        )

        y_val = (
            y_92[
                val_idx
            ]
        )

        y_test = (
            y_92[
                test_idx
            ]
        )

        scaler = (
            StandardScaler()
        )

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

        (
            model,
            selected,
            candidates,
        ) = select_ridge(
            seed=seed,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

        candidate_rows.extend(
            candidates
        )

        # Test evaluated only after
        # validation selection.
        test_prediction = (
            clip_prediction(
                model.predict(
                    x_test
                )
            )
        )

        metrics = (
            calculate_metrics(
                y_test,
                test_prediction,
            )
        )

        row = {
            "seed": seed,
            "descriptor_set": (
                "oracle_full"
            ),
            "n_features": 92,
            "model": "ridge",
            "selected_candidate": (
                selected[
                    "candidate"
                ]
            ),
            "selected_alpha": (
                selected[
                    "alpha"
                ]
            ),
            "val_mse": (
                selected[
                    "val_mse"
                ]
            ),
            "val_mae": (
                selected[
                    "val_mae"
                ]
            ),
            **metrics,
        }

        r92_rows.append(
            row
        )

        damage_rows.extend(
            damage_bin_metrics(
                seed=seed,
                truth=y_test,
                prediction=(
                    test_prediction
                ),
            )
        )

        np.savez_compressed(
            prediction_root
            / (
                f"seed_{seed}_"
                "92d_ridge_predictions.npz"
            ),
            seed=np.asarray(
                seed
            ),
            test_idx=test_idx,
            y_test=y_test,
            prediction=(
                test_prediction
            ),
        )

        print(
            "  selected:",
            selected[
                "candidate"
            ],
        )

        print(
            "  val MSE:",
            f"{selected['val_mse']:.10f}",
        )

        print(
            "  test MAE:",
            f"{metrics['test_mae']:.10f}",
        )

        print(
            "  high-damage MAE:",
            f"{metrics['high_damage_mae']:.10f}",
        )

        print(
            "  high-damage underestimation:",
            (
                f"{metrics[
                    'high_damage_underestimation_ratio'
                ]:.10f}"
            ),
        )

        print()

    r92 = validate_seed_frame(
        pd.DataFrame(
            r92_rows
        ),
        "92D Ridge",
    )

    candidates_frame = (
        pd.DataFrame(
            candidate_rows
        )
    )

    damage_frame = (
        pd.DataFrame(
            damage_rows
        )
    )

    # -------------------------------------------------
    # Ridge alpha stability
    # -------------------------------------------------

    ridge_frequency = (
        r92[
            "selected_alpha"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "selected_alpha"
        )
        .reset_index(
            name="count"
        )
    )

    # -------------------------------------------------
    # Build complete paired 2x2 factorial
    # -------------------------------------------------

    r78_index = (
        r78.set_index(
            "seed"
        )
    )

    s78_index = (
        s78.set_index(
            "seed"
        )
    )

    r92_index = (
        r92.set_index(
            "seed"
        )
    )

    s92_index = (
        s92.set_index(
            "seed"
        )
    )

    factorial_rows = []

    for seed in SEEDS:

        for metric in (
            PRIMARY_METRICS
        ):

            A = float(
                r78_index.loc[
                    seed,
                    metric,
                ]
            )

            B = float(
                s78_index.loc[
                    seed,
                    metric,
                ]
            )

            C = float(
                r92_index.loc[
                    seed,
                    metric,
                ]
            )

            D = float(
                s92_index.loc[
                    seed,
                    metric,
                ]
            )

            information_ridge = (
                C - A
            )

            information_svr = (
                D - B
            )

            model_78d = (
                B - A
            )

            model_92d = (
                D - C
            )

            interaction = (
                information_svr
                - information_ridge
            )

            factorial_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "78d_ridge": A,
                    "78d_svr": B,
                    "92d_ridge": C,
                    "92d_svr": D,
                    "information_effect_under_ridge": (
                        information_ridge
                    ),
                    "information_effect_under_svr": (
                        information_svr
                    ),
                    "model_effect_on_78d": (
                        model_78d
                    ),
                    "model_effect_on_92d": (
                        model_92d
                    ),
                    "interaction_difference_in_differences": (
                        interaction
                    ),
                    "information_improvement_ridge_percent": (
                        (
                            A - C
                        )
                        / A
                        * 100.0
                    ),
                    "information_improvement_svr_percent": (
                        (
                            B - D
                        )
                        / B
                        * 100.0
                    ),
                    "model_improvement_78d_percent": (
                        (
                            A - B
                        )
                        / A
                        * 100.0
                    ),
                    "model_improvement_92d_percent": (
                        (
                            C - D
                        )
                        / C
                        * 100.0
                    ),
                }
            )

    factorial_frame = (
        pd.DataFrame(
            factorial_rows
        )
    )

    # -------------------------------------------------
    # Mean 2x2 table
    # -------------------------------------------------

    mean_rows = []

    for metric in (
        PRIMARY_METRICS
    ):

        subset = (
            factorial_frame.loc[
                factorial_frame[
                    "metric"
                ]
                == metric
            ]
        )

        mean_rows.append(
            {
                "metric": metric,
                "78D_Ridge": float(
                    subset[
                        "78d_ridge"
                    ].mean()
                ),
                "78D_SVR": float(
                    subset[
                        "78d_svr"
                    ].mean()
                ),
                "92D_Ridge": float(
                    subset[
                        "92d_ridge"
                    ].mean()
                ),
                "92D_SVR": float(
                    subset[
                        "92d_svr"
                    ].mean()
                ),
            }
        )

    mean_frame = (
        pd.DataFrame(
            mean_rows
        )
    )

    # -------------------------------------------------
    # Aggregate factorial effects
    # -------------------------------------------------

    effect_specs = {
        "information_effect_under_ridge": {
            "column": (
                "information_effect_under_ridge"
            ),
            "relative_column": (
                "information_improvement_ridge_percent"
            ),
            "definition": (
                "92D_Ridge_minus_78D_Ridge"
            ),
        },
        "information_effect_under_svr": {
            "column": (
                "information_effect_under_svr"
            ),
            "relative_column": (
                "information_improvement_svr_percent"
            ),
            "definition": (
                "92D_SVR_minus_78D_SVR"
            ),
        },
        "model_effect_on_78d": {
            "column": (
                "model_effect_on_78d"
            ),
            "relative_column": (
                "model_improvement_78d_percent"
            ),
            "definition": (
                "78D_SVR_minus_78D_Ridge"
            ),
        },
        "model_effect_on_92d": {
            "column": (
                "model_effect_on_92d"
            ),
            "relative_column": (
                "model_improvement_92d_percent"
            ),
            "definition": (
                "92D_SVR_minus_92D_Ridge"
            ),
        },
        "interaction_difference_in_differences": {
            "column": (
                "interaction_difference_in_differences"
            ),
            "relative_column": None,
            "definition": (
                "(92D_SVR-78D_SVR)"
                "-(92D_Ridge-78D_Ridge)"
            ),
        },
    }

    effect_rows = []

    for metric in (
        PRIMARY_METRICS
    ):

        metric_frame = (
            factorial_frame.loc[
                factorial_frame[
                    "metric"
                ]
                == metric
            ]
        )

        for (
            effect_name,
            spec,
        ) in effect_specs.items():

            values = (
                metric_frame[
                    spec[
                        "column"
                    ]
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            (
                ci_low,
                ci_high,
            ) = confidence_interval(
                values
            )

            t_result = (
                ttest_1samp(
                    values,
                    popmean=0.0,
                )
            )

            (
                w_stat,
                w_p,
            ) = safe_wilcoxon(
                values
            )

            relative_column = (
                spec[
                    "relative_column"
                ]
            )

            if (
                relative_column
                is None
            ):
                mean_relative = (
                    np.nan
                )
            else:
                mean_relative = float(
                    metric_frame[
                        relative_column
                    ].mean()
                )

            effect_rows.append(
                {
                    "metric": metric,
                    "effect": (
                        effect_name
                    ),
                    "definition": (
                        spec[
                            "definition"
                        ]
                    ),
                    "n_seeds": int(
                        len(
                            values
                        )
                    ),
                    "mean_effect": float(
                        np.mean(
                            values
                        )
                    ),
                    "std_effect": float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),
                    "ci95_low": (
                        ci_low
                    ),
                    "ci95_high": (
                        ci_high
                    ),
                    "negative_seeds": int(
                        np.sum(
                            values < 0.0
                        )
                    ),
                    "positive_seeds": int(
                        np.sum(
                            values > 0.0
                        )
                    ),
                    "mean_relative_improvement_percent": (
                        mean_relative
                    ),
                    "one_sample_t_statistic": float(
                        t_result.statistic
                    ),
                    "one_sample_t_pvalue": float(
                        t_result.pvalue
                    ),
                    "wilcoxon_statistic": (
                        w_stat
                    ),
                    "wilcoxon_pvalue": (
                        w_p
                    ),
                }
            )

    effect_frame = (
        pd.DataFrame(
            effect_rows
        )
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    r92_metrics_path = (
        args.output_root
        / "oracle92_ridge_seed_metrics.csv"
    )

    r92_candidates_path = (
        args.output_root
        / "oracle92_ridge_candidates.csv"
    )

    r92_damage_path = (
        args.output_root
        / "oracle92_ridge_damage_bins.csv"
    )

    ridge_frequency_path = (
        args.output_root
        / "oracle92_ridge_alpha_frequency.csv"
    )

    factorial_seed_path = (
        args.output_root
        / "repeated_factorial_seed_effects.csv"
    )

    mean_path = (
        args.output_root
        / "repeated_factorial_mean_table.csv"
    )

    effect_path = (
        args.output_root
        / "repeated_factorial_effect_summary.csv"
    )

    report_path = (
        args.output_root
        / "repeated_factorial_report.json"
    )

    r92.to_csv(
        r92_metrics_path,
        index=False,
    )

    candidates_frame.to_csv(
        r92_candidates_path,
        index=False,
    )

    damage_frame.to_csv(
        r92_damage_path,
        index=False,
    )

    ridge_frequency.to_csv(
        ridge_frequency_path,
        index=False,
    )

    factorial_frame.to_csv(
        factorial_seed_path,
        index=False,
    )

    mean_frame.to_csv(
        mean_path,
        index=False,
    )

    effect_frame.to_csv(
        effect_path,
        index=False,
    )

    report = {
        "experiment": (
            "10_seed_complete_2x2_"
            "78D_92D_x_Ridge_SVR"
        ),
        "seeds": SEEDS,
        "missing_cell_trained": (
            "92D_oracle_full_Ridge"
        ),
        "previous_cells_reused": [
            "78D_Ridge",
            "78D_RBF_SVR",
            "92D_RBF_SVR",
        ],
        "canonical_alignment_passed": (
            True
        ),
        "same_split_files_reused": (
            True
        ),
        "per_seed_92d_scaling": (
            True
        ),
        "ridge_selection": (
            "clipped validation MSE"
        ),
        "ridge_grid": (
            RIDGE_ALPHAS
        ),
        "interaction_definition": (
            "(92D_SVR - 78D_SVR) - "
            "(92D_Ridge - 78D_Ridge)"
        ),
        "interaction_interpretation": (
            "For lower-is-better metrics, "
            "negative interaction means the "
            "92D information benefit is larger "
            "under SVR than under Ridge."
        ),
        "statistical_note": (
            "Ten paired repeated holdout splits "
            "reuse the same 3000 simulated cases. "
            "Effect sizes, paired consistency and "
            "intervals are primary; p-values are "
            "supportive and should not be treated "
            "as ten independent experiments."
        ),
        "output_files": {
            "92d_ridge_metrics": str(
                r92_metrics_path
            ),
            "92d_ridge_candidates": str(
                r92_candidates_path
            ),
            "92d_ridge_damage_bins": str(
                r92_damage_path
            ),
            "92d_ridge_alpha_frequency": str(
                ridge_frequency_path
            ),
            "factorial_seed_effects": str(
                factorial_seed_path
            ),
            "factorial_mean_table": str(
                mean_path
            ),
            "factorial_effect_summary": str(
                effect_path
            ),
        },
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

    # -------------------------------------------------
    # Console output
    # -------------------------------------------------

    print()
    print(
        "===== 92D RIDGE SEED RESULTS ====="
    )

    print(
        r92[
            [
                "seed",
                "selected_candidate",
                "val_mse",
                "test_mae",
                "test_rmse",
                "high_damage_mae",
                "high_damage_abs_bias",
                "high_damage_underestimation_ratio",
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
        "===== 92D RIDGE ALPHA FREQUENCY ====="
    )

    print(
        ridge_frequency.to_string(
            index=False,
        )
    )

    print()
    print(
        "===== 10-SEED 2x2 MEAN TABLE ====="
    )

    print(
        mean_frame.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== FACTORIAL EFFECT SUMMARY ====="
    )

    print(
        effect_frame[
            [
                "metric",
                "effect",
                "mean_effect",
                "ci95_low",
                "ci95_high",
                "negative_seeds",
                "positive_seeds",
                "mean_relative_improvement_percent",
                "one_sample_t_pvalue",
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
        "===== TEST MAE INTERACTION DETAIL ====="
    )

    mae_interaction = (
        factorial_frame.loc[
            factorial_frame[
                "metric"
            ]
            == "test_mae",
            [
                "seed",
                "information_effect_under_ridge",
                "information_effect_under_svr",
                "model_effect_on_78d",
                "model_effect_on_92d",
                "interaction_difference_in_differences",
                "information_improvement_ridge_percent",
                "information_improvement_svr_percent",
                "model_improvement_78d_percent",
                "model_improvement_92d_percent",
            ],
        ]
    )

    print(
        mae_interaction.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
        )
    )

    print()
    print(
        "===== HIGH-DAMAGE INTERACTION DETAIL ====="
    )

    high_interaction = (
        factorial_frame.loc[
            factorial_frame[
                "metric"
            ]
            == "high_damage_mae",
            [
                "seed",
                "information_effect_under_ridge",
                "information_effect_under_svr",
                "model_effect_on_78d",
                "model_effect_on_92d",
                "interaction_difference_in_differences",
            ],
        ]
    )

    print(
        high_interaction.to_string(
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
        "Canonical targets aligned:",
        True,
    )

    print(
        "78D Ridge rows:",
        len(
            r78
        ),
    )

    print(
        "78D SVR rows:",
        len(
            s78
        ),
    )

    print(
        "92D Ridge rows:",
        len(
            r92
        ),
    )

    print(
        "92D SVR rows:",
        len(
            s92
        ),
    )

    print(
        "Exact previous split files reused:",
        True,
    )

    print(
        "Per-seed 92D scaling:",
        True,
    )

    print(
        "Frozen Ridge grid:",
        True,
    )

    print()
    print(
        "CHECK PASSED: complete "
        "10-seed 2x2 factorial analysis "
        "completed."
    )

    print(
        "Mean table:",
        mean_path,
    )

    print(
        "Effect summary:",
        effect_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
