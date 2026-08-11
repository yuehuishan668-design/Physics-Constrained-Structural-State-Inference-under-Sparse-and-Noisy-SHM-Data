"""
Generate Figure 4 for the reconstructed JCSHM manuscript.

Figure 4:
Damage-severity-dependent inference failure and asymmetric calibration.

Scientific structure
--------------------
(a) MAE across damage severity for Ridge and RBF-SVR.
(b) Signed bias across damage severity for Ridge and RBF-SVR.
(c) Severity-specific MAE trade-off after asymmetric calibration.
(d) Targeted high-damage benefits of asymmetric calibration.

IMPORTANT
---------
This script performs:

- NO model training;
- NO hyperparameter tuning;
- NO calibration fitting;
- NO statistical re-testing;
- NO damage-bin redesign;
- NO modification of frozen experimental evidence.

Descriptive means and standard deviations in panels (a)-(b) are calculated
directly from the ten already-frozen repeated-split results.

Panels (c)-(d) use the already-frozen paired calibration summaries.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jcshm_figure_style import (
    DOUBLE_COLUMN_WIDTH,
    FIGURE_DATA_ROOT,
    add_panel_label,
    apply_manuscript_style,
    save_figure,
)


# ============================================================
# Frozen manuscript data
# ============================================================

SEVERITY_PATH = (
    FIGURE_DATA_ROOT
    / "F04_severity_performance.csv"
)

CALIBRATION_PATH = (
    FIGURE_DATA_ROOT
    / "F04_calibration_effects.csv"
)


# ============================================================
# Frozen categorical definitions
# ============================================================

DAMAGE_ORDER = [
    "zero",
    "low",
    "medium",
    "high",
]

DAMAGE_LABELS = [
    "Zero",
    "Low",
    "Medium",
    "High",
]

MODEL_ORDER = [
    "ridge",
    "rbf_svr",
]

MODEL_LABELS = {
    "ridge": "Ridge",
    "rbf_svr": "RBF-SVR",
}

MODEL_MARKERS = {
    "ridge": "o",
    "rbf_svr": "s",
}

MODEL_LINESTYLES = {
    "ridge": "-",
    "rbf_svr": "--",
}


# ============================================================
# Frozen numerical anchors
# ============================================================

CALIBRATION_ANCHORS = {
    "standard_high_mae": 0.0603373847,
    "calibrated_high_mae": 0.0556957518,
    "high_mae_relative_improvement": 7.62974596,

    "standard_high_abs_bias": 0.0474323965,
    "calibrated_high_abs_bias": 0.0339341074,
    "high_abs_bias_relative_improvement": 29.27290328,

    "standard_high_underestimation": 0.7610611425,
    "calibrated_high_underestimation": 0.6573552415,

    "zero_mae_relative_improvement": -1.01874841,
    "low_mae_relative_improvement": -7.71089582,
    "medium_mae_relative_improvement": 0.58993253,
}


def assert_close(
    observed: float,
    expected: float,
    name: str,
    atol: float = 5e-8,
) -> None:
    """
    Fail immediately if a frozen numerical anchor is violated.
    """

    if not np.isclose(
        observed,
        expected,
        atol=atol,
        rtol=1e-8,
    ):
        raise AssertionError(
            f"Numerical anchor failed for {name}: "
            f"observed={observed}, expected={expected}"
        )


def get_calibration_row(
    frame: pd.DataFrame,
    metric: str,
) -> pd.Series:
    """
    Retrieve exactly one frozen calibration-summary row.
    """

    selected = frame.loc[
        frame["metric"] == metric
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one calibration row for "
            f"{metric!r}; found {len(selected)}."
        )

    return selected.iloc[0]


def severity_summary(
    frame: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Descriptively summarize one severity metric across the ten
    already-frozen repeated splits.

    No inferential test is performed.
    """

    summary = (
        frame.groupby(
            [
                "model",
                "damage_bin",
            ],
            as_index=False,
        )[metric]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    return summary


def get_summary_value(
    summary: pd.DataFrame,
    model: str,
    damage_bin: str,
    statistic: str,
) -> float:

    row = summary.loc[
        (summary["model"] == model)
        & (
            summary["damage_bin"]
            == damage_bin
        )
    ]

    if len(row) != 1:
        raise ValueError(
            "Expected exactly one severity summary row for "
            f"{model=}, {damage_bin=}."
        )

    return float(
        row.iloc[0][statistic]
    )


def plot_severity_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    ylabel: str,
    panel_label: str,
    *,
    zero_reference: bool = False,
    force_zero_bottom: bool = False,
) -> None:
    """
    Plot mean ± SD across the ten frozen repeated splits.
    """

    x = np.arange(
        len(
            DAMAGE_ORDER
        )
    )

    for model in MODEL_ORDER:

        means = np.array(
            [
                get_summary_value(
                    summary,
                    model,
                    damage_bin,
                    "mean",
                )
                for damage_bin
                in DAMAGE_ORDER
            ]
        )

        stds = np.array(
            [
                get_summary_value(
                    summary,
                    model,
                    damage_bin,
                    "std",
                )
                for damage_bin
                in DAMAGE_ORDER
            ]
        )

        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker=MODEL_MARKERS[
                model
            ],
            linestyle=MODEL_LINESTYLES[
                model
            ],
            capsize=2.5,
            label=MODEL_LABELS[
                model
            ],
        )

    ax.set_xticks(
        x,
        DAMAGE_LABELS,
    )

    ax.set_xlabel(
        "Damage severity"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    if zero_reference:

        ax.axhline(
            0.0,
            linewidth=0.8,
            linestyle=":",
            color="0.25",
        )

    if force_zero_bottom:

        ax.set_ylim(
            bottom=0.0
        )

    add_panel_label(
        ax,
        panel_label,
    )


def main() -> None:

    apply_manuscript_style()

    severity = pd.read_csv(
        SEVERITY_PATH
    )

    calibration = pd.read_csv(
        CALIBRATION_PATH
    )

    # ========================================================
    # Schema and integrity checks
    # ========================================================

    required_severity_columns = {
        "seed",
        "model",
        "damage_bin",
        "n_entries",
        "mae",
        "rmse",
        "bias",
        "mean_true",
        "mean_prediction",
        "underestimation_ratio",
    }

    if not required_severity_columns.issubset(
        severity.columns
    ):
        raise ValueError(
            "F04 severity schema does not match "
            "the frozen manuscript schema."
        )

    required_calibration_columns = {
        "metric",
        "difference_definition",
        "n_seeds",
        "standard_mean",
        "calibrated_mean",
        "mean_difference",
        "std_difference",
        "ci95_difference_low",
        "ci95_difference_high",
        "calibrated_wins",
        "standard_wins",
        "mean_relative_improvement_percent",
    }

    if not required_calibration_columns.issubset(
        calibration.columns
    ):
        raise ValueError(
            "F04 calibration schema does not match "
            "the frozen manuscript schema."
        )

    if len(
        severity
    ) != 80:
        raise AssertionError(
            f"Expected 80 severity rows; found {len(severity)}."
        )

    if set(
        severity["seed"].unique()
    ) != set(
        range(10)
    ):
        raise AssertionError(
            "Expected repeated-split seeds 0-9."
        )

    if set(
        severity["model"].unique()
    ) != set(
        MODEL_ORDER
    ):
        raise AssertionError(
            "Unexpected model labels in severity data."
        )

    if set(
        severity["damage_bin"].unique()
    ) != set(
        DAMAGE_ORDER
    ):
        raise AssertionError(
            "Unexpected damage-bin labels."
        )

    expected_combinations = (
        severity[
            [
                "seed",
                "model",
                "damage_bin",
            ]
        ]
        .drop_duplicates()
    )

    if len(
        expected_combinations
    ) != 80:
        raise AssertionError(
            "Severity data do not contain one unique "
            "seed × model × damage-bin combination."
        )

    if set(
        calibration[
            "difference_definition"
        ].unique()
    ) != {
        "calibrated_minus_standard"
    }:
        raise AssertionError(
            "Unexpected calibration difference definition."
        )

    # ========================================================
    # Frozen calibration anchor checks
    # ========================================================

    high_mae_row = get_calibration_row(
        calibration,
        "high_mae",
    )

    high_bias_row = get_calibration_row(
        calibration,
        "high_abs_bias",
    )

    high_under_row = get_calibration_row(
        calibration,
        "high_underestimation_ratio",
    )

    zero_row = get_calibration_row(
        calibration,
        "zero_mae",
    )

    low_row = get_calibration_row(
        calibration,
        "low_mae",
    )

    medium_row = get_calibration_row(
        calibration,
        "medium_mae",
    )

    assert_close(
        float(
            high_mae_row[
                "standard_mean"
            ]
        ),
        CALIBRATION_ANCHORS[
            "standard_high_mae"
        ],
        "standard high-damage MAE",
    )

    assert_close(
        float(
            high_mae_row[
                "calibrated_mean"
            ]
        ),
        CALIBRATION_ANCHORS[
            "calibrated_high_mae"
        ],
        "calibrated high-damage MAE",
    )

    assert_close(
        float(
            high_mae_row[
                "mean_relative_improvement_percent"
            ]
        ),
        CALIBRATION_ANCHORS[
            "high_mae_relative_improvement"
        ],
        "high-damage MAE relative improvement",
    )

    assert_close(
        float(
            high_bias_row[
                "standard_mean"
            ]
        ),
        CALIBRATION_ANCHORS[
            "standard_high_abs_bias"
        ],
        "standard high-damage absolute bias",
    )

    assert_close(
        float(
            high_bias_row[
                "calibrated_mean"
            ]
        ),
        CALIBRATION_ANCHORS[
            "calibrated_high_abs_bias"
        ],
        "calibrated high-damage absolute bias",
    )

    assert_close(
        float(
            high_bias_row[
                "mean_relative_improvement_percent"
            ]
        ),
        CALIBRATION_ANCHORS[
            "high_abs_bias_relative_improvement"
        ],
        "high-damage absolute-bias relative improvement",
    )

    assert_close(
        float(
            high_under_row[
                "standard_mean"
            ]
        ),
        CALIBRATION_ANCHORS[
            "standard_high_underestimation"
        ],
        "standard high-damage underestimation",
    )

    assert_close(
        float(
            high_under_row[
                "calibrated_mean"
            ]
        ),
        CALIBRATION_ANCHORS[
            "calibrated_high_underestimation"
        ],
        "calibrated high-damage underestimation",
    )

    assert_close(
        float(
            zero_row[
                "mean_relative_improvement_percent"
            ]
        ),
        CALIBRATION_ANCHORS[
            "zero_mae_relative_improvement"
        ],
        "zero-damage calibration effect",
    )

    assert_close(
        float(
            low_row[
                "mean_relative_improvement_percent"
            ]
        ),
        CALIBRATION_ANCHORS[
            "low_mae_relative_improvement"
        ],
        "low-damage calibration effect",
    )

    assert_close(
        float(
            medium_row[
                "mean_relative_improvement_percent"
            ]
        ),
        CALIBRATION_ANCHORS[
            "medium_mae_relative_improvement"
        ],
        "medium-damage calibration effect",
    )

    # ========================================================
    # Descriptive repeated-split summaries
    # ========================================================

    mae_summary = severity_summary(
        severity,
        "mae",
    )

    bias_summary = severity_summary(
        severity,
        "bias",
    )

    # ========================================================
    # Figure
    # ========================================================

    fig = plt.figure(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            5.85,
        )
    )

    grid = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        bottom=0.10,
        top=0.955,
        wspace=0.42,
        hspace=0.42,
    )

    ax_a = fig.add_subplot(
        grid[0, 0]
    )

    ax_b = fig.add_subplot(
        grid[0, 1]
    )

    ax_c = fig.add_subplot(
        grid[1, 0]
    )

    ax_d = fig.add_subplot(
        grid[1, 1]
    )

    # ========================================================
    # Panel (a): MAE across damage severity
    # ========================================================

    plot_severity_metric(
        ax_a,
        mae_summary,
        "MAE",
        "(a)",
        force_zero_bottom=True,
    )

    # ========================================================
    # Panel (b): signed bias across damage severity
    # ========================================================

    plot_severity_metric(
        ax_b,
        bias_summary,
        "Signed bias",
        "(b)",
        zero_reference=True,
    )

    # Shared legend for panels (a)-(b).
    handles, labels = (
        ax_a.get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.50,
            1.005,
        ),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=2.0,
    )

    # ========================================================
    # Panel (c): calibration MAE trade-off across severity
    #
    # Frozen source convention:
    #
    #     calibrated - standard
    #
    # We reverse the sign for visualization:
    #
    #     standard - calibrated
    #
    # Positive values therefore indicate MAE reduction.
    # ========================================================

    calibration_severity_specs = [
        (
            "zero_mae",
            "Zero",
        ),
        (
            "low_mae",
            "Low",
        ),
        (
            "medium_mae",
            "Medium",
        ),
        (
            "high_mae",
            "High",
        ),
    ]

    calibration_rows = []

    for metric, label in (
        calibration_severity_specs
    ):

        row = get_calibration_row(
            calibration,
            metric,
        )

        reduction = -float(
            row[
                "mean_difference"
            ]
        )

        ci_low = -float(
            row[
                "ci95_difference_high"
            ]
        )

        ci_high = -float(
            row[
                "ci95_difference_low"
            ]
        )

        calibration_rows.append(
            {
                "metric": metric,
                "label": label,
                "reduction": reduction,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "relative_improvement":
                    float(
                        row[
                            "mean_relative_improvement_percent"
                        ]
                    ),
                "calibrated_wins":
                    int(
                        row[
                            "calibrated_wins"
                        ]
                    ),
            }
        )

    tradeoff = pd.DataFrame(
        calibration_rows
    )

    y = np.arange(
        len(
            tradeoff
        )
    )

    x = tradeoff[
        "reduction"
    ].to_numpy()

    xerr_low = (
        x
        - tradeoff[
            "ci_low"
        ].to_numpy()
    )

    xerr_high = (
        tradeoff[
            "ci_high"
        ].to_numpy()
        - x
    )

    ax_c.errorbar(
        x,
        y,
        xerr=np.vstack(
            [
                xerr_low,
                xerr_high,
            ]
        ),
        fmt="D",
        linestyle="none",
        color="0.2",
        capsize=3.0,
    )

    ax_c.axvline(
        0.0,
        linewidth=0.8,
        linestyle=":",
        color="0.25",
    )

    ax_c.set_yticks(
        y,
        tradeoff[
            "label"
        ],
    )

    ax_c.invert_yaxis()

    ax_c.set_xlabel(
        "MAE reduction after calibration"
    )

    ax_c.set_ylabel(
        "Damage severity"
    )

    ax_c.grid(
        axis="x",
        linewidth=0.45,
        alpha=0.25,
    )

    add_panel_label(
        ax_c,
        "(c)",
    )

    # Add source-defined relative-improvement labels.
    span = max(
        abs(
            tradeoff[
                "ci_low"
            ].min()
        ),
        abs(
            tradeoff[
                "ci_high"
            ].max()
        ),
    )

    text_offset = max(
        0.00015,
        0.035
        * span,
    )

    for idx, row in (
        tradeoff.iterrows()
    ):

        value = float(
            row[
                "relative_improvement"
            ]
        )

        if row[
            "ci_high"
        ] >= 0:

            x_text = (
                row[
                    "ci_high"
                ]
                + text_offset
            )

            ha = "left"

        else:

            x_text = (
                row[
                    "ci_low"
                ]
                - text_offset
            )

            ha = "right"

        ax_c.text(
            x_text,
            idx,
            f"{value:+.1f}%",
            va="center",
            ha=ha,
            fontsize=7.5,
        )

    # ========================================================
    # Panel (d): targeted high-damage calibration benefits
    #
    # Values are source-defined mean relative improvements
    # across the ten repeated splits.
    # ========================================================

    high_specs = [
        (
            "high_mae",
            "High-damage MAE",
        ),
        (
            "high_abs_bias",
            "High-damage |bias|",
        ),
        (
            "high_underestimation_ratio",
            "Underestimation ratio",
        ),
    ]

    high_rows = []

    for metric, label in (
        high_specs
    ):

        row = get_calibration_row(
            calibration,
            metric,
        )

        high_rows.append(
            {
                "metric": metric,
                "label": label,
                "relative":
                    float(
                        row[
                            "mean_relative_improvement_percent"
                        ]
                    ),
                "wins":
                    int(
                        row[
                            "calibrated_wins"
                        ]
                    ),
                "standard_mean":
                    float(
                        row[
                            "standard_mean"
                        ]
                    ),
                "calibrated_mean":
                    float(
                        row[
                            "calibrated_mean"
                        ]
                    ),
                "mean_difference":
                    float(
                        row[
                            "mean_difference"
                        ]
                    ),
            }
        )

    high_effects = pd.DataFrame(
        high_rows
    )

    y_high = np.arange(
        len(
            high_effects
        )
    )

    relative_values = high_effects[
        "relative"
    ].to_numpy()

    for y_value, x_value in zip(
        y_high,
        relative_values,
    ):

        ax_d.hlines(
            y=y_value,
            xmin=0.0,
            xmax=x_value,
            linewidth=1.2,
            color="0.35",
        )

    ax_d.plot(
        relative_values,
        y_high,
        linestyle="none",
        marker="D",
        color="0.2",
    )

    ax_d.set_yticks(
        y_high,
        high_effects[
            "label"
        ],
    )

    ax_d.invert_yaxis()

    ax_d.set_xlim(
        0.0,
        max(
            34.0,
            float(
                relative_values.max()
                * 1.20
            ),
        ),
    )

    ax_d.set_xlabel(
        "Mean relative improvement (%)"
    )

    ax_d.set_ylabel(
        ""
    )

    ax_d.grid(
        axis="x",
        linewidth=0.45,
        alpha=0.25,
    )

    add_panel_label(
        ax_d,
        "(d)",
    )

    # Annotate improvement and split consistency.
    for idx, row in (
        high_effects.iterrows()
    ):

        text = (
            f'{row["relative"]:.1f}%'
            f'  ({int(row["wins"])}/10)'
        )

        if (
            row[
                "metric"
            ]
            == "high_underestimation_ratio"
        ):

            pp_change = (
                row[
                    "calibrated_mean"
                ]
                - row[
                    "standard_mean"
                ]
            ) * 100.0

            text += (
                f"\n{pp_change:.1f} pp"
            )

        ax_d.text(
            row[
                "relative"
            ]
            + 0.8,
            idx,
            text,
            va="center",
            ha="left",
            fontsize=7.4,
        )

    # ========================================================
    # Output
    # ========================================================

    stem = (
        "F04_severity_failure_calibration"
    )

    save_figure(
        fig,
        stem,
    )

    plt.close(
        fig
    )

    # ========================================================
    # Console audit
    # ========================================================

    ridge_high_bias = (
        get_summary_value(
            bias_summary,
            "ridge",
            "high",
            "mean",
        )
    )

    svr_high_bias = (
        get_summary_value(
            bias_summary,
            "rbf_svr",
            "high",
            "mean",
        )
    )

    print()
    print(
        "=" * 92
    )

    print(
        "FIGURE 4 CHECK"
    )

    print(
        "=" * 92
    )

    print(
        "Training performed:",
        False,
    )

    print(
        "Hyperparameter tuning performed:",
        False,
    )

    print(
        "Calibration fitting performed:",
        False,
    )

    print(
        "Statistical re-testing performed:",
        False,
    )

    print(
        "Damage bins changed:",
        False,
    )

    print(
        "Repeated severity rows:",
        len(
            severity
        ),
    )

    print(
        "Repeated seeds:",
        severity[
            "seed"
        ].nunique(),
    )

    print(
        "Models:",
        sorted(
            severity[
                "model"
            ].unique()
        ),
    )

    print(
        "Damage bins:",
        DAMAGE_ORDER,
    )

    print()

    print(
        "Mean high-damage signed bias | Ridge:",
        f"{ridge_high_bias:.10f}",
    )

    print(
        "Mean high-damage signed bias | RBF-SVR:",
        f"{svr_high_bias:.10f}",
    )

    print()

    print(
        "Calibration high-damage MAE:",
        f'{float(high_mae_row["standard_mean"]):.10f}',
        "->",
        f'{float(high_mae_row["calibrated_mean"]):.10f}',
    )

    print(
        "Calibration high-damage |bias|:",
        f'{float(high_bias_row["standard_mean"]):.10f}',
        "->",
        f'{float(high_bias_row["calibrated_mean"]):.10f}',
    )

    print(
        "Calibration high-damage underestimation:",
        f'{float(high_under_row["standard_mean"]):.10f}',
        "->",
        f'{float(high_under_row["calibrated_mean"]):.10f}',
    )

    print()

    print(
        "Zero-damage relative calibration effect:",
        f'{float(zero_row["mean_relative_improvement_percent"]):+.4f}%',
    )

    print(
        "Low-damage relative calibration effect:",
        f'{float(low_row["mean_relative_improvement_percent"]):+.4f}%',
    )

    print(
        "Medium-damage relative calibration effect:",
        f'{float(medium_row["mean_relative_improvement_percent"]):+.4f}%',
    )

    print(
        "High-damage relative calibration effect:",
        f'{float(high_mae_row["mean_relative_improvement_percent"]):+.4f}%',
    )

    print()

    high_calibration_checks = {
        "high_mae_10_of_10":
            int(
                high_mae_row[
                    "calibrated_wins"
                ]
            )
            == 10,

        "high_abs_bias_10_of_10":
            int(
                high_bias_row[
                    "calibrated_wins"
                ]
            )
            == 10,

        "high_underestimation_10_of_10":
            int(
                high_under_row[
                    "calibrated_wins"
                ]
            )
            == 10,

        "high_mae_CI_beneficial":
            float(
                high_mae_row[
                    "ci95_difference_high"
                ]
            )
            < 0.0,

        "high_abs_bias_CI_beneficial":
            float(
                high_bias_row[
                    "ci95_difference_high"
                ]
            )
            < 0.0,

        "high_underestimation_CI_beneficial":
            float(
                high_under_row[
                    "ci95_difference_high"
                ]
            )
            < 0.0,

        "low_damage_CI_detrimental":
            float(
                low_row[
                    "ci95_difference_low"
                ]
            )
            > 0.0,

        "medium_damage_CI_crosses_zero":
            (
                float(
                    medium_row[
                        "ci95_difference_low"
                    ]
                )
                < 0.0
                < float(
                    medium_row[
                        "ci95_difference_high"
                    ]
                )
            ),
    }

    for name, passed in (
        high_calibration_checks.items()
    ):

        print(
            f"{name}:",
            passed,
        )

    overall_passed = all(
        high_calibration_checks.values()
    )

    print()
    print(
        "Numerical anchors passed:",
        True,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    if not overall_passed:

        raise AssertionError(
            "Figure 4 scientific audit failed."
        )


if __name__ == "__main__":
    main()
