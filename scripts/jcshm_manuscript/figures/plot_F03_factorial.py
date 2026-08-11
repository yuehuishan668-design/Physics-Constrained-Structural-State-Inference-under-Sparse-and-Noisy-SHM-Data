"""
Generate Figure 3 for the reconstructed JCSHM manuscript.

Figure 3:
Complementary effects of information availability and estimator flexibility.

Scientific purpose
------------------
Panels (a)-(c):
    Compare the 78D signal-derived representation and the 92D
    privileged-information reference under Ridge and RBF-SVR.

Panel (d):
    Show the repeated-split overall-MAE factorial effects and their
    95% confidence intervals.

IMPORTANT
---------
This script performs:

- NO model training;
- NO hyperparameter tuning;
- NO statistical re-testing;
- NO test-driven model selection;
- NO modification of experimental evidence.

All plotted values are read from the frozen manuscript data products.
"""

from __future__ import annotations

from pathlib import Path

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

PERFORMANCE_PATH = (
    FIGURE_DATA_ROOT
    / "F03_factorial_performance.csv"
)

EFFECT_PATH = (
    FIGURE_DATA_ROOT
    / "F03_factorial_effects.csv"
)


# ============================================================
# Frozen numerical anchors
# ============================================================

ANCHORS = {
    "78D_Ridge_test_mae": 0.0448582946,
    "78D_SVR_test_mae": 0.0314500006,
    "92D_Ridge_test_mae": 0.0395454944,
    "92D_SVR_test_mae": 0.0231048479,
    "overall_mae_interaction": -0.0030323525,
}


def get_metric_row(
    frame: pd.DataFrame,
    metric: str,
) -> pd.Series:
    """
    Return exactly one metric row from the factorial-performance table.
    """

    selected = frame.loc[
        frame["metric"] == metric
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one row for metric={metric!r}; "
            f"found {len(selected)}."
        )

    return selected.iloc[0]


def get_effect_row(
    frame: pd.DataFrame,
    metric: str,
    effect: str,
) -> pd.Series:
    """
    Return exactly one factorial-effect row.
    """

    selected = frame.loc[
        (frame["metric"] == metric)
        & (frame["effect"] == effect)
    ]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one effect row for "
            f"metric={metric!r}, effect={effect!r}; "
            f"found {len(selected)}."
        )

    return selected.iloc[0]


def assert_close(
    observed: float,
    expected: float,
    name: str,
    atol: float = 5e-9,
) -> None:
    """
    Fail immediately if a frozen numerical anchor is violated.
    """

    if not np.isclose(
        observed,
        expected,
        atol=atol,
        rtol=1e-9,
    ):
        raise AssertionError(
            f"Numerical anchor failed for {name}: "
            f"observed={observed}, expected={expected}"
        )


def plot_interaction_panel(
    ax: plt.Axes,
    row: pd.Series,
    ylabel: str,
    panel_label: str,
    *,
    percent_axis: bool = False,
) -> None:
    """
    Plot 78D -> 92D interaction lines for Ridge and RBF-SVR.
    """

    x = np.array(
        [
            0.0,
            1.0,
        ]
    )

    ridge = np.array(
        [
            float(
                row["78D_Ridge"]
            ),
            float(
                row["92D_Ridge"]
            ),
        ]
    )

    svr = np.array(
        [
            float(
                row["78D_SVR"]
            ),
            float(
                row["92D_SVR"]
            ),
        ]
    )

    ax.plot(
        x,
        ridge,
        marker="o",
        linestyle="-",
        label="Ridge",
    )

    ax.plot(
        x,
        svr,
        marker="s",
        linestyle="--",
        label="RBF-SVR",
    )

    ax.set_xticks(
        x,
        [
            "78D\nSignal-derived",
            "92D\nPrivileged",
        ],
    )

    ax.set_xlim(
        -0.15,
        1.15,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlabel(
        "Information representation"
    )

    ax.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )
    ax.set_ylim(
    bottom=0.0
    )

    if percent_axis:

        from matplotlib.ticker import PercentFormatter

        ax.yaxis.set_major_formatter(
            PercentFormatter(
                xmax=1.0,
                decimals=0,
            )
        )

    add_panel_label(
        ax,
        panel_label,
    )


def main() -> None:

    apply_manuscript_style()

    performance = pd.read_csv(
        PERFORMANCE_PATH
    )

    effects = pd.read_csv(
        EFFECT_PATH
    )

    # ========================================================
    # Integrity checks
    # ========================================================

    required_performance_columns = {
        "metric",
        "78D_Ridge",
        "78D_SVR",
        "92D_Ridge",
        "92D_SVR",
    }

    if not required_performance_columns.issubset(
        performance.columns
    ):
        raise ValueError(
            "F03 factorial-performance schema does not match "
            "the frozen manuscript schema."
        )

    required_effect_columns = {
        "metric",
        "effect",
        "mean_effect",
        "ci95_low",
        "ci95_high",
        "mean_relative_improvement_percent",
    }

    if not required_effect_columns.issubset(
        effects.columns
    ):
        raise ValueError(
            "F03 factorial-effects schema does not match "
            "the frozen manuscript schema."
        )

    test_mae = get_metric_row(
        performance,
        "test_mae",
    )

    high_mae = get_metric_row(
        performance,
        "high_damage_mae",
    )

    high_abs_bias = get_metric_row(
        performance,
        "high_damage_abs_bias",
    )

    assert_close(
        float(
            test_mae["78D_Ridge"]
        ),
        ANCHORS["78D_Ridge_test_mae"],
        "78D Ridge repeated MAE",
    )

    assert_close(
        float(
            test_mae["78D_SVR"]
        ),
        ANCHORS["78D_SVR_test_mae"],
        "78D SVR repeated MAE",
    )

    assert_close(
        float(
            test_mae["92D_Ridge"]
        ),
        ANCHORS["92D_Ridge_test_mae"],
        "92D Ridge repeated MAE",
    )

    assert_close(
        float(
            test_mae["92D_SVR"]
        ),
        ANCHORS["92D_SVR_test_mae"],
        "92D SVR repeated MAE",
    )

    interaction_row = get_effect_row(
        effects,
        "test_mae",
        "interaction_difference_in_differences",
    )

    assert_close(
        float(
            interaction_row["mean_effect"]
        ),
        ANCHORS["overall_mae_interaction"],
        "overall-MAE factorial interaction",
    )

    # ========================================================
    # Figure layout
    # ========================================================

    fig = plt.figure(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            5.75,
        )
    )

    grid = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        bottom=0.105,
        top=0.965,
        wspace=0.33,
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
    # Panel (a): overall MAE
    # ========================================================

    plot_interaction_panel(
        ax_a,
        test_mae,
        "Overall MAE",
        "(a)",
    )

    # ========================================================
    # Panel (b): high-damage MAE
    # ========================================================

    plot_interaction_panel(
        ax_b,
        high_mae,
        "High-damage MAE",
        "(b)",
    )

    # ========================================================
    # Panel (c): high-damage absolute signed bias
    # ========================================================

    plot_interaction_panel(
        ax_c,
        high_abs_bias,
        "High-damage |bias|",
        "(c)",
    )

    # ========================================================
    # Shared legend
    # ========================================================

    handles, labels = ax_a.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.50,
            1.015,
        ),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=2.0,
    )

    # ========================================================
    # Panel (d): overall-MAE effect decomposition
    #
    # Original effect definitions are expressed as:
    #
    #     comparison condition - reference condition
    #
    # Negative values therefore represent lower MAE.
    #
    # For visual clarity we reverse the sign and display:
    #
    #     absolute MAE reduction
    #
    # so that positive values indicate improvement.
    # ========================================================

    effect_specs = [
    (
        "information_effect_under_ridge",
        "Information effect | Ridge",
    ),
    (
        "information_effect_under_svr",
        "Information effect | RBF-SVR",
    ),
    (
        "model_effect_on_78d",
        "Estimator effect | 78D",
    ),
    (
        "model_effect_on_92d",
        "Estimator effect | 92D",
    ),
    (
        "interaction_difference_in_differences",
        "Interaction",
    ),
]

    effect_rows = []

    for effect_name, display_name in effect_specs:

        row = get_effect_row(
            effects,
            "test_mae",
            effect_name,
        )

        # Reverse sign:
        # negative original difference = positive reduction.
        reduction = -float(
            row["mean_effect"]
        )

        ci_low = -float(
            row["ci95_high"]
        )

        ci_high = -float(
            row["ci95_low"]
        )

        relative = row[
            "mean_relative_improvement_percent"
        ]

        effect_rows.append(
            {
                "effect": effect_name,
                "label": display_name,
                "reduction": reduction,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "relative": (
                    float(relative)
                    if pd.notna(relative)
                    else np.nan
                ),
            }
        )

    effect_plot = pd.DataFrame(
        effect_rows
    )

    # Put first conceptual effect at top.
    effect_plot = effect_plot.iloc[
        ::-1
    ].reset_index(
        drop=True
    )

    y = np.arange(
        len(
            effect_plot
        )
    )

    x = effect_plot[
        "reduction"
    ].to_numpy()

    xerr_lower = (
        x
        - effect_plot[
            "ci_low"
        ].to_numpy()
    )

    xerr_upper = (
        effect_plot[
            "ci_high"
        ].to_numpy()
        - x
    )

    ax_d.errorbar(
        x,
        y,
        xerr=np.vstack(
            [
                xerr_lower,
                xerr_upper,
            ]
        ),
        fmt="D",
        capsize=3.0,
        linestyle="none",
    )

    ax_d.axvline(
        0.0,
        linewidth=0.8,
        linestyle=":",
    )

    ax_d.set_yticks(
        y,
        effect_plot[
            "label"
        ],
    )
    
    ax_d.tick_params(
    axis="y",
    labelsize=7.5,
    )

    ax_d.set_xlabel(
        "Reduction in overall MAE"
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

    # Percentage labels for the four direct effects.
    x_span = max(
        effect_plot[
            "ci_high"
        ].max(),
        1e-6,
    )

    text_offset = (
        0.035
        * x_span
    )

    for idx, row in (
        effect_plot.iterrows()
    ):

        if np.isfinite(
            row["relative"]
        ):

            ax_d.text(
                row["ci_high"]
                + text_offset,
                idx,
                f'{row["relative"]:.1f}%',
                va="center",
                ha="left",
                fontsize=7.5,
            )

    # Give annotation space on the right.
    ax_d.set_xlim(
        left=0.0,
        right=(
            effect_plot[
                "ci_high"
            ].max()
            * 1.30
        ),
    )

    # ========================================================
    # Output
    # ========================================================

    stem = (
        "F03_information_estimator_factorial"
    )

    save_figure(
        fig,
        stem,
    )

    plt.close(
        fig
    )

    print()
    print(
        "=" * 90
    )

    print(
        "FIGURE 3 CHECK"
    )

    print(
        "=" * 90
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
        "Statistical re-testing performed:",
        False,
    )

    print(
        "Numerical anchors passed:",
        True,
    )

    print(
        "Performance rows used:",
        [
            "test_mae",
            "high_damage_mae",
            "high_damage_abs_bias",
        ],
    )

    print(
        "Effect metric used:",
        "test_mae",
    )

    print(
        "Factorial effects displayed:",
        len(
            effect_plot
        ),
    )

    print()

    print(
        "78D Ridge MAE:",
        f'{float(test_mae["78D_Ridge"]):.10f}',
    )

    print(
        "78D SVR MAE:",
        f'{float(test_mae["78D_SVR"]):.10f}',
    )

    print(
        "92D Ridge MAE:",
        f'{float(test_mae["92D_Ridge"]):.10f}',
    )

    print(
        "92D SVR MAE:",
        f'{float(test_mae["92D_SVR"]):.10f}',
    )

    print(
        "Interaction:",
        f'{float(interaction_row["mean_effect"]):.10f}',
    )

    print()
    print(
        "OVERALL PASSED: True"
    )


if __name__ == "__main__":
    main()
