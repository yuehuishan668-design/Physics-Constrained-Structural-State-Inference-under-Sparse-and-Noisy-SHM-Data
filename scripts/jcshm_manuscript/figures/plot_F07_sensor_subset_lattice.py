"""
Generate Figure 7 for the reconstructed JCSHM manuscript.

Figure 7:
Complete structural-sensor subset lattice and marginal
value of additional sensing.

Scientific structure
--------------------
(a) Paired overall-MAE improvement for consecutive
    sensor-count increases: 1->2, 2->3, and 3->4.

(b) Paired high-damage-MAE improvement for the same
    sensor-count increases.

(c) Complete 15-node / 28-edge subset lattice, with edge
    thickness proportional to overall-MAE reduction.

(d) The same complete subset lattice, with edge thickness
    proportional to high-damage-MAE reduction.

IMPORTANT
---------
This script performs:

- NO model training;
- NO hyperparameter tuning;
- NO statistical re-testing;
- NO sensor-layout selection;
- NO sensor-placement optimization;
- NO descriptor redesign.

All paired confidence intervals and marginal effects are
read directly from the frozen bootstrap-closure results.

Bootstrap intervals are conditional case-level resampling
intervals for the current 450-case simulated test population.
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


COUNT_EFFECT_PATH = (
    FIGURE_DATA_ROOT
    / "F07_sensor_count_effects.csv"
)

EDGE_PATH = (
    FIGURE_DATA_ROOT
    / "F07_marginal_sensor_edges.csv"
)


# ============================================================
# Complete non-empty four-sensor subset lattice
# ============================================================

EXPECTED_NODES = {
    "1",
    "2",
    "3",
    "4",
    "12",
    "13",
    "14",
    "23",
    "24",
    "34",
    "123",
    "124",
    "134",
    "234",
    "1234",
}


# Manual Hasse-diagram coordinates.
#
# Vertical coordinate = structural sensor count.
# Therefore every legitimate one-sensor addition moves upward.
NODE_POSITIONS = {
    # 1 sensor
    "1": (-1.8, 0.0),
    "2": (-0.6, 0.0),
    "3": (0.6, 0.0),
    "4": (1.8, 0.0),

    # 2 sensors
    "12": (-2.5, 1.0),
    "13": (-1.5, 1.0),
    "14": (-0.5, 1.0),
    "23": (0.5, 1.0),
    "24": (1.5, 1.0),
    "34": (2.5, 1.0),

    # 3 sensors
    "123": (-1.8, 2.0),
    "124": (-0.6, 2.0),
    "134": (0.6, 2.0),
    "234": (1.8, 2.0),

    # 4 sensors
    "1234": (0.0, 3.0),
}


# ============================================================
# Frozen numerical anchors
# ============================================================

COUNT_ANCHORS = {
    "overall_1_to_2":
        0.0151877699,

    "overall_2_to_3":
        0.0181453620,

    "overall_3_to_4":
        0.0179941446,

    "high_1_to_2":
        0.0731757552,

    "high_2_to_3":
        0.0614395713,

    "high_3_to_4":
        0.0473097719,
}


def assert_close(
    observed: float,
    expected: float,
    name: str,
    atol: float = 5e-8,
) -> None:

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


def bool_series(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(
            bool
        )

    return (
        series.astype(
            str
        )
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(
            False
        )
        .astype(
            bool
        )
    )


def layout_to_set(
    value: str,
) -> set[int]:

    return {
        int(
            character
        )
        for character in str(
            value
        )
        if character.isdigit()
    }


def get_count_row(
    frame: pd.DataFrame,
    from_count: int,
    to_count: int,
) -> pd.Series:

    selected = frame.loc[
        (
            frame[
                "from_sensor_count"
            ]
            == from_count
        )
        & (
            frame[
                "to_sensor_count"
            ]
            == to_count
        )
    ]

    if len(
        selected
    ) != 1:

        raise ValueError(
            "Expected one sensor-count contrast for "
            f"{from_count}->{to_count}; "
            f"found {len(selected)}."
        )

    return selected.iloc[
        0
    ]


def asymmetric_error(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:

    return np.vstack(
        [
            point - lower,
            upper - point,
        ]
    )


def scaled_edge_widths(
    values: np.ndarray,
    minimum: float = 0.8,
    maximum: float = 3.0,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    value_min = float(
        values.min()
    )

    value_max = float(
        values.max()
    )

    if np.isclose(
        value_min,
        value_max,
    ):

        return np.full(
            len(
                values
            ),
            (
                minimum
                + maximum
            )
            / 2.0,
        )

    normalized = (
        values
        - value_min
    ) / (
        value_max
        - value_min
    )

    return (
        minimum
        + normalized
        * (
            maximum
            - minimum
        )
    )


def draw_lattice(
    ax: plt.Axes,
    edges: pd.DataFrame,
    improvement_column: str,
    panel_label: str,
    metric_name: str,
) -> None:
    """
    Draw the complete four-sensor subset lattice.

    Edge thickness is a descriptive encoding of the frozen
    marginal improvement point estimate.

    Statistical support is NOT inferred from line thickness.
    """

    values = edges[
        improvement_column
    ].to_numpy(
        dtype=float
    )

    widths = scaled_edge_widths(
        values
    )

    for (
        (_, row),
        width,
    ) in zip(
        edges.iterrows(),
        widths,
    ):

        base = str(
            row[
                "base_layout"
            ]
        )

        augmented = str(
            row[
                "augmented_layout"
            ]
        )

        x0, y0 = (
            NODE_POSITIONS[
                base
            ]
        )

        x1, y1 = (
            NODE_POSITIONS[
                augmented
            ]
        )

        ax.plot(
            [
                x0,
                x1,
            ],
            [
                y0,
                y1,
            ],
            linewidth=width,
            color="0.55",
            alpha=0.75,
            zorder=1,
        )

    # Nodes grouped by sensor count.
    for layout, (
        x,
        y,
    ) in NODE_POSITIONS.items():

        sensor_count = len(
            layout
        )

        # Layout tags have exactly one digit per structural
        # sensor, so character count equals sensor count.
        # 1234 -> four sensors, etc.
        if layout == "1234":
            sensor_count = 4

        ax.scatter(
            [
                x
            ],
            [
                y
            ],
            s=105,
            facecolor="white",
            edgecolor="0.15",
            linewidth=1.0,
            zorder=3,
        )

        ax.text(
            x,
            y,
            layout,
            ha="center",
            va="center",
            fontsize=6.7,
            zorder=4,
        )

    ax.set_xlim(
        -2.9,
        2.9,
    )

    ax.set_ylim(
        -0.35,
        3.35,
    )

    ax.set_xticks(
        []
    )

    ax.set_yticks(
        [
            0,
            1,
            2,
            3,
        ],
        [
            "1 sensor",
            "2 sensors",
            "3 sensors",
            "4 sensors",
        ],
    )

    ax.set_ylabel(
        "Structural sensing level"
    )

    ax.set_xlabel(
        "One-sensor additions progress upward"
    )

    ax.spines[
        "bottom"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_visible(
        False
    )

    ax.tick_params(
        axis="y",
        length=0,
    )

    ax.grid(
        False
    )

    ax.text(
        0.02,
        0.98,
        "28/28 paired CIs beneficial",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )

    ax.text(
        0.98,
        0.02,
        "Thicker edge = larger reduction",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
    )

    add_panel_label(
        ax,
        panel_label,
    )


def main() -> None:

    apply_manuscript_style()

    count_effects = pd.read_csv(
        COUNT_EFFECT_PATH
    )

    edges = pd.read_csv(
        EDGE_PATH,
        dtype={
            "base_layout": str,
            "augmented_layout": str,
        },
    )

    # ========================================================
    # Schema checks
    # ========================================================

    required_count_columns = {
        "from_sensor_count",
        "to_sensor_count",
        "mae_improvement_point",
        "mae_improvement_percent_point",
        "mae_ci_lower",
        "mae_ci_upper",
        "mae_bootstrap_positive_fraction",
        "high_mae_improvement_point",
        "high_mae_improvement_percent_point",
        "high_mae_ci_lower",
        "high_mae_ci_upper",
        "high_mae_bootstrap_positive_fraction",
    }

    required_edge_columns = {
        "base_layout",
        "base_sensor_count",
        "added_sensor",
        "augmented_layout",
        "mae_improvement_point",
        "mae_ci_lower",
        "mae_ci_upper",
        "mae_ci_entirely_positive",
        "mae_bootstrap_positive_fraction",
        "high_mae_improvement_point",
        "high_mae_ci_lower",
        "high_mae_ci_upper",
        "high_mae_ci_entirely_positive",
        "high_mae_bootstrap_positive_fraction",
    }

    if not required_count_columns.issubset(
        count_effects.columns
    ):

        raise ValueError(
            "F07 sensor-count-effect schema mismatch."
        )

    if not required_edge_columns.issubset(
        edges.columns
    ):

        raise ValueError(
            "F07 marginal-edge schema mismatch."
        )

    if len(
        count_effects
    ) != 3:

        raise AssertionError(
            "Expected exactly three sensor-count contrasts."
        )

    if len(
        edges
    ) != 28:

        raise AssertionError(
            "Expected exactly 28 one-sensor-addition edges."
        )

    # ========================================================
    # Count-effect numerical anchors
    # ========================================================

    row_12 = get_count_row(
        count_effects,
        1,
        2,
    )

    row_23 = get_count_row(
        count_effects,
        2,
        3,
    )

    row_34 = get_count_row(
        count_effects,
        3,
        4,
    )

    assert_close(
        float(
            row_12[
                "mae_improvement_point"
            ]
        ),
        COUNT_ANCHORS[
            "overall_1_to_2"
        ],
        "overall 1->2",
    )

    assert_close(
        float(
            row_23[
                "mae_improvement_point"
            ]
        ),
        COUNT_ANCHORS[
            "overall_2_to_3"
        ],
        "overall 2->3",
    )

    assert_close(
        float(
            row_34[
                "mae_improvement_point"
            ]
        ),
        COUNT_ANCHORS[
            "overall_3_to_4"
        ],
        "overall 3->4",
    )

    assert_close(
        float(
            row_12[
                "high_mae_improvement_point"
            ]
        ),
        COUNT_ANCHORS[
            "high_1_to_2"
        ],
        "high 1->2",
    )

    assert_close(
        float(
            row_23[
                "high_mae_improvement_point"
            ]
        ),
        COUNT_ANCHORS[
            "high_2_to_3"
        ],
        "high 2->3",
    )

    assert_close(
        float(
            row_34[
                "high_mae_improvement_point"
            ]
        ),
        COUNT_ANCHORS[
            "high_3_to_4"
        ],
        "high 3->4",
    )

    # ========================================================
    # Complete-lattice topology audit
    # ========================================================

    all_nodes = set()

    valid_edges = []

    for _, row in (
        edges.iterrows()
    ):

        base = str(
            row[
                "base_layout"
            ]
        )

        augmented = str(
            row[
                "augmented_layout"
            ]
        )

        base_set = layout_to_set(
            base
        )

        augmented_set = layout_to_set(
            augmented
        )

        added = (
            augmented_set
            - base_set
        )

        valid = (
            base_set.issubset(
                augmented_set
            )
            and len(
                augmented_set
            )
            == len(
                base_set
            ) + 1
            and len(
                added
            )
            == 1
        )

        valid_edges.append(
            valid
        )

        all_nodes.add(
            base
        )

        all_nodes.add(
            augmented
        )

    lattice_complete = (
        all(
            valid_edges
        )
        and all_nodes
        == EXPECTED_NODES
    )

    if not lattice_complete:

        raise AssertionError(
            "F07 complete subset-lattice topology failed."
        )

    # ========================================================
    # Beneficial-direction audits
    # ========================================================

    mae_ci_flags = bool_series(
        edges[
            "mae_ci_entirely_positive"
        ]
    )

    high_ci_flags = bool_series(
        edges[
            "high_mae_ci_entirely_positive"
        ]
    )

    edge_checks = {
        "28_MAE_point_positive":
            bool(
                (
                    edges[
                        "mae_improvement_point"
                    ]
                    > 0
                ).all()
            ),

        "28_MAE_CI_positive":
            bool(
                mae_ci_flags.all()
            ),

        "28_MAE_bootstrap_fraction_1":
            bool(
                np.allclose(
                    edges[
                        "mae_bootstrap_positive_fraction"
                    ],
                    1.0,
                    atol=1e-12,
                )
            ),

        "28_high_MAE_point_positive":
            bool(
                (
                    edges[
                        "high_mae_improvement_point"
                    ]
                    > 0
                ).all()
            ),

        "28_high_MAE_CI_positive":
            bool(
                high_ci_flags.all()
            ),

        "28_high_MAE_bootstrap_fraction_1":
            bool(
                np.allclose(
                    edges[
                        "high_mae_bootstrap_positive_fraction"
                    ],
                    1.0,
                    atol=1e-12,
                )
            ),
    }

    count_checks = {
        "3_overall_count_CI_positive":
            bool(
                (
                    count_effects[
                        "mae_ci_lower"
                    ]
                    > 0
                ).all()
            ),

        "3_high_count_CI_positive":
            bool(
                (
                    count_effects[
                        "high_mae_ci_lower"
                    ]
                    > 0
                ).all()
            ),

        "3_overall_bootstrap_fraction_1":
            bool(
                np.allclose(
                    count_effects[
                        "mae_bootstrap_positive_fraction"
                    ],
                    1.0,
                    atol=1e-12,
                )
            ),

        "3_high_bootstrap_fraction_1":
            bool(
                np.allclose(
                    count_effects[
                        "high_mae_bootstrap_positive_fraction"
                    ],
                    1.0,
                    atol=1e-12,
                )
            ),
    }

    # ========================================================
    # Figure
    # ========================================================

    fig = plt.figure(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            6.6,
        )
    )

    grid = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        bottom=0.09,
        top=0.96,
        wspace=0.38,
        hspace=0.42,
        height_ratios=[
            0.72,
            1.28,
        ],
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
    # Panel (a): overall count-step effects
    # ========================================================

    x = np.arange(
        3
    )

    labels = [
        "1→2",
        "2→3",
        "3→4",
    ]

    overall_point = (
        count_effects[
            "mae_improvement_point"
        ]
        .to_numpy()
    )

    overall_error = asymmetric_error(
        overall_point,
        count_effects[
            "mae_ci_lower"
        ].to_numpy(),
        count_effects[
            "mae_ci_upper"
        ].to_numpy(),
    )

    ax_a.errorbar(
        x,
        overall_point,
        yerr=overall_error,
        fmt="D",
        linestyle="none",
        color="0.2",
        capsize=3.0,
    )

    ax_a.set_xticks(
        x,
        labels,
    )

    ax_a.set_ylim(
        bottom=0.0
    )

    ax_a.set_xlabel(
        "Sensor-count increase"
    )

    ax_a.set_ylabel(
        "Overall MAE reduction"
    )

    ax_a.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    for index, row in (
        count_effects.iterrows()
    ):

        ax_a.text(
            index,
            float(
                row[
                    "mae_ci_upper"
                ]
            )
            + 0.00055,
            (
                f'{float(row["mae_improvement_percent_point"]):.1f}%'
            ),
            ha="center",
            va="bottom",
            fontsize=7.3,
        )

    add_panel_label(
        ax_a,
        "(a)",
    )

    # ========================================================
    # Panel (b): high-damage count-step effects
    # ========================================================

    high_point = (
        count_effects[
            "high_mae_improvement_point"
        ]
        .to_numpy()
    )

    high_error = asymmetric_error(
        high_point,
        count_effects[
            "high_mae_ci_lower"
        ].to_numpy(),
        count_effects[
            "high_mae_ci_upper"
        ].to_numpy(),
    )

    ax_b.errorbar(
        x,
        high_point,
        yerr=high_error,
        fmt="D",
        linestyle="none",
        color="0.2",
        capsize=3.0,
    )

    ax_b.set_xticks(
        x,
        labels,
    )

    ax_b.set_ylim(
        bottom=0.0
    )

    ax_b.set_xlabel(
        "Sensor-count increase"
    )

    ax_b.set_ylabel(
        "High-damage MAE reduction"
    )

    ax_b.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    for index, row in (
        count_effects.iterrows()
    ):

        ax_b.text(
            index,
            float(
                row[
                    "high_mae_ci_upper"
                ]
            )
            + 0.002,
            (
                f'{float(row["high_mae_improvement_percent_point"]):.1f}%'
            ),
            ha="center",
            va="bottom",
            fontsize=7.3,
        )

    add_panel_label(
        ax_b,
        "(b)",
    )

    # ========================================================
    # Panel (c): complete overall-MAE lattice
    # ========================================================

    draw_lattice(
        ax_c,
        edges,
        "mae_improvement_point",
        "(c)",
        "Overall MAE",
    )

    # ========================================================
    # Panel (d): complete high-damage-MAE lattice
    # ========================================================

    draw_lattice(
        ax_d,
        edges,
        "high_mae_improvement_point",
        "(d)",
        "High-damage MAE",
    )

    # ========================================================
    # Output
    # ========================================================

    stem = (
        "F07_sensor_subset_lattice"
    )

    save_figure(
        fig,
        stem,
    )

    plt.close(
        fig
    )

    # ========================================================
    # Console scientific audit
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "FIGURE 7 CHECK"
    )

    print(
        "=" * 100
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
        "Sensor-layout optimization performed:",
        False,
    )

    print()

    print(
        "Sensor-count effects:"
    )

    print(
        count_effects.to_string(
            index=False
        )
    )

    print()

    print(
        "Lattice nodes:",
        len(
            all_nodes
        ),
    )

    print(
        "Lattice edges:",
        len(
            edges
        ),
    )

    print(
        "All one-sensor additions valid:",
        all(
            valid_edges
        ),
    )

    print(
        "Complete 15-node lattice:",
        all_nodes
        == EXPECTED_NODES,
    )

    print()

    print(
        "Overall-MAE marginal edge range:"
    )

    print(
        "  min:",
        f'{float(edges["mae_improvement_point"].min()):.10f}',
    )

    print(
        "  max:",
        f'{float(edges["mae_improvement_point"].max()):.10f}',
    )

    print()

    print(
        "High-damage-MAE marginal edge range:"
    )

    print(
        "  min:",
        f'{float(edges["high_mae_improvement_point"].min()):.10f}',
    )

    print(
        "  max:",
        f'{float(edges["high_mae_improvement_point"].max()):.10f}',
    )

    print()

    for name, passed in (
        count_checks.items()
    ):

        print(
            f"{name}:",
            bool(
                passed
            ),
        )

    for name, passed in (
        edge_checks.items()
    ):

        print(
            f"{name}:",
            bool(
                passed
            ),
        )

    overall_passed = bool(
        lattice_complete
        and all(
            count_checks.values()
        )
        and all(
            edge_checks.values()
        )
    )

    print()

    print(
        "28/28 overall-MAE edges beneficial:",
        int(
            (
                mae_ci_flags
                & (
                    edges[
                        "mae_improvement_point"
                    ]
                    > 0
                )
            ).sum()
        ),
        "/ 28",
    )

    print(
        "28/28 high-damage-MAE edges beneficial:",
        int(
            (
                high_ci_flags
                & (
                    edges[
                        "high_mae_improvement_point"
                    ]
                    > 0
                )
            ).sum()
        ),
        "/ 28",
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
            "Figure 7 scientific audit failed."
        )


if __name__ == "__main__":
    main()
