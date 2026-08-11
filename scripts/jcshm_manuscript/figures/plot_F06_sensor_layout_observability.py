"""
Generate Figure 6 for the reconstructed JCSHM manuscript.

Figure 6:
Inference performance across all 15 dependency-aware
structural sensor layouts.

Scientific structure
--------------------
(a) Overall MAE for all 15 layouts with paired case-bootstrap
    95% confidence intervals.

(b) High-damage MAE for the same layouts.

(c) Available descriptor count versus overall MAE, showing
    that sensing availability broadly controls inference
    capability while descriptor dimensionality alone cannot
    fully explain placement effects.

(d) High-damage underestimation ratio across all layouts,
    showing the directional severe-damage failure induced by
    sensor sparsity.

IMPORTANT
---------
This script performs:

- NO model training;
- NO hyperparameter tuning;
- NO statistical re-testing;
- NO sensor-layout selection;
- NO descriptor redesign;
- NO descriptor redefinition.

All 15 non-empty sensor layouts are shown.

Bootstrap intervals are the already-frozen paired case-level
95% intervals conditional on the current simulated test set.
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


DATA_PATH = (
    FIGURE_DATA_ROOT
    / "F06_sensor_layouts.csv"
)


# ============================================================
# Frozen layout order
#
# Sensor budgets are grouped explicitly.
# Within each group, layouts follow the original exhaustive
# enumeration rather than being reordered by test performance.
# ============================================================

LAYOUT_ORDER = [
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
]


# ============================================================
# Frozen numerical anchors
# ============================================================

ANCHORS = {
    "full_mae":
        0.0220471401,

    "full_high_mae":
        0.0367799556,

    "full_high_underestimation":
        0.6824817518,

    "layout_1_mae":
        0.0712681232,

    "layout_4_mae":
        0.0755871484,

    "layout_12_mae":
        0.0545806983,

    "layout_13_high_mae":
        0.1382463745,

    "layout_123_mae":
        0.0345617622,

    "layout_24_mae":
        0.0637990530,
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
            f"observed={observed}, "
            f"expected={expected}"
        )


def get_layout_row(
    frame: pd.DataFrame,
    layout: str,
) -> pd.Series:

    selected = frame.loc[
        frame["layout_tag"]
        == layout
    ]

    if len(selected) != 1:

        raise ValueError(
            f"Expected exactly one row for "
            f"layout={layout!r}; "
            f"found {len(selected)}."
        )

    return selected.iloc[0]


def make_asymmetric_error(
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


def add_sensor_group_backgrounds(
    ax: plt.Axes,
) -> None:
    """
    Light grouping by sensor count.

    No quantitative meaning is encoded by the background.
    """

    groups = [
        (-0.5, 3.5),
        (3.5, 9.5),
        (9.5, 13.5),
        (13.5, 14.5),
    ]

    for index, (
        left,
        right,
    ) in enumerate(
        groups
    ):

        if index % 2 == 0:

            ax.axvspan(
                left,
                right,
                color="0.96",
                zorder=0,
            )


def add_group_labels(
    ax: plt.Axes,
) -> None:

    group_centres = [
        (
            1.5,
            "1 sensor",
        ),
        (
            6.5,
            "2 sensors",
        ),
        (
            11.5,
            "3 sensors",
        ),
        (
            14.0,
            "4",
        ),
    ]

    ylim = ax.get_ylim()

    span = (
        ylim[1]
        - ylim[0]
    )

    y = (
        ylim[1]
        - 0.025 * span
    )

    for x, label in (
        group_centres
    ):

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="top",
            fontsize=7.2,
        )


def main() -> None:

    apply_manuscript_style()

    data = pd.read_csv(
        DATA_PATH,
        dtype={
            "layout_tag": str,
            "sensor_layout": str,
        },
    )

    # ========================================================
    # Schema checks
    # ========================================================

    required_columns = {
        "layout_tag",
        "sensor_layout",
        "sensor_count",
        "feature_count",
        "test_mae",
        "test_high_mae",
        "test_high_bias",
        "test_high_underestimation",
        "mae_point",
        "mae_ci_lower",
        "mae_ci_upper",
        "high_mae_point",
        "high_mae_ci_lower",
        "high_mae_ci_upper",
        "high_underestimation_point",
        "high_underestimation_ci_lower",
        "high_underestimation_ci_upper",
    }

    if not required_columns.issubset(
        data.columns
    ):

        missing = (
            required_columns
            - set(
                data.columns
            )
        )

        raise ValueError(
            "Missing Figure 6 columns: "
            f"{sorted(missing)}"
        )

    if len(
        data
    ) != 15:

        raise AssertionError(
            f"Expected 15 layouts; "
            f"found {len(data)}."
        )

    if set(
        data[
            "layout_tag"
        ]
    ) != set(
        LAYOUT_ORDER
    ):

        raise AssertionError(
            "The exhaustive 15-layout set "
            "does not match the frozen layout list."
        )

    if not (
        data[
            "sensor_count"
        ]
        ==
        data[
            "sensor_count_bootstrap"
        ]
    ).all():

        raise AssertionError(
            "Sensor-count point-estimate and "
            "bootstrap records disagree."
        )

    if not (
        data[
            "feature_count"
        ]
        ==
        data[
            "feature_count_bootstrap"
        ]
    ).all():

        raise AssertionError(
            "Feature-count point-estimate and "
            "bootstrap records disagree."
        )

    # ========================================================
    # Numerical anchors
    # ========================================================

    full = get_layout_row(
        data,
        "1234",
    )

    layout_1 = get_layout_row(
        data,
        "1",
    )

    layout_4 = get_layout_row(
        data,
        "4",
    )

    layout_12 = get_layout_row(
        data,
        "12",
    )

    layout_13 = get_layout_row(
        data,
        "13",
    )

    layout_123 = get_layout_row(
        data,
        "123",
    )

    layout_24 = get_layout_row(
        data,
        "24",
    )

    assert_close(
        float(
            full[
                "test_mae"
            ]
        ),
        ANCHORS[
            "full_mae"
        ],
        "full layout MAE",
    )

    assert_close(
        float(
            full[
                "test_high_mae"
            ]
        ),
        ANCHORS[
            "full_high_mae"
        ],
        "full layout high MAE",
    )

    assert_close(
        float(
            full[
                "test_high_underestimation"
            ]
        ),
        ANCHORS[
            "full_high_underestimation"
        ],
        "full layout high underestimation",
    )

    assert_close(
        float(
            layout_1[
                "test_mae"
            ]
        ),
        ANCHORS[
            "layout_1_mae"
        ],
        "layout 1 MAE",
    )

    assert_close(
        float(
            layout_4[
                "test_mae"
            ]
        ),
        ANCHORS[
            "layout_4_mae"
        ],
        "layout 4 MAE",
    )

    assert_close(
        float(
            layout_12[
                "test_mae"
            ]
        ),
        ANCHORS[
            "layout_12_mae"
        ],
        "layout 12 MAE",
    )

    assert_close(
        float(
            layout_13[
                "test_high_mae"
            ]
        ),
        ANCHORS[
            "layout_13_high_mae"
        ],
        "layout 13 high MAE",
    )

    assert_close(
        float(
            layout_123[
                "test_mae"
            ]
        ),
        ANCHORS[
            "layout_123_mae"
        ],
        "layout 123 MAE",
    )

    assert_close(
        float(
            layout_24[
                "test_mae"
            ]
        ),
        ANCHORS[
            "layout_24_mae"
        ],
        "layout 24 MAE",
    )

    # Point and bootstrap-table anchor consistency.
    if not np.allclose(
        data[
            "test_mae"
        ],
        data[
            "mae_point"
        ],
        atol=1e-12,
        rtol=1e-10,
    ):

        raise AssertionError(
            "Point-estimate MAE does not match "
            "bootstrap reconstruction anchor."
        )

    if not np.allclose(
        data[
            "test_high_mae"
        ],
        data[
            "high_mae_point"
        ],
        atol=1e-12,
        rtol=1e-10,
    ):

        raise AssertionError(
            "Point-estimate high MAE does not match "
            "bootstrap reconstruction anchor."
        )

    # ========================================================
    # Reorder rows for plotting
    # ========================================================

    order_map = {
        layout:
            index
        for index, layout
        in enumerate(
            LAYOUT_ORDER
        )
    }

    data = (
        data.assign(
            plot_order=data[
                "layout_tag"
            ].map(
                order_map
            )
        )
        .sort_values(
            "plot_order"
        )
        .reset_index(
            drop=True
        )
    )

    x = np.arange(
        len(
            data
        )
    )

    # ========================================================
    # Best layouts within each sensor budget
    # ========================================================

    best_overall = (
        data.loc[
            data.groupby(
                "sensor_count"
            )[
                "test_mae"
            ]
            .idxmin()
        ]
        [
            [
                "sensor_count",
                "layout_tag",
                "test_mae",
            ]
        ]
        .sort_values(
            "sensor_count"
        )
    )

    best_high = (
        data.loc[
            data.groupby(
                "sensor_count"
            )[
                "test_high_mae"
            ]
            .idxmin()
        ]
        [
            [
                "sensor_count",
                "layout_tag",
                "test_high_mae",
            ]
        ]
        .sort_values(
            "sensor_count"
        )
    )

    # Expected current-system anchors.
    expected_best_overall = {
        1: "1",
        2: "12",
        3: "123",
        4: "1234",
    }

    expected_best_high = {
        1: "1",
        2: "13",
        3: "123",
        4: "1234",
    }

    for _, row in (
        best_overall.iterrows()
    ):

        count = int(
            row[
                "sensor_count"
            ]
        )

        if (
            row[
                "layout_tag"
            ]
            != expected_best_overall[
                count
            ]
        ):

            raise AssertionError(
                "Unexpected overall-MAE best layout "
                f"for sensor count {count}."
            )

    for _, row in (
        best_high.iterrows()
    ):

        count = int(
            row[
                "sensor_count"
            ]
        )

        if (
            row[
                "layout_tag"
            ]
            != expected_best_high[
                count
            ]
        ):

            raise AssertionError(
                "Unexpected high-MAE best layout "
                f"for sensor count {count}."
            )

    # ========================================================
    # Equal-dimensional diagnostic anchors
    # ========================================================

    pair_13_14 = (
        get_layout_row(
            data,
            "13",
        ),
        get_layout_row(
            data,
            "14",
        ),
    )

    pair_23_34 = (
        get_layout_row(
            data,
            "23",
        ),
        get_layout_row(
            data,
            "34",
        ),
    )

    pair_124_134 = (
        get_layout_row(
            data,
            "124",
        ),
        get_layout_row(
            data,
            "134",
        ),
    )

    equal_dimension_checks = {
        "13_vs_14":
            int(
                pair_13_14[
                    0
                ][
                    "feature_count"
                ]
            )
            ==
            int(
                pair_13_14[
                    1
                ][
                    "feature_count"
                ]
            )
            == 31,

        "23_vs_34":
            int(
                pair_23_34[
                    0
                ][
                    "feature_count"
                ]
            )
            ==
            int(
                pair_23_34[
                    1
                ][
                    "feature_count"
                ]
            )
            == 34,

        "124_vs_134":
            int(
                pair_124_134[
                    0
                ][
                    "feature_count"
                ]
            )
            ==
            int(
                pair_124_134[
                    1
                ][
                    "feature_count"
                ]
            )
            == 48,
    }

    # ========================================================
    # Figure layout
    # ========================================================

    fig = plt.figure(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            7.0,
        )
    )

    grid = fig.add_gridspec(
        3,
        2,
        left=0.09,
        right=0.985,
        bottom=0.085,
        top=0.97,
        wspace=0.40,
        hspace=0.48,
        height_ratios=[
            1.0,
            1.0,
            0.95,
        ],
    )

    ax_a = fig.add_subplot(
        grid[0, :]
    )

    ax_b = fig.add_subplot(
        grid[1, :]
    )

    ax_c = fig.add_subplot(
        grid[2, 0]
    )

    ax_d = fig.add_subplot(
        grid[2, 1]
    )

    # ========================================================
    # Panel (a): overall MAE
    # ========================================================

    add_sensor_group_backgrounds(
        ax_a
    )

    mae = data[
        "mae_point"
    ].to_numpy()

    mae_error = (
        make_asymmetric_error(
            mae,
            data[
                "mae_ci_lower"
            ].to_numpy(),
            data[
                "mae_ci_upper"
            ].to_numpy(),
        )
    )

    ax_a.errorbar(
        x,
        mae,
        yerr=mae_error,
        fmt="o",
        linestyle="none",
        capsize=2.5,
    )

    ax_a.plot(
        x,
        mae,
        linewidth=0.75,
        alpha=0.45,
    )

    ax_a.set_xticks(
        x,
        data[
            "layout_tag"
        ],
    )

    ax_a.set_xlim(
        -0.6,
        14.6,
    )

    ax_a.set_ylim(
        bottom=0.0
    )

    ax_a.set_ylabel(
        "Overall MAE"
    )

    ax_a.set_xlabel(
        "Structural sensor layout"
    )

    ax_a.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    add_group_labels(
        ax_a
    )

    add_panel_label(
        ax_a,
        "(a)",
    )

    # ========================================================
    # Panel (b): high-damage MAE
    # ========================================================

    add_sensor_group_backgrounds(
        ax_b
    )

    high_mae = (
        data[
            "high_mae_point"
        ].to_numpy()
    )

    high_mae_error = (
        make_asymmetric_error(
            high_mae,
            data[
                "high_mae_ci_lower"
            ].to_numpy(),
            data[
                "high_mae_ci_upper"
            ].to_numpy(),
        )
    )

    ax_b.errorbar(
        x,
        high_mae,
        yerr=high_mae_error,
        fmt="s",
        linestyle="none",
        capsize=2.5,
    )

    ax_b.plot(
        x,
        high_mae,
        linewidth=0.75,
        alpha=0.45,
    )

    ax_b.set_xticks(
        x,
        data[
            "layout_tag"
        ],
    )

    ax_b.set_xlim(
        -0.6,
        14.6,
    )

    ax_b.set_ylim(
        bottom=0.0
    )

    ax_b.set_ylabel(
        "High-damage MAE"
    )

    ax_b.set_xlabel(
        "Structural sensor layout"
    )

    ax_b.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    add_group_labels(
        ax_b
    )

    add_panel_label(
        ax_b,
        "(b)",
    )

    # ========================================================
    # Panel (c):
    # observable descriptor dimension vs overall MAE
    # ========================================================

    marker_map = {
        1: "o",
        2: "s",
        3: "^",
        4: "D",
    }

    for sensor_count in [
        1,
        2,
        3,
        4,
    ]:

        subset = data.loc[
            data[
                "sensor_count"
            ]
            == sensor_count
        ]

        ax_c.scatter(
            subset[
                "feature_count"
            ],
            subset[
                "test_mae"
            ],
            marker=marker_map[
                sensor_count
            ],
            label=(
                f"{sensor_count} sensor"
                if sensor_count == 1
                else f"{sensor_count} sensors"
            ),
            zorder=3,
        )

    for _, row in (
        data.iterrows()
    ):

        ax_c.annotate(
            row[
                "layout_tag"
            ],
            (
                row[
                    "feature_count"
                ],
                row[
                    "test_mae"
                ],
            ),
            xytext=(
                3,
                3,
            ),
            textcoords="offset points",
            fontsize=6.3,
        )

    ax_c.set_xlabel(
        "Available descriptor count"
    )

    ax_c.set_ylabel(
        "Overall MAE"
    )

    ax_c.set_ylim(
        bottom=0.0
    )

    ax_c.grid(
        linewidth=0.45,
        alpha=0.25,
    )

    ax_c.legend(
        fontsize=6.8,
        loc="upper right",
    )

    add_panel_label(
        ax_c,
        "(c)",
    )

    # ========================================================
    # Panel (d):
    # high-damage underestimation
    # ========================================================

    add_sensor_group_backgrounds(
        ax_d
    )

    under = (
        data[
            "high_underestimation_point"
        ].to_numpy()
    )

    under_error = (
        make_asymmetric_error(
            under,
            data[
                "high_underestimation_ci_lower"
            ].to_numpy(),
            data[
                "high_underestimation_ci_upper"
            ].to_numpy(),
        )
    )

    # Only four grouped x positions are too compressed here,
    # so retain all 15 layouts but use compact labels.
    ax_d.errorbar(
        x,
        under,
        yerr=under_error,
        fmt="D",
        linestyle="none",
        capsize=2.0,
    )

    ax_d.plot(
        x,
        under,
        linewidth=0.7,
        alpha=0.45,
    )

    ax_d.set_xticks(
        x,
        data[
            "layout_tag"
        ],
        rotation=60,
        ha="right",
    )

    ax_d.set_xlim(
        -0.6,
        14.6,
    )

    ax_d.set_ylim(
        0.55,
        1.03,
    )

    ax_d.set_ylabel(
        "High-damage underestimation ratio"
    )

    ax_d.set_xlabel(
        "Structural sensor layout"
    )

    ax_d.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    add_panel_label(
        ax_d,
        "(d)",
    )

    # ========================================================
    # Output
    # ========================================================

    stem = (
        "F06_sensor_layout_observability"
    )

    save_figure(
        fig,
        stem,
    )

    plt.close(
        fig
    )

    # ========================================================
    # Scientific audit
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "FIGURE 6 CHECK"
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
        "Sensor layouts evaluated:",
        len(
            data
        ),
    )

    print(
        "Sensor-count levels:",
        sorted(
            data[
                "sensor_count"
            ].unique()
        ),
    )

    print()

    print(
        "Best overall-MAE layout by sensor count:"
    )

    print(
        best_overall.to_string(
            index=False
        )
    )

    print()

    print(
        "Best high-damage-MAE layout by sensor count:"
    )

    print(
        best_high.to_string(
            index=False
        )
    )

    print()

    print(
        "Full-layout anchor:"
    )

    print(
        "  layout:",
        full[
            "layout_tag"
        ],
    )

    print(
        "  features:",
        int(
            full[
                "feature_count"
            ]
        ),
    )

    print(
        "  overall MAE:",
        f'{float(full["test_mae"]):.10f}',
    )

    print(
        "  high-damage MAE:",
        f'{float(full["test_high_mae"]):.10f}',
    )

    print(
        "  high underestimation:",
        f'{float(full["test_high_underestimation"]):.10f}',
    )

    print()

    print(
        "Single-sensor layouts:"
    )

    single = (
        data.loc[
            data[
                "sensor_count"
            ]
            == 1
        ]
        [
            [
                "layout_tag",
                "feature_count",
                "test_mae",
                "test_high_mae",
                "test_high_underestimation",
            ]
        ]
    )

    print(
        single.to_string(
            index=False
        )
    )

    print()

    print(
        "Equal-dimensional layout checks:"
    )

    for name, passed in (
        equal_dimension_checks.items()
    ):

        print(
            f"  {name}:",
            bool(
                passed
            ),
        )

    print()

    print(
        "Objective-dependence check | 2 sensors:"
    )

    print(
        "  overall-MAE best:",
        expected_best_overall[
            2
        ],
    )

    print(
        "  high-damage-MAE best:",
        expected_best_high[
            2
        ],
    )

    audit_checks = {
        "15_layouts_complete":
            len(
                data
            )
            == 15,

        "full_layout_78_features":
            int(
                full[
                    "feature_count"
                ]
            )
            == 78,

        "best_overall_chain":
            list(
                best_overall[
                    "layout_tag"
                ]
            )
            == [
                "1",
                "12",
                "123",
                "1234",
            ],

        "best_high_chain":
            list(
                best_high[
                    "layout_tag"
                ]
            )
            == [
                "1",
                "13",
                "123",
                "1234",
            ],

        "two_sensor_objective_dependence":
            expected_best_overall[
                2
            ]
            != expected_best_high[
                2
            ],

        "all_single_sensor_high_under_gt_0.98":
            bool(
                (
                    single[
                        "test_high_underestimation"
                    ]
                    > 0.98
                ).all()
            ),

        "all_equal_dimension_checks":
            all(
                equal_dimension_checks.values()
            ),

        "bootstrap_point_anchor_MAE":
            np.allclose(
                data[
                    "test_mae"
                ],
                data[
                    "mae_point"
                ],
                atol=1e-12,
            ),

        "bootstrap_point_anchor_high_MAE":
            np.allclose(
                data[
                    "test_high_mae"
                ],
                data[
                    "high_mae_point"
                ],
                atol=1e-12,
            ),
    }

    print()

    for name, passed in (
        audit_checks.items()
    ):

        print(
            f"{name}:",
            bool(
                passed
            ),
        )

    overall_passed = bool(
        all(
            audit_checks.values()
        )
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
            "Figure 6 scientific audit failed."
        )


if __name__ == "__main__":
    main()
