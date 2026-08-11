"""
Generate Figure 2 for the reconstructed JCSHM manuscript.

Figure 2:
Information provenance and dependency-aware descriptor observability.

Scientific structure
--------------------
(a) Hierarchy of information representations:
    92D privileged -> 86D legacy -> 78D signal-derived
    -> 59D structural-response-only.

(b) Composition of the 92D descriptor space by information source.

(c) Dependency-aware descriptor availability rule:
        descriptor j observable under sensor layout S
        iff D_j is a subset of S.

(d) Representative sensor layouts and resulting observable
    descriptor dimensions.

IMPORTANT
---------
This script performs:

- NO model training;
- NO hyperparameter tuning;
- NO statistical testing;
- NO descriptor redesign;
- NO sensor masking;
- NO descriptor redefinition.

All dimensions are read from frozen manuscript data or
derived exactly from the frozen nested descriptor hierarchy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from jcshm_figure_style import (
    DOUBLE_COLUMN_WIDTH,
    add_panel_label,
    apply_manuscript_style,
    save_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANUSCRIPT_ROOT = (
    PROJECT_ROOT
    / "manuscript"
    / "jcshm_reconstruction"
)

PROVENANCE_PATH = (
    MANUSCRIPT_ROOT
    / "tables"
    / "T01_descriptor_provenance.csv"
)

LAYOUT_PATH = (
    MANUSCRIPT_ROOT
    / "figures"
    / "data"
    / "F06_sensor_layouts.csv"
)


# ============================================================
# Presentation helpers
# ============================================================

def box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    face="0.96",
    edge="0.25",
    fontsize=8.0,
    weight="normal",
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9,
    )

    ax.add_patch(
        patch
    )

    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )


def arrow(
    ax,
    start,
    end,
):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.9,
            "color": "0.3",
        },
    )


def sensor_icon(
    ax,
    x,
    y,
    label,
    *,
    active=True,
):
    face = (
        "white"
        if active
        else "0.92"
    )

    edge = (
        "0.15"
        if active
        else "0.7"
    )

    ax.scatter(
        [x],
        [y],
        s=120,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        zorder=3,
    )

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=6.8,
        zorder=4,
        color=(
            "0.1"
            if active
            else "0.6"
        ),
    )


def get_layout_dimension(
    layouts,
    layout,
):
    selected = layouts.loc[
        layouts["layout_tag"]
        == layout
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one layout row for {layout!r}."
        )

    return int(
        selected.iloc[0][
            "feature_count"
        ]
    )


# ============================================================
# Main
# ============================================================

def main():

    apply_manuscript_style()

    provenance = pd.read_csv(
        PROVENANCE_PATH
    )

    layouts = pd.read_csv(
        LAYOUT_PATH,
        dtype={
            "layout_tag": str,
        },
    )

    # ========================================================
    # Frozen hierarchy checks
    # ========================================================

    expected_labels = [
        "92D",
        "86D legacy",
        "78D",
        "59D",
    ]

    observed_labels = provenance[
        "short_label"
    ].tolist()

    if observed_labels != expected_labels:
        raise AssertionError(
            "Descriptor hierarchy does not match frozen order."
        )

    dimensions = provenance[
        "dimension"
    ].astype(int).tolist()

    if dimensions != [
        92,
        86,
        78,
        59,
    ]:
        raise AssertionError(
            f"Unexpected descriptor dimensions: {dimensions}"
        )

    # Exact decomposition from nested hierarchy.
    #
    # 92 -> 86 : 6 generator metadata descriptors
    # 86 -> 78 : 8 exact generator-frequency-derived descriptors
    # 78 -> 59 : 19 base-input-dependent signal descriptors
    # 59       : structural-response-derived descriptors
    source_counts = {
        "Structural-response-derived":
            59,

        "Base-input-dependent":
            19,

        "Generator-frequency-derived":
            8,

        "Generator metadata":
            6,
    }

    if sum(
        source_counts.values()
    ) != 92:
        raise AssertionError(
            "Descriptor-source decomposition does not sum to 92."
        )

    # ========================================================
    # Representative layout anchors
    # ========================================================

    representative_layouts = [
        "1",
        "12",
        "24",
        "123",
        "1234",
    ]

    expected_dimensions = {
        "1": 19,
        "12": 36,
        "24": 29,
        "123": 53,
        "1234": 78,
    }

    observed_dimensions = {
        layout:
            get_layout_dimension(
                layouts,
                layout,
            )
        for layout in representative_layouts
    }

    if (
        observed_dimensions
        != expected_dimensions
    ):
        raise AssertionError(
            "Representative layout dimensions "
            "do not match frozen anchors."
        )

    # ========================================================
    # Figure layout
    # ========================================================

    fig = plt.figure(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            6.0,
        )
    )

    grid = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.07,
        top=0.97,
        wspace=0.26,
        hspace=0.28,
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

    for ax in [
        ax_a,
        ax_b,
        ax_c,
        ax_d,
    ]:
        ax.set_axis_off()

    # ========================================================
    # Panel (a)
    # Information-provenance hierarchy
    # ========================================================

    ax_a.set_xlim(
        0,
        1,
    )

    ax_a.set_ylim(
        0,
        1,
    )

    hierarchy = [
        (
            0.80,
            "92D  Privileged-information reference",
            "simulation-privileged",
        ),
        (
            0.59,
            "86D  Legacy simulation-informed",
            "− 6 generator metadata",
        ),
        (
            0.38,
            "78D  Signal-derived representation",
            "− 8 exact generator-frequency-derived",
        ),
        (
            0.17,
            "59D  Structural-response-only",
            "− 19 base-input-dependent",
        ),
    ]

    for index, (
        y,
        label,
        note,
    ) in enumerate(
        hierarchy
    ):

        box(
            ax_a,
            (
                0.13,
                y,
            ),
            0.74,
            0.105,
            label,
            face=(
                "0.92"
                if index < 2
                else "white"
            ),
            weight=(
                "bold"
                if index == 2
                else "normal"
            ),
        )

        if index < 3:

            arrow(
                ax_a,
                (
                    0.50,
                    y,
                ),
                (
                    0.50,
                    hierarchy[
                        index + 1
                    ][0]
                    + 0.105,
                ),
            )

            ax_a.text(
                0.53,
                (
                    y
                    + hierarchy[
                        index + 1
                    ][0]
                    + 0.105
                )
                / 2,
                hierarchy[
                    index + 1
                ][2],
                ha="left",
                va="center",
                fontsize=6.7,
                color="0.35",
            )

    ax_a.text(
        0.50,
        0.035,
        "Primary manuscript representation: 78D",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
    )

    add_panel_label(
        ax_a,
        "(a)",
    )

    # ========================================================
    # Panel (b)
    # Descriptor-source composition
    # ========================================================

    ax_b.set_xlim(
        0,
        1,
    )

    ax_b.set_ylim(
        0,
        1,
    )

    total = 92.0

    x0 = 0.12
    y0 = 0.49
    width = 0.76
    height = 0.16

    cumulative = x0

    shades = [
        "0.20",
        "0.45",
        "0.68",
        "0.86",
    ]

    categories = list(
        source_counts.items()
    )

    for (
        category,
        count,
    ), shade in zip(
        categories,
        shades,
    ):

        segment_width = (
            width
            * count
            / total
        )

        rectangle = Rectangle(
            (
                cumulative,
                y0,
            ),
            segment_width,
            height,
            facecolor=shade,
            edgecolor="white",
            linewidth=0.8,
        )

        ax_b.add_patch(
            rectangle
        )

        if segment_width > 0.08:

            ax_b.text(
                cumulative
                + segment_width / 2,
                y0
                + height / 2,
                str(
                    count
                ),
                ha="center",
                va="center",
                fontsize=7.0,
                color=(
                    "white"
                    if shade
                    in [
                        "0.20",
                        "0.45",
                    ]
                    else "0.1"
                ),
                fontweight="bold",
            )

        cumulative += (
            segment_width
        )

    legend_y = [
        0.34,
        0.25,
        0.16,
        0.07,
    ]

    for (
        category,
        count,
    ), shade, y in zip(
        categories,
        shades,
        legend_y,
    ):

        ax_b.add_patch(
            Rectangle(
                (
                    0.14,
                    y,
                ),
                0.035,
                0.035,
                facecolor=shade,
                edgecolor="0.25",
                linewidth=0.5,
            )
        )

        ax_b.text(
            0.20,
            y + 0.0175,
            f"{category} ({count})",
            ha="left",
            va="center",
            fontsize=7.0,
        )

    ax_b.text(
        0.50,
        0.78,
        "Nested provenance decomposition of the 92D space",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )

    ax_b.text(
        0.50,
        0.70,
        "59 + 19 + 8 + 6 = 92 descriptors",
        ha="center",
        va="center",
        fontsize=7.1,
    )

    add_panel_label(
        ax_b,
        "(b)",
    )

    # ========================================================
    # Panel (c)
    # Dependency-aware availability rule
    # ========================================================

    ax_c.set_xlim(
        0,
        1,
    )

    ax_c.set_ylim(
        0,
        1,
    )

    ax_c.text(
        0.50,
        0.90,
        r"$f_j$ observable under layout $S$ iff $D_j \subseteq S$",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
    )

    ax_c.text(
        0.50,
        0.82,
        (
            r"$D_j$: structural sensors required by "
            r"descriptor $j$"
        ),
        ha="center",
        va="center",
        fontsize=7.1,
    )

    rules = [
        (
            "Ground-only descriptor",
            r"$D_j=\varnothing$",
        ),
        (
            "Story i statistics / spectrum",
            r"$D_j=\{i\}$",
        ),
        (
            "Story 1 relative-to-lower",
            r"$D_j=\{1\}$ + base input",
        ),
        (
            "Story i>1 relative-to-lower",
            r"$D_j=\{i-1,i\}$",
        ),
        (
            "Adjacent ratio / correlation",
            r"$D_j=\{i-1,i\}$",
        ),
        (
            "Spatial response fraction",
            r"$D_j=\{1,2,3,4\}$",
        ),
    ]

    start_y = 0.68
    dy = 0.105

    for index, (
        descriptor,
        dependency,
    ) in enumerate(
        rules
    ):

        y = (
            start_y
            - index * dy
        )

        ax_c.text(
            0.08,
            y,
            descriptor,
            ha="left",
            va="center",
            fontsize=6.8,
        )

        ax_c.text(
            0.92,
            y,
            dependency,
            ha="right",
            va="center",
            fontsize=7.0,
        )

        ax_c.plot(
            [
                0.08,
                0.92,
            ],
            [
                y - 0.045,
                y - 0.045,
            ],
            linewidth=0.4,
            color="0.86",
        )

    ax_c.text(
        0.50,
        0.025,
        "No zero masking • No descriptor redefinition",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
    )

    add_panel_label(
        ax_c,
        "(c)",
    )

    # ========================================================
    # Panel (d)
    # Example sensor layouts and observable dimensions
    # ========================================================

    ax_d.set_xlim(
        0,
        1,
    )

    ax_d.set_ylim(
        0,
        1,
    )

    ax_d.text(
        0.50,
        0.92,
        "Representative dependency-aware layouts",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )

    examples = [
        (
            "1",
            19,
        ),
        (
            "12",
            36,
        ),
        (
            "24",
            29,
        ),
        (
            "123",
            53,
        ),
        (
            "1234",
            78,
        ),
    ]

    row_y = [
        0.76,
        0.61,
        0.46,
        0.31,
        0.16,
    ]

    sensor_x = [
        0.17,
        0.27,
        0.37,
        0.47,
    ]

    for (
        layout,
        dimension,
    ), y in zip(
        examples,
        row_y,
    ):

        active_set = {
            int(
                character
            )
            for character
            in layout
        }

        ax_d.text(
            0.06,
            y,
            f"{{{','.join(layout)}}}",
            ha="left",
            va="center",
            fontsize=7.2,
            fontweight="bold",
        )

        for sensor_id, x_sensor in enumerate(
            sensor_x,
            start=1,
        ):

            sensor_icon(
                ax_d,
                x_sensor,
                y,
                str(
                    sensor_id
                ),
                active=(
                    sensor_id
                    in active_set
                ),
            )

        ax_d.text(
            0.60,
            y,
            "→",
            ha="center",
            va="center",
            fontsize=10.0,
        )

        ax_d.text(
            0.72,
            y,
            f"{dimension}D",
            ha="center",
            va="center",
            fontsize=8.3,
            fontweight="bold",
        )

        if layout == "1234":

            ax_d.text(
                0.87,
                y,
                "full 78D",
                ha="center",
                va="center",
                fontsize=6.5,
                color="0.35",
            )

    ax_d.text(
        0.50,
        0.035,
        "Base-input sensor assumed available for the primary 78D protocol",
        ha="center",
        va="center",
        fontsize=6.7,
    )

    add_panel_label(
        ax_d,
        "(d)",
    )

    # ========================================================
    # Output
    # ========================================================

    stem = (
        "F02_information_provenance_observability"
    )

    save_figure(
        fig,
        stem,
    )

    plt.close(
        fig
    )

    # ========================================================
    # Audit
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "FIGURE 2 CHECK"
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
        "Statistical testing performed:",
        False,
    )

    print(
        "Sensor zero masking used:",
        False,
    )

    print(
        "Descriptor redefinition used:",
        False,
    )

    print()

    print(
        "Descriptor hierarchy:"
    )

    for label, dimension in zip(
        observed_labels,
        dimensions,
    ):

        print(
            f"  {label}: {dimension}D"
        )

    print()

    print(
        "Source decomposition:"
    )

    for name, count in (
        source_counts.items()
    ):

        print(
            f"  {name}: {count}"
        )

    print(
        "  total:",
        sum(
            source_counts.values()
        ),
    )

    print()

    print(
        "Representative layout dimensions:"
    )

    for layout, dimension in (
        observed_dimensions.items()
    ):

        print(
            f"  {layout}: {dimension}D"
        )

    audit_checks = {
        "hierarchy_92_86_78_59":
            dimensions
            == [
                92,
                86,
                78,
                59,
            ],

        "source_decomposition_sums_92":
            sum(
                source_counts.values()
            )
            == 92,

        "92_minus_86_is_6":
            92 - 86
            == 6,

        "86_minus_78_is_8":
            86 - 78
            == 8,

        "78_minus_59_is_19":
            78 - 59
            == 19,

        "representative_dimensions":
            observed_dimensions
            == expected_dimensions,

        "full_layout_is_78D":
            observed_dimensions[
                "1234"
            ]
            == 78,

        "noncontiguous_24_is_29D":
            observed_dimensions[
                "24"
            ]
            == 29,
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
        "OVERALL PASSED:",
        overall_passed,
    )

    if not overall_passed:

        raise AssertionError(
            "Figure 2 scientific audit failed."
        )


if __name__ == "__main__":
    main()
