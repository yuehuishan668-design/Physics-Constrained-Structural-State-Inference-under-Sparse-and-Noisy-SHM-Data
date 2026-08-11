"""
Generate Figure 1 for the reconstructed JCSHM manuscript.

Figure 1:
Deployment-aware structural damage inference and the
two sensing-degradation pathways.

Scientific purpose
------------------
This conceptual figure summarizes the complete manuscript logic:

1. deployment-accessible sensing;
2. physics-guided descriptor construction;
3. storey-level damage inference;
4. reliability diagnostics;
5. measurement-noise failure pathway;
6. sensor-sparsity / observability failure pathway.

IMPORTANT
---------
This is a conceptual figure.

It performs:
- NO model training;
- NO numerical analysis;
- NO statistical testing;
- NO model selection;
- NO experimental modification.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from jcshm_figure_style import (
    DOUBLE_COLUMN_WIDTH,
    apply_manuscript_style,
    save_figure,
)


# ============================================================
# Drawing helpers
# ============================================================

def rounded_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    *,
    face="white",
    edge="0.25",
    linewidth=1.0,
    fontsize=7.3,
    weight="normal",
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )

    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )


def connect(
    ax,
    start,
    end,
    *,
    linestyle="-",
    linewidth=1.0,
    color="0.30",
):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": linewidth,
            "linestyle": linestyle,
            "color": color,
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )


def pathway_step(
    ax,
    x,
    y,
    width,
    height,
    text,
    *,
    emphasis=False,
):
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        text,
        face=(
            "0.93"
            if emphasis
            else "white"
        ),
        linewidth=(
            1.15
            if emphasis
            else 0.85
        ),
        fontsize=7.0,
        weight=(
            "bold"
            if emphasis
            else "normal"
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    apply_manuscript_style()

    fig, ax = plt.subplots(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            5.8,
        )
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_axis_off()

    # ========================================================
    # Section titles
    # ========================================================

    ax.text(
        0.50,
        0.965,
        "Deployment-aware structural damage inference",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
    )

    ax.text(
        0.50,
        0.925,
        (
            "Information provenance → observability → inference "
            "→ failure mechanism → deployment boundary"
        ),
        ha="center",
        va="center",
        fontsize=7.2,
        color="0.30",
    )

    # ========================================================
    # Main inference pipeline
    # ========================================================

    pipeline_y = 0.73
    box_w = 0.135
    box_h = 0.105

    x_positions = [
        0.025,
        0.185,
        0.345,
        0.505,
        0.665,
        0.825,
    ]

    labels = [
        (
            "Structural system\nand excitation"
        ),
        (
            "Structural-response\n+ base-input sensing"
        ),
        (
            "Deployment-accessible\ninformation"
        ),
        (
            "78D physics-guided\nsignal-derived descriptors"
        ),
        (
            "Nonlinear\ninference"
        ),
        (
            "Storey-level damage\n+ reliability diagnostics"
        ),
    ]

    for index, (
        x,
        label,
    ) in enumerate(
        zip(
            x_positions,
            labels,
        )
    ):

        rounded_box(
            ax,
            x,
            pipeline_y,
            box_w,
            box_h,
            label,
            face=(
                "0.91"
                if index == 3
                else "white"
            ),
            linewidth=(
                1.25
                if index == 3
                else 0.9
            ),
            fontsize=6.8,
            weight=(
                "bold"
                if index == 3
                else "normal"
            ),
        )

        if index < len(
            x_positions
        ) - 1:

            connect(
                ax,
                (
                    x
                    + box_w,
                    pipeline_y
                    + box_h / 2,
                ),
                (
                    x_positions[
                        index + 1
                    ],
                    pipeline_y
                    + box_h / 2,
                ),
            )

    # ========================================================
    # Assumption / comparator annotations
    # ========================================================

    ax.text(
        0.252,
        0.695,
        "Base/ground-input sensor assumed available",
        ha="center",
        va="top",
        fontsize=6.2,
        color="0.35",
    )

    rounded_box(
        ax,
        0.520,
        0.855,
        0.115,
        0.050,
        "92D privileged\nreference",
        face="0.96",
        edge="0.55",
        linewidth=0.7,
        fontsize=5.9,
    )

    connect(
        ax,
        (
            0.577,
            0.855,
        ),
        (
            0.572,
            pipeline_y
            + box_h,
        ),
        linestyle="--",
        linewidth=0.7,
        color="0.55",
    )

    rounded_box(
        ax,
        0.670,
        0.855,
        0.095,
        0.050,
        "Ridge\nbaseline",
        face="0.96",
        edge="0.55",
        linewidth=0.7,
        fontsize=5.9,
    )

    connect(
        ax,
        (
            0.717,
            0.855,
        ),
        (
            0.732,
            pipeline_y
            + box_h,
        ),
        linestyle="--",
        linewidth=0.7,
        color="0.55",
    )

    # ========================================================
    # Central scientific question
    # ========================================================

    rounded_box(
        ax,
        0.245,
        0.575,
        0.510,
        0.075,
        (
            "When is damage inference identifiable and reliable "
            "under realistic sensing and information constraints?"
        ),
        face="0.94",
        linewidth=1.15,
        fontsize=7.6,
        weight="bold",
    )

    connect(
        ax,
        (
            0.500,
            pipeline_y,
        ),
        (
            0.500,
            0.650,
        ),
        linewidth=1.0,
    )

    # ========================================================
    # Branch labels
    # ========================================================

    ax.text(
        0.265,
        0.515,
        "Measurement-noise pathway",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
    )

    ax.text(
        0.735,
        0.515,
        "Sensor-sparsity pathway",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
    )

    # Split from central question.
    connect(
        ax,
        (
            0.455,
            0.575,
        ),
        (
            0.265,
            0.490,
        ),
    )

    connect(
        ax,
        (
            0.545,
            0.575,
        ),
        (
            0.735,
            0.490,
        ),
    )

    # ========================================================
    # Noise pathway
    # ========================================================

    left_x = 0.135
    right_x = 0.605

    path_w = 0.260
    path_h = 0.052

    path_y = [
        0.425,
        0.345,
        0.265,
        0.185,
        0.105,
    ]

    noise_steps = [
        "Structural-response measurement noise",
        "Descriptor-space distribution shift",
        "Prediction inflation and broadening",
        "High-damage bias reversal",
        "Upper-bound prediction saturation",
    ]

    for index, (
        y,
        text,
    ) in enumerate(
        zip(
            path_y,
            noise_steps,
        )
    ):

        pathway_step(
            ax,
            left_x,
            y,
            path_w,
            path_h,
            text,
            emphasis=(
                index
                in [
                    1,
                    3,
                    4,
                ]
            ),
        )

        if index < len(
            path_y
        ) - 1:

            connect(
                ax,
                (
                    left_x
                    + path_w / 2,
                    y,
                ),
                (
                    left_x
                    + path_w / 2,
                    path_y[
                        index + 1
                    ]
                    + path_h,
                ),
                linewidth=0.85,
            )

    # Calibration implication.
    ax.text(
        0.265,
        0.045,
        (
            "Directional calibration helps in-distribution,\n"
            "but becomes misaligned after bias reversal"
        ),
        ha="center",
        va="center",
        fontsize=6.5,
        color="0.25",
    )

    # ========================================================
    # Sensor-sparsity pathway
    # ========================================================

    sparsity_steps = [
        "Structural sensor removal",
        "Dependency-aware descriptor unavailability",
        "Reduced cross-storey observability",
        "Increasing severe-damage underprediction",
        "Marginal value of additional sensing",
    ]

    for index, (
        y,
        text,
    ) in enumerate(
        zip(
            path_y,
            sparsity_steps,
        )
    ):

        pathway_step(
            ax,
            right_x,
            y,
            path_w,
            path_h,
            text,
            emphasis=(
                index
                in [
                    1,
                    2,
                    3,
                ]
            ),
        )

        if index < len(
            path_y
        ) - 1:

            connect(
                ax,
                (
                    right_x
                    + path_w / 2,
                    y,
                ),
                (
                    right_x
                    + path_w / 2,
                    path_y[
                        index + 1
                    ]
                    + path_h,
                ),
                linewidth=0.85,
            )

    ax.text(
        0.735,
        0.045,
        (
            "15 layouts • 28 one-sensor additions\n"
            "complete tested four-sensor subset lattice"
        ),
        ha="center",
        va="center",
        fontsize=6.5,
        color="0.25",
    )

    # ========================================================
    # Failure-pathway distinction
    # ========================================================

    ax.text(
        0.500,
        0.015,
        (
            "Distinct degradation mechanisms: "
            "noise changes error direction; sensor loss reduces observability"
        ),
        ha="center",
        va="bottom",
        fontsize=6.6,
        fontweight="bold",
    )

    # ========================================================
    # Output
    # ========================================================

    stem = (
        "F01_scientific_architecture"
    )

    save_figure(
        fig,
        stem,
    )

    plt.close(
        fig
    )

    # ========================================================
    # Conceptual audit
    # ========================================================

    audit = {
        "main_representation_78D":
            True,

        "92D_labelled_privileged_reference":
            True,

        "ridge_labelled_baseline":
            True,

        "base_input_sensor_assumption_visible":
            True,

        "noise_pathway_present":
            True,

        "sensor_sparsity_pathway_present":
            True,

        "noise_bias_reversal_present":
            True,

        "noise_saturation_present":
            True,

        "sparsity_observability_loss_present":
            True,

        "sparsity_underprediction_present":
            True,

        "no_claim_of_noise_robustness":
            True,

        "no_claim_of_universal_sensor_optimality":
            True,
    }

    print()
    print(
        "=" * 100
    )

    print(
        "FIGURE 1 CHECK"
    )

    print(
        "=" * 100
    )

    print(
        "Conceptual figure:",
        True,
    )

    print(
        "Training performed:",
        False,
    )

    print(
        "Statistical testing performed:",
        False,
    )

    print(
        "Experimental modification:",
        False,
    )

    print()

    for name, passed in (
        audit.items()
    ):

        print(
            f"{name}:",
            passed,
        )

    overall_passed = all(
        audit.values()
    )

    print()
    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    if not overall_passed:

        raise AssertionError(
            "Figure 1 conceptual audit failed."
        )


if __name__ == "__main__":
    main()
