"""
Unified plotting style for the reconstructed JCSHM manuscript.

This module contains presentation settings only.

It performs:
- no statistical calculation;
- no model training;
- no data selection;
- no modification of frozen experimental evidence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANUSCRIPT_ROOT = (
    PROJECT_ROOT
    / "manuscript"
    / "jcshm_reconstruction"
)

FIGURE_DATA_ROOT = (
    MANUSCRIPT_ROOT
    / "figures"
    / "data"
)

FIGURE_PREVIEW_ROOT = (
    MANUSCRIPT_ROOT
    / "figures"
    / "preview"
)

FIGURE_FINAL_ROOT = (
    MANUSCRIPT_ROOT
    / "figures"
    / "final"
)

FIGURE_PREVIEW_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_FINAL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# General manuscript plotting constants
# ============================================================

SINGLE_COLUMN_WIDTH = 3.50
DOUBLE_COLUMN_WIDTH = 7.20

FONT_FAMILY = "DejaVu Sans"

BASE_FONT_SIZE = 8.5
LABEL_FONT_SIZE = 9.0
TITLE_FONT_SIZE = 9.5
PANEL_FONT_SIZE = 9.5
LEGEND_FONT_SIZE = 8.0

LINE_WIDTH = 1.35
AXIS_LINE_WIDTH = 0.8
MARKER_SIZE = 5.0
CAP_SIZE = 3.0


def apply_manuscript_style() -> None:
    """
    Apply a clean journal-style plotting configuration.
    """

    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,

            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,

            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,

            "legend.fontsize": LEGEND_FONT_SIZE,

            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,

            "xtick.direction": "out",
            "ytick.direction": "out",

            "xtick.major.width": AXIS_LINE_WIDTH,
            "ytick.major.width": AXIS_LINE_WIDTH,

            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,

            "axes.spines.top": False,
            "axes.spines.right": False,

            "figure.dpi": 120,
            "savefig.dpi": 600,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            "axes.unicode_minus": True,

            "legend.frameon": False,
        }
    )


def add_panel_label(
    ax: plt.Axes,
    label: str,
) -> None:
    """
    Add a consistent panel label such as (a), (b), ...
    """

    ax.text(
        -0.15,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
) -> None:
    """
    Save both internal raster preview and publication master files.
    """

    preview_path = (
        FIGURE_PREVIEW_ROOT
        / f"{stem}.png"
    )

    pdf_path = (
        FIGURE_FINAL_ROOT
        / f"{stem}.pdf"
    )

    svg_path = (
        FIGURE_FINAL_ROOT
        / f"{stem}.svg"
    )

    fig.savefig(
        preview_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    print(
        "Preview:",
        preview_path,
    )

    print(
        "PDF:",
        pdf_path,
    )

    print(
        "SVG:",
        svg_path,
    )
