"""
Generate Figure 5 for the reconstructed JCSHM manuscript.

Figure 5:
Measurement-noise-induced descriptor shift, bias reversal,
and prediction saturation.

Scientific structure
--------------------
(a) Noise-induced degradation of overall and high-damage MAE
    for the clean-trained standard RBF-SVR.

(b) High-damage signed-bias reversal for standard and
    asymmetrically calibrated RBF-SVR.

(c) Upper-bound prediction saturation for all predictions
    and high-damage predictions using the standard RBF-SVR.

(d) Standardized descriptor-space shift for the six
    descriptors with the largest mean absolute shift at
    20% measurement noise.

IMPORTANT
---------
This script performs:

- NO model training;
- NO hyperparameter tuning;
- NO calibration fitting;
- NO statistical re-testing;
- NO descriptor redesign;
- NO feature ablation;
- NO test-driven model selection.

The top-six descriptor selection in panel (d) is a fixed
visualization rule based only on the already-frozen 20%
noise descriptor-shift ranking.

This figure provides mechanism-consistent diagnostic
evidence, not feature-level causal proof.
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

NOISE_PERFORMANCE_PATH = (
    FIGURE_DATA_ROOT
    / "F05_noise_performance.csv"
)

BIAS_REVERSAL_PATH = (
    FIGURE_DATA_ROOT
    / "F05_bias_reversal.csv"
)

SATURATION_PATH = (
    FIGURE_DATA_ROOT
    / "F05_prediction_saturation.csv"
)

DESCRIPTOR_SHIFT_PATH = (
    FIGURE_DATA_ROOT
    / "F05_descriptor_shift.csv"
)


# ============================================================
# Frozen definitions
# ============================================================

NOISE_LEVELS = [
    0,
    5,
    10,
    20,
]

NONZERO_NOISE_LEVELS = [
    5,
    10,
    20,
]

TOP_N_DESCRIPTORS = 6


METHOD_LABELS = {
    "standard_svr": "Standard RBF-SVR",
    "calibrated_svr": "Calibrated RBF-SVR",
}


# ============================================================
# Frozen numerical anchors
# ============================================================

ANCHORS = {
    # Controlled clean-trained standard SVR.
    "clean_overall_mae":
        0.0220471401,

    "noise_5_overall_mae":
        0.0443593523,

    "noise_10_overall_mae":
        0.1118056216,

    "noise_20_overall_mae":
        0.2838931183,

    "clean_high_mae":
        0.0367799556,

    "noise_20_high_mae":
        0.2069806408,

    # High-damage signed bias.
    "standard_bias_0":
        -0.0213373048,

    "standard_bias_5":
        -0.0047894942,

    "standard_bias_10":
        0.0425749208,

    "standard_bias_20":
        0.1285342352,

    "calibrated_bias_0":
        -0.0028639517,

    "calibrated_bias_5":
        0.0128237925,

    "calibrated_bias_10":
        0.0570517229,

    "calibrated_bias_20":
        0.1323378766,
}


# ============================================================
# Utilities
# ============================================================

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


def one_row(
    frame: pd.DataFrame,
    **conditions,
) -> pd.Series:

    mask = pd.Series(
        True,
        index=frame.index,
    )

    for column, value in (
        conditions.items()
    ):

        mask &= (
            frame[column]
            == value
        )

    selected = frame.loc[
        mask
    ]

    if len(selected) != 1:

        raise ValueError(
            "Expected exactly one row for "
            f"{conditions}; found {len(selected)}."
        )

    return selected.iloc[0]


def clean_feature_label(
    name: str,
) -> str:
    """
    Convert machine descriptor names to compact plot labels
    without altering descriptor identity.
    """

    label = (
        name
        .replace(
            "story_",
            "S",
        )
        .replace(
            "_",
            " ",
        )
    )

    return label


def detect_sign_flip_interval(
    frame: pd.DataFrame,
    method: str,
) -> tuple[int, int] | None:

    subset = (
        frame.loc[
            frame["method"]
            == method
        ]
        .sort_values(
            "noise_percent"
        )
    )

    noise = subset[
        "noise_percent"
    ].to_numpy()

    bias = subset[
        "high_signed_bias"
    ].to_numpy()

    for index in range(
        len(
            bias
        )
        - 1
    ):

        if (
            bias[index]
            == 0
        ):

            return (
                int(
                    noise[index]
                ),
                int(
                    noise[index]
                ),
            )

        if (
            np.sign(
                bias[index]
            )
            != np.sign(
                bias[
                    index + 1
                ]
            )
        ):

            return (
                int(
                    noise[index]
                ),
                int(
                    noise[
                        index + 1
                    ]
                ),
            )

    return None


# ============================================================
# Main
# ============================================================

def main() -> None:

    apply_manuscript_style()

    performance = pd.read_csv(
        NOISE_PERFORMANCE_PATH
    )

    bias = pd.read_csv(
        BIAS_REVERSAL_PATH
    )

    saturation = pd.read_csv(
        SATURATION_PATH
    )

    descriptor_shift = pd.read_csv(
        DESCRIPTOR_SHIFT_PATH
    )

    # ========================================================
    # Schema checks
    # ========================================================

    required_performance = {
        "method",
        "noise_percent",
        "n_replicates",
        "mae_mean",
        "mae_std",
        "high_mae_mean",
        "high_mae_std",
    }

    required_bias = {
        "method",
        "noise_percent",
        "high_signed_bias",
        "bias_sign",
        "clean_bias",
        "sign_differs_from_clean",
    }

    required_saturation = {
        "method",
        "noise_percent",
        "severity",
        "n_replicates",
        "prediction_q50_mean",
        "clip_high_ratio_mean",
        "clip_high_ratio_std",
    }

    required_shift = {
        "noise_percent",
        "feature_index",
        "feature_name",
        "n_replicates",
        "mean_abs_delta_z_mean",
        "mean_abs_delta_z_std",
        "fraction_abs_delta_z_gt_1_mean",
        "fraction_abs_delta_z_gt_2_mean",
        "rank_by_mean_abs_shift",
    }

    checks = [
        (
            required_performance,
            performance.columns,
            "noise performance",
        ),
        (
            required_bias,
            bias.columns,
            "bias reversal",
        ),
        (
            required_saturation,
            saturation.columns,
            "prediction saturation",
        ),
        (
            required_shift,
            descriptor_shift.columns,
            "descriptor shift",
        ),
    ]

    for required, columns, name in checks:

        if not required.issubset(
            columns
        ):

            missing = (
                required
                - set(
                    columns
                )
            )

            raise ValueError(
                f"Missing {name} columns: "
                f"{sorted(missing)}"
            )

    if len(
        performance
    ) != 8:

        raise AssertionError(
            "Expected 8 noise-performance rows."
        )

    if len(
        bias
    ) != 8:

        raise AssertionError(
            "Expected 8 bias-reversal rows."
        )

    if len(
        saturation
    ) != 48:

        raise AssertionError(
            "Expected 48 saturation rows."
        )

    if len(
        descriptor_shift
    ) != 312:

        raise AssertionError(
            "Expected 312 descriptor-shift rows."
        )

    if set(
        descriptor_shift[
            "feature_index"
        ].unique()
    ) != set(
        range(
            78
        )
    ):

        raise AssertionError(
            "Descriptor shift does not cover "
            "all 78 canonical descriptors."
        )

    # ========================================================
    # Numerical anchor checks
    # ========================================================

    standard_perf = (
        performance.loc[
            performance["method"]
            == "standard_svr"
        ]
    )

    perf_0 = one_row(
        standard_perf,
        noise_percent=0,
    )

    perf_5 = one_row(
        standard_perf,
        noise_percent=5,
    )

    perf_10 = one_row(
        standard_perf,
        noise_percent=10,
    )

    perf_20 = one_row(
        standard_perf,
        noise_percent=20,
    )

    assert_close(
        float(
            perf_0[
                "mae_mean"
            ]
        ),
        ANCHORS[
            "clean_overall_mae"
        ],
        "clean overall MAE",
    )

    assert_close(
        float(
            perf_5[
                "mae_mean"
            ]
        ),
        ANCHORS[
            "noise_5_overall_mae"
        ],
        "5% overall MAE",
    )

    assert_close(
        float(
            perf_10[
                "mae_mean"
            ]
        ),
        ANCHORS[
            "noise_10_overall_mae"
        ],
        "10% overall MAE",
    )

    assert_close(
        float(
            perf_20[
                "mae_mean"
            ]
        ),
        ANCHORS[
            "noise_20_overall_mae"
        ],
        "20% overall MAE",
    )

    assert_close(
        float(
            perf_0[
                "high_mae_mean"
            ]
        ),
        ANCHORS[
            "clean_high_mae"
        ],
        "clean high-damage MAE",
    )

    assert_close(
        float(
            perf_20[
                "high_mae_mean"
            ]
        ),
        ANCHORS[
            "noise_20_high_mae"
        ],
        "20% high-damage MAE",
    )

    for method in [
        "standard_svr",
        "calibrated_svr",
    ]:

        for noise_level in (
            NOISE_LEVELS
        ):

            row = one_row(
                bias,
                method=method,
                noise_percent=noise_level,
            )

            key = (
                (
                    "standard"
                    if method
                    == "standard_svr"
                    else "calibrated"
                )
                + "_bias_"
                + str(
                    noise_level
                )
            )

            assert_close(
                float(
                    row[
                        "high_signed_bias"
                    ]
                ),
                ANCHORS[
                    key
                ],
                key,
            )

    # ========================================================
    # Bias-sign reversal checks
    # ========================================================

    standard_flip = (
        detect_sign_flip_interval(
            bias,
            "standard_svr",
        )
    )

    calibrated_flip = (
        detect_sign_flip_interval(
            bias,
            "calibrated_svr",
        )
    )

    if standard_flip != (
        5,
        10,
    ):

        raise AssertionError(
            "Standard SVR bias reversal "
            "is not between 5% and 10%."
        )

    if calibrated_flip != (
        0,
        5,
    ):

        raise AssertionError(
            "Calibrated SVR bias reversal "
            "is not between 0% and 5%."
        )

    # ========================================================
    # Saturation data
    # ========================================================

    saturation_standard = (
        saturation.loc[
            saturation["method"]
            == "standard_svr"
        ]
    )

    saturation_all = (
        saturation_standard.loc[
            saturation_standard[
                "severity"
            ]
            == "all"
        ]
        .sort_values(
            "noise_percent"
        )
    )

    saturation_high = (
        saturation_standard.loc[
            saturation_standard[
                "severity"
            ]
            == "high"
        ]
        .sort_values(
            "noise_percent"
        )
    )

    if len(
        saturation_all
    ) != 4:

        raise AssertionError(
            "Expected four all-case "
            "standard-SVR saturation rows."
        )

    if len(
        saturation_high
    ) != 4:

        raise AssertionError(
            "Expected four high-damage "
            "standard-SVR saturation rows."
        )

    sat_20_all = one_row(
        saturation_all,
        noise_percent=20,
    )

    sat_20_high = one_row(
        saturation_high,
        noise_percent=20,
    )

    # ========================================================
    # Descriptor-space shift
    #
    # Select top six descriptors ONCE using the frozen
    # 20%-noise rank, then track the same descriptors at
    # 5%, 10%, and 20%.
    # ========================================================

    descriptor_20 = (
        descriptor_shift.loc[
            descriptor_shift[
                "noise_percent"
            ]
            == 20
        ]
        .sort_values(
            [
                "rank_by_mean_abs_shift",
                "feature_index",
            ]
        )
    )

    if len(
        descriptor_20
    ) != 78:

        raise AssertionError(
            "Expected 78 descriptor rows at 20% noise."
        )

    top_descriptors = (
        descriptor_20
        .head(
            TOP_N_DESCRIPTORS
        )
        [
            [
                "feature_index",
                "feature_name",
                "rank_by_mean_abs_shift",
                "mean_abs_delta_z_mean",
                "fraction_abs_delta_z_gt_1_mean",
                "fraction_abs_delta_z_gt_2_mean",
            ]
        ]
        .copy()
    )

    top_indices = (
        top_descriptors[
            "feature_index"
        ]
        .astype(int)
        .tolist()
    )

    tracked_shift = (
        descriptor_shift.loc[
            (
                descriptor_shift[
                    "feature_index"
                ]
                .isin(
                    top_indices
                )
            )
            & (
                descriptor_shift[
                    "noise_percent"
                ]
                .isin(
                    NONZERO_NOISE_LEVELS
                )
            )
        ]
        .copy()
    )

    if len(
        tracked_shift
    ) != (
        TOP_N_DESCRIPTORS
        * len(
            NONZERO_NOISE_LEVELS
        )
    ):

        raise AssertionError(
            "Top-descriptor tracking table "
            "is incomplete."
        )

    # Preserve top-20% rank order in the heatmap.
    feature_order = (
        top_descriptors[
            "feature_name"
        ]
        .tolist()
    )

    heatmap = (
        tracked_shift.pivot(
            index="feature_name",
            columns="noise_percent",
            values="mean_abs_delta_z_mean",
        )
        .reindex(
            index=feature_order,
            columns=NONZERO_NOISE_LEVELS,
        )
    )

    if heatmap.isna().any().any():

        raise AssertionError(
            "Descriptor-shift heatmap contains "
            "missing values."
        )

    # ========================================================
    # Figure
    # ========================================================

    fig = plt.figure(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            5.9,
        )
    )

    grid = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        bottom=0.10,
        top=0.955,
        wspace=0.40,
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
    # Panel (a)
    # Standard SVR error degradation
    # ========================================================

    x = (
        standard_perf[
            "noise_percent"
        ]
        .to_numpy()
    )

    overall_mae = (
        standard_perf[
            "mae_mean"
        ]
        .to_numpy()
    )

    overall_std = (
        standard_perf[
            "mae_std"
        ]
        .to_numpy()
    )

    high_mae = (
        standard_perf[
            "high_mae_mean"
        ]
        .to_numpy()
    )

    high_std = (
        standard_perf[
            "high_mae_std"
        ]
        .to_numpy()
    )

    order = np.argsort(
        x
    )

    x = x[
        order
    ]

    overall_mae = (
        overall_mae[
            order
        ]
    )

    overall_std = (
        overall_std[
            order
        ]
    )

    high_mae = (
        high_mae[
            order
        ]
    )

    high_std = (
        high_std[
            order
        ]
    )

    ax_a.errorbar(
        x,
        overall_mae,
        yerr=overall_std,
        marker="o",
        linestyle="-",
        capsize=2.5,
        label="Overall MAE",
    )

    ax_a.errorbar(
        x,
        high_mae,
        yerr=high_std,
        marker="s",
        linestyle="--",
        capsize=2.5,
        label="High-damage MAE",
    )

    ax_a.set_xlabel(
        "Response-measurement noise (%)"
    )

    ax_a.set_ylabel(
        "MAE"
    )

    ax_a.set_xticks(
        NOISE_LEVELS
    )

    ax_a.set_ylim(
        bottom=0.0
    )

    ax_a.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    ax_a.legend(
        loc="upper left"
    )

    add_panel_label(
        ax_a,
        "(a)",
    )

    # ========================================================
    # Panel (b)
    # High-damage signed-bias reversal
    # ========================================================

    for method, marker, linestyle in [
        (
            "standard_svr",
            "o",
            "-",
        ),
        (
            "calibrated_svr",
            "s",
            "--",
        ),
    ]:

        subset = (
            bias.loc[
                bias["method"]
                == method
            ]
            .sort_values(
                "noise_percent"
            )
        )

        ax_b.plot(
            subset[
                "noise_percent"
            ],
            subset[
                "high_signed_bias"
            ],
            marker=marker,
            linestyle=linestyle,
            label=METHOD_LABELS[
                method
            ],
        )

    ax_b.axhline(
        0.0,
        linewidth=0.8,
        linestyle=":",
        color="0.25",
    )

    ax_b.set_xlabel(
        "Response-measurement noise (%)"
    )

    ax_b.set_ylabel(
        "High-damage signed bias"
    )

    ax_b.set_xticks(
        NOISE_LEVELS
    )

    ax_b.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    ax_b.legend(
        loc="upper left"
    )

    add_panel_label(
        ax_b,
        "(b)",
    )

    # ========================================================
    # Panel (c)
    # Upper-bound prediction saturation
    # ========================================================

    ax_c.errorbar(
        saturation_all[
            "noise_percent"
        ],
        saturation_all[
            "clip_high_ratio_mean"
        ],
        yerr=saturation_all[
            "clip_high_ratio_std"
        ],
        marker="o",
        linestyle="-",
        capsize=2.5,
        label="All predictions",
    )

    ax_c.errorbar(
        saturation_high[
            "noise_percent"
        ],
        saturation_high[
            "clip_high_ratio_mean"
        ],
        yerr=saturation_high[
            "clip_high_ratio_std"
        ],
        marker="s",
        linestyle="--",
        capsize=2.5,
        label="High-damage predictions",
    )

    ax_c.set_xlabel(
        "Response-measurement noise (%)"
    )

    ax_c.set_ylabel(
        "Upper-bound clipping ratio"
    )

    ax_c.set_xticks(
        NOISE_LEVELS
    )

    ax_c.set_ylim(
        bottom=0.0
    )

    ax_c.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    ax_c.legend(
        loc="upper left"
    )

    add_panel_label(
        ax_c,
        "(c)",
    )

    # ========================================================
    # Panel (d)
    # Descriptor-space shift
    #
    # Heatmap values:
    # mean absolute change in standardized descriptor value.
    # ========================================================

    matrix = (
        heatmap.to_numpy(
            dtype=float
        )
    )

    image = ax_d.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    ax_d.set_xticks(
        np.arange(
            len(
                NONZERO_NOISE_LEVELS
            )
        ),
        [
            f"{value}%"
            for value in
            NONZERO_NOISE_LEVELS
        ],
    )

    ax_d.set_yticks(
        np.arange(
            len(
                feature_order
            )
        ),
        [
            clean_feature_label(
                value
            )
            for value in
            feature_order
        ],
    )

    ax_d.set_xlabel(
        "Response-measurement noise"
    )

    ax_d.set_ylabel(
        "Top shifted descriptors"
    )

    ax_d.tick_params(
        axis="y",
        labelsize=7.0,
    )

    for row_index in range(
        matrix.shape[0]
    ):

        for col_index in range(
            matrix.shape[1]
        ):

            value = matrix[
                row_index,
                col_index,
            ]

            ax_d.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.8,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax_d,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        r"Mean $|\Delta z|$"
    )

    add_panel_label(
        ax_d,
        "(d)",
    )

    # ========================================================
    # Output
    # ========================================================

    stem = (
        "F05_noise_failure_mechanism"
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

    print()
    print(
        "=" * 96
    )

    print(
        "FIGURE 5 CHECK"
    )

    print(
        "=" * 96
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
        "Descriptor ablation performed:",
        False,
    )

    print(
        "Descriptor redesign performed:",
        False,
    )

    print()

    print(
        "Standard-SVR overall MAE:"
    )

    for noise_level in (
        NOISE_LEVELS
    ):

        row = one_row(
            standard_perf,
            noise_percent=noise_level,
        )

        print(
            f"  {noise_level:>2}%:",
            f'{float(row["mae_mean"]):.10f}',
        )

    print()

    print(
        "Standard-SVR high-damage MAE:"
    )

    for noise_level in (
        NOISE_LEVELS
    ):

        row = one_row(
            standard_perf,
            noise_percent=noise_level,
        )

        print(
            f"  {noise_level:>2}%:",
            f'{float(row["high_mae_mean"]):.10f}',
        )

    print()

    print(
        "High-damage signed bias | Standard:"
    )

    for noise_level in (
        NOISE_LEVELS
    ):

        row = one_row(
            bias,
            method="standard_svr",
            noise_percent=noise_level,
        )

        print(
            f"  {noise_level:>2}%:",
            f'{float(row["high_signed_bias"]):+.10f}',
        )

    print()

    print(
        "High-damage signed bias | Calibrated:"
    )

    for noise_level in (
        NOISE_LEVELS
    ):

        row = one_row(
            bias,
            method="calibrated_svr",
            noise_percent=noise_level,
        )

        print(
            f"  {noise_level:>2}%:",
            f'{float(row["high_signed_bias"]):+.10f}',
        )

    print()

    print(
        "Standard bias sign flip interval:",
        standard_flip,
    )

    print(
        "Calibrated bias sign flip interval:",
        calibrated_flip,
    )

    print()

    print(
        "20% standard-SVR saturation:"
    )

    print(
        "  All predictions upper clipping:",
        f'{float(sat_20_all["clip_high_ratio_mean"]):.10f}',
    )

    print(
        "  High-damage upper clipping:",
        f'{float(sat_20_high["clip_high_ratio_mean"]):.10f}',
    )

    print(
        "  High-damage prediction median:",
        f'{float(sat_20_high["prediction_q50_mean"]):.10f}',
    )

    print()

    print(
        "Top six descriptors by mean |delta z| "
        "at 20% noise:"
    )

    print(
        top_descriptors.to_string(
            index=False
        )
    )

    print()

    top1 = (
        top_descriptors.iloc[
            0
        ]
    )

    print(
        "Top descriptor:"
    )

    print(
        "  name:",
        top1[
            "feature_name"
        ],
    )

    print(
        "  mean |delta z|:",
        f'{float(top1["mean_abs_delta_z_mean"]):.10f}',
    )

    print(
        "  fraction |delta z| > 1:",
        f'{float(top1["fraction_abs_delta_z_gt_1_mean"]):.10f}',
    )

    print(
        "  fraction |delta z| > 2:",
        f'{float(top1["fraction_abs_delta_z_gt_2_mean"]):.10f}',
    )

    print()

    audit_checks = {
        "standard_flip_5_to_10":
            standard_flip
            == (
                5,
                10,
            ),

        "calibrated_flip_0_to_5":
            calibrated_flip
            == (
                0,
                5,
            ),

        "top6_count":
            len(
                top_descriptors
            )
            == 6,

        "tracked_shift_rows":
            len(
                tracked_shift
            )
            == 18,

        "heatmap_complete":
            not heatmap
            .isna()
            .any()
            .any(),

        "all_case_saturation_increases":
            (
                saturation_all[
                    "clip_high_ratio_mean"
                ]
                .to_numpy()
                [1:]
                >=
                saturation_all[
                    "clip_high_ratio_mean"
                ]
                .to_numpy()
                [:-1]
            )
            .all(),

        "high_damage_saturation_increases":
            (
                saturation_high[
                    "clip_high_ratio_mean"
                ]
                .to_numpy()
                [1:]
                >=
                saturation_high[
                    "clip_high_ratio_mean"
                ]
                .to_numpy()
                [:-1]
            )
            .all(),

        "high_20pct_median_at_upper_bound":
            np.isclose(
                float(
                    sat_20_high[
                        "prediction_q50_mean"
                    ]
                ),
                0.5,
                atol=1e-9,
            ),
    }

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
            "Figure 5 scientific audit failed."
        )


if __name__ == "__main__":
    main()
