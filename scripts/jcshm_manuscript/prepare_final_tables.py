"""
Prepare publication-oriented Tables 1-4 for the JCSHM manuscript.

IMPORTANT
---------
This script performs:

- NO model training;
- NO hyperparameter tuning;
- NO statistical re-testing;
- NO model selection;
- NO sensor-layout optimization;
- NO modification of experimental source files.

It only reformats frozen manuscript-level evidence into
publication-oriented tables.

Raw frozen source files remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANUSCRIPT_ROOT = (
    PROJECT_ROOT
    / "manuscript"
    / "jcshm_reconstruction"
)

TABLE_SOURCE_ROOT = (
    MANUSCRIPT_ROOT
    / "tables"
)

FIGURE_DATA_ROOT = (
    MANUSCRIPT_ROOT
    / "figures"
    / "data"
)

FINAL_TABLE_ROOT = (
    TABLE_SOURCE_ROOT
    / "final"
)

SUPPLEMENTARY_TABLE_ROOT = (
    MANUSCRIPT_ROOT
    / "supplementary"
    / "tables"
)

FINAL_TABLE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SUPPLEMENTARY_TABLE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Frozen source paths
# ============================================================

SOURCE_PATHS = {
    "T01":
        TABLE_SOURCE_ROOT
        / "T01_descriptor_provenance.csv",

    "T02_performance":
        TABLE_SOURCE_ROOT
        / "T02_factorial_performance.csv",

    "T02_effects":
        TABLE_SOURCE_ROOT
        / "T02_factorial_effects.csv",

    "T03_noise":
        TABLE_SOURCE_ROOT
        / "T03_matched_noise_summary.csv",

    "T04_count":
        TABLE_SOURCE_ROOT
        / "T04_sensor_observability.csv",

    "F05_bias":
        FIGURE_DATA_ROOT
        / "F05_bias_reversal.csv",

    "F05_saturation":
        FIGURE_DATA_ROOT
        / "F05_prediction_saturation.csv",

    "F06_layouts":
        FIGURE_DATA_ROOT
        / "F06_sensor_layouts.csv",
}


# ============================================================
# Utilities
# ============================================================

def read_csv(
    key: str,
    **kwargs,
) -> pd.DataFrame:

    path = SOURCE_PATHS[key]

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen source: {path}"
        )

    return pd.read_csv(
        path,
        **kwargs,
    )


def save_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )


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


def yes_no(
    value,
) -> str:

    if isinstance(
        value,
        str,
    ):

        return (
            "Yes"
            if value.strip().lower()
            in {
                "true",
                "1",
                "yes",
            }
            else "No"
        )

    return (
        "Yes"
        if bool(value)
        else "No"
    )


def format_mean_sd(
    mean: float,
    sd: float,
    n: int,
    decimals: int = 4,
) -> str:

    if int(n) <= 1:

        return (
            f"{float(mean):.{decimals}f}"
        )

    return (
        f"{float(mean):.{decimals}f}"
        f" ± "
        f"{float(sd):.{decimals}f}"
    )


def format_point_ci(
    point: float,
    lower: float,
    upper: float,
    decimals: int = 4,
) -> str:

    return (
        f"{float(point):.{decimals}f} "
        f"[{float(lower):.{decimals}f}, "
        f"{float(upper):.{decimals}f}]"
    )


def sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 100
    )

    print(
        "JCSHM FINAL TABLE PREPARATION"
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

    print()

    # ========================================================
    # Load frozen evidence
    # ========================================================

    provenance = read_csv(
        "T01"
    )

    factorial_performance = read_csv(
        "T02_performance"
    )

    factorial_effects = read_csv(
        "T02_effects"
    )

    noise = read_csv(
        "T03_noise"
    )

    count_summary = read_csv(
        "T04_count",
        dtype={
            "best_layout": str,
            "worst_layout": str,
        },
    )

    bias = read_csv(
        "F05_bias"
    )

    saturation = read_csv(
        "F05_saturation"
    )

    layouts = read_csv(
        "F06_layouts",
        dtype={
            "layout_tag": str,
            "sensor_layout": str,
        },
    )

    # ========================================================
    # TABLE 1
    # Information provenance hierarchy
    # ========================================================

    role_map = {
        "92D":
            "Privileged-information reference",

        "86D legacy":
            "Historical simulation-informed reference",

        "78D":
            "Primary deployment-oriented representation",

        "59D":
            "Response-only pressure-test baseline",
    }

    T01_rows = []

    for _, row in (
        provenance.iterrows()
    ):

        T01_rows.append(
            {
                "Representation":
                    row[
                        "short_label"
                    ],

                "Dimension":
                    int(
                        row[
                            "dimension"
                        ]
                    ),

                "Generator metadata":
                    yes_no(
                        row[
                            "generator_metadata"
                        ]
                    ),

                "Exact generator-frequency-derived":
                    yes_no(
                        row[
                            "exact_generator_frequency_derived"
                        ]
                    ),

                "Base-input dependent":
                    yes_no(
                        row[
                            "base_input_dependent_descriptors"
                        ]
                    ),

                "Structural-response derived":
                    yes_no(
                        row[
                            "structural_response_descriptors"
                        ]
                    ),

                "Manuscript role":
                    role_map[
                        row[
                            "short_label"
                        ]
                    ],
            }
        )

    T01 = pd.DataFrame(
        T01_rows
    )

    T01_path = (
        FINAL_TABLE_ROOT
        / "T01_information_provenance.csv"
    )

    save_csv(
        T01,
        T01_path,
    )

    # ========================================================
    # TABLE 2A
    # Repeated factorial performance
    # ========================================================

    performance_index = (
        factorial_performance
        .set_index(
            "metric"
        )
    )

    conditions = [
        (
            "78D",
            "Ridge",
            "78D_Ridge",
        ),
        (
            "78D",
            "RBF-SVR",
            "78D_SVR",
        ),
        (
            "92D",
            "Ridge",
            "92D_Ridge",
        ),
        (
            "92D",
            "RBF-SVR",
            "92D_SVR",
        ),
    ]

    T02A_rows = []

    for (
        representation,
        estimator,
        source_column,
    ) in conditions:

        T02A_rows.append(
            {
                "Representation":
                    representation,

                "Estimator":
                    estimator,

                "Overall MAE":
                    round(
                        float(
                            performance_index.loc[
                                "test_mae",
                                source_column,
                            ]
                        ),
                        4,
                    ),

                "High-damage MAE":
                    round(
                        float(
                            performance_index.loc[
                                "high_damage_mae",
                                source_column,
                            ]
                        ),
                        4,
                    ),

                "High-damage |bias|":
                    round(
                        float(
                            performance_index.loc[
                                "high_damage_abs_bias",
                                source_column,
                            ]
                        ),
                        4,
                    ),

                "High-damage underestimation":
                    round(
                        float(
                            performance_index.loc[
                                "high_damage_underestimation_ratio",
                                source_column,
                            ]
                        ),
                        3,
                    ),
            }
        )

    T02A = pd.DataFrame(
        T02A_rows
    )

    T02A_path = (
        FINAL_TABLE_ROOT
        / "T02A_information_estimator_performance.csv"
    )

    save_csv(
        T02A,
        T02A_path,
    )

    # ========================================================
    # TABLE 2B
    # Overall-MAE paired factorial effects
    #
    # Source effects use:
    # comparison - reference
    #
    # Negative values therefore indicate reduced MAE.
    #
    # For publication presentation, signs are reversed so that:
    # positive = MAE reduction.
    # ========================================================

    effect_labels = {
        "information_effect_under_ridge":
            "Information effect | Ridge",

        "information_effect_under_svr":
            "Information effect | RBF-SVR",

        "model_effect_on_78d":
            "Estimator effect | 78D",

        "model_effect_on_92d":
            "Estimator effect | 92D",

        "interaction_difference_in_differences":
            "Interaction",
    }

    effect_order = list(
        effect_labels.keys()
    )

    overall_effects = (
        factorial_effects.loc[
            factorial_effects[
                "metric"
            ]
            == "test_mae"
        ]
        .copy()
    )

    T02B_rows = []

    for effect_name in (
        effect_order
    ):

        row = one_row(
            overall_effects,
            effect=effect_name,
        )

        reduction = -float(
            row[
                "mean_effect"
            ]
        )

        ci_lower = -float(
            row[
                "ci95_high"
            ]
        )

        ci_upper = -float(
            row[
                "ci95_low"
            ]
        )

        relative = (
            ""
            if pd.isna(
                row[
                    "mean_relative_improvement_percent"
                ]
            )
            else
            f'{float(row["mean_relative_improvement_percent"]):.1f}'
        )

        T02B_rows.append(
            {
                "Effect":
                    effect_labels[
                        effect_name
                    ],

                "Overall MAE reduction":
                    round(
                        reduction,
                        6,
                    ),

                "Paired 95% CI":
                    (
                        f"[{ci_lower:.6f}, "
                        f"{ci_upper:.6f}]"
                    ),

                "Relative improvement (%)":
                    relative,

                "Beneficial direction / 10 splits":
                    (
                        f'{int(row["negative_seeds"])}/10'
                    ),
            }
        )

    T02B = pd.DataFrame(
        T02B_rows
    )

    T02B_path = (
        FINAL_TABLE_ROOT
        / "T02B_factorial_effects_overall_mae.csv"
    )

    save_csv(
        T02B,
        T02B_path,
    )

    # ========================================================
    # TABLE 3
    # Controlled measurement-noise failure
    #
    # Main table uses STANDARD clean-trained RBF-SVR.
    #
    # Calibration is treated separately in Figure 4/5 and
    # should not obscure the primary noise-failure mechanism.
    # ========================================================

    standard_noise = (
        noise.loc[
            noise[
                "method"
            ]
            == "standard_svr"
        ]
        .sort_values(
            "noise_percent"
        )
    )

    standard_bias = (
        bias.loc[
            bias[
                "method"
            ]
            == "standard_svr"
        ]
    )

    high_saturation = (
        saturation.loc[
            (
                saturation[
                    "method"
                ]
                == "standard_svr"
            )
            & (
                saturation[
                    "severity"
                ]
                == "high"
            )
        ]
    )

    T03_rows = []

    for _, row in (
        standard_noise.iterrows()
    ):

        noise_percent = int(
            row[
                "noise_percent"
            ]
        )

        bias_row = one_row(
            standard_bias,
            noise_percent=noise_percent,
        )

        saturation_row = one_row(
            high_saturation,
            noise_percent=noise_percent,
        )

        n_replicates = int(
            row[
                "n_replicates"
            ]
        )

        T03_rows.append(
            {
                "Response noise (%)":
                    noise_percent,

                "Noise realizations":
                    n_replicates,

                "Overall MAE":
                    format_mean_sd(
                        row[
                            "mae_mean"
                        ],
                        row[
                            "mae_std"
                        ],
                        n_replicates,
                        decimals=4,
                    ),

                "High-damage MAE":
                    format_mean_sd(
                        row[
                            "high_mae_mean"
                        ],
                        row[
                            "high_mae_std"
                        ],
                        n_replicates,
                        decimals=4,
                    ),

                "High-damage signed bias":
                    f'{float(bias_row["high_signed_bias"]):+.4f}',

                "High-damage underestimation":
                    format_mean_sd(
                        row[
                            "high_underestimation_mean"
                        ],
                        row[
                            "high_underestimation_std"
                        ],
                        n_replicates,
                        decimals=3,
                    ),

                "High-damage upper clipping":
                    format_mean_sd(
                        saturation_row[
                            "clip_high_ratio_mean"
                        ],
                        saturation_row[
                            "clip_high_ratio_std"
                        ],
                        n_replicates,
                        decimals=3,
                    ),
            }
        )

    T03 = pd.DataFrame(
        T03_rows
    )

    T03_path = (
        FINAL_TABLE_ROOT
        / "T03_measurement_noise_failure.csv"
    )

    save_csv(
        T03,
        T03_path,
    )

    # ========================================================
    # TABLE 4
    # Sensor-count observability summary
    #
    # Use count-level bootstrap summary, then add the
    # high-damage-optimal layout from the exhaustive frozen
    # 15-layout results.
    # ========================================================

    T04_rows = []

    for _, row in (
        count_summary
        .sort_values(
            "sensor_count"
        )
        .iterrows()
    ):

        count = int(
            row[
                "sensor_count"
            ]
        )

        subset = (
            layouts.loc[
                layouts[
                    "sensor_count"
                ]
                == count
            ]
        )

        high_best_index = (
            subset[
                "test_high_mae"
            ]
            .idxmin()
        )

        best_high_layout = str(
            subset.loc[
                high_best_index,
                "layout_tag",
            ]
        )

        T04_rows.append(
            {
                "Structural sensors":
                    count,

                "Layouts":
                    int(
                        row[
                            "n_layouts"
                        ]
                    ),

                "Mean overall MAE [95% CI]":
                    format_point_ci(
                        row[
                            "mae_point"
                        ],
                        row[
                            "mae_ci_lower"
                        ],
                        row[
                            "mae_ci_upper"
                        ],
                        decimals=4,
                    ),

                "Mean high-damage MAE [95% CI]":
                    format_point_ci(
                        row[
                            "high_mae_point"
                        ],
                        row[
                            "high_mae_ci_lower"
                        ],
                        row[
                            "high_mae_ci_upper"
                        ],
                        decimals=4,
                    ),

                "Best overall-MAE layout":
                    str(
                        row[
                            "best_layout"
                        ]
                    ),

                "Best high-damage-MAE layout":
                    best_high_layout,

                "Worst overall-MAE layout":
                    str(
                        row[
                            "worst_layout"
                        ]
                    ),

                "Overall placement spread (%)":
                    round(
                        float(
                            row[
                                "relative_placement_spread_percent"
                            ]
                        ),
                        1,
                    ),
            }
        )

    T04 = pd.DataFrame(
        T04_rows
    )

    T04_path = (
        FINAL_TABLE_ROOT
        / "T04_sensor_observability_summary.csv"
    )

    save_csv(
        T04,
        T04_path,
    )

    # ========================================================
    # SUPPLEMENTARY TABLE S01
    # Complete 15-layout results
    # ========================================================

    best_overall_layouts = {}

    best_high_layouts = {}

    for count, subset in (
        layouts.groupby(
            "sensor_count"
        )
    ):

        best_overall_layouts[
            int(
                count
            )
        ] = str(
            subset.loc[
                subset[
                    "test_mae"
                ].idxmin(),
                "layout_tag",
            ]
        )

        best_high_layouts[
            int(
                count
            )
        ] = str(
            subset.loc[
                subset[
                    "test_high_mae"
                ].idxmin(),
                "layout_tag",
            ]
        )

    layout_order = [
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

    order_map = {
        layout:
            index
        for index, layout
        in enumerate(
            layout_order
        )
    }

    layouts_ordered = (
        layouts.assign(
            _order=layouts[
                "layout_tag"
            ].map(
                order_map
            )
        )
        .sort_values(
            "_order"
        )
    )

    S01_rows = []

    for _, row in (
        layouts_ordered.iterrows()
    ):

        layout = str(
            row[
                "layout_tag"
            ]
        )

        count = int(
            row[
                "sensor_count"
            ]
        )

        S01_rows.append(
            {
                "Layout":
                    layout,

                "Sensors":
                    count,

                "Observable descriptors":
                    int(
                        row[
                            "feature_count"
                        ]
                    ),

                "Overall MAE [95% CI]":
                    format_point_ci(
                        row[
                            "mae_point"
                        ],
                        row[
                            "mae_ci_lower"
                        ],
                        row[
                            "mae_ci_upper"
                        ],
                        decimals=4,
                    ),

                "High-damage MAE [95% CI]":
                    format_point_ci(
                        row[
                            "high_mae_point"
                        ],
                        row[
                            "high_mae_ci_lower"
                        ],
                        row[
                            "high_mae_ci_upper"
                        ],
                        decimals=4,
                    ),

                "High-damage underestimation [95% CI]":
                    format_point_ci(
                        row[
                            "high_underestimation_point"
                        ],
                        row[
                            "high_underestimation_ci_lower"
                        ],
                        row[
                            "high_underestimation_ci_upper"
                        ],
                        decimals=3,
                    ),

                "Best overall within sensor count":
                    (
                        "Yes"
                        if layout
                        == best_overall_layouts[
                            count
                        ]
                        else "No"
                    ),

                "Best high-damage within sensor count":
                    (
                        "Yes"
                        if layout
                        == best_high_layouts[
                            count
                        ]
                        else "No"
                    ),
            }
        )

    S01 = pd.DataFrame(
        S01_rows
    )

    S01_path = (
        SUPPLEMENTARY_TABLE_ROOT
        / "S01_complete_sensor_layout_results.csv"
    )

    save_csv(
        S01,
        S01_path,
    )

    # ========================================================
    # Numerical anchor audit
    # ========================================================

    checks = {}

    # Table 1
    checks[
        "T01_dimensions_92_86_78_59"
    ] = (
        provenance[
            "dimension"
        ]
        .astype(int)
        .tolist()
        == [
            92,
            86,
            78,
            59,
        ]
    )

    # Table 2
    assert_close(
        float(
            performance_index.loc[
                "test_mae",
                "78D_SVR",
            ]
        ),
        0.0314500006,
        "78D SVR repeated MAE",
    )

    assert_close(
        float(
            performance_index.loc[
                "test_mae",
                "92D_SVR",
            ]
        ),
        0.0231048479,
        "92D SVR repeated MAE",
    )

    interaction_source = one_row(
        overall_effects,
        effect="interaction_difference_in_differences",
    )

    assert_close(
        float(
            interaction_source[
                "mean_effect"
            ]
        ),
        -0.0030323525,
        "overall-MAE interaction",
    )

    checks[
        "T02_rows_4_plus_5"
    ] = (
        len(
            T02A
        )
        == 4
        and len(
            T02B
        )
        == 5
    )

    # Table 3
    noise20 = one_row(
        standard_noise,
        noise_percent=20,
    )

    bias20 = one_row(
        standard_bias,
        noise_percent=20,
    )

    saturation20 = one_row(
        high_saturation,
        noise_percent=20,
    )

    assert_close(
        float(
            noise20[
                "mae_mean"
            ]
        ),
        0.2838931183,
        "20% overall MAE",
    )

    assert_close(
        float(
            noise20[
                "high_mae_mean"
            ]
        ),
        0.2069806408,
        "20% high-damage MAE",
    )

    assert_close(
        float(
            bias20[
                "high_signed_bias"
            ]
        ),
        0.1285342352,
        "20% high signed bias",
    )

    assert_close(
        float(
            saturation20[
                "clip_high_ratio_mean"
            ]
        ),
        0.6408759124,
        "20% high upper clipping",
    )

    checks[
        "T03_four_noise_levels"
    ] = (
        len(
            T03
        )
        == 4
    )

    # Table 4
    count1 = one_row(
        count_summary,
        sensor_count=1,
    )

    count4 = one_row(
        count_summary,
        sensor_count=4,
    )

    assert_close(
        float(
            count1[
                "mae_point"
            ]
        ),
        0.0733744166,
        "1-sensor mean MAE",
    )

    assert_close(
        float(
            count4[
                "mae_point"
            ]
        ),
        0.0220471401,
        "4-sensor MAE",
    )

    checks[
        "T04_four_sensor_counts"
    ] = (
        len(
            T04
        )
        == 4
    )

    checks[
        "T04_two_sensor_objective_dependence"
    ] = (
        T04.loc[
            T04[
                "Structural sensors"
            ]
            == 2,
            "Best overall-MAE layout",
        ].iloc[
            0
        ]
        == "12"
        and
        T04.loc[
            T04[
                "Structural sensors"
            ]
            == 2,
            "Best high-damage-MAE layout",
        ].iloc[
            0
        ]
        == "13"
    )

    checks[
        "S01_all_15_layouts"
    ] = (
        len(
            S01
        )
        == 15
    )

    checks[
        "S01_full_layout_78_descriptors"
    ] = (
        int(
            S01.loc[
                S01[
                    "Layout"
                ]
                == "1234",
                "Observable descriptors",
            ].iloc[
                0
            ]
        )
        == 78
    )

    overall_passed = bool(
        all(
            checks.values()
        )
    )

    # ========================================================
    # Manifest
    # ========================================================

    output_paths = [
        T01_path,
        T02A_path,
        T02B_path,
        T03_path,
        T04_path,
        S01_path,
    ]

    manifest_rows = []

    for path in (
        output_paths
    ):

        frame = pd.read_csv(
            path
        )

        manifest_rows.append(
            {
                "file":
                    str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "rows":
                    int(
                        frame.shape[
                            0
                        ]
                    ),

                "columns":
                    int(
                        frame.shape[
                            1
                        ]
                    ),

                "sha256":
                    sha256(
                        path
                    ),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        MANUSCRIPT_ROOT
        / "revision_notes"
        / "20_FINAL_TABLE_MANIFEST.csv"
    )

    save_csv(
        manifest,
        manifest_path,
    )

    report = {
        "training_performed":
            False,

        "hyperparameter_tuning_performed":
            False,

        "statistical_retesting_performed":
            False,

        "experimental_sources_modified":
            False,

        "checks":
            checks,

        "overall_passed":
            overall_passed,

        "output_files":
            [
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                )
                for path in
                output_paths
            ],
    }

    report_path = (
        MANUSCRIPT_ROOT
        / "revision_notes"
        / "20_FINAL_TABLE_REPORT.json"
    )

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

    # ========================================================
    # Console output
    # ========================================================

    print(
        "=" * 100
    )

    print(
        "FINAL TABLE AUDIT"
    )

    print(
        "=" * 100
    )

    for name, passed in (
        checks.items()
    ):

        print(
            f"{name}:",
            bool(
                passed
            ),
        )

    print()

    print(
        "=" * 100
    )

    print(
        "GENERATED TABLES"
    )

    print(
        "=" * 100
    )

    print(
        manifest[
            [
                "file",
                "rows",
                "columns",
            ]
        ].to_string(
            index=False
        )
    )

    print()

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
        "Experimental sources modified:",
        False,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Report:",
        report_path,
    )

    if not overall_passed:

        raise AssertionError(
            "Final JCSHM table preparation failed."
        )


if __name__ == "__main__":
    main()
