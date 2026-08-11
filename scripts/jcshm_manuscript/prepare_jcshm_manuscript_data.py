"""
Prepare frozen experimental evidence for the JCSHM manuscript.

IMPORTANT
---------
This script performs:

- NO model training;
- NO hyperparameter tuning;
- NO descriptor redesign;
- NO test-driven selection;
- NO modification of experimental source files.

It only transforms already-frozen result CSV files into manuscript-level
table and figure datasets.

All headline numerical claims are checked against frozen source evidence
before manuscript products are accepted.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "sss_fast_revision"
)

MANUSCRIPT_ROOT = (
    PROJECT_ROOT
    / "manuscript"
    / "jcshm_reconstruction"
)

TABLE_ROOT = (
    MANUSCRIPT_ROOT
    / "tables"
)

FIGURE_DATA_ROOT = (
    MANUSCRIPT_ROOT
    / "figures"
    / "data"
)

TABLE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DATA_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Frozen sources
# ============================================================

SOURCES = {
    "descriptor_audit":
        RESULT_ROOT
        / "tables"
        / "T01_descriptor_audit.csv",

    "factorial_mean":
        RESULT_ROOT
        / "repeated_factorial_78d_92d_ridge_svr"
        / "repeated_factorial_mean_table.csv",

    "factorial_effects":
        RESULT_ROOT
        / "repeated_factorial_78d_92d_ridge_svr"
        / "repeated_factorial_effect_summary.csv",

    "ridge_svr_damage_bins":
        RESULT_ROOT
        / "repeated_split_78d_ridge_svr"
        / "repeated_split_damage_bins.csv",

    "calibration_statistics":
        RESULT_ROOT
        / "repeated_split_asymmetric_calibration_78d"
        / "calibration_paired_statistics.csv",

    "noise_level_summary":
        RESULT_ROOT
        / "matched_noise_robustness_clean_trained"
        / "noise_level_summary.csv",

    "noise_condition_metrics":
        RESULT_ROOT
        / "matched_noise_robustness_clean_trained"
        / "condition_metrics.csv",

    "noise_directional_summary":
        RESULT_ROOT
        / "matched_noise_failure_mechanism"
        / "severity_directional_summary.csv",

    "noise_bias_reversal":
        RESULT_ROOT
        / "matched_noise_failure_mechanism"
        / "high_damage_bias_reversal.csv",

    "noise_prediction_distribution":
        RESULT_ROOT
        / "matched_noise_failure_mechanism"
        / "prediction_distribution_summary.csv",

    "noise_descriptor_shift":
        RESULT_ROOT
        / "matched_noise_failure_mechanism"
        / "descriptor_shift_summary.csv",

    "sensor_layout_results":
        RESULT_ROOT
        / "exhaustive_sensor_layout_svr"
        / "sensor_layout_results.csv",

    "sensor_placement_spread":
        RESULT_ROOT
        / "exhaustive_sensor_layout_svr"
        / "sensor_placement_spread.csv",

    "sensor_layout_bootstrap":
        RESULT_ROOT
        / "sensor_layout_paired_bootstrap_closure"
        / "layout_bootstrap_ci.csv",

    "sensor_count_bootstrap":
        RESULT_ROOT
        / "sensor_layout_paired_bootstrap_closure"
        / "sensor_count_bootstrap_ci.csv",

    "sensor_count_contrasts":
        RESULT_ROOT
        / "sensor_layout_paired_bootstrap_closure"
        / "sensor_count_step_contrasts.csv",

    "sensor_best_layout_stability":
        RESULT_ROOT
        / "sensor_layout_paired_bootstrap_closure"
        / "best_layout_bootstrap_stability.csv",

    "sensor_marginal_edges":
        RESULT_ROOT
        / "sensor_layout_paired_bootstrap_closure"
        / "marginal_sensor_edge_bootstrap_ci.csv",
}


# ============================================================
# Utilities
# ============================================================

def read_csv(
    key: str,
    **kwargs,
) -> pd.DataFrame:

    path = SOURCES[key]

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


def numeric_contains(
    frame: pd.DataFrame,
    target: float,
    *,
    atol: float = 5e-9,
    rtol: float = 1e-9,
) -> bool:

    numeric = frame.select_dtypes(
        include=[
            np.number,
        ]
    )

    if numeric.empty:
        return False

    values = numeric.to_numpy(
        dtype=np.float64,
    )

    return bool(
        np.isclose(
            values,
            target,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        ).any()
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
        series.astype(str)
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
        .fillna(False)
        .astype(bool)
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
        "JCSHM MANUSCRIPT DATA PREPARATION"
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
        "Experimental source files modified:",
        False,
    )

    print()

    # --------------------------------------------------------
    # Load frozen evidence.
    # --------------------------------------------------------

    descriptor_audit = read_csv(
        "descriptor_audit"
    )

    factorial_mean = read_csv(
        "factorial_mean"
    )

    factorial_effects = read_csv(
        "factorial_effects"
    )

    ridge_svr_damage_bins = read_csv(
        "ridge_svr_damage_bins"
    )

    calibration_statistics = read_csv(
        "calibration_statistics"
    )

    noise_level_summary = read_csv(
        "noise_level_summary"
    )

    noise_condition_metrics = read_csv(
        "noise_condition_metrics"
    )

    noise_directional_summary = read_csv(
        "noise_directional_summary"
    )

    noise_bias_reversal = read_csv(
        "noise_bias_reversal"
    )

    noise_prediction_distribution = read_csv(
        "noise_prediction_distribution"
    )

    noise_descriptor_shift = read_csv(
        "noise_descriptor_shift"
    )

    sensor_layout_results = read_csv(
        "sensor_layout_results",
        dtype={
            "layout_tag": str,
            "sensor_layout": str,
        },
    )

    sensor_placement_spread = read_csv(
        "sensor_placement_spread",
        dtype={
            "best_layout": str,
            "worst_layout": str,
        },
    )

    sensor_layout_bootstrap = read_csv(
        "sensor_layout_bootstrap",
        dtype={
            "layout_tag": str,
        },
    )

    sensor_count_bootstrap = read_csv(
        "sensor_count_bootstrap"
    )

    sensor_count_contrasts = read_csv(
        "sensor_count_contrasts"
    )

    sensor_best_layout_stability = read_csv(
        "sensor_best_layout_stability",
        dtype={
            "layout_tag": str,
        },
    )

    sensor_marginal_edges = read_csv(
        "sensor_marginal_edges",
        dtype={
            "base_layout": str,
            "augmented_layout": str,
        },
    )

    # ========================================================
    # TABLE 1
    # Descriptor provenance hierarchy.
    #
    # This is a methodological synthesis table.
    # The dimensions and provenance labels are frozen by the
    # descriptor audit/reconstruction programme.
    # ========================================================

    T01 = pd.DataFrame(
        [
            {
                "descriptor_set": "Privileged-information reference",
                "short_label": "92D",
                "dimension": 92,
                "generator_metadata": True,
                "exact_generator_frequency_derived": True,
                "base_input_dependent_descriptors": True,
                "structural_response_descriptors": True,
                "deployment_interpretation":
                    "Simulation-privileged reference; not deployment representation",
                "manuscript_role":
                    "Privileged-information reference",
            },
            {
                "descriptor_set": "Legacy simulation-informed reference",
                "short_label": "86D legacy",
                "dimension": 86,
                "generator_metadata": False,
                "exact_generator_frequency_derived": True,
                "base_input_dependent_descriptors": True,
                "structural_response_descriptors": True,
                "deployment_interpretation":
                    "Legacy reference retaining exact generator-frequency-derived ratios",
                "manuscript_role":
                    "Historical comparison / supplementary support",
            },
            {
                "descriptor_set": "Signal-derived with sensed base input",
                "short_label": "78D",
                "dimension": 78,
                "generator_metadata": False,
                "exact_generator_frequency_derived": False,
                "base_input_dependent_descriptors": True,
                "structural_response_descriptors": True,
                "deployment_interpretation":
                    "Primary signal-derived representation under assumed sensed base input",
                "manuscript_role":
                    "Primary deployment-oriented representation",
            },
            {
                "descriptor_set": "Structural-response-only",
                "short_label": "59D",
                "dimension": 59,
                "generator_metadata": False,
                "exact_generator_frequency_derived": False,
                "base_input_dependent_descriptors": False,
                "structural_response_descriptors": True,
                "deployment_interpretation":
                    "Strict output-only representation without base-input sensing",
                "manuscript_role":
                    "Pressure-test baseline / supplementary support",
            },
        ]
    )

    T01_path = (
        TABLE_ROOT
        / "T01_descriptor_provenance.csv"
    )

    save_csv(
        T01,
        T01_path,
    )

    # Preserve full descriptor audit as traceable source.
    save_csv(
        descriptor_audit,
        TABLE_ROOT
        / "source_data"
        / "T01_full_descriptor_audit.csv",
    )

    # ========================================================
    # TABLE 2 / FIGURE 3
    # Repeated factorial.
    # ========================================================

    T02_performance_path = (
        TABLE_ROOT
        / "T02_factorial_performance.csv"
    )

    T02_effects_path = (
        TABLE_ROOT
        / "T02_factorial_effects.csv"
    )

    save_csv(
        factorial_mean,
        T02_performance_path,
    )

    save_csv(
        factorial_effects,
        T02_effects_path,
    )

    save_csv(
        factorial_mean,
        FIGURE_DATA_ROOT
        / "F03_factorial_performance.csv",
    )

    save_csv(
        factorial_effects,
        FIGURE_DATA_ROOT
        / "F03_factorial_effects.csv",
    )

    # ========================================================
    # FIGURE 4
    # Severity + calibration.
    # ========================================================

    save_csv(
        ridge_svr_damage_bins,
        FIGURE_DATA_ROOT
        / "F04_severity_performance.csv",
    )

    save_csv(
        calibration_statistics,
        FIGURE_DATA_ROOT
        / "F04_calibration_effects.csv",
    )

    # ========================================================
    # TABLE 3 / FIGURE 5
    # Matched noise.
    #
    # Keep source components separate at this stage.
    # Final formatted manuscript table will draw from these.
    # ========================================================

    T03_path = (
        TABLE_ROOT
        / "T03_matched_noise_summary.csv"
    )

    save_csv(
        noise_level_summary,
        T03_path,
    )

    save_csv(
        noise_level_summary,
        FIGURE_DATA_ROOT
        / "F05_noise_performance.csv",
    )

    save_csv(
        noise_bias_reversal,
        FIGURE_DATA_ROOT
        / "F05_bias_reversal.csv",
    )

    save_csv(
        noise_prediction_distribution,
        FIGURE_DATA_ROOT
        / "F05_prediction_saturation.csv",
    )

    save_csv(
        noise_descriptor_shift,
        FIGURE_DATA_ROOT
        / "F05_descriptor_shift.csv",
    )

    # Preserve directional source for future table assembly.
    save_csv(
        noise_directional_summary,
        TABLE_ROOT
        / "source_data"
        / "T03_noise_directional_summary.csv",
    )

    # ========================================================
    # TABLE 4
    # Sensor-count observability summary.
    # ========================================================

    placement_for_merge = (
        sensor_placement_spread.copy()
    )

    T04 = sensor_count_bootstrap.merge(
        placement_for_merge,
        on="sensor_count",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_placement",
        ),
    )

    T04_path = (
        TABLE_ROOT
        / "T04_sensor_observability.csv"
    )

    save_csv(
        T04,
        T04_path,
    )

    # ========================================================
    # FIGURE 6
    # All 15 layouts + bootstrap CI.
    # ========================================================

    F06 = sensor_layout_results.merge(
        sensor_layout_bootstrap,
        on="layout_tag",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_bootstrap",
        ),
    )

    F06_path = (
        FIGURE_DATA_ROOT
        / "F06_sensor_layouts.csv"
    )

    save_csv(
        F06,
        F06_path,
    )

    # ========================================================
    # FIGURE 7
    # Count effects + complete subset lattice.
    # ========================================================

    F07_count_path = (
        FIGURE_DATA_ROOT
        / "F07_sensor_count_effects.csv"
    )

    save_csv(
        sensor_count_contrasts,
        F07_count_path,
    )

    F07_edges_path = (
        FIGURE_DATA_ROOT
        / "F07_marginal_sensor_edges.csv"
    )

    save_csv(
        sensor_marginal_edges,
        F07_edges_path,
    )

    # Best-layout stability retained as source data.
    save_csv(
        sensor_best_layout_stability,
        FIGURE_DATA_ROOT
        / "F07_best_layout_stability.csv",
    )

    # ========================================================
    # NUMERICAL CLAIM ANCHORS
    # ========================================================

    checks = {}

    # P1 — repeated factorial anchors.
    expected_factorial_values = {
        "78D_Ridge_MAE": 0.0448582946,
        "78D_SVR_MAE": 0.0314500006,
        "92D_Ridge_MAE": 0.0395454944,
        "92D_SVR_MAE": 0.0231048479,
    }

    for name, value in (
        expected_factorial_values.items()
    ):

        checks[
            f"factorial_{name}"
        ] = numeric_contains(
            factorial_mean,
            value,
        )

    # P3 — clean controlled anchor.
    checks[
        "controlled_clean_MAE_0.0220471401"
    ] = numeric_contains(
        noise_condition_metrics,
        0.0220471401,
    )

    # P3 — noise-induced high-damage bias reversal anchor.
    checks[
        "noise_20pct_high_bias_0.1285342352"
    ] = (
        numeric_contains(
            noise_bias_reversal,
            0.1285342352,
        )
        or numeric_contains(
            noise_directional_summary,
            0.1285342352,
        )
    )

    # P4/P5 — sensor anchors.
    checks[
        "sensor_1_mean_MAE_0.0733744166"
    ] = numeric_contains(
        sensor_count_bootstrap,
        0.0733744166,
    )

    checks[
        "sensor_4_MAE_0.0220471401"
    ] = numeric_contains(
        sensor_count_bootstrap,
        0.0220471401,
    )

    checks[
        "sensor_layout_count_15"
    ] = (
        len(
            sensor_layout_results
        )
        == 15
    )

    checks[
        "sensor_count_rows_4"
    ] = (
        len(
            sensor_count_bootstrap
        )
        == 4
    )

    checks[
        "sensor_count_contrasts_3"
    ] = (
        len(
            sensor_count_contrasts
        )
        == 3
    )

    checks[
        "marginal_sensor_edges_28"
    ] = (
        len(
            sensor_marginal_edges
        )
        == 28
    )

    # Consecutive count improvements:
    # all paired CI lower bounds must be beneficial.
    required_count_columns = {
        "mae_ci_lower",
        "high_mae_ci_lower",
    }

    if required_count_columns.issubset(
        sensor_count_contrasts.columns
    ):

        checks[
            "all_count_MAE_step_CI_positive"
        ] = bool(
            (
                sensor_count_contrasts[
                    "mae_ci_lower"
                ]
                > 0.0
            ).all()
        )

        checks[
            "all_count_high_MAE_step_CI_positive"
        ] = bool(
            (
                sensor_count_contrasts[
                    "high_mae_ci_lower"
                ]
                > 0.0
            ).all()
        )

    else:

        checks[
            "all_count_MAE_step_CI_positive"
        ] = False

        checks[
            "all_count_high_MAE_step_CI_positive"
        ] = False

    # Complete 28-edge lattice.
    required_edge_columns = {
        "mae_ci_entirely_positive",
        "high_mae_ci_entirely_positive",
        "mae_improvement_point",
        "high_mae_improvement_point",
    }

    if required_edge_columns.issubset(
        sensor_marginal_edges.columns
    ):

        mae_ci_positive = bool_series(
            sensor_marginal_edges[
                "mae_ci_entirely_positive"
            ]
        )

        high_ci_positive = bool_series(
            sensor_marginal_edges[
                "high_mae_ci_entirely_positive"
            ]
        )

        checks[
            "28_of_28_MAE_point_positive"
        ] = bool(
            (
                sensor_marginal_edges[
                    "mae_improvement_point"
                ]
                > 0.0
            ).all()
        )

        checks[
            "28_of_28_MAE_CI_positive"
        ] = bool(
            mae_ci_positive.all()
        )

        checks[
            "28_of_28_high_MAE_point_positive"
        ] = bool(
            (
                sensor_marginal_edges[
                    "high_mae_improvement_point"
                ]
                > 0.0
            ).all()
        )

        checks[
            "28_of_28_high_MAE_CI_positive"
        ] = bool(
            high_ci_positive.all()
        )

    else:

        checks[
            "28_of_28_MAE_point_positive"
        ] = False

        checks[
            "28_of_28_MAE_CI_positive"
        ] = False

        checks[
            "28_of_28_high_MAE_point_positive"
        ] = False

        checks[
            "28_of_28_high_MAE_CI_positive"
        ] = False

    overall_passed = bool(
        all(
            checks.values()
        )
    )

    # ========================================================
    # Output inventory + SHA256 manifest
    # ========================================================

    generated_files = sorted(
        [
            path
            for path in (
                list(
                    TABLE_ROOT.glob(
                        "T*.csv"
                    )
                )
                + list(
                    FIGURE_DATA_ROOT.glob(
                        "F*.csv"
                    )
                )
            )
            if path.is_file()
        ],
        key=lambda value: str(
            value
        ),
    )

    manifest_rows = []

    for path in generated_files:

        frame = pd.read_csv(
            path
        )

        manifest_rows.append(
            {
                "file": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "rows": int(
                    frame.shape[0]
                ),
                "columns": int(
                    frame.shape[1]
                ),
                "sha256": sha256(
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
        / "06_MANUSCRIPT_DATA_MANIFEST.csv"
    )

    save_csv(
        manifest,
        manifest_path,
    )

    # ========================================================
    # Preparation report
    # ========================================================

    report = {
        "experiment_status": (
            "frozen"
        ),
        "training_performed": (
            False
        ),
        "hyperparameter_tuning_performed": (
            False
        ),
        "experimental_source_files_modified": (
            False
        ),
        "source_file_count": (
            len(
                SOURCES
            )
        ),
        "generated_manuscript_file_count": (
            len(
                generated_files
            )
        ),
        "numerical_anchor_checks": (
            checks
        ),
        "overall_passed": (
            overall_passed
        ),
        "manifest": str(
            manifest_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    report_path = (
        MANUSCRIPT_ROOT
        / "revision_notes"
        / "06_MANUSCRIPT_DATA_PREPARATION_REPORT.json"
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
    # Console summary
    # ========================================================

    print(
        "=" * 100
    )

    print(
        "NUMERICAL CLAIM ANCHOR CHECK"
    )

    print(
        "=" * 100
    )

    for name, passed in (
        checks.items()
    ):

        print(
            f"{name}:",
            passed,
        )

    print()

    print(
        "=" * 100
    )

    print(
        "GENERATED MANUSCRIPT DATA"
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
        "=" * 100
    )

    print(
        "FINAL MANUSCRIPT DATA CHECK"
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
        "Experimental source files modified:",
        False,
    )

    print(
        "Generated manuscript files:",
        len(
            generated_files
        ),
    )

    print(
        "All numerical anchors passed:",
        overall_passed,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    if overall_passed:

        print(
            "CHECK PASSED: frozen experimental evidence "
            "has been converted into JCSHM manuscript data."
        )

        print(
            "Figure/table production may proceed."
        )

    else:

        print(
            "CHECK FAILED: do not generate manuscript "
            "figures or formatted tables until the "
            "failed numerical anchor is resolved."
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


if __name__ == "__main__":
    main()
