"""
Audit schemas of the frozen primary evidence files used by the JCSHM manuscript.

This script:
- performs NO model training;
- performs NO hyperparameter tuning;
- modifies NO experimental result;
- prints CSV shapes, column names and representative rows;
- creates a machine-readable schema inventory for manuscript data preparation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "manuscript"
    / "jcshm_reconstruction"
    / "tables"
    / "source_data"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SOURCES = {
    # --------------------------------------------------------
    # Table 1 / Figure 2
    # Descriptor provenance
    # --------------------------------------------------------
    "descriptor_audit": (
        "results/sss_fast_revision/"
        "tables/T01_descriptor_audit.csv"
    ),

    # --------------------------------------------------------
    # Table 2 / Figure 3
    # Repeated information × estimator factorial
    # --------------------------------------------------------
    "factorial_mean": (
        "results/sss_fast_revision/"
        "repeated_factorial_78d_92d_ridge_svr/"
        "repeated_factorial_mean_table.csv"
    ),

    "factorial_effects": (
        "results/sss_fast_revision/"
        "repeated_factorial_78d_92d_ridge_svr/"
        "repeated_factorial_effect_summary.csv"
    ),

    "factorial_seed_effects": (
        "results/sss_fast_revision/"
        "repeated_factorial_78d_92d_ridge_svr/"
        "repeated_factorial_seed_effects.csv"
    ),

    # --------------------------------------------------------
    # Figure 4
    # Severity / calibration
    # --------------------------------------------------------
    "ridge_svr_damage_bins": (
        "results/sss_fast_revision/"
        "repeated_split_78d_ridge_svr/"
        "repeated_split_damage_bins.csv"
    ),

    "ridge_svr_metric_summary": (
        "results/sss_fast_revision/"
        "repeated_split_78d_ridge_svr/"
        "repeated_split_metric_summary.csv"
    ),

    "calibration_statistics": (
        "results/sss_fast_revision/"
        "repeated_split_asymmetric_calibration_78d/"
        "calibration_paired_statistics.csv"
    ),

    "calibration_test_results": (
        "results/sss_fast_revision/"
        "repeated_split_asymmetric_calibration_78d/"
        "calibration_paired_test_results.csv"
    ),

    # --------------------------------------------------------
    # Table 3 / Figure 5
    # Controlled matched-noise experiment
    # --------------------------------------------------------
    "noise_level_summary": (
        "results/sss_fast_revision/"
        "matched_noise_robustness_clean_trained/"
        "noise_level_summary.csv"
    ),

    "noise_directional_summary": (
        "results/sss_fast_revision/"
        "matched_noise_failure_mechanism/"
        "severity_directional_summary.csv"
    ),

    "noise_bias_reversal": (
        "results/sss_fast_revision/"
        "matched_noise_failure_mechanism/"
        "high_damage_bias_reversal.csv"
    ),

    "noise_prediction_distribution": (
        "results/sss_fast_revision/"
        "matched_noise_failure_mechanism/"
        "prediction_distribution_summary.csv"
    ),

    "noise_descriptor_shift": (
        "results/sss_fast_revision/"
        "matched_noise_failure_mechanism/"
        "descriptor_shift_summary.csv"
    ),

    # --------------------------------------------------------
    # Table 4 / Figures 6–7
    # Sensor observability
    # --------------------------------------------------------
    "sensor_layout_results": (
        "results/sss_fast_revision/"
        "exhaustive_sensor_layout_svr/"
        "sensor_layout_results.csv"
    ),

    "sensor_placement_spread": (
        "results/sss_fast_revision/"
        "exhaustive_sensor_layout_svr/"
        "sensor_placement_spread.csv"
    ),

    "sensor_count_bootstrap": (
        "results/sss_fast_revision/"
        "sensor_layout_paired_bootstrap_closure/"
        "sensor_count_bootstrap_ci.csv"
    ),

    "sensor_count_contrasts": (
        "results/sss_fast_revision/"
        "sensor_layout_paired_bootstrap_closure/"
        "sensor_count_step_contrasts.csv"
    ),

    "sensor_best_layout_stability": (
        "results/sss_fast_revision/"
        "sensor_layout_paired_bootstrap_closure/"
        "best_layout_bootstrap_stability.csv"
    ),

    "sensor_marginal_edges": (
        "results/sss_fast_revision/"
        "sensor_layout_paired_bootstrap_closure/"
        "marginal_sensor_edge_bootstrap_ci.csv"
    ),

    "sensor_marginal_summary": (
        "results/sss_fast_revision/"
        "sensor_layout_paired_bootstrap_closure/"
        "marginal_sensor_bootstrap_summary.csv"
    ),

    "sensor_pairwise_contrasts": (
        "results/sss_fast_revision/"
        "sensor_layout_paired_bootstrap_closure/"
        "within_count_pairwise_layout_contrasts.csv"
    ),
}


def main() -> None:

    schema_inventory = {}

    print("=" * 100)
    print("JCSHM PRIMARY EVIDENCE SCHEMA AUDIT")
    print("=" * 100)
    print()
    print("Training/tuning performed: False")
    print("Experimental files modified: False")
    print()

    all_found = True

    for source_name, relative_path in SOURCES.items():

        path = (
            PROJECT_ROOT
            / relative_path
        ).resolve()

        print("=" * 100)
        print(source_name)
        print("=" * 100)
        print("File:", relative_path)

        if not path.is_file():

            print("STATUS: MISSING")
            print()

            schema_inventory[source_name] = {
                "file": relative_path,
                "exists": False,
            }

            all_found = False
            continue

        frame = pd.read_csv(
            path
        )

        columns = [
            str(column)
            for column in frame.columns
        ]

        print(
            "Shape:",
            frame.shape,
        )

        print(
            "Columns:"
        )

        for index, column in enumerate(
            columns
        ):

            print(
                f"  [{index:02d}] {column}"
            )

        print()
        print("First 3 rows:")

        if len(frame) == 0:

            print("<EMPTY DATAFRAME>")

        else:

            print(
                frame.head(
                    3
                ).to_string(
                    index=False
                )
            )

        print()

        schema_inventory[source_name] = {
            "file": relative_path,
            "exists": True,
            "n_rows": int(
                frame.shape[0]
            ),
            "n_columns": int(
                frame.shape[1]
            ),
            "columns": columns,
        }

    inventory_path = (
        OUTPUT_DIR
        / "primary_evidence_schema_inventory.json"
    )

    with inventory_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            schema_inventory,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 100)
    print("FINAL SCHEMA CHECK")
    print("=" * 100)

    print(
        "Expected primary source files:",
        len(
            SOURCES
        ),
    )

    print(
        "All source files found:",
        all_found,
    )

    print(
        "Training/tuning performed:",
        False,
    )

    print(
        "Experimental files modified:",
        False,
    )

    print(
        "Schema inventory:",
        inventory_path,
    )

    print()

    if all_found:

        print(
            "CHECK PASSED: primary evidence schemas "
            "are ready for manuscript data preparation."
        )

    else:

        print(
            "CHECK FAILED: one or more source files "
            "are missing."
        )


if __name__ == "__main__":
    main()
