"""
Build exhaustive dependency-aware structural sensor-layout descriptor datasets.

Purpose
-------
Construct all 15 non-empty subsets of the four structural acceleration
sensors using the canonical clean 78D signal-derived-with-ground
descriptor representation.

Important
---------
This script DOES NOT zero-mask unavailable sensor channels.

Instead, each descriptor is retained only when all structural sensors
required by its original physical definition are present.

The ground/base-input signal is assumed available for every layout.

No descriptor is redefined under sensor sparsity.

Examples
--------
story_3_rms
    requires {3}

story_1_relative_to_lower_rms
    requires {1}; lower reference is ground

story_3_relative_to_lower_rms
    requires {2, 3}

story_3_to_story_2_rms_ratio
    requires {2, 3}

story_2_rms_spatial_fraction
    requires {1, 2, 3, 4}

story_3_ground_correlation
    requires {3}

Outputs
-------
15 layout-specific NPZ files
feature dependency manifest
layout manifest
build report
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


STRUCTURAL_SENSORS = (
    1,
    2,
    3,
    4,
)

EXPECTED_FULL_DIM = 78

EXPECTED_LAYOUT_DIMENSIONS = {
    (1,): 19,
    (2,): 17,
    (3,): 17,
    (4,): 17,

    (1, 2): 36,
    (1, 3): 31,
    (1, 4): 31,
    (2, 3): 34,
    (2, 4): 29,
    (3, 4): 34,

    (1, 2, 3): 53,
    (1, 2, 4): 48,
    (1, 3, 4): 48,
    (2, 3, 4): 51,

    (1, 2, 3, 4): 78,
}


def all_layouts() -> list[tuple[int, ...]]:

    layouts = []

    for count in range(
        1,
        len(STRUCTURAL_SENSORS) + 1,
    ):

        layouts.extend(
            combinations(
                STRUCTURAL_SENSORS,
                count,
            )
        )

    return [
        tuple(layout)
        for layout in layouts
    ]


def required_structural_sensors(
    feature_name: str,
) -> frozenset[int]:
    """
    Return structural sensors required by the descriptor's
    ORIGINAL physical definition.

    Ground/base-input availability is not included here because
    ground is assumed available in every sensor-layout experiment.
    """

    # --------------------------------------------------
    # Ground-only descriptors.
    # --------------------------------------------------

    ground_only = {
        "ground_max_abs",
        "ground_rms",
        "ground_dominant_frequency",
        "ground_spectral_centroid",
        "ground_band_energy",
    }

    if feature_name in ground_only:

        return frozenset()

    # --------------------------------------------------
    # Spatial fractions.
    #
    # Denominator uses responses from all four stories.
    # Therefore these descriptors are available only in
    # the full four-sensor layout.
    # --------------------------------------------------

    match = re.fullmatch(
        r"story_(\d+)_(?:max_abs|rms)_spatial_fraction",
        feature_name,
    )

    if match:

        return frozenset(
            STRUCTURAL_SENSORS
        )

    # --------------------------------------------------
    # Relative-to-lower descriptors.
    #
    # Story 1 uses ground as lower reference.
    # Stories 2-4 require their immediately lower story.
    # --------------------------------------------------

    match = re.fullmatch(
        r"story_(\d+)_relative_to_lower_(?:max_abs|rms)",
        feature_name,
    )

    if match:

        upper = int(
            match.group(1)
        )

        if upper == 1:

            return frozenset(
                {
                    1,
                }
            )

        return frozenset(
            {
                upper - 1,
                upper,
            }
        )

    # --------------------------------------------------
    # Adjacent-story amplitude/RMS ratios.
    # --------------------------------------------------

    match = re.fullmatch(
        r"story_(\d+)_to_story_(\d+)_(?:max_abs|rms)_ratio",
        feature_name,
    )

    if match:

        upper = int(
            match.group(1)
        )

        lower = int(
            match.group(2)
        )

        return frozenset(
            {
                upper,
                lower,
            }
        )

    # --------------------------------------------------
    # Adjacent-story correlations.
    # --------------------------------------------------

    match = re.fullmatch(
        r"story_(\d+)_story_(\d+)_correlation",
        feature_name,
    )

    if match:

        upper = int(
            match.group(1)
        )

        lower = int(
            match.group(2)
        )

        return frozenset(
            {
                upper,
                lower,
            }
        )

    # --------------------------------------------------
    # Story-ground correlation.
    # Ground is globally available.
    # --------------------------------------------------

    match = re.fullmatch(
        r"story_(\d+)_ground_correlation",
        feature_name,
    )

    if match:

        story = int(
            match.group(1)
        )

        return frozenset(
            {
                story,
            }
        )

    # --------------------------------------------------
    # Every remaining canonical story descriptor depends
    # only on that story response, with ground optionally
    # supplied globally.
    #
    # Includes:
    # basic statistics,
    # floor-ground amplification,
    # dominant frequency,
    # spectral centroid,
    # band energy.
    # --------------------------------------------------

    match = re.match(
        r"story_(\d+)_",
        feature_name,
    )

    if match:

        story = int(
            match.group(1)
        )

        if story not in STRUCTURAL_SENSORS:

            raise RuntimeError(
                f"Invalid story ID in feature: "
                f"{feature_name}"
            )

        return frozenset(
            {
                story,
            }
        )

    raise RuntimeError(
        "Could not determine sensor dependency "
        f"for feature: {feature_name}"
    )


def dependency_string(
    required: frozenset[int],
) -> str:

    if not required:
        return "ground_only"

    return ",".join(
        str(sensor)
        for sensor in sorted(
            required
        )
    )


def layout_tag(
    layout: tuple[int, ...],
) -> str:

    return "".join(
        str(sensor)
        for sensor in layout
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean-78d",
        type=Path,
        default=Path(
            "data_processed/"
            "sss_fast_revision/"
            "matched_noise_78d/"
            "matched_noise_000_rep_0.npz"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data_processed/"
            "sss_fast_revision/"
            "sensor_layouts_dependency_aware"
        ),
    )

    args = parser.parse_args()

    clean_path = (
        args.clean_78d
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if not clean_path.is_file():

        raise FileNotFoundError(
            clean_path
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # Load canonical clean 78D dataset.
    # ==================================================

    with np.load(
        clean_path,
        allow_pickle=False,
    ) as data:

        clean = {
            key: np.asarray(
                data[key]
            )
            for key in data.files
        }

    required_keys = {
        "F_78_raw",
        "feature_names_78",
        "y_damage",
        "case_id",
        "train_idx",
        "val_idx",
        "test_idx",
        "dt",
    }

    missing = sorted(
        required_keys
        - set(
            clean.keys()
        )
    )

    if missing:

        raise RuntimeError(
            f"Clean dataset missing keys: "
            f"{missing}"
        )

    F_full = np.asarray(
        clean[
            "F_78_raw"
        ],
        dtype=np.float64,
    )

    feature_names = [
        str(value)
        for value
        in clean[
            "feature_names_78"
        ].tolist()
    ]

    if (
        F_full.shape
        != (
            3000,
            EXPECTED_FULL_DIM,
        )
    ):

        raise RuntimeError(
            "Unexpected canonical clean "
            f"78D shape: {F_full.shape}"
        )

    if (
        len(
            feature_names
        )
        != EXPECTED_FULL_DIM
    ):

        raise RuntimeError(
            "Expected exactly 78 "
            "canonical feature names."
        )

    if (
        len(
            set(
                feature_names
            )
        )
        != EXPECTED_FULL_DIM
    ):

        raise RuntimeError(
            "Duplicate canonical feature names."
        )

    # ==================================================
    # Build explicit dependency manifest.
    # ==================================================

    dependencies = []

    dependency_rows = []

    for index, name in enumerate(
        feature_names
    ):

        required = (
            required_structural_sensors(
                name
            )
        )

        dependencies.append(
            required
        )

        dependency_rows.append(
            {
                "feature_index_78d": (
                    index
                ),
                "feature_name": (
                    name
                ),
                "required_structural_sensors": (
                    dependency_string(
                        required
                    )
                ),
                "required_sensor_count": (
                    len(
                        required
                    )
                ),
                "requires_story_1": (
                    1 in required
                ),
                "requires_story_2": (
                    2 in required
                ),
                "requires_story_3": (
                    3 in required
                ),
                "requires_story_4": (
                    4 in required
                ),
            }
        )

    dependency_frame = (
        pd.DataFrame(
            dependency_rows
        )
    )

    dependency_path = (
        output_root
        / "sensor_feature_dependency_manifest.csv"
    )

    dependency_frame.to_csv(
        dependency_path,
        index=False,
    )

    # --------------------------------------------------
    # Dependency-level sanity checks.
    # --------------------------------------------------

    ground_only_count = int(
        sum(
            len(required) == 0
            for required
            in dependencies
        )
    )

    full_dependency_count = int(
        sum(
            required
            == frozenset(
                STRUCTURAL_SENSORS
            )
            for required
            in dependencies
        )
    )

    if (
        ground_only_count
        != 5
    ):

        raise RuntimeError(
            "Expected exactly 5 "
            "ground-only descriptors, "
            f"got {ground_only_count}."
        )

    if (
        full_dependency_count
        != 8
    ):

        raise RuntimeError(
            "Expected exactly 8 "
            "all-story spatial-fraction "
            "descriptors, "
            f"got {full_dependency_count}."
        )

    print(
        "=" * 80
    )

    print(
        "EXHAUSTIVE SENSOR-LAYOUT "
        "DESCRIPTOR BUILDER"
    )

    print(
        "=" * 80
    )

    print(
        "Canonical clean dataset:",
        clean_path,
    )

    print(
        "Canonical dimension:",
        F_full.shape[1],
    )

    print(
        "Ground/base sensor:",
        "always available",
    )

    print(
        "Structural sensors:",
        STRUCTURAL_SENSORS,
    )

    print(
        "Ground-only descriptors:",
        ground_only_count,
    )

    print(
        "All-story descriptors:",
        full_dependency_count,
    )

    print()

    # ==================================================
    # Build all 15 layouts.
    # ==================================================

    manifest_rows = []

    layouts = all_layouts()

    if len(
        layouts
    ) != 15:

        raise RuntimeError(
            "Expected 15 non-empty layouts."
        )

    full_layout_exact = False

    for layout_number, layout in enumerate(
        layouts,
        start=1,
    ):

        available = frozenset(
            layout
        )

        selected_indices = np.asarray(
            [
                index
                for index, required
                in enumerate(
                    dependencies
                )
                if required.issubset(
                    available
                )
            ],
            dtype=np.int64,
        )

        selected_names = [
            feature_names[index]
            for index
            in selected_indices
        ]

        F_layout = (
            F_full[
                :,
                selected_indices
            ]
        )

        expected_dimension = (
            EXPECTED_LAYOUT_DIMENSIONS[
                layout
            ]
        )

        dimension_ok = bool(
            F_layout.shape
            == (
                3000,
                expected_dimension,
            )
        )

        if not dimension_ok:

            raise RuntimeError(
                f"Layout {layout} "
                f"dimension mismatch: "
                f"got {F_layout.shape[1]}, "
                f"expected {expected_dimension}."
            )

        tag = layout_tag(
            layout
        )

        output_path = (
            output_root
            / (
                f"sensor_layout_{tag}_"
                f"clean_descriptors.npz"
            )
        )

        save_payload = {
            "F_raw": (
                F_layout
            ),
            "feature_names": np.asarray(
                selected_names
            ),
            "feature_indices_in_full_78d": (
                selected_indices
            ),
            "sensor_keep_1based": np.asarray(
                layout,
                dtype=np.int64,
            ),
            "sensor_count": np.asarray(
                len(
                    layout
                ),
                dtype=np.int64,
            ),
            "y_damage": (
                clean[
                    "y_damage"
                ]
            ),
            "case_id": (
                clean[
                    "case_id"
                ]
            ),
            "train_idx": (
                clean[
                    "train_idx"
                ]
            ),
            "val_idx": (
                clean[
                    "val_idx"
                ]
            ),
            "test_idx": (
                clean[
                    "test_idx"
                ]
            ),
            "dt": (
                clean[
                    "dt"
                ]
            ),
        }

        optional_passthrough = [
            "amplitude_g",
            "frequency_hz",
            "phase",
            "healthy_first_frequency_hz",
        ]

        for key in (
            optional_passthrough
        ):

            if key in clean:

                save_payload[
                    key
                ] = clean[
                    key
                ]

        np.savez_compressed(
            output_path,
            **save_payload,
        )

        full_anchor_error = None

        if (
            layout
            == STRUCTURAL_SENSORS
        ):

            full_anchor_error = float(
                np.max(
                    np.abs(
                        F_layout
                        - F_full
                    )
                )
            )

            full_names_exact = bool(
                selected_names
                == feature_names
            )

            full_indices_exact = bool(
                np.array_equal(
                    selected_indices,
                    np.arange(
                        EXPECTED_FULL_DIM,
                        dtype=np.int64,
                    ),
                )
            )

            full_layout_exact = bool(
                full_anchor_error
                == 0.0
                and full_names_exact
                and full_indices_exact
            )

            if not full_layout_exact:

                raise RuntimeError(
                    "FULL-LAYOUT ANCHOR FAILED."
                )

        print(
            f"{layout_number:2d}/15 "
            f"layout={layout} "
            f"count={len(layout)} "
            f"features={F_layout.shape[1]}"
        )

        manifest_rows.append(
            {
                "layout_number": (
                    layout_number
                ),
                "layout_tag": (
                    tag
                ),
                "sensor_layout": (
                    ",".join(
                        str(sensor)
                        for sensor
                        in layout
                    )
                ),
                "sensor_count": (
                    len(
                        layout
                    )
                ),
                "feature_count": (
                    F_layout.shape[1]
                ),
                "expected_feature_count": (
                    expected_dimension
                ),
                "dimension_ok": (
                    dimension_ok
                ),
                "full_anchor_max_abs_error": (
                    full_anchor_error
                ),
                "file": str(
                    output_path
                ),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        output_root
        / "sensor_layout_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    # ==================================================
    # Count-level audit.
    # ==================================================

    count_summary = (
        manifest.groupby(
            "sensor_count"
        )
        .agg(
            n_layouts=(
                "layout_tag",
                "count",
            ),
            min_feature_count=(
                "feature_count",
                "min",
            ),
            max_feature_count=(
                "feature_count",
                "max",
            ),
            mean_feature_count=(
                "feature_count",
                "mean",
            ),
        )
        .reset_index()
    )

    expected_layout_counts = {
        1: 4,
        2: 6,
        3: 4,
        4: 1,
    }

    count_structure_ok = bool(
        all(
            int(
                count_summary.loc[
                    count_summary[
                        "sensor_count"
                    ]
                    == sensor_count,
                    "n_layouts",
                ].iloc[0]
            )
            == expected_count
            for (
                sensor_count,
                expected_count,
            )
            in expected_layout_counts.items()
        )
    )

    all_dimensions_ok = bool(
        manifest[
            "dimension_ok"
        ].all()
    )

    overall_passed = bool(
        len(
            manifest
        )
        == 15
        and count_structure_ok
        and all_dimensions_ok
        and full_layout_exact
    )

    # ==================================================
    # Save report.
    # ==================================================

    report = {
        "experiment": (
            "dependency_aware_exhaustive_"
            "sensor_layout_descriptor_build"
        ),
        "canonical_source": str(
            clean_path
        ),
        "structural_sensors": list(
            STRUCTURAL_SENSORS
        ),
        "ground_sensor_assumed_available": (
            True
        ),
        "zero_masking_used": (
            False
        ),
        "descriptor_redefinition_used": (
            False
        ),
        "availability_rule": (
            "retain descriptor only if all "
            "required structural sensors "
            "are contained in the layout"
        ),
        "n_layouts": int(
            len(
                manifest
            )
        ),
        "expected_layout_counts": (
            expected_layout_counts
        ),
        "ground_only_descriptor_count": (
            ground_only_count
        ),
        "all_story_descriptor_count": (
            full_dependency_count
        ),
        "expected_layout_dimensions": {
            ",".join(
                str(sensor)
                for sensor
                in layout
            ): dimension
            for (
                layout,
                dimension,
            )
            in (
                EXPECTED_LAYOUT_DIMENSIONS
                .items()
            )
        },
        "count_structure_ok": (
            count_structure_ok
        ),
        "all_dimensions_ok": (
            all_dimensions_ok
        ),
        "full_layout_anchor_exact": (
            full_layout_exact
        ),
        "overall_passed": (
            overall_passed
        ),
        "dependency_manifest": str(
            dependency_path
        ),
        "layout_manifest": str(
            manifest_path
        ),
    }

    report_path = (
        output_root
        / "sensor_layout_build_report.json"
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

    # ==================================================
    # Console summary.
    # ==================================================

    print()
    print(
        "=" * 80
    )

    print(
        "LAYOUT MANIFEST"
    )

    print(
        "=" * 80
    )

    print(
        manifest[
            [
                "layout_tag",
                "sensor_layout",
                "sensor_count",
                "feature_count",
                "expected_feature_count",
                "dimension_ok",
            ]
        ].to_string(
            index=False,
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "SENSOR-COUNT SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        count_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL BUILD CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Layouts:",
        len(
            manifest
        ),
    )

    print(
        "Layout count structure correct:",
        count_structure_ok,
    )

    print(
        "All layout dimensions correct:",
        all_dimensions_ok,
    )

    print(
        "Full 4-sensor descriptor anchor exact:",
        full_layout_exact,
    )

    print(
        "Zero masking used:",
        False,
    )

    print(
        "Descriptor redefinition used:",
        False,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    if overall_passed:

        print(
            "CHECK PASSED: all 15 "
            "dependency-aware sensor-layout "
            "descriptor datasets were built."
        )

        print(
            "Exhaustive layout-specific "
            "SVR evaluation may proceed."
        )

    else:

        print(
            "CHECK FAILED: do not run "
            "sensor-layout models."
        )

    print()

    print(
        "Dependency manifest:",
        dependency_path,
    )

    print(
        "Layout manifest:",
        manifest_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
