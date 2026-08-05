"""
Verify canonical Paper 0 descriptor sets and write tracked manifests.

验证92、86、78、59维描述符集合，并生成可追溯配置清单。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Add the repository root to Python's module search path when this
# file is executed directly.
#
# 当通过 python scripts/... 直接运行本文件时，
# 将项目根目录加入模块搜索路径，以便正常导入 src 包。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.preprocessing.descriptor_sets import (
    DescriptorSetName,
    build_descriptor_manifest,
    descriptor_set_counts,
    descriptor_source_counts,
    get_descriptor_indices,
)


EXPECTED_SET_COUNTS = {
    "oracle_full": 92,
    "legacy_no_meta": 86,
    "signal_derived_with_ground": 78,
    "structural_response_only": 59,
}

EXPECTED_SOURCE_COUNTS = {
    "generator_metadata": 6,
    "generator_frequency_derived": 8,
    "measured_ground_signal": 19,
    "structural_response_signal": 59,
}


def resolve_name_column(frame: pd.DataFrame) -> str:
    """Resolve the feature-name column."""
    preferred = [
        "feature_name",
        "name",
        "descriptor_name",
        "feature",
    ]

    for candidate in preferred:
        if candidate in frame.columns:
            return candidate

    object_columns = [
        column
        for column in frame.columns
        if frame[column].dtype == object
    ]

    if len(object_columns) == 1:
        return str(object_columns[0])

    raise ValueError(
        "Could not identify feature-name column. "
        f"Available columns: {list(frame.columns)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-names",
        type=Path,
        default=Path(
            "data_inputs/aes_frozen/"
            "debug_plus_3000_physics_feature_names.csv"
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "configs/sss_fast_revision/"
            "descriptor_set_manifest.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "configs/sss_fast_revision/"
            "descriptor_set_summary.json"
        ),
    )
    args = parser.parse_args()

    if not args.feature_names.is_file():
        raise FileNotFoundError(args.feature_names)

    frame = pd.read_csv(args.feature_names)
    name_column = resolve_name_column(frame)

    feature_names = (
        frame[name_column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(feature_names) != 92:
        raise AssertionError(
            f"Expected 92 descriptors, found {len(feature_names)}."
        )

    set_counts = descriptor_set_counts(feature_names)
    source_counts = descriptor_source_counts(feature_names)

    print("===== DESCRIPTOR SET COUNTS =====")
    for name, count in set_counts.items():
        print(f"{name:32s}: {count}")

    print()
    print("===== DESCRIPTOR SOURCE COUNTS =====")
    for name, count in source_counts.items():
        print(f"{name:32s}: {count}")

    if set_counts != EXPECTED_SET_COUNTS:
        raise AssertionError(
            "Descriptor-set counts do not match expectations.\n"
            f"Expected: {EXPECTED_SET_COUNTS}\n"
            f"Actual:   {set_counts}"
        )

    if source_counts != EXPECTED_SOURCE_COUNTS:
        raise AssertionError(
            "Descriptor-source counts do not match expectations.\n"
            f"Expected: {EXPECTED_SOURCE_COUNTS}\n"
            f"Actual:   {source_counts}"
        )

    index_sets = {
        descriptor_set.value: set(
            get_descriptor_indices(
                feature_names,
                descriptor_set,
            ).tolist()
        )
        for descriptor_set in DescriptorSetName
    }

    if not (
        index_sets["structural_response_only"]
        <= index_sets["signal_derived_with_ground"]
        <= index_sets["legacy_no_meta"]
        <= index_sets["oracle_full"]
    ):
        raise AssertionError(
            "Expected nested descriptor-set relationship failed."
        )

    manifest = build_descriptor_manifest(feature_names)

    args.manifest_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        args.manifest_output,
        index=False,
    )

    summary = {
        "source_feature_name_file": str(args.feature_names),
        "descriptor_count": len(feature_names),
        "descriptor_set_counts": set_counts,
        "descriptor_source_counts": source_counts,
        "descriptor_sets": {},
    }

    for descriptor_set in DescriptorSetName:
        indices = get_descriptor_indices(
            feature_names,
            descriptor_set,
        )

        selected_names = [
            feature_names[int(index)]
            for index in indices
        ]

        summary["descriptor_sets"][descriptor_set.value] = {
            "n_features": len(selected_names),
            "indices": [
                int(index)
                for index in indices
            ],
            "feature_names": selected_names,
        }

    with args.summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("===== SET DIFFERENCES =====")

    comparisons = [
        ("oracle_full", "legacy_no_meta"),
        ("legacy_no_meta", "signal_derived_with_ground"),
        (
            "signal_derived_with_ground",
            "structural_response_only",
        ),
    ]

    for larger, smaller in comparisons:
        removed_indices = sorted(
            index_sets[larger] - index_sets[smaller]
        )

        print()
        print(
            f"{larger} -> {smaller}: "
            f"remove {len(removed_indices)}"
        )

        for index in removed_indices:
            print(
                f"  {index:3d}  {feature_names[index]}"
            )

    print()
    print("CHECK PASSED: all descriptor sets are valid.")
    print("Manifest:", args.manifest_output)
    print("Summary:", args.summary_output)


if __name__ == "__main__":
    main()
