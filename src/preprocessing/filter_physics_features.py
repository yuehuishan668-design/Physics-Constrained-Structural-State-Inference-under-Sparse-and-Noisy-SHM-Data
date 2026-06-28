#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature filtering utility for physics-feature ablation experiments.

Purpose
-------
This script reads a physics-feature dataset, selects or removes columns by
interpretable feature groups, and writes a new .npz feature dataset plus a
matching feature-name CSV.

Why this file is needed
-----------------------
The current project has already extracted many physics-guided features from
acceleration responses. A single "full feature" result is not enough for a
paper-level experiment, because it cannot show which feature groups are useful.
This script enables ablation studies such as:
    - full features
    - remove metadata features
    - response-only features
    - response + spatial features
    - response + correlation features
    - response + frequency features

Expected input .npz keys
------------------------
At minimum:
    F_train, F_val, F_test
    y_train, y_val, y_test

All other keys are copied unchanged unless they are feature-dimension dependent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


def feature_groups_for_name(name: str) -> Set[str]:
    """Return interpretable groups assigned to one feature name."""
    n = name.lower()
    groups: Set[str] = set()

    # Metadata / generation-condition features.
    # 中文：这些是数据生成条件类特征，例如输入频率、噪声水平、输入幅值。
    if (
        n.startswith("input_")
        or "noise_level" in n
        or "amplitude_g" in n
        or "frequency_hz" in n
        or "amp_g" in n
        or n in {"noise", "freq", "frequency", "amplitude"}
    ):
        groups.add("meta")

    # Basic time-domain response statistics.
    # 中文：每层响应的基本时域统计量。
    if any(
        key in n
        for key in [
            "_mean",
            "_std",
            "_max_abs",
            "_rms",
            "_peak_to_peak",
            "_crest_factor",
        ]
    ):
        groups.add("response_basic")

    # Ground-motion related features.
    # 中文：和地震/地面输入运动有关的特征。
    if n.startswith("ground_") or "_ground_" in n:
        groups.add("ground")

    # Amplification / ratio to input motion.
    # 中文：结构响应相对于输入地震动的放大关系。
    if "ground_amplification" in n or "to_input_ratio" in n:
        groups.add("amplification")
        groups.add("ratio")

    # Spatial response distribution across stories.
    # 中文：各楼层响应在空间上的分布比例。
    if "spatial_fraction" in n:
        groups.add("spatial")

    # Story-to-story ratio features.
    # 中文：楼层之间的响应比值特征。
    if "_to_story_" in n or n.endswith("_ratio") or "_ratio" in n:
        groups.add("ratio")

    # Frequency-domain features.
    # 中文：频域特征，例如主频、频谱质心、频带能量。
    if (
        "spectral" in n
        or "dominant_frequency" in n
        or "band_energy" in n
        or "centroid" in n
        or "frequency_to_input" in n
    ):
        groups.add("frequency")

    # Correlation features.
    # 中文：楼层响应或地面输入之间的相关性。
    if "correlation" in n or "corr" in n:
        groups.add("correlation")

    if n.startswith("story_"):
        groups.add("story_local")

    if not groups:
        groups.add("other")

    return groups


KNOWN_GROUPS = [
    "meta",
    "response_basic",
    "ground",
    "amplification",
    "spatial",
    "ratio",
    "frequency",
    "correlation",
    "story_local",
    "other",
]


def read_feature_names(path: Path) -> List[str]:
    """Read feature names from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Feature name CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if "feature_name" in fieldnames:
            name_col = "feature_name"
        elif "name" in fieldnames:
            name_col = "name"
        elif len(fieldnames) == 1:
            name_col = fieldnames[0]
        else:
            raise ValueError(f"Cannot infer feature-name column from CSV header: {fieldnames}")

        names = [row[name_col].strip() for row in reader if row.get(name_col, "").strip()]

    if not names:
        raise ValueError(f"No feature names were read from {path}")

    return names


def write_feature_names(path: Path, names: Sequence[str]) -> None:
    """Write selected feature names to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "feature_name"])
        for i, name in enumerate(names):
            writer.writerow([i, name])


def parse_group_list(value: str | None) -> Set[str]:
    """Parse comma-separated or space-separated group names."""
    if value is None or value.strip() == "":
        return set()
    return {x.strip() for x in value.replace(",", " ").split() if x.strip()}


def group_count_summary(names: Sequence[str]) -> Dict[str, int]:
    """Count selected features by group."""
    counts = {g: 0 for g in KNOWN_GROUPS}
    for name in names:
        for g in feature_groups_for_name(name):
            counts[g] = counts.get(g, 0) + 1
    return counts


def select_feature_indices(
    names: Sequence[str],
    include_groups: Set[str],
    exclude_groups: Set[str],
    include_regex: Sequence[str] | None = None,
    exclude_regex: Sequence[str] | None = None,
) -> Tuple[List[int], List[str]]:
    """Select feature indices using group inclusion/exclusion rules."""
    import re

    include_regex = include_regex or []
    exclude_regex = exclude_regex or []

    selected: List[int] = []
    selected_names: List[str] = []

    for i, name in enumerate(names):
        groups = feature_groups_for_name(name)
        keep = True

        # If include_groups is non-empty, a feature must belong to at least one included group.
        if include_groups:
            keep = bool(groups & include_groups)

        # Exclusion wins over inclusion.
        if groups & exclude_groups:
            keep = False

        for pattern in include_regex:
            if re.search(pattern, name):
                keep = True

        for pattern in exclude_regex:
            if re.search(pattern, name):
                keep = False

        if keep:
            selected.append(i)
            selected_names.append(name)

    if not selected:
        raise ValueError(
            "Feature selection produced zero columns. "
            f"include_groups={sorted(include_groups)}, exclude_groups={sorted(exclude_groups)}"
        )

    return selected, selected_names


def filter_npz_features(input_npz: Path, output_npz: Path, selected_indices: Sequence[int]) -> Dict[str, object]:
    """Filter F_train/F_val/F_test columns and copy other arrays."""
    if not input_npz.exists():
        raise FileNotFoundError(f"Input feature dataset not found: {input_npz}")

    data = np.load(input_npz, allow_pickle=True)
    output = {}

    required = ["F_train", "F_val", "F_test", "y_train", "y_val", "y_test"]
    for key in required:
        if key not in data:
            raise KeyError(f"Input NPZ is missing required key: {key}")

    for key in data.files:
        arr = data[key]
        if key in {"F_train", "F_val", "F_test"}:
            if arr.ndim != 2:
                raise ValueError(f"{key} must be 2D, got shape={arr.shape}")
            output[key] = arr[:, selected_indices]
        else:
            output[key] = arr

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **output)

    return {
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "n_selected_features": int(len(selected_indices)),
        "F_train_shape": list(output["F_train"].shape),
        "F_val_shape": list(output["F_val"].shape),
        "F_test_shape": list(output["F_test"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter physics-feature columns for ablation experiments.")
    parser.add_argument("--input", required=True, type=Path, help="Input feature .npz")
    parser.add_argument("--feature-names", required=True, type=Path, help="Input feature-name CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output filtered .npz")
    parser.add_argument("--output-feature-names", required=True, type=Path, help="Output filtered feature-name CSV")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON")
    parser.add_argument("--include-groups", default="", help=f"Groups to keep. Known groups: {', '.join(KNOWN_GROUPS)}")
    parser.add_argument("--exclude-groups", default="", help=f"Groups to remove. Known groups: {', '.join(KNOWN_GROUPS)}")
    parser.add_argument("--include-regex", nargs="*", default=[], help="Optional regex patterns. Matching features are kept.")
    parser.add_argument("--exclude-regex", nargs="*", default=[], help="Optional regex patterns. Matching features are removed.")
    args = parser.parse_args()

    names = read_feature_names(args.feature_names)
    include_groups = parse_group_list(args.include_groups)
    exclude_groups = parse_group_list(args.exclude_groups)

    unknown = (include_groups | exclude_groups) - set(KNOWN_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")

    selected_indices, selected_names = select_feature_indices(
        names=names,
        include_groups=include_groups,
        exclude_groups=exclude_groups,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    )

    npz_summary = filter_npz_features(args.input, args.output, selected_indices)
    write_feature_names(args.output_feature_names, selected_names)

    summary = {
        **npz_summary,
        "input_feature_name_csv": str(args.feature_names),
        "output_feature_name_csv": str(args.output_feature_names),
        "include_groups": sorted(include_groups),
        "exclude_groups": sorted(exclude_groups),
        "selected_group_counts": group_count_summary(selected_names),
        "first_30_selected_features": selected_names[:30],
    }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Feature filtering completed.")
    print(f"Input feature dataset: {args.input}")
    print(f"Output feature dataset: {args.output}")
    print(f"Input feature count: {len(names)}")
    print(f"Selected feature count: {len(selected_names)}")
    print(f"F_train shape: {npz_summary['F_train_shape']}")
    print("Selected group counts:")
    for key, value in summary["selected_group_counts"].items():
        print(f"  {key}: {value}")
    if args.summary_json is not None:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
