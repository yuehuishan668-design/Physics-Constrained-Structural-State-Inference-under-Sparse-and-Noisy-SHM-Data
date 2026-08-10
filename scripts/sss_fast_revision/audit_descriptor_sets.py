"""
Audit the 92-dimensional descriptor set used in Paper 0.

Outputs:
1. Descriptor taxonomy and provisional deployment status.
2. Duplicate feature-name checks.
3. Zero/near-zero variance checks.
4. Exact duplicate numerical-column checks.
5. Summary printed to the terminal.

This script does not modify the frozen input data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def resolve_name_column(frame: pd.DataFrame) -> str:
    """Find the column containing descriptor names."""
    preferred = [
        "feature_name",
        "name",
        "descriptor_name",
        "feature",
    ]

    lower_to_original = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }

    for candidate in preferred:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    object_columns = [
        column
        for column in frame.columns
        if frame[column].dtype == object
    ]

    if len(object_columns) == 1:
        return str(object_columns[0])

    raise ValueError(
        "Could not identify the feature-name column. "
        f"Available columns: {list(frame.columns)}"
    )


def classify_descriptor(name: str) -> tuple[str, str, str]:
    """
    Return:
        descriptor_family,
        information_status,
        review_note
    """
    text = name.strip().lower()

    # Privileged labels or generator-known variables.
    privileged_tokens = [
        "noise_level",
        "true_noise",
        "injected_noise",
        "damage_label",
        "true_damage",
        "case_id",
    ]

    # Input variables may be deployable only if measured or estimated
    # from an actual ground/input sensor rather than copied directly
    # from the simulation generator.
    input_tokens = [
        "amplitude_g",
        "frequency_hz",
        "input_amplitude",
        "input_frequency",
        "ground_amplitude",
        "ground_frequency",
        "excitation_amplitude",
        "excitation_frequency",
    ]

    # These require a healthy/reference model or baseline record.
    reference_tokens = [
        "healthy",
        "reference",
        "baseline",
        "proximity",
        "detuning",
        "resonance_distance",
    ]

    correlation_tokens = [
        "corr",
        "correlation",
        "coherence",
        "cross_channel",
        "cross-channel",
    ]

    spatial_tokens = [
        "spatial",
        "storey_ratio",
        "story_ratio",
        "interstorey",
        "inter_storey",
        "interstory",
        "inter_story",
        "adjacent",
        "floor_ratio",
        "channel_ratio",
    ]

    frequency_tokens = [
        "frequency",
        "freq",
        "fft",
        "psd",
        "spectral",
        "spectrum",
        "band_energy",
        "dominant_peak",
        "peak_frequency",
    ]

    if any(token in text for token in privileged_tokens):
        return (
            "privileged_simulation_metadata",
            "oracle_only",
            "Exclude from deployable main model.",
        )

    if any(token in text for token in input_tokens):
        return (
            "input_condition",
            "conditional",
            "Deployable only if obtained from a measured input signal; "
            "not deployable when copied from generator parameters.",
        )

    if any(token in text for token in reference_tokens):
        return (
            "reference_dependent",
            "conditional",
            "Requires an available healthy baseline or calibrated reference model.",
        )

    if any(token in text for token in correlation_tokens):
        return (
            "response_correlation",
            "deployable",
            "Derived from measured structural responses.",
        )

    if any(token in text for token in spatial_tokens):
        return (
            "response_spatial",
            "deployable",
            "Derived from the spatial distribution of measured responses.",
        )

    if any(token in text for token in frequency_tokens):
        return (
            "response_frequency",
            "deployable",
            "Provisionally response-derived; verify that no generator parameter is used.",
        )

    return (
        "response_time_or_statistical",
        "deployable",
        "Provisionally derived from measured structural responses.",
    )


def concatenate_feature_arrays(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Concatenate one non-overlapping set of sample-level feature arrays.

    Prefer raw train/validation/test descriptors. This avoids:
    1. double-counting raw and standardised samples;
    2. treating F_mean and F_std as observations.
    """
    candidate_groups = [
        [
            "F_train_raw",
            "F_val_raw",
            "F_test_raw",
        ],
        [
            "F_train",
            "F_val",
            "F_test",
        ],
    ]

    for keys in candidate_groups:
        if not all(key in arrays for key in keys):
            continue

        if not all(arrays[key].ndim == 2 for key in keys):
            continue

        feature_dims = {
            arrays[key].shape[1]
            for key in keys
        }

        if len(feature_dims) != 1:
            raise ValueError(
                "Inconsistent feature dimensions across "
                f"{keys}: {feature_dims}"
            )

        matrix = np.concatenate(
            [arrays[key] for key in keys],
            axis=0,
        )

        return matrix, keys

    available = {
        key: array.shape
        for key, array in arrays.items()
    }

    raise KeyError(
        "Could not find a complete raw or standardised "
        "train/validation/test feature group. "
        f"Available arrays: {available}"
    )


def find_exact_duplicate_columns(
    matrix: np.ndarray,
    names: list[str],
    atol: float = 1e-12,
) -> list[tuple[int, int, str, str]]:
    """Find numerically identical descriptor columns."""
    duplicates: list[tuple[int, int, str, str]] = []

    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            if np.allclose(
                matrix[:, left],
                matrix[:, right],
                rtol=0.0,
                atol=atol,
                equal_nan=True,
            ):
                duplicates.append(
                    (left, right, names[left], names[right])
                )

    return duplicates


def format_counts(values: Iterable[str]) -> str:
    series = pd.Series(list(values), dtype="object")
    return series.value_counts(dropna=False).to_string()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data_inputs/aes_frozen/"
            "debug_plus_3000_physics_features_mlp.npz"
        ),
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        default=Path(
            "data_inputs/aes_frozen/"
            "debug_plus_3000_physics_feature_names.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/sss_fast_revision/tables/"
            "T01_descriptor_audit.csv"
        ),
    )
    args = parser.parse_args()

    if not args.features.is_file():
        raise FileNotFoundError(args.features)

    if not args.feature_names.is_file():
        raise FileNotFoundError(args.feature_names)

    names_frame = pd.read_csv(args.feature_names)
    name_column = resolve_name_column(names_frame)

    feature_names = (
        names_frame[name_column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    with np.load(args.features, allow_pickle=False) as loaded:
        arrays = {
            key: loaded[key]
            for key in loaded.files
        }

    feature_matrix, used_keys = concatenate_feature_arrays(arrays)

    if feature_matrix.shape[1] != len(feature_names):
        raise ValueError(
            "Feature dimension mismatch: "
            f"NPZ has {feature_matrix.shape[1]} columns, "
            f"CSV has {len(feature_names)} names."
        )

    rows: list[dict[str, object]] = []

    variances = np.nanvar(feature_matrix, axis=0)
    finite_ratios = np.mean(np.isfinite(feature_matrix), axis=0)

    for index, name in enumerate(feature_names):
        family, status, note = classify_descriptor(name)

        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "descriptor_family": family,
                "information_status": status,
                "provisional_response_only": status == "deployable",
                "provisional_with_conditions": status in {
                    "deployable",
                    "conditional",
                },
                "oracle_full": True,
                "variance": float(variances[index]),
                "near_zero_variance": bool(variances[index] < 1e-12),
                "finite_ratio": float(finite_ratios[index]),
                "manual_review_required": status == "conditional",
                "review_note": note,
            }
        )

    audit = pd.DataFrame(rows)

    duplicate_name_mask = audit["feature_name"].duplicated(
        keep=False
    )
    audit["duplicate_feature_name"] = duplicate_name_mask

    duplicate_columns = find_exact_duplicate_columns(
        feature_matrix,
        feature_names,
    )

    duplicate_column_indices: set[int] = set()
    for left, right, _, _ in duplicate_columns:
        duplicate_column_indices.add(left)
        duplicate_column_indices.add(right)

    audit["exact_duplicate_numeric_column"] = (
        audit["feature_index"].isin(duplicate_column_indices)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)

    duplicate_output = args.output.with_name(
        "T02_exact_duplicate_descriptor_columns.csv"
    )

    duplicate_frame = pd.DataFrame(
        duplicate_columns,
        columns=[
            "left_index",
            "right_index",
            "left_name",
            "right_name",
        ],
    )
    duplicate_frame.to_csv(duplicate_output, index=False)

    print("===== DESCRIPTOR AUDIT COMPLETE =====")
    print(f"Feature file: {args.features}")
    print(f"Feature-name file: {args.feature_names}")
    print(f"Feature arrays used: {used_keys}")
    print(f"Combined matrix shape: {feature_matrix.shape}")
    print(f"Descriptor count: {len(feature_names)}")

    print()
    print("===== FAMILY COUNTS =====")
    print(format_counts(audit["descriptor_family"]))

    print()
    print("===== INFORMATION-STATUS COUNTS =====")
    print(format_counts(audit["information_status"]))

    print()
    print("===== PROVISIONAL SET SIZES =====")
    print(
        "Response-only deployable:",
        int(audit["provisional_response_only"].sum()),
    )
    print(
        "Deployable plus conditional:",
        int(audit["provisional_with_conditions"].sum()),
    )
    print("Oracle full:", int(audit["oracle_full"].sum()))

    print()
    print("===== QUALITY CHECKS =====")
    print(
        "Duplicate names:",
        int(audit["duplicate_feature_name"].sum()),
    )
    print(
        "Near-zero variance columns:",
        int(audit["near_zero_variance"].sum()),
    )
    print(
        "Columns containing non-finite values:",
        int((audit["finite_ratio"] < 1.0).sum()),
    )
    print(
        "Exact duplicate numeric pairs:",
        len(duplicate_columns),
    )

    print()
    print("===== CONDITIONAL / ORACLE FEATURES =====")
    review_columns = [
        "feature_index",
        "feature_name",
        "descriptor_family",
        "information_status",
        "review_note",
    ]
    review_rows = audit[
        audit["information_status"].isin(
            ["conditional", "oracle_only"]
        )
    ][review_columns]

    if review_rows.empty:
        print("None")
    else:
        print(review_rows.to_string(index=False))

    print()
    print("Audit table:", args.output)
    print("Duplicate-column table:", duplicate_output)


if __name__ == "__main__":
    main()
