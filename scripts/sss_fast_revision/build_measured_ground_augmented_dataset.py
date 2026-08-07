"""
Build an 86-dimensional measured-ground-augmented descriptor dataset.

Starting point
--------------
78 signal-derived descriptors using structural responses and measured
ground-input histories.

Added descriptors
-----------------
For each storey:
1. response dominant frequency / measured-ground dominant frequency
2. response spectral centroid / measured-ground spectral centroid

The exact simulation-generator frequency is never used.

构建86维实测地面输入增强描述符集。
新增特征完全由已有地面输入与楼层响应描述符计算，
不使用仿真生成器输入频率或其他特权元数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_FEATURE_COUNT = 78
AUGMENTED_FEATURE_COUNT = 86

DEFAULT_RATIO_EPSILON = 1.0e-12
DEFAULT_DENOMINATOR_THRESHOLD = 1.0e-8
STANDARD_DEVIATION_EPSILON = 1.0e-12


GROUND_DOMINANT_FREQUENCY = (
    "ground_dominant_frequency"
)

GROUND_SPECTRAL_CENTROID = (
    "ground_spectral_centroid"
)


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def percentile_summary(
    values: np.ndarray,
) -> dict[str, float]:
    """Return robust numerical distribution statistics."""
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if values.size == 0:
        raise ValueError(
            "Cannot summarise an empty array."
        )

    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            "Distribution contains NaN or infinity."
        )

    return {
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(
            np.std(values)
        ),
    }


def analyse_design_matrix(
    matrix: np.ndarray,
) -> dict[str, Any]:
    """
    Analyse rank and conditioning of the standardised training matrix.

    分析标准化训练矩阵的数值秩和条件数。
    """
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    centred = matrix - np.mean(
        matrix,
        axis=0,
        keepdims=True,
    )

    singular_values = np.linalg.svd(
        centred,
        full_matrices=False,
        compute_uv=False,
    )

    tolerance = (
        np.finfo(np.float64).eps
        * max(centred.shape)
        * singular_values[0]
    )

    nonzero = singular_values[
        singular_values > tolerance
    ]

    numerical_rank = int(nonzero.size)

    if numerical_rank == 0:
        condition_number = float("inf")
        smallest_nonzero = float("nan")
    else:
        smallest_nonzero = float(nonzero[-1])
        condition_number = float(
            nonzero[0] / nonzero[-1]
        )

    return {
        "sample_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "numerical_rank": numerical_rank,
        "full_column_rank": (
            numerical_rank == matrix.shape[1]
        ),
        "largest_singular_value": float(
            singular_values[0]
        ),
        "smallest_nonzero_singular_value": (
            smallest_nonzero
        ),
        "condition_number_nonzero_spectrum": (
            condition_number
        ),
        "rank_tolerance": float(tolerance),
    }


def load_source_dataset(
    path: Path,
) -> dict[str, np.ndarray]:
    """Load and validate the 78-dimensional source dataset."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as loaded:
        arrays = {
            key: loaded[key]
            for key in loaded.files
        }

    required = {
        "F_train",
        "F_val",
        "F_test",
        "F_train_raw",
        "F_val_raw",
        "F_test_raw",
        "F_mean",
        "F_std",
        "feature_names",
        "y_train",
        "y_val",
        "y_test",
    }

    missing = sorted(
        required - set(arrays)
    )

    if missing:
        raise KeyError(
            f"Source dataset is missing keys: {missing}"
        )

    feature_names = [
        str(name)
        for name in arrays["feature_names"].tolist()
    ]

    if len(feature_names) != BASE_FEATURE_COUNT:
        raise ValueError(
            f"Expected {BASE_FEATURE_COUNT} source features, "
            f"found {len(feature_names)}."
        )

    for split in ["train", "val", "test"]:
        raw = np.asarray(
            arrays[f"F_{split}_raw"],
            dtype=np.float64,
        )

        standardised = np.asarray(
            arrays[f"F_{split}"],
            dtype=np.float64,
        )

        if raw.shape[1] != BASE_FEATURE_COUNT:
            raise ValueError(
                f"F_{split}_raw shape mismatch: {raw.shape}"
            )

        if standardised.shape[1] != BASE_FEATURE_COUNT:
            raise ValueError(
                f"F_{split} shape mismatch: {standardised.shape}"
            )

        if not np.all(np.isfinite(raw)):
            raise FloatingPointError(
                f"F_{split}_raw contains non-finite values."
            )

        if not np.all(np.isfinite(standardised)):
            raise FloatingPointError(
                f"F_{split} contains non-finite values."
            )

    return arrays


def get_feature_index(
    feature_names: list[str],
    feature_name: str,
) -> int:
    """Return the unique index of one named feature."""
    matches = [
        index
        for index, name in enumerate(feature_names)
        if name == feature_name
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one occurrence of "
            f"'{feature_name}', found {matches}."
        )

    return int(matches[0])


def derive_augmented_raw_features(
    raw_matrix: np.ndarray,
    feature_names: list[str],
    ratio_epsilon: float,
    denominator_threshold: float,
    split_name: str,
) -> tuple[
    np.ndarray,
    list[str],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Construct the eight measured-ground-normalised descriptors.

    使用地面输入主频和谱质心构造八个新描述符。
    """
    raw_matrix = np.asarray(
        raw_matrix,
        dtype=np.float64,
    )

    ground_dom_index = get_feature_index(
        feature_names,
        GROUND_DOMINANT_FREQUENCY,
    )

    ground_centroid_index = get_feature_index(
        feature_names,
        GROUND_SPECTRAL_CENTROID,
    )

    ground_dom = raw_matrix[
        :,
        ground_dom_index,
    ]

    ground_centroid = raw_matrix[
        :,
        ground_centroid_index,
    ]

    denominator_audit = {
        "ground_dominant_frequency": {
            **percentile_summary(ground_dom),
            "count_abs_le_1e_minus_12": int(
                np.sum(
                    np.abs(ground_dom)
                    <= 1.0e-12
                )
            ),
            "count_abs_le_1e_minus_8": int(
                np.sum(
                    np.abs(ground_dom)
                    <= 1.0e-8
                )
            ),
            "count_abs_le_1e_minus_6": int(
                np.sum(
                    np.abs(ground_dom)
                    <= 1.0e-6
                )
            ),
        },
        "ground_spectral_centroid": {
            **percentile_summary(
                ground_centroid
            ),
            "count_abs_le_1e_minus_12": int(
                np.sum(
                    np.abs(ground_centroid)
                    <= 1.0e-12
                )
            ),
            "count_abs_le_1e_minus_8": int(
                np.sum(
                    np.abs(ground_centroid)
                    <= 1.0e-8
                )
            ),
            "count_abs_le_1e_minus_6": int(
                np.sum(
                    np.abs(ground_centroid)
                    <= 1.0e-6
                )
            ),
        },
    }

    unsafe_dom = np.abs(
        ground_dom
    ) <= denominator_threshold

    unsafe_centroid = np.abs(
        ground_centroid
    ) <= denominator_threshold

    if np.any(unsafe_dom):
        raise ValueError(
            f"{split_name}: ground dominant frequency "
            f"has {int(np.sum(unsafe_dom))} values at or "
            f"below denominator threshold "
            f"{denominator_threshold:.3e}."
        )

    if np.any(unsafe_centroid):
        raise ValueError(
            f"{split_name}: ground spectral centroid "
            f"has {int(np.sum(unsafe_centroid))} values at or "
            f"below denominator threshold "
            f"{denominator_threshold:.3e}."
        )

    new_columns: list[np.ndarray] = []
    new_names: list[str] = []
    audit_rows: list[dict[str, Any]] = []

    for story_id in range(1, 5):
        story_dom_name = (
            f"story_{story_id}_dominant_frequency"
        )

        story_centroid_name = (
            f"story_{story_id}_spectral_centroid"
        )

        story_dom_index = get_feature_index(
            feature_names,
            story_dom_name,
        )

        story_centroid_index = get_feature_index(
            feature_names,
            story_centroid_name,
        )

        story_dom = raw_matrix[
            :,
            story_dom_index,
        ]

        story_centroid = raw_matrix[
            :,
            story_centroid_index,
        ]

        dominant_ratio = (
            story_dom
            / (
                ground_dom
                + ratio_epsilon
            )
        )

        centroid_ratio = (
            story_centroid
            / (
                ground_centroid
                + ratio_epsilon
            )
        )

        dominant_ratio_name = (
            f"story_{story_id}_"
            "dominant_frequency_to_"
            "measured_ground_ratio"
        )

        centroid_ratio_name = (
            f"story_{story_id}_"
            "centroid_to_"
            "measured_ground_ratio"
        )

        for name, values, numerator, denominator in [
            (
                dominant_ratio_name,
                dominant_ratio,
                story_dom_name,
                GROUND_DOMINANT_FREQUENCY,
            ),
            (
                centroid_ratio_name,
                centroid_ratio,
                story_centroid_name,
                GROUND_SPECTRAL_CENTROID,
            ),
        ]:
            if not np.all(np.isfinite(values)):
                raise FloatingPointError(
                    f"{split_name}: derived feature "
                    f"'{name}' contains NaN or infinity."
                )

            stats = percentile_summary(values)

            audit_rows.append(
                {
                    "split": split_name,
                    "feature_name": name,
                    "numerator_feature": numerator,
                    "denominator_feature": denominator,
                    "ratio_epsilon": ratio_epsilon,
                    **stats,
                }
            )

            new_columns.append(
                values.reshape(-1, 1)
            )

            new_names.append(name)

    augmented = np.hstack(
        new_columns
    )

    if augmented.shape[1] != 8:
        raise AssertionError(
            f"Expected eight derived columns, "
            f"found {augmented.shape[1]}."
        )

    if len(new_names) != len(set(new_names)):
        raise ValueError(
            "Duplicate augmented feature names detected."
        )

    return (
        augmented,
        new_names,
        audit_rows,
        denominator_audit,
    )


def find_exact_duplicate_columns(
    original_matrix: np.ndarray,
    augmented_matrix: np.ndarray,
    original_names: list[str],
    augmented_names: list[str],
) -> list[dict[str, str]]:
    """Check whether a new column exactly duplicates an existing one."""
    duplicates: list[dict[str, str]] = []

    for new_index, new_name in enumerate(
        augmented_names
    ):
        new_values = augmented_matrix[
            :,
            new_index,
        ]

        for original_index, original_name in enumerate(
            original_names
        ):
            original_values = original_matrix[
                :,
                original_index,
            ]

            if np.array_equal(
                new_values,
                original_values,
            ):
                duplicates.append(
                    {
                        "new_feature": new_name,
                        "duplicate_existing_feature": (
                            original_name
                        ),
                    }
                )

    return duplicates


def calculate_max_existing_correlation(
    original_matrix: np.ndarray,
    augmented_matrix: np.ndarray,
    original_names: list[str],
    augmented_names: list[str],
) -> list[dict[str, Any]]:
    """
    Report the strongest linear association between each new feature
    and the original 78 descriptors.
    """
    results: list[dict[str, Any]] = []

    original_matrix = np.asarray(
        original_matrix,
        dtype=np.float64,
    )

    augmented_matrix = np.asarray(
        augmented_matrix,
        dtype=np.float64,
    )

    for new_index, new_name in enumerate(
        augmented_names
    ):
        new_values = augmented_matrix[
            :,
            new_index,
        ]

        new_std = float(
            np.std(new_values)
        )

        best_name = ""
        best_correlation = 0.0

        for original_index, original_name in enumerate(
            original_names
        ):
            original_values = original_matrix[
                :,
                original_index,
            ]

            original_std = float(
                np.std(original_values)
            )

            if (
                new_std <= STANDARD_DEVIATION_EPSILON
                or original_std
                <= STANDARD_DEVIATION_EPSILON
            ):
                correlation = 0.0
            else:
                correlation = float(
                    np.corrcoef(
                        new_values,
                        original_values,
                    )[0, 1]
                )

            if abs(correlation) > abs(
                best_correlation
            ):
                best_correlation = correlation
                best_name = original_name

        results.append(
            {
                "feature_name": new_name,
                "most_correlated_existing_feature": (
                    best_name
                ),
                "correlation": best_correlation,
                "absolute_correlation": abs(
                    best_correlation
                ),
            }
        )

    return results


def build_standardised_matrices(
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Standardise every split using training statistics only."""
    mean = np.mean(
        train_raw,
        axis=0,
        keepdims=True,
    )

    standard_deviation = np.std(
        train_raw,
        axis=0,
        keepdims=True,
    )

    standard_deviation = np.where(
        standard_deviation
        < STANDARD_DEVIATION_EPSILON,
        1.0,
        standard_deviation,
    )

    train = (
        train_raw - mean
    ) / standard_deviation

    val = (
        val_raw - mean
    ) / standard_deviation

    test = (
        test_raw - mean
    ) / standard_deviation

    return (
        train,
        val,
        test,
        mean,
        standard_deviation,
    )


def write_feature_name_csv(
    path: Path,
    feature_names: list[str],
) -> None:
    """Write the complete augmented feature-name table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(
        {
            "feature_index": np.arange(
                len(feature_names),
                dtype=int,
            ),
            "feature_name": feature_names,
            "feature_origin": [
                (
                    "measured_ground_derived"
                    if index >= BASE_FEATURE_COUNT
                    else "signal_derived_base"
                )
                for index in range(
                    len(feature_names)
                )
            ],
        }
    )

    frame.to_csv(
        path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_processed/sss_fast_revision/"
            "descriptor_sets/"
            "signal_derived_with_ground/"
            "debug_plus_3000_"
            "signal_derived_with_ground_features.npz"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_processed/sss_fast_revision/"
            "descriptor_sets/"
            "measured_ground_augmented/"
            "debug_plus_3000_"
            "measured_ground_augmented_features.npz"
        ),
    )

    parser.add_argument(
        "--output-feature-names",
        type=Path,
        default=Path(
            "data_processed/sss_fast_revision/"
            "descriptor_sets/"
            "measured_ground_augmented/"
            "debug_plus_3000_"
            "measured_ground_augmented_feature_names.csv"
        ),
    )

    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path(
            "results/sss_fast_revision/"
            "tables/"
            "measured_ground_augmented_"
            "feature_audit.csv"
        ),
    )

    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "configs/sss_fast_revision/"
            "measured_ground_augmented_"
            "build_manifest.json"
        ),
    )

    parser.add_argument(
        "--ratio-epsilon",
        type=float,
        default=DEFAULT_RATIO_EPSILON,
    )

    parser.add_argument(
        "--denominator-threshold",
        type=float,
        default=DEFAULT_DENOMINATOR_THRESHOLD,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. "
            "Use --overwrite to replace it."
        )

    arrays = load_source_dataset(
        args.input
    )

    original_feature_names = [
        str(name)
        for name in arrays[
            "feature_names"
        ].tolist()
    ]

    augmented_raw_by_split: dict[
        str,
        np.ndarray,
    ] = {}

    denominator_audit: dict[
        str,
        Any,
    ] = {}

    all_audit_rows: list[
        dict[str, Any]
    ] = []

    augmented_feature_names: list[str] | None = None

    for split in [
        "train",
        "val",
        "test",
    ]:
        source_raw = np.asarray(
            arrays[f"F_{split}_raw"],
            dtype=np.float64,
        )

        (
            new_raw,
            new_names,
            audit_rows,
            split_denominator_audit,
        ) = derive_augmented_raw_features(
            raw_matrix=source_raw,
            feature_names=original_feature_names,
            ratio_epsilon=args.ratio_epsilon,
            denominator_threshold=(
                args.denominator_threshold
            ),
            split_name=split,
        )

        if augmented_feature_names is None:
            augmented_feature_names = (
                new_names
            )
        elif augmented_feature_names != new_names:
            raise RuntimeError(
                "Augmented feature names changed across splits."
            )

        augmented_raw_by_split[
            split
        ] = np.hstack(
            [
                source_raw,
                new_raw,
            ]
        )

        denominator_audit[
            split
        ] = split_denominator_audit

        all_audit_rows.extend(
            audit_rows
        )

    if augmented_feature_names is None:
        raise RuntimeError(
            "No augmented feature names were generated."
        )

    complete_feature_names = (
        original_feature_names
        + augmented_feature_names
    )

    if len(complete_feature_names) != (
        AUGMENTED_FEATURE_COUNT
    ):
        raise AssertionError(
            f"Expected {AUGMENTED_FEATURE_COUNT} total "
            f"features, found {len(complete_feature_names)}."
        )

    train_new_only = (
        augmented_raw_by_split["train"][
            :,
            BASE_FEATURE_COUNT:,
        ]
    )

    exact_duplicates = (
        find_exact_duplicate_columns(
            original_matrix=np.asarray(
                arrays["F_train_raw"],
                dtype=np.float64,
            ),
            augmented_matrix=train_new_only,
            original_names=original_feature_names,
            augmented_names=augmented_feature_names,
        )
    )

    if exact_duplicates:
        raise ValueError(
            "Exact duplicate augmented features detected: "
            f"{exact_duplicates}"
        )

    correlation_audit = (
        calculate_max_existing_correlation(
            original_matrix=np.asarray(
                arrays["F_train_raw"],
                dtype=np.float64,
            ),
            augmented_matrix=train_new_only,
            original_names=original_feature_names,
            augmented_names=augmented_feature_names,
        )
    )

    (
        F_train,
        F_val,
        F_test,
        F_mean,
        F_std,
    ) = build_standardised_matrices(
        train_raw=augmented_raw_by_split[
            "train"
        ],
        val_raw=augmented_raw_by_split[
            "val"
        ],
        test_raw=augmented_raw_by_split[
            "test"
        ],
    )

    # Verify that rebuilding the first 78 columns does not alter
    # the existing standardised data.
    #
    # 验证重新标准化后原78维数据保持一致。
    base_standardisation_errors = {}

    for split, rebuilt in [
        ("train", F_train),
        ("val", F_val),
        ("test", F_test),
    ]:
        original_standardised = np.asarray(
            arrays[f"F_{split}"],
            dtype=np.float64,
        )

        max_error = float(
            np.max(
                np.abs(
                    rebuilt[
                        :,
                        :BASE_FEATURE_COUNT,
                    ]
                    - original_standardised
                )
            )
        )

        base_standardisation_errors[
            split
        ] = max_error

        if not np.allclose(
            rebuilt[
                :,
                :BASE_FEATURE_COUNT,
            ],
            original_standardised,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"{split}: rebuilding standardisation changed "
                f"the original 78 features; "
                f"max_error={max_error:.6e}."
            )

    for key, matrix in {
        "F_train": F_train,
        "F_val": F_val,
        "F_test": F_test,
        "F_train_raw": (
            augmented_raw_by_split["train"]
        ),
        "F_val_raw": (
            augmented_raw_by_split["val"]
        ),
        "F_test_raw": (
            augmented_raw_by_split["test"]
        ),
        "F_mean": F_mean,
        "F_std": F_std,
    }.items():
        if matrix.shape[-1] != AUGMENTED_FEATURE_COUNT:
            raise ValueError(
                f"{key} has shape {matrix.shape}; "
                f"expected final dimension "
                f"{AUGMENTED_FEATURE_COUNT}."
            )

        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError(
                f"{key} contains NaN or infinity."
            )

    near_zero_std_indices = np.flatnonzero(
        F_std.reshape(-1)
        <= STANDARD_DEVIATION_EPSILON
    ).tolist()

    if near_zero_std_indices:
        raise ValueError(
            "Augmented dataset contains zero-variance features "
            f"at indices: {near_zero_std_indices}"
        )

    output_arrays = {
        key: value
        for key, value in arrays.items()
        if key not in {
            "F_train",
            "F_val",
            "F_test",
            "F_train_raw",
            "F_val_raw",
            "F_test_raw",
            "F_mean",
            "F_std",
            "feature_names",
        }
    }

    output_arrays.update(
        {
            "F_train": F_train,
            "F_val": F_val,
            "F_test": F_test,
            "F_train_raw": (
                augmented_raw_by_split["train"]
            ),
            "F_val_raw": (
                augmented_raw_by_split["val"]
            ),
            "F_test_raw": (
                augmented_raw_by_split["test"]
            ),
            "F_mean": F_mean,
            "F_std": F_std,
            "feature_names": np.asarray(
                complete_feature_names,
                dtype=str,
            ),
        }
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        **output_arrays,
    )

    write_feature_name_csv(
        path=args.output_feature_names,
        feature_names=complete_feature_names,
    )

    args.audit_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_frame = pd.DataFrame(
        all_audit_rows
    )

    correlation_frame = pd.DataFrame(
        correlation_audit
    )

    audit_frame = audit_frame.merge(
        correlation_frame,
        on="feature_name",
        how="left",
        validate="many_to_one",
    )

    audit_frame.to_csv(
        args.audit_csv,
        index=False,
    )

    design_diagnostics = (
        analyse_design_matrix(
            F_train
        )
    )

    manifest = {
        "source_dataset": str(args.input),
        "source_sha256": sha256_file(
            args.input
        ),
        "output_dataset": str(args.output),
        "output_sha256": sha256_file(
            args.output
        ),
        "output_feature_names": str(
            args.output_feature_names
        ),
        "audit_csv": str(args.audit_csv),
        "base_feature_count": (
            BASE_FEATURE_COUNT
        ),
        "added_feature_count": len(
            augmented_feature_names
        ),
        "total_feature_count": len(
            complete_feature_names
        ),
        "ratio_epsilon": (
            args.ratio_epsilon
        ),
        "denominator_threshold": (
            args.denominator_threshold
        ),
        "added_feature_names": (
            augmented_feature_names
        ),
        "added_feature_formulas": {
            name: (
                (
                    "story dominant frequency / "
                    "measured-ground dominant frequency"
                )
                if "dominant_frequency" in name
                else (
                    "story spectral centroid / "
                    "measured-ground spectral centroid"
                )
            )
            for name in augmented_feature_names
        },
        "denominator_audit": (
            denominator_audit
        ),
        "exact_duplicate_columns": (
            exact_duplicates
        ),
        "maximum_existing_correlations": (
            correlation_audit
        ),
        "base_standardisation_max_abs_error": (
            base_standardisation_errors
        ),
        "design_matrix_diagnostics": (
            design_diagnostics
        ),
        "matrix_shapes": {
            "F_train": list(F_train.shape),
            "F_val": list(F_val.shape),
            "F_test": list(F_test.shape),
            "F_train_raw": list(
                augmented_raw_by_split[
                    "train"
                ].shape
            ),
            "F_val_raw": list(
                augmented_raw_by_split[
                    "val"
                ].shape
            ),
            "F_test_raw": list(
                augmented_raw_by_split[
                    "test"
                ].shape
            ),
            "F_mean": list(F_mean.shape),
            "F_std": list(F_std.shape),
        },
    }

    args.manifest_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.manifest_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "===== MEASURED-GROUND AUGMENTED DATASET ====="
    )
    print(
        "Source feature count:",
        BASE_FEATURE_COUNT,
    )
    print(
        "Added feature count:",
        len(augmented_feature_names),
    )
    print(
        "Total feature count:",
        len(complete_feature_names),
    )

    print()
    print(
        "===== ADDED FEATURES ====="
    )

    for index, name in enumerate(
        augmented_feature_names,
        start=BASE_FEATURE_COUNT,
    ):
        print(f"{index:3d}  {name}")

    print()
    print(
        "===== DENOMINATOR AUDIT ====="
    )

    for split, split_audit in (
        denominator_audit.items()
    ):
        print()
        print(split)

        for name, statistics in (
            split_audit.items()
        ):
            print(
                f"  {name}: "
                f"min={statistics['minimum']:.10f}, "
                f"p01={statistics['p01']:.10f}, "
                f"median={statistics['median']:.10f}, "
                f"max={statistics['maximum']:.10f}, "
                f"count<=1e-8="
                f"{statistics['count_abs_le_1e_minus_8']}"
            )

    print()
    print(
        "===== AUGMENTED FEATURE RANGE ====="
    )

    train_audit = audit_frame.loc[
        audit_frame["split"] == "train"
    ]

    print(
        train_audit[
            [
                "feature_name",
                "minimum",
                "p01",
                "median",
                "p99",
                "maximum",
                "standard_deviation",
                "most_correlated_existing_feature",
                "absolute_correlation",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "===== STANDARDISATION CHECK ====="
    )

    for split, error in (
        base_standardisation_errors.items()
    ):
        print(
            f"{split}: max_abs_error="
            f"{error:.12e}"
        )

    print()
    print(
        "===== DESIGN MATRIX DIAGNOSTICS ====="
    )

    for key, value in (
        design_diagnostics.items()
    ):
        print(f"{key}: {value}")

    print()
    print(
        "CHECK PASSED: measured-ground-augmented "
        "dataset was built and validated."
    )
    print("Output:", args.output)
    print(
        "Feature names:",
        args.output_feature_names,
    )
    print("Audit table:", args.audit_csv)
    print(
        "Manifest:",
        args.manifest_output,
    )


if __name__ == "__main__":
    main()
