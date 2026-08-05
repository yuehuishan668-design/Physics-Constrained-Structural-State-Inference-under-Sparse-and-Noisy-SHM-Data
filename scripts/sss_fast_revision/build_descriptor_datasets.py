"""
Build internally consistent descriptor datasets for the SSS revision.

The script filters every feature-dimension-dependent array, including:
- F_train / F_val / F_test
- F_train_raw / F_val_raw / F_test_raw
- F_mean / F_std
- embedded feature_names

It copies labels, case IDs and other non-feature arrays unchanged.

为SSS修订版生成内部一致的四套描述符数据。
所有依赖特征维数的数组将同步筛选。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing.descriptor_sets import (  # noqa: E402
    DescriptorSetName,
    get_descriptor_indices,
)


FEATURE_MATRIX_KEYS = {
    "F_train",
    "F_val",
    "F_test",
    "F_train_raw",
    "F_val_raw",
    "F_test_raw",
}

FEATURE_STATISTIC_KEYS = {
    "F_mean",
    "F_std",
}

FEATURE_NAME_KEYS = {
    "feature_names",
}


def resolve_name_column(frame: pd.DataFrame) -> str:
    """Resolve the descriptor-name column."""
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


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for one file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def subset_feature_array(
    key: str,
    array: np.ndarray,
    selected_indices: np.ndarray,
    original_feature_count: int,
) -> np.ndarray:
    """
    Filter arrays whose axes represent descriptors.

    筛选所有以描述符为列或元素的数组。
    """
    if key in FEATURE_MATRIX_KEYS:
        if array.ndim != 2:
            raise ValueError(
                f"{key} must be 2D, got shape={array.shape}"
            )

        if array.shape[1] != original_feature_count:
            raise ValueError(
                f"{key} has {array.shape[1]} columns, expected "
                f"{original_feature_count}."
            )

        return array[:, selected_indices]

    if key in FEATURE_STATISTIC_KEYS:
        if array.ndim == 1:
            if array.shape[0] != original_feature_count:
                raise ValueError(
                    f"{key} has shape={array.shape}, expected "
                    f"({original_feature_count},)."
                )

            return array[selected_indices]

        if array.ndim == 2:
            if array.shape[1] != original_feature_count:
                raise ValueError(
                    f"{key} has shape={array.shape}, expected "
                    f"second dimension={original_feature_count}."
                )

            return array[:, selected_indices]

        raise ValueError(
            f"{key} must be 1D or 2D, got shape={array.shape}"
        )

    if key in FEATURE_NAME_KEYS:
        if array.ndim != 1:
            raise ValueError(
                f"{key} must be 1D, got shape={array.shape}"
            )

        if array.shape[0] != original_feature_count:
            raise ValueError(
                f"{key} has {array.shape[0]} entries, expected "
                f"{original_feature_count}."
            )

        return array[selected_indices]

    return array


def normalize_statistic_shape(
    array: np.ndarray,
) -> np.ndarray:
    """Convert feature statistics to shape (1, n_features)."""
    result = np.asarray(array, dtype=np.float64)

    if result.ndim == 1:
        result = result.reshape(1, -1)

    if result.ndim != 2 or result.shape[0] != 1:
        raise ValueError(
            "Feature statistic must have shape (n_features,) "
            f"or (1, n_features), got {result.shape}."
        )

    return result


def validate_filtered_dataset(
    arrays: dict[str, np.ndarray],
    expected_feature_count: int,
    expected_feature_names: list[str],
) -> dict[str, Any]:
    """
    Validate shapes, finite values and standardisation consistency.

    验证维数、有限值和标准化一致性。
    """
    required_keys = {
        "F_train",
        "F_val",
        "F_test",
        "y_train",
        "y_val",
        "y_test",
    }

    missing = sorted(required_keys - set(arrays))

    if missing:
        raise KeyError(
            f"Filtered dataset is missing required keys: {missing}"
        )

    shape_summary: dict[str, list[int]] = {}

    for key in sorted(FEATURE_MATRIX_KEYS):
        if key not in arrays:
            continue

        array = np.asarray(arrays[key])

        if array.ndim != 2:
            raise ValueError(
                f"{key} must be 2D, got shape={array.shape}"
            )

        if array.shape[1] != expected_feature_count:
            raise ValueError(
                f"{key} has {array.shape[1]} features, expected "
                f"{expected_feature_count}."
            )

        if not np.all(np.isfinite(array)):
            raise FloatingPointError(
                f"{key} contains NaN or infinity."
            )

        shape_summary[key] = list(array.shape)

    if "feature_names" in arrays:
        embedded_names = [
            str(name)
            for name in np.asarray(arrays["feature_names"]).tolist()
        ]

        if embedded_names != expected_feature_names:
            raise ValueError(
                "Embedded feature names do not match selected names."
            )

    standardisation_errors: dict[str, float] = {}

    standardisation_keys = {
        "F_mean",
        "F_std",
        "F_train_raw",
        "F_val_raw",
        "F_test_raw",
        "F_train",
        "F_val",
        "F_test",
    }

    if standardisation_keys.issubset(arrays):
        mean = normalize_statistic_shape(arrays["F_mean"])
        std = normalize_statistic_shape(arrays["F_std"])

        if mean.shape[1] != expected_feature_count:
            raise ValueError(
                f"F_mean has {mean.shape[1]} features, expected "
                f"{expected_feature_count}."
            )

        if std.shape[1] != expected_feature_count:
            raise ValueError(
                f"F_std has {std.shape[1]} features, expected "
                f"{expected_feature_count}."
            )

        if np.any(std <= 0.0):
            raise ValueError(
                "F_std contains non-positive values."
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

            reconstructed = (raw - mean) / std

            max_error = float(
                np.max(
                    np.abs(
                        reconstructed - standardised
                    )
                )
            )

            standardisation_errors[split] = max_error

            if not np.allclose(
                reconstructed,
                standardised,
                rtol=1e-7,
                atol=1e-9,
            ):
                raise ValueError(
                    f"Standardisation consistency check failed "
                    f"for {split}: max_error={max_error:.6e}"
                )

    label_shapes = {
        key: list(np.asarray(arrays[key]).shape)
        for key in ["y_train", "y_val", "y_test"]
    }

    return {
        "feature_count": expected_feature_count,
        "feature_matrix_shapes": shape_summary,
        "label_shapes": label_shapes,
        "standardisation_max_abs_error": (
            standardisation_errors
        ),
    }


def write_feature_names(
    output_path: Path,
    selected_names: list[str],
) -> None:
    """Write selected descriptor names."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(
        {
            "feature_index": np.arange(
                len(selected_names),
                dtype=int,
            ),
            "feature_name": selected_names,
        }
    )

    frame.to_csv(
        output_path,
        index=False,
    )


def build_one_dataset(
    source_arrays: dict[str, np.ndarray],
    original_feature_names: list[str],
    descriptor_set: DescriptorSetName,
    output_root: Path,
    source_stem: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Build and validate one descriptor dataset."""
    selected_indices = get_descriptor_indices(
        original_feature_names,
        descriptor_set,
    )

    selected_names = [
        original_feature_names[int(index)]
        for index in selected_indices
    ]

    set_root = output_root / descriptor_set.value
    set_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_npz = (
        set_root
        / f"{source_stem}_{descriptor_set.value}_features.npz"
    )

    output_names = (
        set_root
        / f"{source_stem}_{descriptor_set.value}"
        "_feature_names.csv"
    )

    if output_npz.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_npz}. "
            "Use --overwrite to replace it."
        )

    filtered_arrays: dict[str, np.ndarray] = {}

    for key, array in source_arrays.items():
        filtered_arrays[key] = subset_feature_array(
            key=key,
            array=np.asarray(array),
            selected_indices=selected_indices,
            original_feature_count=len(
                original_feature_names
            ),
        )

    # Ensure the embedded names are canonical even if the input
    # NPZ did not contain feature_names.
    filtered_arrays["feature_names"] = np.asarray(
        selected_names,
        dtype=str,
    )

    validation = validate_filtered_dataset(
        arrays=filtered_arrays,
        expected_feature_count=len(selected_names),
        expected_feature_names=selected_names,
    )

    np.savez_compressed(
        output_npz,
        **filtered_arrays,
    )

    write_feature_names(
        output_path=output_names,
        selected_names=selected_names,
    )

    return {
        "descriptor_set": descriptor_set.value,
        "n_features": len(selected_names),
        "selected_original_indices": [
            int(index)
            for index in selected_indices
        ],
        "output_npz": str(output_npz),
        "output_feature_names": str(output_names),
        "output_npz_size_bytes": output_npz.stat().st_size,
        "output_npz_sha256": sha256_file(output_npz),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
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
        "--output-root",
        type=Path,
        default=Path(
            "data_processed/sss_fast_revision/"
            "descriptor_sets"
        ),
    )

    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "configs/sss_fast_revision/"
            "descriptor_dataset_build_manifest.json"
        ),
    )

    parser.add_argument(
        "--source-stem",
        type=str,
        default="debug_plus_3000",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if not args.feature_names.is_file():
        raise FileNotFoundError(args.feature_names)

    name_frame = pd.read_csv(args.feature_names)
    name_column = resolve_name_column(name_frame)

    original_feature_names = (
        name_frame[name_column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(original_feature_names) != 92:
        raise ValueError(
            f"Expected 92 original features, found "
            f"{len(original_feature_names)}."
        )

    with np.load(
        args.input,
        allow_pickle=False,
    ) as loaded:
        source_arrays = {
            key: loaded[key]
            for key in loaded.files
        }

    print("===== SOURCE DATASET =====")
    print("Input:", args.input)
    print("Feature-name file:", args.feature_names)
    print("Available keys:", sorted(source_arrays))
    print("Original feature count:", len(original_feature_names))

    build_results: dict[str, Any] = {}

    for descriptor_set in DescriptorSetName:
        result = build_one_dataset(
            source_arrays=source_arrays,
            original_feature_names=original_feature_names,
            descriptor_set=descriptor_set,
            output_root=args.output_root,
            source_stem=args.source_stem,
            overwrite=args.overwrite,
        )

        build_results[descriptor_set.value] = result

        print()
        print(
            f"===== {descriptor_set.value} ====="
        )
        print(
            "Feature count:",
            result["n_features"],
        )
        print(
            "Output:",
            result["output_npz"],
        )
        print(
            "SHA-256:",
            result["output_npz_sha256"],
        )
        print(
            "Standardisation error:",
            result["validation"][
                "standardisation_max_abs_error"
            ],
        )

    manifest = {
        "source_npz": str(args.input),
        "source_npz_sha256": sha256_file(args.input),
        "source_feature_names": str(args.feature_names),
        "source_feature_names_sha256": sha256_file(
            args.feature_names
        ),
        "source_feature_count": len(
            original_feature_names
        ),
        "outputs": build_results,
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

    print()
    print(
        "CHECK PASSED: all descriptor datasets "
        "were built and validated."
    )
    print("Manifest:", args.manifest_output)


if __name__ == "__main__":
    main()
