"""
Historical 92D descriptor reproduction lock.

Purpose
-------
Reconstruct the historical 92D physics-guided descriptor dataset
directly from the original 3000-case raw dataset and verify that the
current extractor reproduces the frozen AES feature dataset.

This is a provenance/integrity experiment only.

IMPORTANT
---------
Historical extractor priority was:

    X_abs_accel
    X_clean_abs_accel
    X

Since the raw dataset contains X_abs_accel, historical descriptors
were extracted from the stored mixed-noise response X_abs_accel.

No new noise is generated here.
No model is trained here.
No model-selection decision is made here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

EXPECTED_CASES = 3000
EXPECTED_FEATURES = 92


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

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


def load_module(
    path: Path,
):

    spec = (
        importlib.util
        .spec_from_file_location(
            "historical_feature_extractor",
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import extractor: {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def infer_cli_default(
    source_path: Path,
    destination: str,
) -> Any:
    """
    Recover argparse default from source code.

    Example:
        --min-fft-freq
        -> destination min_fft_freq
    """

    source = source_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        function = node.func

        if not (
            isinstance(
                function,
                ast.Attribute,
            )
            and function.attr
            == "add_argument"
        ):
            continue

        option_strings = []

        for argument in node.args:

            try:
                value = ast.literal_eval(
                    argument
                )
            except Exception:
                continue

            if isinstance(
                value,
                str,
            ):
                option_strings.append(
                    value
                )

        explicit_dest = None
        default_node = None

        for keyword in (
            node.keywords
        ):

            if (
                keyword.arg
                == "dest"
            ):

                try:
                    explicit_dest = (
                        ast.literal_eval(
                            keyword.value
                        )
                    )
                except Exception:
                    pass

            if (
                keyword.arg
                == "default"
            ):
                default_node = (
                    keyword.value
                )

        if (
            explicit_dest
            is not None
        ):

            inferred_dest = str(
                explicit_dest
            )

        else:

            long_options = [
                option
                for option
                in option_strings
                if option.startswith(
                    "--"
                )
            ]

            if not long_options:
                continue

            inferred_dest = (
                long_options[0][2:]
                .replace(
                    "-",
                    "_",
                )
            )

        if (
            inferred_dest
            != destination
        ):
            continue

        if default_node is None:

            raise RuntimeError(
                f"Argument {destination} "
                "has no explicit default."
            )

        try:

            return ast.literal_eval(
                default_node
            )

        except Exception as exc:

            raise RuntimeError(
                "Could not statically recover "
                f"default for {destination}."
            ) from exc

    raise KeyError(
        f"Could not locate argparse "
        f"destination: {destination}"
    )


def numeric_comparison(
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float,
    rtol: float,
) -> dict[str, Any]:

    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    expected = np.asarray(
        expected,
        dtype=np.float64,
    )

    if (
        actual.shape
        != expected.shape
    ):

        return {
            "shape_match": False,
            "actual_shape": list(
                actual.shape
            ),
            "expected_shape": list(
                expected.shape
            ),
            "allclose": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_scaled_error": None,
        }

    difference = (
        actual
        - expected
    )

    abs_difference = np.abs(
        difference
    )

    scale = np.maximum(
        1.0,
        np.abs(
            expected
        ),
    )

    scaled = (
        abs_difference
        / scale
    )

    return {
        "shape_match": True,
        "actual_shape": list(
            actual.shape
        ),
        "expected_shape": list(
            expected.shape
        ),
        "allclose": bool(
            np.allclose(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
        ),
        "max_abs_error": float(
            np.max(
                abs_difference
            )
        ),
        "mean_abs_error": float(
            np.mean(
                abs_difference
            )
        ),
        "max_scaled_error": float(
            np.max(
                scaled
            )
        ),
    }


def exact_array_match(
    actual: np.ndarray,
    expected: np.ndarray,
) -> bool:

    return bool(
        np.array_equal(
            np.asarray(
                actual
            ),
            np.asarray(
                expected
            ),
        )
    )


def main() -> None:

    raw_env = os.environ.get(
        "RAW3000"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=(
            Path(raw_env)
            if raw_env
            else None
        ),
    )

    parser.add_argument(
        "--frozen-features",
        type=Path,
        default=Path(
            "data_inputs/"
            "aes_frozen/"
            "debug_plus_3000_"
            "physics_features_mlp.npz"
        ),
    )

    parser.add_argument(
        "--extractor",
        type=Path,
        default=Path(
            "src/"
            "preprocessing/"
            "extract_physics_features.py"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "sss_fast_revision/"
            "historical_92d_reproduction"
        ),
    )

    parser.add_argument(
        "--min-fft-freq",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--max-fft-freq",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
    )

    args = parser.parse_args()

    if args.raw_dataset is None:

        raise RuntimeError(
            "Raw dataset path is missing.\n"
            "Set RAW3000 or pass "
            "--raw-dataset explicitly."
        )

    raw_path = (
        args.raw_dataset
        .expanduser()
        .resolve()
    )

    frozen_path = (
        args.frozen_features
        .expanduser()
        .resolve()
    )

    extractor_path = (
        args.extractor
        .expanduser()
        .resolve()
    )

    for path in [
        raw_path,
        frozen_path,
        extractor_path,
    ]:

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Recover extractor numerical defaults.
    # --------------------------------------------------

    min_fft_freq = (
        args.min_fft_freq
        if args.min_fft_freq
        is not None
        else float(
            infer_cli_default(
                extractor_path,
                "min_fft_freq",
            )
        )
    )

    max_fft_freq = (
        args.max_fft_freq
        if args.max_fft_freq
        is not None
        else float(
            infer_cli_default(
                extractor_path,
                "max_fft_freq",
            )
        )
    )

    eps = (
        args.eps
        if args.eps
        is not None
        else float(
            infer_cli_default(
                extractor_path,
                "eps",
            )
        )
    )

    extractor = load_module(
        extractor_path
    )

    print(
        "=" * 80
    )

    print(
        "HISTORICAL 92D REPRODUCTION LOCK"
    )

    print(
        "=" * 80
    )

    print(
        "Raw dataset:",
        raw_path,
    )

    print(
        "Frozen features:",
        frozen_path,
    )

    print(
        "Extractor:",
        extractor_path,
    )

    print(
        "Extractor SHA256:",
        sha256_file(
            extractor_path
        ),
    )

    print()

    print(
        "Recovered numerical settings:"
    )

    print(
        "  min_fft_freq:",
        min_fft_freq,
    )

    print(
        "  max_fft_freq:",
        max_fft_freq,
    )

    print(
        "  eps:",
        eps,
    )

    print()

    # --------------------------------------------------
    # Load frozen reference.
    # --------------------------------------------------

    with np.load(
        frozen_path,
        allow_pickle=False,
    ) as frozen:

        frozen_data = {
            key: np.asarray(
                frozen[key]
            )
            for key
            in frozen.files
        }

    train_idx = np.asarray(
        frozen_data[
            "train_idx"
        ],
        dtype=np.int64,
    )

    val_idx = np.asarray(
        frozen_data[
            "val_idx"
        ],
        dtype=np.int64,
    )

    test_idx = np.asarray(
        frozen_data[
            "test_idx"
        ],
        dtype=np.int64,
    )

    healthy_frequency = float(
        frozen_data[
            "healthy_first_frequency_hz"
        ]
    )

    frozen_dt = float(
        frozen_data[
            "dt"
        ]
    )

    frozen_names = [
        str(name)
        for name
        in frozen_data[
            "feature_names"
        ].tolist()
    ]

    # --------------------------------------------------
    # Split integrity.
    # --------------------------------------------------

    combined_idx = np.concatenate(
        [
            train_idx,
            val_idx,
            test_idx,
        ]
    )

    split_integrity = bool(
        len(train_idx)
        == 2100
        and len(val_idx)
        == 450
        and len(test_idx)
        == 450
        and len(
            np.unique(
                combined_idx
            )
        )
        == EXPECTED_CASES
        and int(
            np.min(
                combined_idx
            )
        )
        == 0
        and int(
            np.max(
                combined_idx
            )
        )
        == (
            EXPECTED_CASES
            - 1
        )
    )

    if not split_integrity:

        raise RuntimeError(
            "Frozen split integrity failed."
        )

    # --------------------------------------------------
    # Load historical raw arrays.
    # IMPORTANT:
    # explicitly choose X_abs_accel.
    # --------------------------------------------------

    print(
        "Loading historical raw arrays ..."
    )

    with np.load(
        raw_path,
        allow_pickle=False,
    ) as raw:

        required_raw = {
            "X_abs_accel",
            "ground_accel",
            "y_damage",
            "amplitude_g",
            "frequency_hz",
            "noise_level",
            "case_id",
            "dt",
        }

        missing = sorted(
            required_raw
            - set(
                raw.files
            )
        )

        if missing:

            raise RuntimeError(
                f"Raw dataset missing: "
                f"{missing}"
            )

        response = np.asarray(
            raw[
                "X_abs_accel"
            ],
            dtype=np.float64,
        )

        ground = np.asarray(
            raw[
                "ground_accel"
            ],
            dtype=np.float64,
        )

        y_damage = np.asarray(
            raw[
                "y_damage"
            ],
            dtype=np.float64,
        )

        amplitude_g = np.asarray(
            raw[
                "amplitude_g"
            ],
            dtype=np.float64,
        ).reshape(
            -1
        )

        frequency_hz = np.asarray(
            raw[
                "frequency_hz"
            ],
            dtype=np.float64,
        ).reshape(
            -1
        )

        noise_level = np.asarray(
            raw[
                "noise_level"
            ],
            dtype=np.float64,
        ).reshape(
            -1
        )

        case_id = np.asarray(
            raw[
                "case_id"
            ],
            dtype=np.int64,
        ).reshape(
            -1
        )

        raw_dt = float(
            raw[
                "dt"
            ]
        )

    if (
        response.shape[0]
        != EXPECTED_CASES
    ):

        raise RuntimeError(
            "Unexpected response case count."
        )

    response = (
        extractor
        .ensure_3d_response(
            response
        )
    )

    ground = (
        extractor
        .ensure_ground_shape(
            ground,
            response.shape[0],
            response.shape[1],
        )
    )

    print(
        "Response source:",
        "X_abs_accel",
    )

    print(
        "Response shape:",
        response.shape,
    )

    print(
        "Ground shape:",
        ground.shape,
    )

    print(
        "Raw dt:",
        raw_dt,
    )

    print(
        "Frozen dt:",
        frozen_dt,
    )

    print(
        "Healthy first frequency:",
        healthy_frequency,
    )

    print()

    dt_match = bool(
        np.isclose(
            raw_dt,
            frozen_dt,
            rtol=0.0,
            atol=0.0,
        )
    )

    if not dt_match:

        raise RuntimeError(
            "Raw/frozen dt mismatch."
        )

    # --------------------------------------------------
    # Metadata and target locks BEFORE extraction.
    # --------------------------------------------------

    metadata_checks = {
        "y_train_exact": (
            exact_array_match(
                y_damage[
                    train_idx
                ],
                frozen_data[
                    "y_train"
                ],
            )
        ),
        "y_val_exact": (
            exact_array_match(
                y_damage[
                    val_idx
                ],
                frozen_data[
                    "y_val"
                ],
            )
        ),
        "y_test_exact": (
            exact_array_match(
                y_damage[
                    test_idx
                ],
                frozen_data[
                    "y_test"
                ],
            )
        ),
        "case_id_train_exact": (
            exact_array_match(
                case_id[
                    train_idx
                ],
                frozen_data[
                    "case_id_train"
                ],
            )
        ),
        "case_id_val_exact": (
            exact_array_match(
                case_id[
                    val_idx
                ],
                frozen_data[
                    "case_id_val"
                ],
            )
        ),
        "case_id_test_exact": (
            exact_array_match(
                case_id[
                    test_idx
                ],
                frozen_data[
                    "case_id_test"
                ],
            )
        ),
        "amplitude_train_exact": (
            exact_array_match(
                amplitude_g[
                    train_idx
                ],
                frozen_data[
                    "amplitude_g_train"
                ],
            )
        ),
        "amplitude_val_exact": (
            exact_array_match(
                amplitude_g[
                    val_idx
                ],
                frozen_data[
                    "amplitude_g_val"
                ],
            )
        ),
        "amplitude_test_exact": (
            exact_array_match(
                amplitude_g[
                    test_idx
                ],
                frozen_data[
                    "amplitude_g_test"
                ],
            )
        ),
        "frequency_train_exact": (
            exact_array_match(
                frequency_hz[
                    train_idx
                ],
                frozen_data[
                    "frequency_hz_train"
                ],
            )
        ),
        "frequency_val_exact": (
            exact_array_match(
                frequency_hz[
                    val_idx
                ],
                frozen_data[
                    "frequency_hz_val"
                ],
            )
        ),
        "frequency_test_exact": (
            exact_array_match(
                frequency_hz[
                    test_idx
                ],
                frozen_data[
                    "frequency_hz_test"
                ],
            )
        ),
        "noise_train_exact": (
            exact_array_match(
                noise_level[
                    train_idx
                ],
                frozen_data[
                    "noise_level_train"
                ],
            )
        ),
        "noise_val_exact": (
            exact_array_match(
                noise_level[
                    val_idx
                ],
                frozen_data[
                    "noise_level_val"
                ],
            )
        ),
        "noise_test_exact": (
            exact_array_match(
                noise_level[
                    test_idx
                ],
                frozen_data[
                    "noise_level_test"
                ],
            )
        ),
    }

    metadata_all_passed = bool(
        all(
            metadata_checks.values()
        )
    )

    print(
        "=" * 80
    )

    print(
        "METADATA / SPLIT LOCK"
    )

    print(
        "=" * 80
    )

    print(
        "Split integrity:",
        split_integrity,
    )

    print(
        "dt exact:",
        dt_match,
    )

    for (
        name,
        passed,
    ) in metadata_checks.items():

        print(
            f"{name}:",
            passed,
        )

    if not metadata_all_passed:

        raise RuntimeError(
            "STOP: historical metadata "
            "does not align with frozen "
            "feature dataset."
        )

    # --------------------------------------------------
    # Re-extract historical 92D features.
    # --------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "RE-EXTRACTING HISTORICAL 92D"
    )

    print(
        "=" * 80
    )

    print(
        "This may take several minutes."
    )

    start = (
        time.perf_counter()
    )

    (
        F_all_raw,
        reproduced_names,
    ) = (
        extractor
        .extract_all_features(
            response=response,
            ground=ground,
            amplitude_g=amplitude_g,
            frequency_hz=frequency_hz,
            noise_level=noise_level,
            dt=raw_dt,
            healthy_first_frequency_hz=(
                healthy_frequency
            ),
            min_fft_freq=(
                min_fft_freq
            ),
            max_fft_freq=(
                max_fft_freq
            ),
            eps=eps,
        )
    )

    elapsed = float(
        time.perf_counter()
        - start
    )

    F_all_raw = np.asarray(
        F_all_raw,
        dtype=np.float64,
    )

    reproduced_names = [
        str(name)
        for name
        in reproduced_names
    ]

    print(
        "Extraction complete."
    )

    print(
        "Elapsed seconds:",
        f"{elapsed:.3f}",
    )

    print(
        "Reproduced shape:",
        F_all_raw.shape,
    )

    if (
        F_all_raw.shape
        != (
            EXPECTED_CASES,
            EXPECTED_FEATURES,
        )
    ):

        raise RuntimeError(
            "Unexpected reproduced "
            "feature shape."
        )

    feature_name_match = bool(
        reproduced_names
        == frozen_names
    )

    print(
        "Feature names exact:",
        feature_name_match,
    )

    if not feature_name_match:

        mismatch_rows = []

        for index, (
            reproduced,
            frozen,
        ) in enumerate(
            zip(
                reproduced_names,
                frozen_names,
            )
        ):

            if reproduced != frozen:

                mismatch_rows.append(
                    {
                        "index": index,
                        "reproduced": (
                            reproduced
                        ),
                        "frozen": frozen,
                    }
                )

        mismatch_path = (
            args.output_root
            / "feature_name_mismatches.csv"
        )

        pd.DataFrame(
            mismatch_rows
        ).to_csv(
            mismatch_path,
            index=False,
        )

        raise RuntimeError(
            "STOP: feature-name sequence "
            "does not match frozen 92D."
        )

    # --------------------------------------------------
    # Compare RAW descriptors.
    # --------------------------------------------------

    F_train_raw = (
        F_all_raw[
            train_idx
        ]
    )

    F_val_raw = (
        F_all_raw[
            val_idx
        ]
    )

    F_test_raw = (
        F_all_raw[
            test_idx
        ]
    )

    raw_comparisons = {
        "F_train_raw": numeric_comparison(
            F_train_raw,
            frozen_data[
                "F_train_raw"
            ],
            args.atol,
            args.rtol,
        ),
        "F_val_raw": numeric_comparison(
            F_val_raw,
            frozen_data[
                "F_val_raw"
            ],
            args.atol,
            args.rtol,
        ),
        "F_test_raw": numeric_comparison(
            F_test_raw,
            frozen_data[
                "F_test_raw"
            ],
            args.atol,
            args.rtol,
        ),
    }

    # --------------------------------------------------
    # Reproduce train-only standardisation.
    # --------------------------------------------------

    (
        F_train,
        F_val,
        F_test,
        F_mean,
        F_std,
    ) = (
        extractor
        .standardize_by_train(
            F_train_raw,
            F_val_raw,
            F_test_raw,
            eps,
        )
    )

    standardised_comparisons = {
        "F_train": numeric_comparison(
            F_train,
            frozen_data[
                "F_train"
            ],
            args.atol,
            args.rtol,
        ),
        "F_val": numeric_comparison(
            F_val,
            frozen_data[
                "F_val"
            ],
            args.atol,
            args.rtol,
        ),
        "F_test": numeric_comparison(
            F_test,
            frozen_data[
                "F_test"
            ],
            args.atol,
            args.rtol,
        ),
        "F_mean": numeric_comparison(
            np.asarray(
                F_mean
            ).reshape(
                1,
                -1,
            ),
            frozen_data[
                "F_mean"
            ],
            args.atol,
            args.rtol,
        ),
        "F_std": numeric_comparison(
            np.asarray(
                F_std
            ).reshape(
                1,
                -1,
            ),
            frozen_data[
                "F_std"
            ],
            args.atol,
            args.rtol,
        ),
    }

    # --------------------------------------------------
    # Per-feature error audit.
    # --------------------------------------------------

    reproduced_raw = np.vstack(
        [
            F_train_raw,
            F_val_raw,
            F_test_raw,
        ]
    )

    frozen_raw = np.vstack(
        [
            frozen_data[
                "F_train_raw"
            ],
            frozen_data[
                "F_val_raw"
            ],
            frozen_data[
                "F_test_raw"
            ],
        ]
    )

    reproduced_std = np.vstack(
        [
            F_train,
            F_val,
            F_test,
        ]
    )

    frozen_std = np.vstack(
        [
            frozen_data[
                "F_train"
            ],
            frozen_data[
                "F_val"
            ],
            frozen_data[
                "F_test"
            ],
        ]
    )

    feature_rows = []

    for feature_index, feature_name in enumerate(
        frozen_names
    ):

        raw_diff = np.abs(
            reproduced_raw[
                :,
                feature_index,
            ]
            - frozen_raw[
                :,
                feature_index,
            ]
        )

        std_diff = np.abs(
            reproduced_std[
                :,
                feature_index,
            ]
            - frozen_std[
                :,
                feature_index,
            ]
        )

        feature_rows.append(
            {
                "feature_index": (
                    feature_index
                ),
                "feature_name": (
                    feature_name
                ),
                "raw_max_abs_error": float(
                    np.max(
                        raw_diff
                    )
                ),
                "raw_mean_abs_error": float(
                    np.mean(
                        raw_diff
                    )
                ),
                "standardized_max_abs_error": float(
                    np.max(
                        std_diff
                    )
                ),
                "standardized_mean_abs_error": float(
                    np.mean(
                        std_diff
                    )
                ),
            }
        )

    feature_error_frame = (
        pd.DataFrame(
            feature_rows
        )
        .sort_values(
            "raw_max_abs_error",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    feature_error_path = (
        args.output_root
        / "per_feature_reproduction_error.csv"
    )

    feature_error_frame.to_csv(
        feature_error_path,
        index=False,
    )

    # --------------------------------------------------
    # Overall PASS/FAIL.
    # --------------------------------------------------

    raw_allclose = bool(
        all(
            comparison[
                "allclose"
            ]
            for comparison
            in raw_comparisons.values()
        )
    )

    standardized_allclose = bool(
        all(
            comparison[
                "allclose"
            ]
            for comparison
            in standardised_comparisons.values()
        )
    )

    overall_passed = bool(
        split_integrity
        and dt_match
        and metadata_all_passed
        and feature_name_match
        and raw_allclose
        and standardized_allclose
    )

    # --------------------------------------------------
    # Save report.
    # --------------------------------------------------

    report = {
        "experiment": (
            "historical_92D_"
            "descriptor_reproduction_lock"
        ),
        "raw_dataset": str(
            raw_path
        ),
        "frozen_feature_dataset": str(
            frozen_path
        ),
        "extractor": str(
            extractor_path
        ),
        "extractor_sha256": (
            sha256_file(
                extractor_path
            )
        ),
        "response_source": (
            "X_abs_accel"
        ),
        "historical_distribution": (
            "mixed measurement-noise "
            "response distribution"
        ),
        "numerical_settings": {
            "dt": raw_dt,
            "healthy_first_frequency_hz": (
                healthy_frequency
            ),
            "min_fft_freq": (
                min_fft_freq
            ),
            "max_fft_freq": (
                max_fft_freq
            ),
            "eps": eps,
            "comparison_atol": (
                args.atol
            ),
            "comparison_rtol": (
                args.rtol
            ),
        },
        "checks": {
            "split_integrity": (
                split_integrity
            ),
            "dt_match": (
                dt_match
            ),
            "metadata": (
                metadata_checks
            ),
            "feature_names_exact": (
                feature_name_match
            ),
            "raw_allclose": (
                raw_allclose
            ),
            "standardized_allclose": (
                standardized_allclose
            ),
        },
        "raw_comparisons": (
            raw_comparisons
        ),
        "standardized_comparisons": (
            standardised_comparisons
        ),
        "elapsed_seconds": (
            elapsed
        ),
        "overall_passed": (
            overall_passed
        ),
    }

    report_path = (
        args.output_root
        / "historical_92d_reproduction_report.json"
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

    # --------------------------------------------------
    # Console summary.
    # --------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "RAW FEATURE REPRODUCTION"
    )

    print(
        "=" * 80
    )

    for (
        name,
        comparison,
    ) in raw_comparisons.items():

        print(
            name,
        )

        print(
            "  allclose:",
            comparison[
                "allclose"
            ],
        )

        print(
            "  max abs error:",
            comparison[
                "max_abs_error"
            ],
        )

        print(
            "  mean abs error:",
            comparison[
                "mean_abs_error"
            ],
        )

        print(
            "  max scaled error:",
            comparison[
                "max_scaled_error"
            ],
        )

    print()
    print(
        "=" * 80
    )

    print(
        "STANDARDIZED FEATURE REPRODUCTION"
    )

    print(
        "=" * 80
    )

    for (
        name,
        comparison,
    ) in (
        standardised_comparisons
        .items()
    ):

        print(
            name,
        )

        print(
            "  allclose:",
            comparison[
                "allclose"
            ],
        )

        print(
            "  max abs error:",
            comparison[
                "max_abs_error"
            ],
        )

        print(
            "  mean abs error:",
            comparison[
                "mean_abs_error"
            ],
        )

        print(
            "  max scaled error:",
            comparison[
                "max_scaled_error"
            ],
        )

    print()
    print(
        "=" * 80
    )

    print(
        "TOP 10 FEATURE REPRODUCTION ERRORS"
    )

    print(
        "=" * 80
    )

    print(
        feature_error_frame.head(
            10
        ).to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.12e}"
            ),
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL REPRODUCTION LOCK"
    )

    print(
        "=" * 80
    )

    print(
        "Split integrity:",
        split_integrity,
    )

    print(
        "Metadata exact:",
        metadata_all_passed,
    )

    print(
        "Feature names exact:",
        feature_name_match,
    )

    print(
        "Raw descriptors allclose:",
        raw_allclose,
    )

    print(
        "Standardized descriptors allclose:",
        standardized_allclose,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    if overall_passed:

        print(
            "CHECK PASSED: historical "
            "92D descriptor pipeline "
            "was reproduced."
        )

        print(
            "Matched-noise dataset "
            "construction may proceed."
        )

    else:

        print(
            "CHECK FAILED: do NOT "
            "construct matched-noise "
            "datasets yet."
        )

    print()

    print(
        "Per-feature errors:",
        feature_error_path,
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
