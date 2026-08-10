"""
Build matched clean/noisy descriptor datasets for controlled
measurement-noise robustness experiments.

Design
------
Canonical structural signal:
    X_clean_abs_accel

Noise levels:
    0%, 5%, 10%, 20%

Noise replicates:
    0%  : one clean condition
    >0% : five deterministic independent replicates

Historical noise definition:
    signal_std = np.std(signal)
    noise_std  = noise_level * signal_std
    noise ~ Normal(0, noise_std)

IMPORTANT:
- np.std(signal) is evaluated over the entire 2000 x 4 response
  matrix of one case, giving one scalar standard deviation.
- Structural acceleration channels are perturbed.
- ground_accel remains clean.
- Damage labels, excitation parameters, case IDs and dt remain fixed.
- Raw noisy time histories are NOT saved.
- Every noisy realization is generated, descriptors are extracted,
  then the noisy response array is discarded.

Outputs
-------
For each condition:
    - full raw 92D descriptors
    - canonical raw 78D signal-derived-with-ground descriptors
    - case IDs / targets / split indices / noise metadata

A manifest and build report are also produced.
"""

from __future__ import annotations

import argparse
import hashlib
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.sss_fast_revision.validate_historical_92d_reproduction import (  # noqa: E402
    infer_cli_default,
    load_module,
    sha256_file,
)


EXPECTED_CASES = 3000
EXPECTED_FEATURES_92D = 92
EXPECTED_FEATURES_78D = 78

NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
]

NONZERO_REPLICATES = 5

BASE_NOISE_SEED = 20260809

ZERO_TOLERANCE = 0.0


def sha256_path(
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


def load_feature_names(
    path: Path,
) -> list[str]:
    """
    Load the canonical descriptor names from a canonical NPZ.

    The canonical 78D dataset is treated as the source of truth
    for descriptor identity/order.
    """

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        candidate_keys = [
            "feature_names",
            "selected_feature_names",
            "descriptor_names",
        ]

        key = None

        for candidate in (
            candidate_keys
        ):

            if candidate in data.files:

                key = candidate
                break

        if key is None:

            raise RuntimeError(
                "Could not find canonical "
                "78D feature-name array.\n"
                f"Available keys: "
                f"{sorted(data.files)}"
            )

        names = [
            str(value)
            for value in (
                data[key]
                .tolist()
            )
        ]

    if len(
        names
    ) != EXPECTED_FEATURES_78D:

        raise RuntimeError(
            "Canonical descriptor-name "
            f"count is {len(names)}, "
            "expected 78."
        )

    if len(
        set(names)
    ) != len(names):

        raise RuntimeError(
            "Duplicate canonical 78D "
            "feature names detected."
        )

    return names


def build_name_mapping(
    full_names: list[str],
    canonical_names: list[str],
) -> np.ndarray:

    lookup: dict[
        str,
        int,
    ] = {}

    for index, name in enumerate(
        full_names
    ):

        if name in lookup:

            raise RuntimeError(
                "Duplicate full 92D "
                f"feature name: {name}"
            )

        lookup[name] = index

    missing = [
        name
        for name
        in canonical_names
        if name not in lookup
    ]

    if missing:

        raise RuntimeError(
            "Canonical 78D names missing "
            "from reproduced 92D:\n"
            + "\n".join(
                missing
            )
        )

    indices = np.asarray(
        [
            lookup[name]
            for name
            in canonical_names
        ],
        dtype=np.int64,
    )

    if len(
        np.unique(indices)
    ) != EXPECTED_FEATURES_78D:

        raise RuntimeError(
            "78D mapping is not unique."
        )

    return indices


def build_all_case_raw(
    frozen: dict[str, np.ndarray],
    prefix: str,
    n_features: int,
) -> np.ndarray | None:
    """
    Reconstruct all-case raw descriptor matrix from split arrays
    when the canonical NPZ contains F_*_raw.
    """

    required = {
        "train_idx",
        "val_idx",
        "test_idx",
        "F_train_raw",
        "F_val_raw",
        "F_test_raw",
    }

    if not required.issubset(
        set(
            frozen.keys()
        )
    ):

        return None

    result = np.empty(
        (
            EXPECTED_CASES,
            n_features,
        ),
        dtype=np.float64,
    )

    for split in [
        "train",
        "val",
        "test",
    ]:

        idx = np.asarray(
            frozen[
                f"{split}_idx"
            ],
            dtype=np.int64,
        )

        values = np.asarray(
            frozen[
                f"F_{split}_raw"
            ],
            dtype=np.float64,
        )

        result[
            idx
        ] = values

    return result


def condition_specs() -> list[
    tuple[float, int]
]:

    specs = [
        (
            0.0,
            0,
        )
    ]

    for level in (
        NOISE_LEVELS
    ):

        if level <= 0.0:
            continue

        for replicate in range(
            NONZERO_REPLICATES
        ):

            specs.append(
                (
                    level,
                    replicate,
                )
            )

    return specs


def level_code(
    noise_level: float,
) -> int:

    return int(
        round(
            noise_level
            * 1000
        )
    )


def deterministic_rng(
    case_id: int,
    noise_level: float,
    replicate: int,
) -> np.random.Generator:
    """
    Case-level deterministic RNG.

    The noise realization for a case is independent of loop order
    and can be reproduced without regenerating preceding cases.
    """

    sequence = np.random.SeedSequence(
        [
            BASE_NOISE_SEED,
            level_code(
                noise_level
            ),
            int(
                replicate
            ),
            int(
                case_id
            ),
        ]
    )

    return np.random.default_rng(
        sequence
    )


def make_noisy_response(
    clean_response: np.ndarray,
    case_ids: np.ndarray,
    noise_level: float,
    replicate: int,
) -> tuple[
    np.ndarray,
    dict[str, float],
]:
    """
    Generate a complete 3000-case matched response condition.
    """

    clean_response = np.asarray(
        clean_response,
        dtype=np.float64,
    )

    if noise_level <= 0.0:

        result = (
            clean_response.copy()
        )

        diagnostics = {
            "target_noise_ratio_mean": (
                0.0
            ),
            "realized_noise_to_signal_std_mean": (
                0.0
            ),
            "realized_noise_to_signal_std_median": (
                0.0
            ),
            "realized_noise_to_signal_std_std": (
                0.0
            ),
            "max_abs_noise": (
                0.0
            ),
        }

        return (
            result,
            diagnostics,
        )

    result = np.empty_like(
        clean_response
    )

    realized_ratios = []

    max_abs_noise = 0.0

    for row in range(
        clean_response.shape[0]
    ):

        clean = (
            clean_response[
                row
            ]
        )

        signal_std = float(
            np.std(
                clean
            )
        )

        noise_std = (
            noise_level
            * signal_std
        )

        rng = deterministic_rng(
            case_id=int(
                case_ids[
                    row
                ]
            ),
            noise_level=(
                noise_level
            ),
            replicate=(
                replicate
            ),
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=clean.shape,
        )

        result[
            row
        ] = (
            clean
            + noise
        )

        realized_std = float(
            np.std(
                noise
            )
        )

        if signal_std > 1e-15:

            realized_ratios.append(
                realized_std
                / signal_std
            )

        max_abs_noise = max(
            max_abs_noise,
            float(
                np.max(
                    np.abs(
                        noise
                    )
                )
            ),
        )

    realized = np.asarray(
        realized_ratios,
        dtype=np.float64,
    )

    diagnostics = {
        "target_noise_ratio_mean": float(
            noise_level
        ),
        "realized_noise_to_signal_std_mean": float(
            np.mean(
                realized
            )
        ),
        "realized_noise_to_signal_std_median": float(
            np.median(
                realized
            )
        ),
        "realized_noise_to_signal_std_std": float(
            np.std(
                realized
            )
        ),
        "max_abs_noise": float(
            max_abs_noise
        ),
    }

    return (
        result,
        diagnostics,
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
        "--historical-92d",
        type=Path,
        default=Path(
            "data_inputs/"
            "aes_frozen/"
            "debug_plus_3000_"
            "physics_features_mlp.npz"
        ),
    )

    parser.add_argument(
        "--canonical-78d",
        type=Path,
        default=Path(
            "data_processed/"
            "sss_fast_revision/"
            "descriptor_sets/"
            "signal_derived_with_ground/"
            "debug_plus_3000_"
            "signal_derived_with_ground_"
            "features.npz"
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
            "data_processed/"
            "sss_fast_revision/"
            "matched_noise_78d"
        ),
    )

    args = parser.parse_args()

    if args.raw_dataset is None:

        raise RuntimeError(
            "Raw dataset path missing. "
            "Set RAW3000 or pass "
            "--raw-dataset."
        )

    raw_path = (
        args.raw_dataset
        .expanduser()
        .resolve()
    )

    historical_path = (
        args.historical_92d
        .expanduser()
        .resolve()
    )

    canonical_78d_path = (
        args.canonical_78d
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
        historical_path,
        canonical_78d_path,
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

    # -----------------------------------------
    # Extractor numerical settings.
    # -----------------------------------------

    min_fft_freq = float(
        infer_cli_default(
            extractor_path,
            "min_fft_freq",
        )
    )

    max_fft_freq = float(
        infer_cli_default(
            extractor_path,
            "max_fft_freq",
        )
    )

    eps = float(
        infer_cli_default(
            extractor_path,
            "eps",
        )
    )

    extractor = load_module(
        extractor_path
    )

    print(
        "=" * 80
    )

    print(
        "MATCHED NOISE DESCRIPTOR BUILDER"
    )

    print(
        "=" * 80
    )

    print(
        "Raw dataset:",
        raw_path,
    )

    print(
        "Extractor SHA256:",
        sha256_file(
            extractor_path
        ),
    )

    print(
        "Base noise seed:",
        BASE_NOISE_SEED,
    )

    print(
        "Noise levels:",
        NOISE_LEVELS,
    )

    print(
        "Nonzero replicates:",
        NONZERO_REPLICATES,
    )

    print(
        "Conditions:",
        len(
            condition_specs()
        ),
    )

    print()

    # -----------------------------------------
    # Load historical frozen reference.
    # -----------------------------------------

    with np.load(
        historical_path,
        allow_pickle=False,
    ) as data:

        historical = {
            key: np.asarray(
                data[key]
            )
            for key in data.files
        }

    train_idx = np.asarray(
        historical[
            "train_idx"
        ],
        dtype=np.int64,
    )

    val_idx = np.asarray(
        historical[
            "val_idx"
        ],
        dtype=np.int64,
    )

    test_idx = np.asarray(
        historical[
            "test_idx"
        ],
        dtype=np.int64,
    )

    historical_names = [
        str(name)
        for name in (
            historical[
                "feature_names"
            ]
            .tolist()
        )
    ]

    healthy_frequency = float(
        historical[
            "healthy_first_frequency_hz"
        ]
    )

    historical_all_raw = np.empty(
        (
            EXPECTED_CASES,
            EXPECTED_FEATURES_92D,
        ),
        dtype=np.float64,
    )

    historical_all_raw[
        train_idx
    ] = historical[
        "F_train_raw"
    ]

    historical_all_raw[
        val_idx
    ] = historical[
        "F_val_raw"
    ]

    historical_all_raw[
        test_idx
    ] = historical[
        "F_test_raw"
    ]

    # -----------------------------------------
    # Load canonical 78D names.
    # -----------------------------------------

    canonical_78d_names = (
        load_feature_names(
            canonical_78d_path
        )
    )

    # Optional canonical raw 78D audit source.
    with np.load(
        canonical_78d_path,
        allow_pickle=False,
    ) as data:

        canonical_78d_npz = {
            key: np.asarray(
                data[key]
            )
            for key in data.files
        }

    canonical_78d_all_raw = (
        build_all_case_raw(
            canonical_78d_npz,
            prefix="",
            n_features=(
                EXPECTED_FEATURES_78D
            ),
        )
    )

    # -----------------------------------------
    # Load canonical clean raw source.
    # -----------------------------------------

    print(
        "Loading canonical clean signals ..."
    )

    with np.load(
        raw_path,
        allow_pickle=False,
    ) as raw:

        required = {
            "X_clean_abs_accel",
            "X_abs_accel",
            "ground_accel",
            "y_damage",
            "amplitude_g",
            "frequency_hz",
            "phase",
            "noise_level",
            "case_id",
            "dt",
        }

        missing = sorted(
            required
            - set(
                raw.files
            )
        )

        if missing:

            raise RuntimeError(
                f"Raw dataset missing: "
                f"{missing}"
            )

        clean_response = np.asarray(
            raw[
                "X_clean_abs_accel"
            ],
            dtype=np.float64,
        )

        stored_response = np.asarray(
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

        phase = np.asarray(
            raw[
                "phase"
            ],
            dtype=np.float64,
        ).reshape(
            -1
        )

        historical_noise_level = np.asarray(
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

        dt = float(
            raw[
                "dt"
            ]
        )

    if (
        clean_response.shape
        != (
            EXPECTED_CASES,
            2000,
            4,
        )
    ):

        raise RuntimeError(
            "Unexpected clean response shape: "
            f"{clean_response.shape}"
        )

    if (
        ground.shape
        != (
            EXPECTED_CASES,
            2000,
        )
    ):

        raise RuntimeError(
            "Unexpected ground shape: "
            f"{ground.shape}"
        )

    if not np.array_equal(
        case_id,
        np.arange(
            EXPECTED_CASES,
            dtype=np.int64,
        ),
    ):

        raise RuntimeError(
            "case_id is not exact 0..2999."
        )

    # -----------------------------------------
    # Condition construction.
    # -----------------------------------------

    manifest_rows = []

    zero_condition_F92 = None
    zero_condition_F78 = None

    full_name_mapping = None

    condition_list = (
        condition_specs()
    )

    for (
        condition_number,
        (
            noise_level,
            replicate,
        ),
    ) in enumerate(
        condition_list,
        start=1,
    ):

        level_percent = int(
            round(
                noise_level
                * 100
            )
        )

        print()
        print(
            "=" * 80
        )

        print(
            "CONDITION "
            f"{condition_number}/"
            f"{len(condition_list)}"
        )

        print(
            "=" * 80
        )

        print(
            "Noise level:",
            f"{level_percent}%"
        )

        print(
            "Replicate:",
            replicate,
        )

        start = (
            time.perf_counter()
        )

        (
            response,
            noise_diagnostics,
        ) = make_noisy_response(
            clean_response=(
                clean_response
            ),
            case_ids=case_id,
            noise_level=(
                noise_level
            ),
            replicate=(
                replicate
            ),
        )

        if (
            noise_level
            == 0.0
        ):

            if not np.array_equal(
                response,
                clean_response,
            ):

                raise RuntimeError(
                    "0% condition differs "
                    "from clean response."
                )

        condition_noise_vector = np.full(
            EXPECTED_CASES,
            noise_level,
            dtype=np.float64,
        )

        (
            F92_raw,
            feature_names,
        ) = (
            extractor
            .extract_all_features(
                response=response,
                ground=ground,
                amplitude_g=(
                    amplitude_g
                ),
                frequency_hz=(
                    frequency_hz
                ),
                noise_level=(
                    condition_noise_vector
                ),
                dt=dt,
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

        F92_raw = np.asarray(
            F92_raw,
            dtype=np.float64,
        )

        feature_names = [
            str(name)
            for name in feature_names
        ]

        if (
            feature_names
            != historical_names
        ):

            raise RuntimeError(
                "92D feature-name order changed."
            )

        if full_name_mapping is None:

            full_name_mapping = (
                build_name_mapping(
                    full_names=(
                        feature_names
                    ),
                    canonical_names=(
                        canonical_78d_names
                    ),
                )
            )

            print(
                "Canonical 78D mapping:",
                full_name_mapping.tolist(),
            )

        F78_raw = (
            F92_raw[
                :,
                full_name_mapping
            ]
        )

        if (
            F78_raw.shape
            != (
                EXPECTED_CASES,
                EXPECTED_FEATURES_78D,
            )
        ):

            raise RuntimeError(
                "Unexpected 78D shape."
            )

        # -------------------------------------
        # 0% source locks.
        # -------------------------------------

        zero_92_error = None
        zero_78_error = None

        if (
            noise_level
            == 0.0
        ):

            zero_condition_F92 = (
                F92_raw.copy()
            )

            zero_condition_F78 = (
                F78_raw.copy()
            )

            historical_zero_mask = np.isclose(
                historical_noise_level,
                0.0,
            )

            zero_count = int(
                np.sum(
                    historical_zero_mask
                )
            )

            if zero_count != 744:

                raise RuntimeError(
                    "Historical zero-noise "
                    f"count changed: {zero_count}"
                )

            zero_92_error = float(
                np.max(
                    np.abs(
                        F92_raw[
                            historical_zero_mask
                        ]
                        - historical_all_raw[
                            historical_zero_mask
                        ]
                    )
                )
            )

            print(
                "Historical 0%-case "
                "92D max abs error:",
                zero_92_error,
            )

            if (
                zero_92_error
                != ZERO_TOLERANCE
            ):

                raise RuntimeError(
                    "STOP: canonical clean "
                    "92D does not exactly "
                    "reproduce historical "
                    "zero-noise cases."
                )

            if (
                canonical_78d_all_raw
                is not None
            ):

                zero_78_error = float(
                    np.max(
                        np.abs(
                            F78_raw[
                                historical_zero_mask
                            ]
                            - canonical_78d_all_raw[
                                historical_zero_mask
                            ]
                        )
                    )
                )

                print(
                    "Historical 0%-case "
                    "canonical 78D max abs error:",
                    zero_78_error,
                )

                if (
                    zero_78_error
                    != ZERO_TOLERANCE
                ):

                    raise RuntimeError(
                        "STOP: canonical clean "
                        "78D does not exactly "
                        "reproduce historical "
                        "zero-noise cases."
                    )

        # -------------------------------------
        # Save descriptor condition.
        # -------------------------------------

        filename = (
            "matched_noise_"
            f"{level_percent:03d}_"
            f"rep_{replicate}.npz"
        )

        output_path = (
            args.output_root
            / filename
        )

        np.savez_compressed(
            output_path,
            F_92_raw=(
                F92_raw
            ),
            F_78_raw=(
                F78_raw
            ),
            feature_names_92=np.asarray(
                feature_names
            ),
            feature_names_78=np.asarray(
                canonical_78d_names
            ),
            y_damage=(
                y_damage
            ),
            case_id=(
                case_id
            ),
            amplitude_g=(
                amplitude_g
            ),
            frequency_hz=(
                frequency_hz
            ),
            phase=(
                phase
            ),
            noise_level=np.asarray(
                noise_level
            ),
            replicate=np.asarray(
                replicate
            ),
            base_noise_seed=np.asarray(
                BASE_NOISE_SEED
            ),
            train_idx=(
                train_idx
            ),
            val_idx=(
                val_idx
            ),
            test_idx=(
                test_idx
            ),
            dt=np.asarray(
                dt
            ),
            healthy_first_frequency_hz=np.asarray(
                healthy_frequency
            ),
        )

        elapsed = float(
            time.perf_counter()
            - start
        )

        digest = sha256_path(
            output_path
        )

        print(
            "Realized noise/std mean:",
            (
                noise_diagnostics[
                    "realized_noise_to_signal_std_mean"
                ]
            ),
        )

        print(
            "Realized noise/std median:",
            (
                noise_diagnostics[
                    "realized_noise_to_signal_std_median"
                ]
            ),
        )

        print(
            "Descriptor output:",
            output_path,
        )

        print(
            "Elapsed seconds:",
            f"{elapsed:.3f}",
        )

        manifest_rows.append(
            {
                "condition_number": (
                    condition_number
                ),
                "noise_level": (
                    noise_level
                ),
                "noise_percent": (
                    level_percent
                ),
                "replicate": (
                    replicate
                ),
                "n_cases": (
                    EXPECTED_CASES
                ),
                "n_features_92": (
                    EXPECTED_FEATURES_92D
                ),
                "n_features_78": (
                    EXPECTED_FEATURES_78D
                ),
                "ground_signal_noisy": (
                    False
                ),
                "target_noise_ratio": (
                    noise_diagnostics[
                        "target_noise_ratio_mean"
                    ]
                ),
                "realized_noise_to_signal_std_mean": (
                    noise_diagnostics[
                        "realized_noise_to_signal_std_mean"
                    ]
                ),
                "realized_noise_to_signal_std_median": (
                    noise_diagnostics[
                        "realized_noise_to_signal_std_median"
                    ]
                ),
                "realized_noise_to_signal_std_std": (
                    noise_diagnostics[
                        "realized_noise_to_signal_std_std"
                    ]
                ),
                "max_abs_noise": (
                    noise_diagnostics[
                        "max_abs_noise"
                    ]
                ),
                "historical_zero_case_92d_max_error": (
                    zero_92_error
                ),
                "historical_zero_case_78d_max_error": (
                    zero_78_error
                ),
                "elapsed_seconds": (
                    elapsed
                ),
                "file": str(
                    output_path
                ),
                "sha256": (
                    digest
                ),
            }
        )

        # Explicitly release the large response.
        del response
        del F92_raw
        del F78_raw

    # -----------------------------------------
    # Build manifest.
    # -----------------------------------------

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        args.output_root
        / "matched_noise_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    # -----------------------------------------
    # Validate condition counts.
    # -----------------------------------------

    expected_conditions = (
        1
        + (
            len(
                [
                    level
                    for level in NOISE_LEVELS
                    if level > 0.0
                ]
            )
            * NONZERO_REPLICATES
        )
    )

    condition_count_ok = bool(
        len(
            manifest
        )
        == expected_conditions
    )

    zero_rows = (
        manifest[
            np.isclose(
                manifest[
                    "noise_level"
                ],
                0.0,
            )
        ]
    )

    nonzero_counts = (
        manifest.loc[
            manifest[
                "noise_level"
            ]
            > 0.0
        ]
        .groupby(
            "noise_level"
        )
        .size()
        .to_dict()
    )

    replicate_count_ok = bool(
        len(
            zero_rows
        )
        == 1
        and all(
            int(
                nonzero_counts.get(
                    level,
                    0,
                )
            )
            == NONZERO_REPLICATES
            for level
            in NOISE_LEVELS
            if level > 0.0
        )
    )

    # -----------------------------------------
    # Build report.
    # -----------------------------------------

    report = {
        "experiment": (
            "matched_clean_noisy_"
            "descriptor_dataset_build"
        ),
        "raw_source": str(
            raw_path
        ),
        "canonical_clean_response": (
            "X_clean_abs_accel"
        ),
        "historical_stored_response": (
            "X_abs_accel"
        ),
        "ground_signal": (
            "ground_accel"
        ),
        "ground_signal_perturbed": (
            False
        ),
        "noise_definition": {
            "distribution": (
                "Gaussian"
            ),
            "mean": 0.0,
            "case_signal_scale": (
                "np.std(clean_response_case)"
            ),
            "noise_std": (
                "noise_level * "
                "np.std(clean_response_case)"
            ),
            "std_scope": (
                "entire 2000 x 4 case matrix"
            ),
        },
        "noise_levels": (
            NOISE_LEVELS
        ),
        "nonzero_replicates": (
            NONZERO_REPLICATES
        ),
        "base_noise_seed": (
            BASE_NOISE_SEED
        ),
        "rng_definition": (
            "SeedSequence([base_seed, "
            "level_code, replicate, case_id])"
        ),
        "n_conditions": int(
            len(
                manifest
            )
        ),
        "condition_count_ok": (
            condition_count_ok
        ),
        "replicate_count_ok": (
            replicate_count_ok
        ),
        "historical_zero_noise_case_count": (
            744
        ),
        "historical_zero_noise_92d_exact": bool(
            (
                manifest[
                    "historical_zero_case_92d_max_error"
                ]
                .dropna()
                == 0.0
            ).all()
        ),
        "canonical_zero_noise_78d_exact": (
            None
            if canonical_78d_all_raw
            is None
            else bool(
                (
                    manifest[
                        "historical_zero_case_78d_max_error"
                    ]
                    .dropna()
                    == 0.0
                ).all()
            )
        ),
        "feature_extractor_sha256": (
            sha256_file(
                extractor_path
            )
        ),
        "canonical_78d_names": (
            canonical_78d_names
        ),
        "canonical_78d_indices_in_92d": (
            []
            if full_name_mapping
            is None
            else (
                full_name_mapping
                .tolist()
            )
        ),
        "large_noisy_raw_arrays_saved": (
            False
        ),
        "manifest": str(
            manifest_path
        ),
    }

    overall_passed = bool(
        condition_count_ok
        and replicate_count_ok
        and report[
            "historical_zero_noise_92d_exact"
        ]
        and (
            report[
                "canonical_zero_noise_78d_exact"
            ]
            in (
                True,
                None,
            )
        )
    )

    report[
        "overall_passed"
    ] = (
        overall_passed
    )

    report_path = (
        args.output_root
        / "matched_noise_build_report.json"
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

    # -----------------------------------------
    # Console summary.
    # -----------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "MATCHED NOISE MANIFEST"
    )

    print(
        "=" * 80
    )

    print(
        manifest[
            [
                "noise_percent",
                "replicate",
                "realized_noise_to_signal_std_mean",
                "realized_noise_to_signal_std_median",
                "historical_zero_case_92d_max_error",
                "historical_zero_case_78d_max_error",
                "elapsed_seconds",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.10f}"
            ),
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
        "Conditions:",
        len(
            manifest
        ),
    )

    print(
        "Condition count correct:",
        condition_count_ok,
    )

    print(
        "Replicate count correct:",
        replicate_count_ok,
    )

    print(
        "Historical 0%-case "
        "92D exact:",
        report[
            "historical_zero_noise_92d_exact"
        ],
    )

    print(
        "Historical 0%-case "
        "canonical 78D exact:",
        report[
            "canonical_zero_noise_78d_exact"
        ],
    )

    print(
        "Ground signal perturbed:",
        False,
    )

    print(
        "Large noisy raw arrays saved:",
        False,
    )

    print(
        "OVERALL PASSED:",
        overall_passed,
    )

    print()

    if overall_passed:

        print(
            "CHECK PASSED: matched "
            "clean/noisy descriptor datasets "
            "were built successfully."
        )

        print(
            "Controlled clean-trained "
            "noise robustness experiments "
            "may proceed."
        )

    else:

        print(
            "CHECK FAILED: do NOT "
            "run noise robustness models."
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
