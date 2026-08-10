"""
Audit raw dataset and historical noise-generation provenance
before constructing matched clean/noisy datasets.

Purpose
-------
This script DOES NOT generate new data.

It reports:
1. Keys / shapes / dtypes in the original clean 3000-case dataset.
2. Keys / shapes in existing historical noise datasets.
3. Dataset-index columns and first rows.
4. Python source locations mentioning noise generation,
   response histories, ground motion, and descriptor extraction.
5. Candidate feature-generation scripts/functions.

The audit is required before defining the paired signal-level
measurement-noise protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEARCH_TERMS = [
    "noise",
    "noise_level",
    "gaussian",
    "normal(",
    "ground",
    "accel",
    "acceleration",
    "response",
    "time_history",
    "physics_feature",
    "physics_features",
    "descriptor",
    "dominant_frequency",
    "spectral_centroid",
]


def print_npz_inventory(
    path: Path,
    title: str,
) -> None:

    print()
    print(
        "=" * 80
    )
    print(title)
    print(
        "=" * 80
    )

    if not path.is_file():
        print(
            "MISSING:",
            path,
        )
        return

    print(
        "Path:",
        path,
    )

    print(
        "Size MB:",
        f"{path.stat().st_size / 1024 / 1024:.3f}",
    )

    with np.load(
        path,
        allow_pickle=False,
        mmap_mode="r",
    ) as data:

        print(
            "Keys:",
            len(data.files),
        )

        for key in data.files:

            try:
                array = data[key]

                print(
                    f"{key:45s}",
                    f"shape={str(array.shape):22s}",
                    f"dtype={array.dtype}",
                )

            except Exception as exc:

                print(
                    f"{key:45s}",
                    "ERROR:",
                    repr(exc),
                )


def find_npz_files(
    roots: list[Path],
) -> list[Path]:

    files = []

    for root in roots:

        if not root.exists():
            continue

        files.extend(
            root.rglob("*.npz")
        )

    unique = sorted(
        {
            path.resolve()
            for path in files
        }
    )

    return [
        Path(path)
        for path in unique
    ]


def is_noise_related(
    path: Path,
) -> bool:

    name = (
        path.name.lower()
    )

    return (
        "noise" in name
        or "noisy" in name
    )


def scan_python_sources(
    project_root: Path,
) -> list[
    tuple[str, int, str]
]:

    results = []

    excluded = {
        ".git",
        ".venv",
        "__pycache__",
        "results",
    }

    for path in sorted(
        project_root.rglob("*.py")
    ):

        if any(
            part in excluded
            for part in path.parts
        ):
            continue

        try:

            lines = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()

        except Exception:
            continue

        for lineno, line in enumerate(
            lines,
            start=1,
        ):

            lower = line.lower()

            matched = [
                term
                for term in SEARCH_TERMS
                if term in lower
            ]

            if matched:

                relative = path.relative_to(
                    project_root
                )

                results.append(
                    (
                        str(relative),
                        lineno,
                        line.strip(),
                    )
                )

    return results


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean-dataset",
        type=Path,
        default=Path(
            "data_processed/"
            "debug_plus_3000_dataset.npz"
        ),
    )

    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path(
            "data_processed/"
            "debug_plus_3000_dataset_index.csv"
        ),
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    args = parser.parse_args()

    project_root = (
        args.project_root
        .resolve()
    )

    print(
        "===== PAIRED-NOISE PROVENANCE AUDIT ====="
    )

    print(
        "Project root:",
        project_root,
    )

    print_npz_inventory(
        args.clean_dataset,
        "ORIGINAL CLEAN DATASET",
    )

    print()
    print(
        "=" * 80
    )
    print(
        "DATASET INDEX"
    )
    print(
        "=" * 80
    )

    if args.index_csv.is_file():

        index = pd.read_csv(
            args.index_csv
        )

        print(
            "Path:",
            args.index_csv,
        )

        print(
            "Shape:",
            index.shape,
        )

        print(
            "Columns:"
        )

        for column in index.columns:
            print(
                "  -",
                column,
            )

        print()
        print(
            "First 8 rows:"
        )

        print(
            index.head(
                8
            ).to_string(
                index=False
            )
        )

        print()
        print(
            "Unique-count summary:"
        )

        for column in index.columns:

            try:
                unique_count = (
                    index[
                        column
                    ]
                    .nunique(
                        dropna=False
                    )
                )

                print(
                    f"{column:40s}",
                    unique_count,
                )

            except Exception:
                pass

    else:

        print(
            "MISSING:",
            args.index_csv,
        )

    # -------------------------------------------------
    # Existing noise datasets
    # -------------------------------------------------

    print()
    print(
        "=" * 80
    )
    print(
        "EXISTING NOISE-RELATED NPZ FILES"
    )
    print(
        "=" * 80
    )

    roots = [
        project_root
        / "data_processed",
        project_root
        / "data_inputs",
        project_root
        / "data",
    ]

    all_npz = find_npz_files(
        roots
    )

    noise_npz = [
        path
        for path in all_npz
        if is_noise_related(
            path
        )
    ]

    print(
        "Noise-related NPZ count:",
        len(
            noise_npz
        ),
    )

    for path in noise_npz:

        try:
            relative = path.relative_to(
                project_root
            )
        except ValueError:
            relative = path

        print_npz_inventory(
            path,
            f"NOISE DATASET: {relative}",
        )

    # -------------------------------------------------
    # Python source audit
    # -------------------------------------------------

    print()
    print(
        "=" * 80
    )
    print(
        "SOURCE-CODE SEARCH"
    )
    print(
        "=" * 80
    )

    source_hits = (
        scan_python_sources(
            project_root
        )
    )

    print(
        "Matched source lines:",
        len(
            source_hits
        ),
    )

    print()

    for (
        path,
        lineno,
        line,
    ) in source_hits:

        print(
            f"{path}:{lineno}: {line}"
        )

    print()
    print(
        "=" * 80
    )
    print(
        "AUDIT COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "No new noise dataset was generated."
    )

    print(
        "Next step: identify the exact "
        "raw sensing arrays, historical "
        "noise definition, and canonical "
        "descriptor-extraction entry point."
    )


if __name__ == "__main__":
    main()
