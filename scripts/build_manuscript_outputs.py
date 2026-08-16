#!/usr/bin/env python3
"""Rebuild manuscript tables and Figs. 3–7 from frozen processed evidence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "scripts/jcshm_manuscript/prepare_jcshm_manuscript_data.py",
    "scripts/jcshm_manuscript/prepare_final_tables.py",
    "scripts/jcshm_manuscript/figures/plot_F03_factorial.py",
    "scripts/jcshm_manuscript/figures/plot_F04_severity_calibration.py",
    "scripts/jcshm_manuscript/figures/plot_F05_noise_failure_mechanism.py",
    "scripts/jcshm_manuscript/figures/plot_F06_sensor_layout_observability.py",
    "scripts/jcshm_manuscript/figures/plot_F07_sensor_subset_lattice.py",
    "scripts/verify_manuscript_results.py",
]


def main() -> int:
    print("SCHM MANUSCRIPT OUTPUT REBUILD FROM FROZEN EVIDENCE")
    print("Training performed: False")
    print("Hyperparameter tuning performed: False")
    print("Figures 1–2: retained conceptual/vector sources; not rebuilt here")
    print()

    for relative in STEPS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"[RUN] {relative}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode != 0:
            print(f"[FAIL] {relative}: exit {completed.returncode}")
            return completed.returncode
        print(f"[PASS] {relative}")

    print("ALL FROZEN MANUSCRIPT OUTPUT STEPS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

