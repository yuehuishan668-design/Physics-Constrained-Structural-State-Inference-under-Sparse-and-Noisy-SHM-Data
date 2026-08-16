#!/usr/bin/env python3
"""Small end-to-end execution test; this does not reproduce manuscript metrics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_generation.generate_debug_dataset import generate_debug_dataset  # noqa: E402
from src.preprocessing.extract_physics_features import extract_all_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".reproduction_work" / "smoke_test",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 20 <= args.n_cases <= 50:
        raise ValueError("Smoke test requires 20–50 cases.")
    if args.duration <= 0 or args.dt <= 0:
        raise ValueError("duration and dt must be positive")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    raw_dir = output / "raw_cases"
    dataset_path = output / "smoke_dataset.npz"
    index_path = output / "smoke_dataset_index.csv"

    print("SCHM END-TO-END SMOKE TEST")
    print("This test checks execution only; it does not verify manuscript metrics.")
    print(f"cases={args.n_cases}, duration={args.duration}, dt={args.dt}, seed={args.seed}")

    generate_debug_dataset(
        n_cases=args.n_cases,
        seed=args.seed,
        duration=args.duration,
        dt=args.dt,
        output_dir_raw=raw_dir,
        output_path_processed=dataset_path,
        index_csv_path=index_path,
    )

    with np.load(dataset_path, allow_pickle=False) as archive:
        response = np.asarray(archive["X_abs_accel"], dtype=np.float64)
        ground = np.asarray(archive["ground_accel"], dtype=np.float64)
        target = np.asarray(archive["y_damage"], dtype=np.float64)
        amplitude = np.asarray(archive["amplitude_g"], dtype=np.float64)
        frequency = np.asarray(archive["frequency_hz"], dtype=np.float64)
        noise = np.asarray(archive["noise_level"], dtype=np.float64)
        dt = float(np.asarray(archive["dt"]).reshape(-1)[0])

    expected_steps = int(round(args.duration / args.dt))
    if response.shape != (args.n_cases, expected_steps, 4):
        raise AssertionError(f"Unexpected response shape: {response.shape}")
    if target.shape != (args.n_cases, 4):
        raise AssertionError(f"Unexpected target shape: {target.shape}")

    features, feature_names = extract_all_features(
        response=response,
        ground=ground,
        amplitude_g=amplitude,
        frequency_hz=frequency,
        noise_level=noise,
        dt=dt,
        healthy_first_frequency_hz=1.0 / 1.2700425053309456,
        min_fft_freq=0.05,
        max_fft_freq=min(8.0, 0.5 / dt),
        eps=1.0e-8,
    )
    if features.shape[0] != args.n_cases or features.shape[1] != len(feature_names):
        raise AssertionError("Feature extraction shape mismatch")
    if not np.all(np.isfinite(features)):
        raise AssertionError("Non-finite smoke-test descriptor detected")

    n_train = int(round(args.n_cases * 0.60))
    n_val = int(round(args.n_cases * 0.20))
    train_end = n_train
    val_end = n_train + n_val
    scaler = StandardScaler().fit(features[:train_end])
    x_train = scaler.transform(features[:train_end])
    x_val = scaler.transform(features[train_end:val_end])
    x_test = scaler.transform(features[val_end:])

    model = Ridge(alpha=1.0).fit(x_train, target[:train_end])
    val_prediction = np.clip(model.predict(x_val), 0.0, 0.5)
    test_prediction = np.clip(model.predict(x_test), 0.0, 0.5)
    val_mae = float(np.mean(np.abs(val_prediction - target[train_end:val_end])))
    test_mae = float(np.mean(np.abs(test_prediction - target[val_end:])))
    if not math.isfinite(val_mae) or not math.isfinite(test_mae):
        raise AssertionError("Smoke-test MAE is not finite")

    report = {
        "purpose": "execution smoke test; not a manuscript metric",
        "n_cases": args.n_cases,
        "n_steps": expected_steps,
        "n_stories": 4,
        "n_descriptors": int(features.shape[1]),
        "split_counts": {
            "train": n_train,
            "validation": n_val,
            "test": args.n_cases - val_end,
        },
        "validation_mae": val_mae,
        "test_mae": test_mae,
        "seed": args.seed,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_test_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[PASS] OpenSeesPy cases generated: {args.n_cases}")
    print(f"[PASS] Response shape: {response.shape}")
    print(f"[PASS] Descriptor matrix: {features.shape}")
    print(f"[PASS] Ridge validation MAE is finite: {val_mae:.6f}")
    print(f"[PASS] Ridge test MAE is finite: {test_mae:.6f}")
    print(f"[PASS] Report: {output / 'smoke_test_report.json'}")
    print("SMOKE TEST PASSED (EXECUTION ONLY; NOT A MANUSCRIPT RESULT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

