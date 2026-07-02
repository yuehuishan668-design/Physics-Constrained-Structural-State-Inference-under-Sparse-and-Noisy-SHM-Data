"""
Create sensor-masked datasets for sensor sparsity stress testing.

English:
This script keeps the original structural damage labels unchanged, but zero-masks
unavailable sensor channels in the response tensors. The tensor shape is preserved
so downstream feature extraction and training scripts remain compatible.

中文：
本脚本用于传感器稀疏性实验。它保持原始结构损伤标签不变，但将未保留的
传感器响应通道置零，并保留原始张量形状，从而保证后续特征提取和训练脚本兼容。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np


RAW_RESPONSE_KEYS = [
    "X_abs_accel",
    "X_clean_abs_accel",
    "story_drift",
]


def parse_keep_sensors(values: List[int], n_channels: int) -> np.ndarray:
    """
    Convert 1-based sensor IDs to 0-based channel indices.

    中文：
    将用户输入的 1-based 传感器编号转换为 Python 使用的 0-based 通道索引。
    """
    keep = np.array(values, dtype=int) - 1

    if keep.size == 0:
        raise ValueError("At least one sensor must be kept.")

    if np.any(keep < 0) or np.any(keep >= n_channels):
        raise ValueError(f"Sensor IDs must be between 1 and {n_channels}. Got {values}")

    return np.unique(keep)


def mask_response_tensor(x: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    """
    Zero-mask unavailable sensor channels while preserving tensor shape.

    中文：
    将未保留的传感器通道置零，但保持张量形状不变。
    """
    out = np.array(x, copy=True)
    all_idx = np.arange(out.shape[-1])
    drop_idx = np.setdiff1d(all_idx, keep_idx)
    out[..., drop_idx] = 0.0
    return out


def split_sample_aligned_metadata(
    raw: Dict[str, np.ndarray],
    split: Dict[str, np.ndarray],
    output: Dict[str, np.ndarray],
    n_cases: int,
    excluded_keys: set,
) -> None:
    """
    Preserve sample-aligned metadata arrays in train/val/test format.

    中文：
    将原始数据中与样本数对齐的元数据字段按照原有 train/val/test 索引重新划分。
    """
    train_idx = output["train_idx"]
    val_idx = output["val_idx"]
    test_idx = output["test_idx"]

    for key, arr in raw.items():
        if key in excluded_keys:
            continue

        arr = np.asarray(arr)

        if arr.ndim >= 1 and arr.shape[0] == n_cases:
            output[f"{key}_train"] = arr[train_idx]
            output[f"{key}_val"] = arr[val_idx]
            output[f"{key}_test"] = arr[test_idx]

    # Preserve existing split keys not overwritten above.
    # 中文：保留原 split 文件中已有但未被重写的字段。
    for key, arr in split.items():
        if key not in output:
            output[key] = arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--split-dataset", required=True, type=Path)
    parser.add_argument("--output-raw", required=True, type=Path)
    parser.add_argument("--output-split", required=True, type=Path)
    parser.add_argument("--keep-sensors", nargs="+", required=True, type=int)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    raw_npz = np.load(args.raw_dataset, allow_pickle=True)
    split_npz = np.load(args.split_dataset, allow_pickle=True)

    raw = {key: raw_npz[key] for key in raw_npz.files}
    split = {key: split_npz[key] for key in split_npz.files}

    if "X_abs_accel" not in raw:
        raise KeyError("Raw dataset must contain X_abs_accel.")

    if "y_damage" not in raw:
        raise KeyError("Raw dataset must contain y_damage.")

    X = np.asarray(raw["X_abs_accel"], dtype=np.float64)
    y = np.asarray(raw["y_damage"], dtype=np.float64)

    if X.ndim != 3:
        raise ValueError(f"Expected X_abs_accel shape (n_cases, n_steps, n_channels), got {X.shape}")

    n_cases, n_steps, n_channels = X.shape
    keep_idx = parse_keep_sensors(args.keep_sensors, n_channels=n_channels)
    drop_idx = np.setdiff1d(np.arange(n_channels), keep_idx)

    if not all(k in split for k in ["train_idx", "val_idx", "test_idx"]):
        raise KeyError("Split dataset must contain train_idx, val_idx, and test_idx.")

    train_idx = split["train_idx"].astype(int)
    val_idx = split["val_idx"].astype(int)
    test_idx = split["test_idx"].astype(int)

    # Create masked raw dataset.
    # 中文：创建传感器置零后的 raw dataset。
    masked_raw = {}
    for key, arr in raw.items():
        arr = np.asarray(arr)
        if key in RAW_RESPONSE_KEYS and arr.shape == X.shape:
            masked_raw[key] = mask_response_tensor(arr, keep_idx)
        else:
            masked_raw[key] = arr

    masked_raw["sensor_keep_1based"] = np.array(args.keep_sensors, dtype=np.int64)
    masked_raw["sensor_keep_0based"] = keep_idx.astype(np.int64)
    masked_raw["sensor_drop_0based"] = drop_idx.astype(np.int64)

    X_masked = masked_raw["X_abs_accel"]

    # Recreate normalized split using the original split indices.
    # 中文：使用原有 train/val/test 索引，基于置零后的响应重新计算标准化。
    x_mean = X_masked[train_idx].mean(axis=(0, 1), keepdims=True)
    x_std = X_masked[train_idx].std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < args.eps, 1.0, x_std)

    X_train = (X_masked[train_idx] - x_mean) / x_std
    X_val = (X_masked[val_idx] - x_mean) / x_std
    X_test = (X_masked[test_idx] - x_mean) / x_std

    masked_split = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "y_test": y[test_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "x_mean": x_mean,
        "x_std": x_std,
        "sensor_keep_1based": np.array(args.keep_sensors, dtype=np.int64),
        "sensor_keep_0based": keep_idx.astype(np.int64),
        "sensor_drop_0based": drop_idx.astype(np.int64),
    }

    split_sample_aligned_metadata(
        raw=masked_raw,
        split=split,
        output=masked_split,
        n_cases=n_cases,
        excluded_keys={"X_abs_accel", "X_clean_abs_accel", "story_drift", "y_damage"},
    )

    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    args.output_split.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(args.output_raw, **masked_raw)
    np.savez_compressed(args.output_split, **masked_split)

    print("Sensor-masked dataset created.")
    print(f"Input raw: {args.raw_dataset}")
    print(f"Input split: {args.split_dataset}")
    print(f"Output raw: {args.output_raw}")
    print(f"Output split: {args.output_split}")
    print(f"Original X shape: {X.shape}")
    print(f"Kept sensors 1-based: {args.keep_sensors}")
    print(f"Kept sensors 0-based: {keep_idx.tolist()}")
    print(f"Dropped sensors 0-based: {drop_idx.tolist()}")
    print(f"Train/Val/Test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")


if __name__ == "__main__":
    main()
