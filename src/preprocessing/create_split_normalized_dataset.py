"""
Create train/validation/test split and normalized response tensors.

English:
This script converts a raw OpenSeesPy-generated dataset into the split-normalized
format used by the training, feature extraction, and evaluation scripts.

中文：
本脚本用于把 OpenSeesPy 生成的原始数据集转换为后续训练、特征提取和评价
脚本所需的 train/val/test 划分与标准化格式。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def find_required_key(data: Dict[str, np.ndarray], candidates: List[str], purpose: str) -> str:
    """
    Find the first existing key from a candidate list.

    中文：
    从候选字段名中寻找第一个存在的字段，用于兼容不同版本数据集的命名差异。
    """
    for key in candidates:
        if key in data:
            return key
    available = ", ".join(sorted(data.keys()))
    raise KeyError(f"Cannot find key for {purpose}. Candidates={candidates}. Available keys={available}")


def make_damage_bins(y_damage: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Create simple damage bins using maximum story damage.

    Bin definition:
    - 0: zero damage
    - 1: low damage, 0 < max_damage <= 0.10
    - 2: medium damage, 0.10 < max_damage <= 0.20
    - 3: high damage, max_damage > 0.20

    中文：
    根据每个样本所有楼层中的最大损伤值生成损伤等级，用于分层划分数据集。
    """
    max_damage = np.max(y_damage, axis=1)
    bins = np.zeros_like(max_damage, dtype=np.int64)
    bins[(max_damage > eps) & (max_damage <= 0.10)] = 1
    bins[(max_damage > 0.10) & (max_damage <= 0.20)] = 2
    bins[max_damage > 0.20] = 3
    return bins


def stratified_split_indices(
    labels: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split without relying on scikit-learn.

    中文：
    不依赖 sklearn，按损伤等级进行分层随机划分，尽量保证 train/val/test
    中 zero/low/medium/high 的比例接近。
    """
    rng = np.random.default_rng(seed)

    train_parts = []
    val_parts = []
    test_parts = []

    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        # Ensure at least one test sample when possible.
        # 中文：在样本数允许时，尽量保证测试集中保留样本。
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        else:
            n_train = max(1, min(n_train, n))
            n_val = max(0, min(n_val, n - n_train))

        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train + n_val])
        test_parts.append(idx[n_train + n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def split_metadata_arrays(
    raw: Dict[str, np.ndarray],
    n_cases: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    excluded_keys: set,
) -> Dict[str, np.ndarray]:
    """
    Split all 1D or sample-aligned metadata arrays.

    中文：
    自动把所有第一维等于样本数的元数据字段同步划分为 train/val/test。
    例如 amplitude_g、frequency_hz、noise_level、case_id 等。
    """
    out = {}

    for key, arr in raw.items():
        if key in excluded_keys:
            continue

        arr = np.asarray(arr)

        if arr.ndim >= 1 and arr.shape[0] == n_cases:
            out[f"{key}_train"] = arr[train_idx]
            out[f"{key}_val"] = arr[val_idx]
            out[f"{key}_test"] = arr[test_idx]

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    if not args.raw_dataset.exists():
        raise FileNotFoundError(f"Raw dataset not found: {args.raw_dataset}")

    raw_npz = np.load(args.raw_dataset, allow_pickle=True)
    raw = {key: raw_npz[key] for key in raw_npz.files}

    x_key = find_required_key(
        raw,
        ["X_abs_accel", "X", "responses", "X_response", "abs_accel"],
        "response tensor",
    )
    y_key = find_required_key(
        raw,
        ["y_damage", "y", "damage", "damage_targets"],
        "damage target",
    )

    X = np.asarray(raw[x_key], dtype=np.float64)
    y = np.asarray(raw[y_key], dtype=np.float64)

    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (n_cases, seq_len, n_channels), got {X.shape}")

    if y.ndim != 2:
        raise ValueError(f"Expected y to have shape (n_cases, n_stories), got {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y sample counts do not match: X={X.shape}, y={y.shape}")

    n_cases = X.shape[0]
    damage_bins = make_damage_bins(y, eps=args.eps)

    train_idx, val_idx, test_idx = stratified_split_indices(
        damage_bins,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    x_mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
    x_std = X[train_idx].std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < args.eps, 1.0, x_std)

    X_train = (X[train_idx] - x_mean) / x_std
    X_val = (X[val_idx] - x_mean) / x_std
    X_test = (X[test_idx] - x_mean) / x_std

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    output = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "x_mean": x_mean,
        "x_std": x_std,
        "damage_bin_train": damage_bins[train_idx],
        "damage_bin_val": damage_bins[val_idx],
        "damage_bin_test": damage_bins[test_idx],
    }

    metadata = split_metadata_arrays(
        raw=raw,
        n_cases=n_cases,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        excluded_keys={x_key, y_key},
    )
    output.update(metadata)

    # Ensure case_id exists even if the raw dataset did not store it.
    # 中文：如果原始数据没有 case_id，则自动生成。
    if "case_id_train" not in output:
        case_id = np.arange(n_cases, dtype=np.int64)
        output["case_id_train"] = case_id[train_idx]
        output["case_id_val"] = case_id[val_idx]
        output["case_id_test"] = case_id[test_idx]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)

    print("Split-normalized dataset saved.")
    print(f"Raw dataset: {args.raw_dataset}")
    print(f"Output: {args.output}")
    print(f"Response key: {x_key}")
    print(f"Damage key: {y_key}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Train/Val/Test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print("Damage-bin counts:")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        unique, counts = np.unique(damage_bins[idx], return_counts=True)
        print(f"  {name}: {dict(zip(unique.tolist(), counts.tolist()))}")
    print("Saved keys:")
    for key in sorted(output.keys()):
        print(f"  {key}: {np.asarray(output[key]).shape}")


if __name__ == "__main__":
    main()
