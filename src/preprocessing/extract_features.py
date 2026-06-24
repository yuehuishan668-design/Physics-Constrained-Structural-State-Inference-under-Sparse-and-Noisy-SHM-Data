"""
File location:
    src/preprocessing/extract_features.py

Purpose:
    Extract simple statistical features from normalized OpenSeesPy response histories
    for the first MLP baseline.

Input:
    data_processed/debug_split_normalized.npz

Output:
    data_processed/debug_features_mlp.npz

中文说明：
    当前 X_train 的形状是 (14, 2000, 4)，即：
        14 个训练样本；
        每个样本 2000 个时间步；
        每个时间步 4 个楼层加速度通道。
    如果直接把 2000×4 展平给 MLP，会得到 8000 个输入维度。
    对当前 20 个 debug 样本来说，这会严重过拟合。
    因此，第一版 MLP baseline 先把每层时程压缩成统计特征：
        mean / std / max / min / rms
    4 层 × 5 个统计量 = 20 个 MLP 输入特征。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

FEATURE_NAMES = ["mean", "std", "max", "min", "rms"]
# FEATURE_NAMES：每个楼层通道要提取的统计量名称。


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load .npz file as a normal dictionary. 中文说明：把 .npz 文件读取为普通字典。"""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.keys()}


def check_split_dataset(data: dict[str, np.ndarray]) -> None:
    """Check required train/val/test arrays. 中文说明：检查必要数组是否存在。"""
    required = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key: {key}. Available keys: {list(data.keys())}")


def check_X(name: str, X: np.ndarray) -> None:
    """Check one time-history array. 中文说明：检查时程数组形状和数值。"""
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape (n_samples, n_steps, n_channels), got {X.shape}")
    if np.isnan(X).any():
        raise ValueError(f"{name} contains NaN.")
    if np.isinf(X).any():
        raise ValueError(f"{name} contains Inf.")


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract mean/std/max/min/rms from X.

    Parameters:
        X: shape = (n_samples, n_steps, n_channels)

    Returns:
        F: shape = (n_samples, n_channels * 5)

    中文说明：
        axis=1 是时间维度，所以这里对每一个样本、每一个楼层通道沿时间方向提取统计量。
    """
    check_X("X", X)

    mean_feature = np.mean(X, axis=1)
    # mean_feature：每个样本、每层响应的时间均值，形状 (n_samples, n_channels)。

    std_feature = np.std(X, axis=1)
    # std_feature：每个样本、每层响应的时间标准差。

    max_feature = np.max(X, axis=1)
    # max_feature：每个样本、每层响应的最大值。

    min_feature = np.min(X, axis=1)
    # min_feature：每个样本、每层响应的最小值。

    rms_feature = np.sqrt(np.mean(X ** 2, axis=1))
    # rms_feature：均方根，反映响应能量水平。

    F = np.concatenate(
        [mean_feature, std_feature, max_feature, min_feature, rms_feature],
        axis=1,
    )
    # concatenate(axis=1)：把五组统计量横向拼接。
    # 若 n_channels=4，则输出维度为 4*5=20。

    return F.astype(np.float64)


def build_feature_names(n_channels: int) -> np.ndarray:
    """Build feature-name array. 中文说明：生成每一列特征的名称。"""
    names = []
    for feature_name in FEATURE_NAMES:
        for channel_idx in range(n_channels):
            names.append(f"story_{channel_idx + 1}_{feature_name}")
    return np.array(names, dtype=str)


def copy_optional_metadata(data: dict[str, np.ndarray], save_dict: dict[str, np.ndarray]) -> None:
    """Copy optional metadata. 中文说明：同步复制样本元数据。"""
    optional_keys = [
        "train_idx", "val_idx", "test_idx",
        "amplitude_g_train", "amplitude_g_val", "amplitude_g_test",
        "frequency_hz_train", "frequency_hz_val", "frequency_hz_test",
        "noise_level_train", "noise_level_val", "noise_level_test",
        "case_id_train", "case_id_val", "case_id_test",
    ]
    for key in optional_keys:
        if key in data:
            save_dict[key] = data[key]


def run(input_path: Path, output_path: Path) -> None:
    """Run full feature extraction. 中文说明：执行完整特征提取流程。"""
    data = load_npz(input_path)
    check_split_dataset(data)

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    F_train = extract_features(X_train)
    F_val = extract_features(X_val)
    F_test = extract_features(X_test)

    feature_names = build_feature_names(X_train.shape[2])

    save_dict = {
        "F_train": F_train,
        "F_val": F_val,
        "F_test": F_test,
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "y_test": data["y_test"],
        "feature_names": feature_names,
    }
    copy_optional_metadata(data, save_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    print("Feature extraction completed.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"F_train shape: {F_train.shape}")
    print(f"F_val shape: {F_val.shape}")
    print(f"F_test shape: {F_test.shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    print(f"Feature names: {feature_names.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data_processed/debug_split_normalized.npz")
    parser.add_argument("--output", type=str, default="data_processed/debug_features_mlp.npz")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
