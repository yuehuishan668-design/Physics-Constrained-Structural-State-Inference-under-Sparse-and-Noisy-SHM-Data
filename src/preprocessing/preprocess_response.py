"""
File location:
    src/preprocessing/preprocess_response.py

Purpose:
    Preprocess the OpenSeesPy debug dataset before PyTorch training.

Input:
    data_processed/debug_dataset.npz

Output:
    data_processed/debug_split_normalized.npz

Main steps:
    1. Load X_abs_accel and y_damage.
    2. Check shape, NaN, and Inf.
    3. Split samples into train / validation / test.
    4. Compute normalization statistics from the training set only.
    5. Normalize X.
    6. Save the preprocessed dataset.

中文说明：
    本文件用于把 OpenSeesPy 生成的 debug 数据集整理成后续 PyTorch 可以直接读取的格式。
    注意：标准化参数只能用训练集计算，不能用验证集或测试集，否则会产生数据泄漏。
"""

from __future__ import annotations
# 延迟类型注解解析；对代码运行逻辑没有直接影响，但能提高兼容性。

import argparse
# argparse：从命令行读取参数，例如 --input、--output、--seed。

from dataclasses import dataclass
# dataclass：用于定义配置类，减少重复的 __init__ 代码。

from pathlib import Path
# Path：用于处理文件路径，比普通字符串路径更稳妥。

from typing import Dict, Tuple
# Dict、Tuple：类型注解，用于说明函数返回值和变量结构。

import numpy as np
# numpy：用于读取 .npz、数组计算、随机划分、保存数据。


@dataclass
class PreprocessConfig:
    """Preprocessing configuration. 中文说明：预处理配置。"""

    input_path: Path = Path("data_processed/debug_dataset.npz")
    # input_path：输入数据集路径。

    output_path: Path = Path("data_processed/debug_split_normalized.npz")
    # output_path：预处理后数据保存路径。

    x_key: str = "X_abs_accel"
    # x_key：输入响应数组的键名；当前使用带噪绝对加速度。

    y_key: str = "y_damage"
    # y_key：标签数组的键名；当前为层刚度退化向量。

    train_ratio: float = 0.70
    # train_ratio：训练集比例。

    val_ratio: float = 0.15
    # val_ratio：验证集比例。

    test_ratio: float = 0.15
    # test_ratio：测试集比例。

    seed: int = 20260622
    # seed：随机种子，用于保证样本划分可复现。

    eps: float = 1.0e-8
    # eps：避免除以 0 的极小数。


def load_npz_dataset(path: Path, x_key: str, y_key: str) -> Dict[str, np.ndarray]:
    """
    Load a .npz dataset and return a normal dictionary.

    中文说明：
        读取 .npz 文件，并转成普通 dict，便于后续处理。
    """

    if not path.exists():
        # 检查输入文件是否存在。
        raise FileNotFoundError(f"Input dataset not found: {path}")

    data = np.load(path, allow_pickle=False)
    # np.load：读取 .npz 文件。
    # allow_pickle=False：更安全，不加载 Python 对象。

    dataset = {key: data[key] for key in data.keys()}
    # 将 NpzFile 转成普通字典。

    if x_key not in dataset:
        # 检查输入数组是否存在。
        raise KeyError(f"Input key '{x_key}' not found. Available keys: {list(dataset.keys())}")

    if y_key not in dataset:
        # 检查标签数组是否存在。
        raise KeyError(f"Label key '{y_key}' not found. Available keys: {list(dataset.keys())}")

    return dataset


def validate_xy(X: np.ndarray, y: np.ndarray) -> None:
    """
    Check shape and numerical validity of X and y.

    中文说明：
        检查输入和标签是否满足训练前的基本要求。
    """

    if X.ndim != 3:
        # X 应为三维：(样本数, 时间步数, 楼层/通道数)。
        raise ValueError(f"X must have shape (n_samples, n_steps, n_channels), got {X.shape}")

    if y.ndim != 2:
        # y 应为二维：(样本数, 损伤变量数)。
        raise ValueError(f"y must have shape (n_samples, n_targets), got {y.shape}")

    if X.shape[0] != y.shape[0]:
        # X 和 y 的样本数必须一致。
        raise ValueError(f"X and y sample counts do not match: {X.shape[0]} vs {y.shape[0]}")

    if np.isnan(X).any() or np.isnan(y).any():
        # 检查 NaN。
        raise ValueError("X or y contains NaN.")

    if np.isinf(X).any() or np.isinf(y).any():
        # 检查 Inf。
        raise ValueError("X or y contains Inf.")

    if X.shape[0] < 5:
        # 样本太少时，train/val/test 划分不可靠。
        raise ValueError(f"Too few samples: {X.shape[0]}")


def split_indices(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train / validation / test sample indices.

    中文说明：
        按样本编号划分数据集。
        注意：不要按时间步划分，否则同一个时程会泄漏到训练集和测试集。
    """

    ratio_sum = train_ratio + val_ratio + test_ratio
    # 计算比例总和。

    if not np.isclose(ratio_sum, 1.0):
        # 三个比例之和必须为 1。
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    rng = np.random.default_rng(seed)
    # 创建随机数生成器。

    indices = np.arange(n_samples)
    # 生成样本编号。

    rng.shuffle(indices)
    # 打乱样本编号。

    n_train = int(round(n_samples * train_ratio))
    # 训练集样本数。

    n_val = int(round(n_samples * val_ratio))
    # 验证集样本数。

    n_test = n_samples - n_train - n_val
    # 测试集样本数。

    if min(n_train, n_val, n_test) <= 0:
        # 防止任何一个集合为空。
        raise ValueError(f"Invalid split sizes: {n_train}, {n_val}, {n_test}")

    train_idx = indices[:n_train]
    # 训练集索引。

    val_idx = indices[n_train:n_train + n_val]
    # 验证集索引。

    test_idx = indices[n_train + n_val:]
    # 测试集索引。

    return train_idx, val_idx, test_idx


def compute_train_stats(X_train: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute channel-wise mean and standard deviation from training data only.

    中文说明：
        只用训练集计算每个楼层/通道的均值和标准差。
    """

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    # 对样本维度和时间维度求均值，保留通道维度。
    # 输出形状：(1, 1, n_channels)。

    std = X_train.std(axis=(0, 1), keepdims=True)
    # 对样本维度和时间维度求标准差。

    std = np.maximum(std, eps)
    # 防止标准差为 0。

    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalize X.

    中文说明：
        用训练集均值和标准差对输入进行标准化。
    """

    return (X - mean) / std
    # 标准化公式：X_norm = (X - mean) / std。


def subset_if_sample_level(
    dataset: Dict[str, np.ndarray],
    key: str,
    indices: np.ndarray,
    n_samples: int,
) -> np.ndarray | None:
    """
    Subset metadata arrays such as noise_level, case_id, amplitude_g.

    中文说明：
        如果某个数组是按样本保存的元数据，则按照 train/val/test 索引同步划分。
    """

    if key not in dataset:
        # 如果 key 不存在，跳过。
        return None

    arr = dataset[key]
    # 读取数组。

    if arr.shape[0] != n_samples:
        # 如果第一维不是样本数，则不是样本级元数据。
        return None

    return arr[indices]
    # 返回对应子集。


def preprocess_dataset(config: PreprocessConfig) -> None:
    """
    Execute the full preprocessing pipeline.

    中文说明：
        执行完整预处理流程。
    """

    dataset = load_npz_dataset(config.input_path, config.x_key, config.y_key)
    # 读取数据集。

    X = dataset[config.x_key].astype(np.float64)
    # 读取输入 X，并转为 float64。

    y = dataset[config.y_key].astype(np.float64)
    # 读取标签 y，并转为 float64。

    validate_xy(X, y)
    # 检查数据质量。

    n_samples = X.shape[0]
    # 样本数。

    train_idx, val_idx, test_idx = split_indices(
        n_samples=n_samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    # 生成训练/验证/测试索引。

    X_train_raw = X[train_idx]
    # 未标准化训练集输入。

    X_val_raw = X[val_idx]
    # 未标准化验证集输入。

    X_test_raw = X[test_idx]
    # 未标准化测试集输入。

    y_train = y[train_idx]
    # 训练集标签。

    y_val = y[val_idx]
    # 验证集标签。

    y_test = y[test_idx]
    # 测试集标签。

    x_mean, x_std = compute_train_stats(X_train_raw, config.eps)
    # 只用训练集计算标准化参数。

    X_train = normalize(X_train_raw, x_mean, x_std)
    # 标准化训练集。

    X_val = normalize(X_val_raw, x_mean, x_std)
    # 标准化验证集。

    X_test = normalize(X_test_raw, x_mean, x_std)
    # 标准化测试集。

    save_dict = {
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
    }
    # 构造保存字典。

    for key in ["amplitude_g", "frequency_hz", "noise_level", "case_id"]:
        # 遍历可选元数据。

        train_part = subset_if_sample_level(dataset, key, train_idx, n_samples)
        # 训练集元数据。

        val_part = subset_if_sample_level(dataset, key, val_idx, n_samples)
        # 验证集元数据。

        test_part = subset_if_sample_level(dataset, key, test_idx, n_samples)
        # 测试集元数据。

        if train_part is not None:
            # 如果存在该元数据，则保存。
            save_dict[f"{key}_train"] = train_part
            save_dict[f"{key}_val"] = val_part
            save_dict[f"{key}_test"] = test_part

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。

    np.savez_compressed(config.output_path, **save_dict)
    # 保存预处理数据。

    print("Preprocessing completed.")
    print(f"Input dataset: {config.input_path}")
    print(f"Output dataset: {config.output_path}")
    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"x_mean shape: {x_mean.shape}")
    print(f"x_std shape: {x_std.shape}")
    print(f"Normalized train mean: {X_train.mean():.6f}")
    print(f"Normalized train std: {X_train.std():.6f}")


def parse_args() -> PreprocessConfig:
    """
    Parse command-line arguments.

    中文说明：
        解析命令行参数；如果不传参数，则使用默认路径。
    """

    parser = argparse.ArgumentParser()
    # 创建命令行解析器。

    parser.add_argument("--input", type=str, default="data_processed/debug_dataset.npz")
    # --input：输入数据路径。

    parser.add_argument("--output", type=str, default="data_processed/debug_split_normalized.npz")
    # --output：输出数据路径。

    parser.add_argument("--seed", type=int, default=20260622)
    # --seed：随机种子。

    parser.add_argument("--train-ratio", type=float, default=0.70)
    # --train-ratio：训练集比例。

    parser.add_argument("--val-ratio", type=float, default=0.15)
    # --val-ratio：验证集比例。

    parser.add_argument("--test-ratio", type=float, default=0.15)
    # --test-ratio：测试集比例。

    args = parser.parse_args()
    # 解析参数。

    return PreprocessConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    # 返回配置对象。


def main() -> None:
    """Command-line entry point. 中文说明：命令行入口。"""

    config = parse_args()
    # 获取配置。

    preprocess_dataset(config)
    # 执行预处理。


if __name__ == "__main__":
    main()
