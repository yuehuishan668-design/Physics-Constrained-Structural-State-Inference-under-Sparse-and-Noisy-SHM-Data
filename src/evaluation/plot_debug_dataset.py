"""
File location:
    src/evaluation/plot_debug_dataset.py

Purpose:
    Visualize the generated debug dataset before model training.

Input:
    data_processed/debug_dataset.npz

Outputs:
    results/figures/debug_dataset/
        case_XXXX_acceleration.png
        case_XXXX_damage_label.png
        dataset_damage_distribution.png
        dataset_noise_distribution.png
        dataset_response_amplitude_summary.png

中文说明：
    本文件用于在训练模型前可视化检查调试数据集。
    这一步不是论文最终作图，而是为了检查数据是否存在明显异常。
"""

from __future__ import annotations
# 延迟类型注解解析。

import argparse
# argparse：读取命令行参数。

from pathlib import Path
# Path：处理文件路径。

from typing import Dict
# Dict：类型注解。

import numpy as np
# numpy：读取 .npz 和数组计算。

import matplotlib.pyplot as plt
# matplotlib：绘图。


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    """
    Load the debug dataset.

    中文说明：
        读取 data_processed/debug_dataset.npz。
    """

    if not path.exists():
        # 检查数据集是否存在。
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path, allow_pickle=False)
    # 读取 .npz 文件。

    return {key: data[key] for key in data.keys()}
    # 转成普通字典。


def check_required_keys(dataset: Dict[str, np.ndarray]) -> None:
    """
    Check required keys.

    中文说明：
        检查可视化需要的关键数组是否存在。
    """

    for key in ["X_abs_accel", "y_damage", "dt"]:
        # 逐个检查必要 key。

        if key not in dataset:
            # 如果缺失则报错。
            raise KeyError(f"Required key '{key}' not found. Available keys: {list(dataset.keys())}")


def select_case_ids(n_samples: int, n_cases: int, seed: int) -> np.ndarray:
    """
    Select random cases for plotting.

    中文说明：
        随机选择若干个样本用于绘图。
    """

    rng = np.random.default_rng(seed)
    # 创建随机数生成器。

    n_cases = min(n_cases, n_samples)
    # 防止要求绘制数量超过总样本数。

    case_ids = rng.choice(np.arange(n_samples), size=n_cases, replace=False)
    # 不重复随机抽样。

    return np.sort(case_ids)
    # 排序后返回，便于查看。


def plot_case_acceleration(X: np.ndarray, case_id: int, dt: float, output_dir: Path) -> None:
    """
    Plot floor acceleration time histories for one case.

    中文说明：
        绘制某个样本的各楼层绝对加速度时程。
    """

    response = X[case_id]
    # 提取该样本响应，形状为 (time_steps, n_story)。

    n_steps = response.shape[0]
    # 时间步数。

    n_story = response.shape[1]
    # 楼层数/通道数。

    time = np.arange(n_steps) * dt
    # 构造时间轴。

    plt.figure(figsize=(10, 5))
    # 创建图像。

    for story_idx in range(n_story):
        # 遍历楼层。

        plt.plot(time, response[:, story_idx], label=f"Story {story_idx + 1}")
        # 绘制该楼层加速度。

    plt.xlabel("Time [s]")
    # 横轴标签。

    plt.ylabel("Absolute acceleration [m/s²]")
    # 纵轴标签。

    plt.title(f"Case {case_id:04d}: floor absolute acceleration")
    # 图标题。

    plt.legend()
    # 图例。

    plt.tight_layout()
    # 自动调整布局。

    plt.savefig(output_dir / f"case_{case_id:04d}_acceleration.png", dpi=200)
    # 保存图像。

    plt.close()
    # 关闭图像，释放内存。


def plot_case_damage_label(y: np.ndarray, case_id: int, output_dir: Path) -> None:
    """
    Plot damage label for one case.

    中文说明：
        绘制某个样本的层刚度退化标签。
    """

    damage = y[case_id]
    # 提取损伤标签。

    story_ids = np.arange(1, damage.shape[0] + 1)
    # 楼层编号，从 1 开始。

    plt.figure(figsize=(6, 4))
    # 创建图像。

    plt.bar(story_ids, damage)
    # 绘制柱状图。

    plt.xlabel("Story")
    # 横轴标签。

    plt.ylabel("Stiffness degradation ratio")
    # 纵轴标签。

    plt.ylim(0.0, max(0.40, float(damage.max()) + 0.05))
    # 设置纵轴范围。

    plt.title(f"Case {case_id:04d}: damage label")
    # 图标题。

    plt.tight_layout()
    # 自动调整布局。

    plt.savefig(output_dir / f"case_{case_id:04d}_damage_label.png", dpi=200)
    # 保存图像。

    plt.close()
    # 关闭图像。


def plot_damage_distribution(y: np.ndarray, output_dir: Path) -> None:
    """
    Plot damage distribution for all cases.

    中文说明：
        绘制各楼层损伤值分布，用于检查数据集是否过于偏斜。
    """

    n_story = y.shape[1]
    # 楼层数量。

    plt.figure(figsize=(8, 4))
    # 创建图像。

    for story_idx in range(n_story):
        # 遍历楼层。

        plt.hist(y[:, story_idx], bins=10, alpha=0.5, label=f"Story {story_idx + 1}")
        # 绘制直方图。

    plt.xlabel("Stiffness degradation ratio")
    # 横轴标签。

    plt.ylabel("Count")
    # 纵轴标签。

    plt.title("Damage distribution by story")
    # 图标题。

    plt.legend()
    # 图例。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(output_dir / "dataset_damage_distribution.png", dpi=200)
    # 保存图像。

    plt.close()
    # 关闭图像。


def plot_noise_distribution(dataset: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plot noise-level distribution if available.

    中文说明：
        如果数据集中有 noise_level，则绘制噪声水平分布。
    """

    if "noise_level" not in dataset:
        # 如果没有 noise_level，则跳过。
        print("Skip noise distribution: key 'noise_level' not found.")
        return

    noise_level = dataset["noise_level"]
    # 读取噪声水平。

    plt.figure(figsize=(6, 4))
    # 创建图像。

    plt.hist(noise_level, bins=[-0.01, 0.025, 0.075, 0.15, 0.25])
    # 绘制噪声水平直方图。

    plt.xlabel("Noise level")
    # 横轴标签。

    plt.ylabel("Count")
    # 纵轴标签。

    plt.title("Noise-level distribution")
    # 图标题。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(output_dir / "dataset_noise_distribution.png", dpi=200)
    # 保存图像。

    plt.close()
    # 关闭图像。


def plot_response_amplitude_summary(X: np.ndarray, output_dir: Path) -> None:
    """
    Plot maximum response amplitude distribution.

    中文说明：
        绘制每个样本最大绝对加速度分布，用于检查是否存在异常爆炸响应。
    """

    max_abs_accel = np.max(np.abs(X), axis=(1, 2))
    # 每个样本在所有时间步和楼层中的最大绝对加速度。

    plt.figure(figsize=(7, 4))
    # 创建图像。

    plt.hist(max_abs_accel, bins=10)
    # 绘制直方图。

    plt.xlabel("Maximum absolute acceleration [m/s²]")
    # 横轴标签。

    plt.ylabel("Count")
    # 纵轴标签。

    plt.title("Response amplitude summary")
    # 图标题。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(output_dir / "dataset_response_amplitude_summary.png", dpi=200)
    # 保存图像。

    plt.close()
    # 关闭图像。


def plot_debug_dataset(dataset_path: Path, output_dir: Path, n_cases: int, seed: int) -> None:
    """
    Main plotting pipeline.

    中文说明：
        可视化主流程。
    """

    dataset = load_dataset(dataset_path)
    # 读取数据集。

    check_required_keys(dataset)
    # 检查关键字段。

    X = dataset["X_abs_accel"]
    # 输入加速度，形状为 (n_samples, n_steps, n_story)。

    y = dataset["y_damage"]
    # 损伤标签，形状为 (n_samples, n_story)。

    dt = float(dataset["dt"])
    # 时间步长。

    output_dir.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。

    case_ids = select_case_ids(X.shape[0], n_cases, seed)
    # 随机选择要绘制的样本。

    for case_id in case_ids:
        # 遍历选中样本。

        plot_case_acceleration(X, int(case_id), dt, output_dir)
        # 绘制加速度时程。

        plot_case_damage_label(y, int(case_id), output_dir)
        # 绘制损伤标签。

    plot_damage_distribution(y, output_dir)
    # 绘制损伤分布。

    plot_noise_distribution(dataset, output_dir)
    # 绘制噪声分布。

    plot_response_amplitude_summary(X, output_dir)
    # 绘制响应幅值分布。

    print("Debug dataset plotting completed.")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Plotted case IDs: {case_ids.tolist()}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    中文说明：
        解析命令行参数。
    """

    parser = argparse.ArgumentParser()
    # 创建解析器。

    parser.add_argument("--dataset", type=str, default="data_processed/debug_dataset.npz")
    # --dataset：输入数据集路径。

    parser.add_argument("--output-dir", type=str, default="results/figures/debug_dataset")
    # --output-dir：图片输出目录。

    parser.add_argument("--n-cases", type=int, default=4)
    # --n-cases：随机绘制样本数量。

    parser.add_argument("--seed", type=int, default=20260622)
    # --seed：随机种子。

    return parser.parse_args()
    # 返回参数。


def main() -> None:
    """Command-line entry point. 中文说明：命令行入口。"""

    args = parse_args()
    # 解析参数。

    plot_debug_dataset(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        n_cases=args.n_cases,
        seed=args.seed,
    )
    # 执行绘图。


if __name__ == "__main__":
    main()
