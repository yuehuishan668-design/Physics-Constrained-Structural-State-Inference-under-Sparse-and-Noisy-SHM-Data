"""
File location:
    src/evaluation/analyze_dataset_distribution.py

Purpose:
    Diagnose the generated OpenSeesPy dataset before moving to sequence models such as LSTM / TCN.

Default input:
    data_processed/debug_plus_100_dataset.npz

Default outputs:
    results/tables/dataset_distribution/debug_plus_100_distribution_summary.json
    results/tables/dataset_distribution/debug_plus_100_top_response_cases.csv
    results/tables/dataset_distribution/debug_plus_100_case_summary.csv

    results/figures/dataset_distribution/debug_plus_100_damage_distribution_by_story.png
    results/figures/dataset_distribution/debug_plus_100_damaged_story_count.png
    results/figures/dataset_distribution/debug_plus_100_excitation_distribution.png
    results/figures/dataset_distribution/debug_plus_100_noise_distribution.png
    results/figures/dataset_distribution/debug_plus_100_max_response_distribution.png
    results/figures/dataset_distribution/debug_plus_100_frequency_vs_max_response.png
    results/figures/dataset_distribution/debug_plus_100_amplitude_vs_max_response.png
    results/figures/dataset_distribution/debug_plus_100_damage_sum_vs_max_response.png
    results/figures/dataset_distribution/debug_plus_100_damage_heatmap_cases_by_story.png

Research role:
    This script checks whether the dataset distribution is reasonable before training
    stronger sequence models.

    It answers:
        1. How many healthy / damaged cases exist?
        2. How many stories are damaged per case?
        3. Which stories are damaged most often?
        4. Are damage values distributed reasonably?
        5. Are excitation amplitude and frequency ranges correct?
        6. Are noise levels balanced?
        7. Are there extreme response cases?
        8. Are extreme responses related to frequency, amplitude, or damage level?
        9. Are train / validation / test splits roughly comparable?

中文说明：
    本文件用于在进入 LSTM / TCN 等时序模型前，对 100-case 数据集进行系统诊断。
    当前 MLP baseline 的结果已经说明，仅用统计特征很难稳定识别损伤位置。
    但在升级模型之前，必须确认数据集本身没有明显偏差或异常：
        1. 损伤样本比例是否合理；
        2. 每层损伤出现频率是否严重不平衡；
        3. 激励幅值与频率是否覆盖预设范围；
        4. 噪声水平是否分布正常；
        5. 是否存在极端响应样本；
        6. train / val / test 划分是否分布相近。
"""

from __future__ import annotations
# 延迟类型注解解析，提高兼容性。

import argparse
# argparse：用于读取命令行参数。

import csv
# csv：用于保存 case summary 和 top response cases。

import json
# json：用于保存总体统计摘要。

from pathlib import Path
# Path：用于处理文件路径。

from typing import Dict, List
# Dict、List：类型注解。

import numpy as np
# numpy：用于读取 .npz 文件和数组统计。

import matplotlib.pyplot as plt
# matplotlib：用于绘图。


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """
    Load a .npz file as a normal dictionary.

    中文说明：
        读取 .npz 文件，并转换成普通 dict。
    """

    if not path.exists():
        # 如果输入文件不存在，直接报错。
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path, allow_pickle=False)
    # np.load：读取 .npz 文件。
    # allow_pickle=False：不允许加载 Python 对象，更安全。

    return {key: data[key] for key in data.keys()}
    # 转成普通字典。


def require_keys(data: Dict[str, np.ndarray], keys: List[str]) -> None:
    """
    Check required keys.

    中文说明：
        检查数据集中是否包含必要字段。
    """

    for key in keys:
        # 遍历必要键名。

        if key not in data:
            # 如果缺少必要键名，报错并显示已有键名。
            raise KeyError(f"Required key '{key}' not found. Available keys: {list(data.keys())}")


def safe_float(value: np.ndarray | float | int) -> float:
    """
    Convert a scalar-like value to Python float.

    中文说明：
        将 NumPy 标量安全转换为 Python float，便于 JSON 保存。
    """

    return float(np.asarray(value))
    # np.asarray 可以兼容 Python 标量和 NumPy 标量。


def compute_case_summary(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute per-case summary arrays.

    中文说明：
        计算每个 case 的样本级统计信息。
    """

    require_keys(
        data,
        [
            "X_abs_accel",
            "y_damage",
            "amplitude_g",
            "frequency_hz",
            "noise_level",
            "case_id",
        ],
    )
    # 检查必要字段。

    X = data["X_abs_accel"]
    # 结构绝对加速度响应，形状：(n_cases, n_steps, n_story)。

    y = data["y_damage"]
    # 损伤标签，形状：(n_cases, n_story)。

    if X.ndim != 3:
        # X 必须是三维数组。
        raise ValueError(f"X_abs_accel must be 3D, got shape {X.shape}")

    if y.ndim != 2:
        # y 必须是二维数组。
        raise ValueError(f"y_damage must be 2D, got shape {y.shape}")

    if X.shape[0] != y.shape[0]:
        # 样本数必须一致。
        raise ValueError(f"X and y have different sample counts: {X.shape[0]} vs {y.shape[0]}")

    max_abs_response = np.max(np.abs(X), axis=(1, 2))
    # 每个 case 的最大绝对加速度。

    rms_response = np.sqrt(np.mean(X ** 2, axis=(1, 2)))
    # 每个 case 的响应 RMS，用于描述整体响应能量。

    damage_sum = np.sum(y, axis=1)
    # 每个 case 的总损伤量。

    damage_max = np.max(y, axis=1)
    # 每个 case 的最大单层损伤。

    damaged_story_count = np.sum(y > 0.0, axis=1)
    # 每个 case 的受损层数。

    is_healthy = damaged_story_count == 0
    # 是否完全健康。

    return {
        "case_id": data["case_id"].astype(int),
        "max_abs_response": max_abs_response,
        "rms_response": rms_response,
        "damage_sum": damage_sum,
        "damage_max": damage_max,
        "damaged_story_count": damaged_story_count.astype(int),
        "is_healthy": is_healthy,
        "amplitude_g": data["amplitude_g"],
        "frequency_hz": data["frequency_hz"],
        "noise_level": data["noise_level"],
        "y_damage": y,
    }
    # 返回样本级摘要。


def compute_global_summary(case_summary: Dict[str, np.ndarray], data: Dict[str, np.ndarray]) -> Dict[str, object]:
    """
    Compute global dataset summary.

    中文说明：
        计算整体数据集摘要，用于保存 JSON。
    """

    y = case_summary["y_damage"]
    # 损伤矩阵。

    n_cases = int(y.shape[0])
    # 样本数。

    n_story = int(y.shape[1])
    # 楼层数。

    damaged_story_count = case_summary["damaged_story_count"]
    # 每个 case 受损层数。

    is_healthy = case_summary["is_healthy"]
    # 健康样本标记。

    max_abs_response = case_summary["max_abs_response"]
    # 每个 case 最大响应。

    story_damage_frequency = np.mean(y > 0.0, axis=0)
    # 每层出现损伤的比例。

    story_damage_mean_all = np.mean(y, axis=0)
    # 每层在所有样本上的平均损伤。

    story_damage_mean_damaged_only = []
    # 每层只在受损样本上的平均损伤。

    for story_idx in range(n_story):
        # 遍历每层。

        damaged_values = y[y[:, story_idx] > 0.0, story_idx]
        # 提取该层非零损伤值。

        if damaged_values.size == 0:
            # 如果该层没有损伤样本。
            story_damage_mean_damaged_only.append(0.0)
        else:
            story_damage_mean_damaged_only.append(float(np.mean(damaged_values)))

    quantiles = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    # 响应分位数。

    max_response_quantiles = {
        str(q): float(np.quantile(max_abs_response, q))
        for q in quantiles
    }
    # 最大响应分位数统计。

    unique_noise, noise_counts = np.unique(case_summary["noise_level"], return_counts=True)
    # 噪声水平计数。

    noise_distribution = {
        str(float(level)): int(count)
        for level, count in zip(unique_noise, noise_counts)
    }
    # 噪声分布字典。

    unique_damaged_counts, damaged_count_counts = np.unique(damaged_story_count, return_counts=True)
    # 受损层数计数。

    damaged_story_count_distribution = {
        str(int(count_key)): int(count_value)
        for count_key, count_value in zip(unique_damaged_counts, damaged_count_counts)
    }
    # 受损层数分布。

    summary = {
        "n_cases": n_cases,
        "n_story": n_story,
        "x_shape": list(data["X_abs_accel"].shape),
        "y_shape": list(data["y_damage"].shape),
        "has_nan_X": bool(np.isnan(data["X_abs_accel"]).any()),
        "has_inf_X": bool(np.isinf(data["X_abs_accel"]).any()),
        "has_nan_y": bool(np.isnan(data["y_damage"]).any()),
        "has_inf_y": bool(np.isinf(data["y_damage"]).any()),
        "healthy_case_count": int(np.sum(is_healthy)),
        "damaged_case_count": int(np.sum(~is_healthy)),
        "healthy_case_ratio": float(np.mean(is_healthy)),
        "damaged_case_ratio": float(np.mean(~is_healthy)),
        "damaged_story_count_distribution": damaged_story_count_distribution,
        "story_damage_frequency": story_damage_frequency.tolist(),
        "story_damage_mean_all_cases": story_damage_mean_all.tolist(),
        "story_damage_mean_damaged_only": story_damage_mean_damaged_only,
        "damage_mean_overall": float(np.mean(y)),
        "damage_max_overall": float(np.max(y)),
        "damage_sum_mean": float(np.mean(case_summary["damage_sum"])),
        "amplitude_g_min": float(np.min(case_summary["amplitude_g"])),
        "amplitude_g_max": float(np.max(case_summary["amplitude_g"])),
        "amplitude_g_mean": float(np.mean(case_summary["amplitude_g"])),
        "frequency_hz_min": float(np.min(case_summary["frequency_hz"])),
        "frequency_hz_max": float(np.max(case_summary["frequency_hz"])),
        "frequency_hz_mean": float(np.mean(case_summary["frequency_hz"])),
        "noise_distribution": noise_distribution,
        "max_abs_response_min": float(np.min(max_abs_response)),
        "max_abs_response_max": float(np.max(max_abs_response)),
        "max_abs_response_mean": float(np.mean(max_abs_response)),
        "max_abs_response_quantiles": max_response_quantiles,
    }
    # 汇总整体统计量。

    return summary


def write_json(path: Path, obj: Dict[str, object]) -> None:
    """
    Write JSON file.

    中文说明：
        保存 JSON 文件。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。

    with path.open("w", encoding="utf-8") as f:
        # 打开 JSON 文件。

        json.dump(obj, f, indent=2, ensure_ascii=False)
        # 写入 JSON。


def write_case_summary_csv(path: Path, case_summary: Dict[str, np.ndarray]) -> None:
    """
    Write per-case summary CSV.

    中文说明：
        保存每个 case 的摘要 CSV。
        这个文件用于快速定位异常样本。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。

    y = case_summary["y_damage"]
    # 损伤矩阵。

    n_story = y.shape[1]
    # 楼层数。

    fieldnames = [
        "case_id",
        "max_abs_response",
        "rms_response",
        "damage_sum",
        "damage_max",
        "damaged_story_count",
        "is_healthy",
        "amplitude_g",
        "frequency_hz",
        "noise_level",
    ]
    # 基础列。

    for story_idx in range(n_story):
        # 添加每层损伤列。
        fieldnames.append(f"damage_story_{story_idx + 1}")

    with path.open("w", newline="", encoding="utf-8") as f:
        # 打开 CSV 文件。

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 创建 CSV 写入器。

        writer.writeheader()
        # 写入表头。

        for i in range(y.shape[0]):
            # 遍历每个 case。

            row = {
                "case_id": int(case_summary["case_id"][i]),
                "max_abs_response": float(case_summary["max_abs_response"][i]),
                "rms_response": float(case_summary["rms_response"][i]),
                "damage_sum": float(case_summary["damage_sum"][i]),
                "damage_max": float(case_summary["damage_max"][i]),
                "damaged_story_count": int(case_summary["damaged_story_count"][i]),
                "is_healthy": bool(case_summary["is_healthy"][i]),
                "amplitude_g": float(case_summary["amplitude_g"][i]),
                "frequency_hz": float(case_summary["frequency_hz"][i]),
                "noise_level": float(case_summary["noise_level"][i]),
            }
            # 当前 case 基础信息。

            for story_idx in range(n_story):
                # 写入每层损伤。
                row[f"damage_story_{story_idx + 1}"] = float(y[i, story_idx])

            writer.writerow(row)
            # 写入一行。


def write_top_response_cases_csv(path: Path, case_summary: Dict[str, np.ndarray], top_k: int) -> None:
    """
    Write top response cases CSV.

    中文说明：
        保存最大响应最高的若干个 case。
        这用于检查极端响应是否来自近共振、高输入幅值或严重损伤。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。

    y = case_summary["y_damage"]
    # 损伤矩阵。

    n_story = y.shape[1]
    # 楼层数。

    order = np.argsort(case_summary["max_abs_response"])[::-1]
    # 按最大响应从大到小排序。

    top_indices = order[: min(top_k, len(order))]
    # 取前 top_k 个。

    fieldnames = [
        "rank",
        "case_id",
        "max_abs_response",
        "rms_response",
        "damage_sum",
        "damage_max",
        "damaged_story_count",
        "amplitude_g",
        "frequency_hz",
        "noise_level",
    ]
    # 基础列。

    for story_idx in range(n_story):
        # 每层损伤列。
        fieldnames.append(f"damage_story_{story_idx + 1}")

    with path.open("w", newline="", encoding="utf-8") as f:
        # 打开 CSV 文件。

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 创建写入器。

        writer.writeheader()
        # 写入表头。

        for rank, i in enumerate(top_indices, start=1):
            # 遍历 top 响应 case。

            row = {
                "rank": rank,
                "case_id": int(case_summary["case_id"][i]),
                "max_abs_response": float(case_summary["max_abs_response"][i]),
                "rms_response": float(case_summary["rms_response"][i]),
                "damage_sum": float(case_summary["damage_sum"][i]),
                "damage_max": float(case_summary["damage_max"][i]),
                "damaged_story_count": int(case_summary["damaged_story_count"][i]),
                "amplitude_g": float(case_summary["amplitude_g"][i]),
                "frequency_hz": float(case_summary["frequency_hz"][i]),
                "noise_level": float(case_summary["noise_level"][i]),
            }
            # 当前 top case 信息。

            for story_idx in range(n_story):
                # 写入每层损伤。
                row[f"damage_story_{story_idx + 1}"] = float(y[i, story_idx])

            writer.writerow(row)
            # 写入行。


def plot_damage_distribution_by_story(case_summary: Dict[str, np.ndarray], path: Path, title_prefix: str) -> None:
    """
    Plot nonzero damage distribution by story.

    中文说明：
        绘制每层非零损伤值分布。
    """

    y = case_summary["y_damage"]
    # 损伤矩阵。

    n_story = y.shape[1]
    # 楼层数。

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建目录。

    plt.figure(figsize=(8, 4))
    # 创建图像。

    for story_idx in range(n_story):
        # 遍历楼层。

        values = y[y[:, story_idx] > 0.0, story_idx]
        # 取该层非零损伤值。

        if values.size > 0:
            # 如果该层有损伤样本。

            plt.hist(values, bins=10, alpha=0.5, label=f"Story {story_idx + 1}")
            # 绘制直方图。

    plt.xlabel("Stiffness degradation ratio")
    # 横轴标签。

    plt.ylabel("Count")
    # 纵轴标签。

    plt.title(f"{title_prefix}: nonzero damage distribution by story")
    # 标题。

    plt.legend()
    # 图例。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(path, dpi=200)
    # 保存图片。

    plt.close()
    # 关闭图像。


def plot_damaged_story_count(case_summary: Dict[str, np.ndarray], path: Path, title_prefix: str) -> None:
    """
    Plot damaged-story-count distribution.

    中文说明：
        绘制每个 case 中受损层数分布。
    """

    counts = case_summary["damaged_story_count"]
    # 每个 case 受损层数。

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建目录。

    plt.figure(figsize=(6, 4))
    # 创建图像。

    bins = np.arange(counts.max() + 3) - 0.5
    # 为整数计数构造 bin。

    plt.hist(counts, bins=bins)
    # 绘制直方图。

    plt.xticks(np.arange(counts.max() + 2))
    # 设置横轴刻度。

    plt.xlabel("Number of damaged stories per case")
    # 横轴标签。

    plt.ylabel("Count")
    # 纵轴标签。

    plt.title(f"{title_prefix}: damaged-story-count distribution")
    # 标题。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(path, dpi=200)
    # 保存图片。

    plt.close()
    # 关闭图像。


def plot_excitation_distribution(case_summary: Dict[str, np.ndarray], path: Path, title_prefix: str) -> None:
    """
    Plot excitation amplitude and frequency distributions.

    中文说明：
        绘制输入激励幅值和频率分布。
    """

    amplitude_g = case_summary["amplitude_g"]
    # 激励幅值。

    frequency_hz = case_summary["frequency_hz"]
    # 激励频率。

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建目录。

    plt.figure(figsize=(7, 4))
    # 创建图像。

    plt.hist(amplitude_g, bins=12, alpha=0.6, label="Amplitude [g]")
    # 绘制幅值分布。

    plt.hist(frequency_hz, bins=12, alpha=0.6, label="Frequency [Hz]")
    # 绘制频率分布。
    # 注意：两者量纲不同，放在同一图只是快速 sanity check。
    # 后续正式论文作图应分开画。

    plt.xlabel("Value")
    # 横轴标签。

    plt.ylabel("Count")
    # 纵轴标签。

    plt.title(f"{title_prefix}: excitation distribution")
    # 标题。

    plt.legend()
    # 图例。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(path, dpi=200)
    # 保存图片。

    plt.close()
    # 关闭图像。


def plot_noise_distribution(case_summary: Dict[str, np.ndarray], path: Path, title_prefix: str) -> None:
    """
    Plot noise-level distribution.

    中文说明：
        绘制噪声水平分布。
    """

    noise_level = case_summary["noise_level"]
    # 噪声水平。

    unique_levels, counts = np.unique(noise_level, return_counts=True)
    # 统计每个噪声水平数量。

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建目录。

    plt.figure(figsize=(6, 4))
    # 创建图像。

    plt.bar(np.arange(len(unique_levels)), counts)
    # 绘制柱状图。

    plt.xticks(np.arange(len(unique_levels)), [str(float(v)) for v in unique_levels])
    # 设置横轴为噪声水平。

    plt.xlabel("Noise level")
    # 横轴标签。

    plt.ylabel("Count")
    # 纵轴标签。

    plt.title(f"{title_prefix}: noise-level distribution")
    # 标题。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(path, dpi=200)
    # 保存图片。

    plt.close()
    # 关闭图像。


def plot_max_response_distribution(case_summary: Dict[str, np.ndarray], path: Path, title_prefix: str) -> None:
    """
    Plot maximum response distribution.

    中文说明：
        绘制每个 case 最大绝对加速度分布。
    """

    max_response = case_summary["max_abs_response"]
    # 最大响应。

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建目录。

    plt.figure(figsize=(7, 4))
    # 创建图像。

    plt.hist(max_response, bins=20)
    # 绘制直方图。

    plt.xlabel("Maximum absolute acceleration [m/s²]")
    # 横轴标签。

    plt.ylabel("Count")
    # 纵轴标签。

    plt.title(f"{title_prefix}: max-response distribution")
    # 标题。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(path, dpi=200)
    # 保存图片。

    plt.close()
    # 关闭图像。


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """
    Plot a simple scatter figure.

    中文说明：
        通用散点图函数。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建目录。

    plt.figure(figsize=(6, 4))
    # 创建图像。

    plt.scatter(x, y)
    # 绘制散点。

    plt.xlabel(xlabel)
    # 横轴标签。

    plt.ylabel(ylabel)
    # 纵轴标签。

    plt.title(title)
    # 标题。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(path, dpi=200)
    # 保存图片。

    plt.close()
    # 关闭图像。


def plot_damage_heatmap_cases_by_story(case_summary: Dict[str, np.ndarray], path: Path, title_prefix: str) -> None:
    """
    Plot case-by-story damage heatmap.

    中文说明：
        绘制 case × story 的损伤热力图。
        用于检查哪些 case 有损伤、损伤集中在哪些楼层。
    """

    y = case_summary["y_damage"]
    # 损伤矩阵。

    case_ids = case_summary["case_id"]
    # case 编号。

    order = np.argsort(np.sum(y, axis=1))[::-1]
    # 按总损伤量从高到低排序。

    y_sorted = y[order]
    # 排序后的损伤矩阵。

    case_ids_sorted = case_ids[order]
    # 排序后的 case_id。

    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建目录。

    plt.figure(figsize=(7, max(5, 0.08 * y.shape[0] + 3)))
    # 样本越多，图越高。

    image = plt.imshow(y_sorted, aspect="auto")
    # 绘制热力图。

    plt.colorbar(image, label="Stiffness degradation ratio")
    # 色条。

    plt.xticks(
        ticks=np.arange(y.shape[1]),
        labels=[f"Story {i + 1}" for i in range(y.shape[1])],
    )
    # 横轴楼层标签。

    if y.shape[0] <= 120:
        # 100-case 规模还可以显示部分 y 轴刻度。
        tick_step = max(1, y.shape[0] // 20)
        tick_positions = np.arange(0, y.shape[0], tick_step)
        plt.yticks(
            ticks=tick_positions,
            labels=[str(int(case_ids_sorted[i])) for i in tick_positions],
        )
    else:
        # 大规模数据集不显示所有 case_id。
        plt.yticks([])

    plt.xlabel("Story")
    # 横轴标签。

    plt.ylabel("Case ID sorted by total damage")
    # 纵轴标签。

    plt.title(f"{title_prefix}: damage heatmap")
    # 标题。

    plt.tight_layout()
    # 调整布局。

    plt.savefig(path, dpi=200)
    # 保存图片。

    plt.close()
    # 关闭图像。


def summarize_split_distribution(
    split_data: Dict[str, np.ndarray],
    output_json: Path,
) -> None:
    """
    Summarize train/val/test distribution if split dataset is provided.

    中文说明：
        如果提供了预处理后的 split 文件，则比较 train / val / test 的分布。
    """

    required = [
        "y_train",
        "y_val",
        "y_test",
        "amplitude_g_train",
        "amplitude_g_val",
        "amplitude_g_test",
        "frequency_hz_train",
        "frequency_hz_val",
        "frequency_hz_test",
        "noise_level_train",
        "noise_level_val",
        "noise_level_test",
    ]
    # 需要的 split 字段。

    missing = [key for key in required if key not in split_data]
    # 检查缺失字段。

    if missing:
        # 如果缺失字段，则跳过 split 摘要。
        print(f"Skip split distribution summary because keys are missing: {missing}")
        return

    summary = {}
    # split 摘要。

    for split in ["train", "val", "test"]:
        # 遍历三个数据集。

        y = split_data[f"y_{split}"]
        # 标签。

        amp = split_data[f"amplitude_g_{split}"]
        # 幅值。

        freq = split_data[f"frequency_hz_{split}"]
        # 频率。

        noise = split_data[f"noise_level_{split}"]
        # 噪声。

        damaged_story_count = np.sum(y > 0.0, axis=1)
        # 每个 case 受损层数。

        unique_noise, noise_counts = np.unique(noise, return_counts=True)
        # 噪声计数。

        summary[split] = {
            "n_samples": int(y.shape[0]),
            "healthy_ratio": float(np.mean(damaged_story_count == 0)),
            "damage_mean": float(np.mean(y)),
            "damage_max": float(np.max(y)),
            "story_damage_frequency": np.mean(y > 0.0, axis=0).tolist(),
            "amplitude_g_mean": float(np.mean(amp)),
            "amplitude_g_min": float(np.min(amp)),
            "amplitude_g_max": float(np.max(amp)),
            "frequency_hz_mean": float(np.mean(freq)),
            "frequency_hz_min": float(np.min(freq)),
            "frequency_hz_max": float(np.max(freq)),
            "noise_distribution": {
                str(float(level)): int(count)
                for level, count in zip(unique_noise, noise_counts)
            },
        }
        # 写入该 split 摘要。

    write_json(output_json, summary)
    # 保存 JSON。

    print(f"Saved split distribution summary to: {output_json}")
    # 打印路径。


def run_analysis(
    dataset_path: Path,
    split_path: Path | None,
    output_prefix: str,
    figures_dir: Path,
    tables_dir: Path,
    top_k: int,
) -> None:
    """
    Run full dataset distribution analysis.

    中文说明：
        执行完整数据集诊断流程。
    """

    data = load_npz(dataset_path)
    # 读取原始合并数据集。

    case_summary = compute_case_summary(data)
    # 计算 case 级摘要。

    global_summary = compute_global_summary(case_summary, data)
    # 计算整体摘要。

    summary_json = tables_dir / f"{output_prefix}_distribution_summary.json"
    # 整体摘要 JSON 路径。

    case_summary_csv = tables_dir / f"{output_prefix}_case_summary.csv"
    # case 摘要 CSV 路径。

    top_cases_csv = tables_dir / f"{output_prefix}_top_response_cases.csv"
    # top 响应 case CSV 路径。

    write_json(summary_json, global_summary)
    # 保存整体摘要。

    write_case_summary_csv(case_summary_csv, case_summary)
    # 保存 case 摘要。

    write_top_response_cases_csv(top_cases_csv, case_summary, top_k=top_k)
    # 保存最大响应前 top_k 个 case。

    plot_damage_distribution_by_story(
        case_summary=case_summary,
        path=figures_dir / f"{output_prefix}_damage_distribution_by_story.png",
        title_prefix=output_prefix,
    )
    # 损伤分布图。

    plot_damaged_story_count(
        case_summary=case_summary,
        path=figures_dir / f"{output_prefix}_damaged_story_count.png",
        title_prefix=output_prefix,
    )
    # 受损层数分布图。

    plot_excitation_distribution(
        case_summary=case_summary,
        path=figures_dir / f"{output_prefix}_excitation_distribution.png",
        title_prefix=output_prefix,
    )
    # 激励分布图。

    plot_noise_distribution(
        case_summary=case_summary,
        path=figures_dir / f"{output_prefix}_noise_distribution.png",
        title_prefix=output_prefix,
    )
    # 噪声分布图。

    plot_max_response_distribution(
        case_summary=case_summary,
        path=figures_dir / f"{output_prefix}_max_response_distribution.png",
        title_prefix=output_prefix,
    )
    # 最大响应分布图。

    plot_scatter(
        x=case_summary["frequency_hz"],
        y=case_summary["max_abs_response"],
        xlabel="Input frequency [Hz]",
        ylabel="Maximum absolute acceleration [m/s²]",
        title=f"{output_prefix}: frequency vs max response",
        path=figures_dir / f"{output_prefix}_frequency_vs_max_response.png",
    )
    # 频率-最大响应散点图。

    plot_scatter(
        x=case_summary["amplitude_g"],
        y=case_summary["max_abs_response"],
        xlabel="Input amplitude [g]",
        ylabel="Maximum absolute acceleration [m/s²]",
        title=f"{output_prefix}: amplitude vs max response",
        path=figures_dir / f"{output_prefix}_amplitude_vs_max_response.png",
    )
    # 幅值-最大响应散点图。

    plot_scatter(
        x=case_summary["damage_sum"],
        y=case_summary["max_abs_response"],
        xlabel="Total stiffness degradation ratio",
        ylabel="Maximum absolute acceleration [m/s²]",
        title=f"{output_prefix}: total damage vs max response",
        path=figures_dir / f"{output_prefix}_damage_sum_vs_max_response.png",
    )
    # 总损伤-最大响应散点图。

    plot_damage_heatmap_cases_by_story(
        case_summary=case_summary,
        path=figures_dir / f"{output_prefix}_damage_heatmap_cases_by_story.png",
        title_prefix=output_prefix,
    )
    # case-story 损伤热力图。

    if split_path is not None:
        # 如果提供了 split 文件。

        split_data = load_npz(split_path)
        # 读取 split 数据。

        summarize_split_distribution(
            split_data=split_data,
            output_json=tables_dir / f"{output_prefix}_split_distribution_summary.json",
        )
        # 保存 split 摘要。

    print("\nDataset distribution analysis completed.")
    print(f"Dataset: {dataset_path}")
    print(f"Summary JSON: {summary_json}")
    print(f"Case summary CSV: {case_summary_csv}")
    print(f"Top response cases CSV: {top_cases_csv}")
    print(f"Figures directory: {figures_dir}")
    print(f"n_cases: {global_summary['n_cases']}")
    print(f"healthy_case_ratio: {global_summary['healthy_case_ratio']:.4f}")
    print(f"damage_mean_overall: {global_summary['damage_mean_overall']:.6f}")
    print(f"damage_max_overall: {global_summary['damage_max_overall']:.6f}")
    print(f"max_abs_response_max: {global_summary['max_abs_response_max']:.6f}")
    print(f"max_abs_response_95_quantile: {global_summary['max_abs_response_quantiles']['0.95']:.6f}")
    # 打印摘要。


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    中文说明：
        解析命令行参数。
    """

    parser = argparse.ArgumentParser()
    # 创建解析器。

    parser.add_argument(
        "--dataset",
        type=str,
        default="data_processed/debug_plus_100_dataset.npz",
    )
    # 原始合并数据集路径。

    parser.add_argument(
        "--split",
        type=str,
        default="data_processed/debug_plus_100_split_normalized.npz",
    )
    # 预处理后 split 数据集路径。
    # 如果不想分析 split，可传 --split ""。

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="debug_plus_100",
    )
    # 输出文件前缀。

    parser.add_argument(
        "--figures-dir",
        type=str,
        default="results/figures/dataset_distribution",
    )
    # 图片输出目录。

    parser.add_argument(
        "--tables-dir",
        type=str,
        default="results/tables/dataset_distribution",
    )
    # 表格输出目录。

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    # 保存最大响应前几个 case。

    return parser.parse_args()
    # 返回参数。


def main() -> None:
    """
    Command-line entry point.

    中文说明：
        命令行入口。
    """

    args = parse_args()
    # 解析参数。

    split_path = None if args.split.strip() == "" else Path(args.split)
    # 如果 --split 为空字符串，则不分析 split。

    if split_path is not None and not split_path.exists():
        # 如果 split 文件不存在，则提示并跳过。
        print(f"Split file not found, skip split distribution summary: {split_path}")
        split_path = None

    run_analysis(
        dataset_path=Path(args.dataset),
        split_path=split_path,
        output_prefix=args.output_prefix,
        figures_dir=Path(args.figures_dir),
        tables_dir=Path(args.tables_dir),
        top_k=args.top_k,
    )
    # 执行分析。


if __name__ == "__main__":
    main()
    # 直接运行该文件时执行 main。
