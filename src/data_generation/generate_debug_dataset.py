"""
File location:
    src/data_generation/generate_debug_dataset.py

Purpose:
    Generate a small debug dataset using the OpenSeesPy 2D frame model.

中文说明：
    生成一个小型调试数据集，用于验证：
    1. OpenSeesPy 是否能连续运行多个损伤案例；
    2. 每个样本的响应维度是否一致；
    3. 损伤标签 y_damage 是否正确保存；
    4. 后续 PyTorch 模型是否可以直接读取该数据。

Important:
    This is NOT the final dataset for the SCI paper.
    This is only the first debug dataset.

中文提醒：
    这不是最终论文数据集。
    这只是第一版调试数据集。
"""

from __future__ import annotations
# 延迟类型注解解析，提高兼容性。

from pathlib import Path
# Path 用于管理文件路径。

from typing import Dict, List
# Dict 和 List 是类型注解，让代码结构更清楚。

import csv
# csv 用于保存每个样本的元信息，例如损伤向量、激励幅值、噪声水平。

import numpy as np
# numpy 用于数组计算、随机数生成、保存 .npz 文件。

from opensees_models.frame_base import FrameConfig
# 导入基础框架配置类。

from opensees_models.run_dynamic_analysis import run_time_history
# 导入已经写好的 OpenSeesPy 动力分析函数。


def create_random_sine_motion(
    rng: np.random.Generator,
    duration: float,
    dt: float,
    amplitude_g_range: tuple[float, float],
    frequency_hz_range: tuple[float, float],
) -> Dict[str, float | np.ndarray]:
    """
    Create one randomized sine ground motion.

    中文说明：
        生成一个随机正弦地面加速度。
        当前仍然是调试用人工激励，不是最终地震波数据集。
    """

    t = np.arange(0.0, duration, dt)
    # 创建时间向量。
    # 例如 duration=20, dt=0.01，则长度为 2000。

    amplitude_g = rng.uniform(amplitude_g_range[0], amplitude_g_range[1])
    # 从给定范围内随机抽取地震动幅值，单位为 g。

    frequency_hz = rng.uniform(frequency_hz_range[0], frequency_hz_range[1])
    # 从给定范围内随机抽取正弦频率，单位为 Hz。

    phase = rng.uniform(0.0, 2.0 * np.pi)
    # 随机相位，使不同样本的激励不完全重合。

    accel_g = amplitude_g * np.sin(2.0 * np.pi * frequency_hz * t + phase)
    # 生成正弦地面加速度，单位为 g。

    return {
        "t": t,
        "accel_g": accel_g,
        "amplitude_g": float(amplitude_g),
        "frequency_hz": float(frequency_hz),
        "phase": float(phase),
    }


def create_random_damage(
    rng: np.random.Generator,
    n_story: int,
    max_damaged_stories: int = 2,
) -> List[float]:
    """
    Create one random story-level damage vector.

    中文说明：
        随机生成一个层刚度退化向量。
        例如：
            [0.00, 0.15, 0.00, 0.25]
        表示第 2 层和第 4 层发生刚度退化。
    """

    damage = [0.0] * n_story
    # 初始化为健康结构，即每一层损伤均为 0。

    n_damaged = rng.integers(0, max_damaged_stories + 1)
    # 随机决定有几层受损。
    # 可能是 0 层、1 层或 2 层。

    if n_damaged == 0:
        # 如果没有损伤，直接返回全 0 向量。
        return damage

    damaged_stories = rng.choice(
        np.arange(n_story),
        size=n_damaged,
        replace=False,
    )
    # 随机选择受损楼层。
    # np.arange(n_story) 生成 [0, 1, 2, 3]。
    # replace=False 表示不会重复选择同一层。

    for story_index in damaged_stories:
        # 遍历每一个受损楼层索引。

        damage_value = rng.uniform(0.05, 0.35)
        # 随机生成刚度退化比例。
        # 当前 debug 数据集使用 5% 到 35%。
        # 后续正式实验再拆分轻微、中等、严重损伤等级。

        damage[int(story_index)] = float(damage_value)
        # 将该楼层损伤值写入 damage 向量。

    return damage


def add_measurement_noise(
    rng: np.random.Generator,
    signal: np.ndarray,
    noise_level: float,
) -> np.ndarray:
    """
    Add Gaussian measurement noise to a response signal.

    中文说明：
        向结构响应中加入高斯测量噪声。
        这模拟传感器测量误差。
    """

    if noise_level <= 0.0:
        # 如果噪声水平为 0，直接返回原始信号副本。
        return signal.copy()

    signal_std = np.std(signal)
    # 计算响应信号的标准差。

    noise_std = noise_level * signal_std
    # 噪声标准差 = 噪声比例 × 信号标准差。

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=signal.shape,
    )
    # 生成与 signal 形状相同的高斯噪声。
    # loc=0.0 表示均值为 0。
    # scale=noise_std 表示标准差为 noise_std。

    return signal + noise
    # 返回带噪声的测量信号。


def generate_debug_dataset(
    n_cases: int = 20,
    seed: int = 20260621,
    duration: float = 20.0,
    dt: float = 0.01,
    output_dir_raw: Path = Path("data_raw/opensees_outputs/debug_cases"),
    output_path_processed: Path = Path("data_processed/debug_dataset.npz"),
    index_csv_path: Path = Path("data_processed/debug_dataset_index.csv"),
) -> None:
    """
    Generate a small debug dataset.

    中文说明：
        批量生成小型调试数据集。
        每一个样本都会：
        1. 随机生成损伤向量；
        2. 随机生成正弦激励；
        3. 调用 OpenSeesPy 运行时程分析；
        4. 对楼层绝对加速度加入测量噪声；
        5. 保存到统一的 debug_dataset.npz。
    """

    rng = np.random.default_rng(seed)
    # 创建随机数生成器。
    # seed 固定后，实验可以复现。

    config = FrameConfig()
    # 使用默认 4 层 2 跨框架配置。

    output_dir_raw.mkdir(parents=True, exist_ok=True)
    # 创建原始单样本输出目录。

    output_path_processed.parent.mkdir(parents=True, exist_ok=True)
    # 创建 processed 数据输出目录。

    index_csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 创建索引 CSV 的输出目录。

    all_abs_accel = []
    all_clean_abs_accel = []
    all_damage = []
    all_story_drift = []
    all_ground_accel = []
    all_amplitude_g = []
    all_frequency_hz = []
    all_noise_level = []
    all_case_id = []
    metadata_rows = []
    # 上面这些列表用于暂存每个样本的数据，最后统一堆叠并保存。

    noise_candidates = [0.0, 0.05, 0.10, 0.20]
    # 噪声水平候选值。
    # 对应 0%、5%、10%、20%。

    for case_id in range(n_cases):
        # 循环生成每一个样本。

        damage = create_random_damage(
            rng=rng,
            n_story=config.n_story,
            max_damaged_stories=2,
        )
        # 随机生成损伤向量。

        motion = create_random_sine_motion(
            rng=rng,
            duration=duration,
            dt=dt,
            amplitude_g_range=(0.05, 0.25),
            frequency_hz_range=(0.5, 2.0),
        )
        # 随机生成调试用正弦激励。

        noise_level = float(rng.choice(noise_candidates))
        # 随机选择噪声水平。

        response = run_time_history(
            damage=damage,
            ground_accel_g=motion["accel_g"],
            dt=dt,
            config=config,
            damping_ratio=0.02,
        )
        # 调用 OpenSeesPy 进行动力时程分析。

        failed_at = int(response["analysis_failed_at"])
        # 读取分析失败步数。
        # -1 表示没有失败。

        if failed_at != -1:
            # 如果某个样本分析失败，暂时跳过。
            print(f"Skip case {case_id}: analysis failed at step {failed_at}")
            continue

        clean_abs_accel = response["abs_accel"]
        # 无噪绝对加速度，形状为 (time_steps, n_story)。

        noisy_abs_accel = add_measurement_noise(
            rng=rng,
            signal=clean_abs_accel,
            noise_level=noise_level,
        )
        # 加入测量噪声后的绝对加速度。
        # 这是后续神经网络的第一版输入 X。

        case_file = output_dir_raw / f"case_{case_id:04d}.npz"
        # 定义单个样本的原始保存路径。
        # 例如 case_0000.npz。

        np.savez_compressed(
            case_file,
            damage=response["damage"],
            ground_accel_mps2=response["ground_accel_mps2"],
            clean_abs_accel=clean_abs_accel,
            noisy_abs_accel=noisy_abs_accel,
            rel_disp=response["rel_disp"],
            rel_accel=response["rel_accel"],
            story_drift=response["story_drift"],
            dt=response["dt"],
            modal_periods=response["modal_periods"],
            amplitude_g=np.array(motion["amplitude_g"]),
            frequency_hz=np.array(motion["frequency_hz"]),
            noise_level=np.array(noise_level),
        )
        # 保存单个样本。
        # 这方便后续单独排查某一个 case。

        all_abs_accel.append(noisy_abs_accel)
        all_clean_abs_accel.append(clean_abs_accel)
        all_damage.append(response["damage"])
        all_story_drift.append(response["story_drift"])
        all_ground_accel.append(response["ground_accel_mps2"])
        all_amplitude_g.append(motion["amplitude_g"])
        all_frequency_hz.append(motion["frequency_hz"])
        all_noise_level.append(noise_level)
        all_case_id.append(case_id)
        # 把当前样本写入总列表。

        metadata_rows.append(
            {
                "case_id": case_id,
                "case_file": str(case_file),
                "damage": response["damage"].tolist(),
                "amplitude_g": motion["amplitude_g"],
                "frequency_hz": motion["frequency_hz"],
                "noise_level": noise_level,
                "modal_periods": response["modal_periods"].tolist(),
            }
        )
        # 保存一行元信息，之后写入 CSV。

        print(
            f"Generated case {case_id:04d}: "
            f"damage={np.round(response['damage'], 3)}, "
            f"amp_g={motion['amplitude_g']:.3f}, "
            f"freq={motion['frequency_hz']:.3f}, "
            f"noise={noise_level:.2f}"
        )
        # 打印当前样本生成进度。

    if len(all_abs_accel) == 0:
        # 如果所有样本都失败，不能继续 stack。
        raise RuntimeError("No successful cases were generated. Check OpenSees analysis settings.")

    X_abs_accel = np.stack(all_abs_accel, axis=0)
    # 将所有样本堆叠成一个三维数组。
    # 形状为 (n_successful_cases, time_steps, n_story)。

    X_clean_abs_accel = np.stack(all_clean_abs_accel, axis=0)
    # 无噪输入版本。

    y_damage = np.stack(all_damage, axis=0)
    # 损伤标签数组，形状为 (n_successful_cases, n_story)。

    story_drift = np.stack(all_story_drift, axis=0)
    # 层间位移角数组。

    ground_accel = np.stack(all_ground_accel, axis=0)
    # 地面加速度数组。

    np.savez_compressed(
        output_path_processed,
        X_abs_accel=X_abs_accel,
        X_clean_abs_accel=X_clean_abs_accel,
        y_damage=y_damage,
        story_drift=story_drift,
        ground_accel=ground_accel,
        amplitude_g=np.array(all_amplitude_g),
        frequency_hz=np.array(all_frequency_hz),
        noise_level=np.array(all_noise_level),
        case_id=np.array(all_case_id),
        dt=np.array(dt),
    )
    # 保存合并后的 debug 数据集。
    # 后续 PyTorch 会优先读取这个文件。

    with index_csv_path.open("w", newline="", encoding="utf-8") as f:
        # 打开 CSV 文件用于写入。

        fieldnames = [
            "case_id",
            "case_file",
            "damage",
            "amplitude_g",
            "frequency_hz",
            "noise_level",
            "modal_periods",
        ]
        # 定义 CSV 表头。

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 创建 CSV 字典写入器。

        writer.writeheader()
        # 写入表头。

        writer.writerows(metadata_rows)
        # 写入所有样本元信息。

    print("\nDebug dataset generation completed.")
    print(f"Saved processed dataset to: {output_path_processed}")
    print(f"Saved index CSV to: {index_csv_path}")
    print(f"X_abs_accel shape: {X_abs_accel.shape}")
    print(f"y_damage shape: {y_damage.shape}")
    # 打印最终结果。


def main() -> None:
    """
    Main entry point.

    中文说明：
        主入口函数。
    """

    generate_debug_dataset(
        n_cases=20,
        seed=20260621,
        duration=20.0,
        dt=0.01,
    )
    # 生成 20 个样本的 debug 数据集。


if __name__ == "__main__":
    # 直接运行本文件时执行 main()。
    main()
