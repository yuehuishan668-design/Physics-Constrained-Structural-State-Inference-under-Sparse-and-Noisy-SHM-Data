"""
File location:
    src/data_generation/generate_dataset.py

Purpose:
    Generate a configurable OpenSeesPy dataset for the next debug-plus stage.

Default output:
    data_processed/debug_plus_100_dataset.npz
    data_processed/debug_plus_100_dataset_index.csv
    data_raw/opensees_outputs/debug_plus_100_cases/

Recommended first run:
    python -m src.data_generation.generate_dataset --n-cases 100 --output-prefix debug_plus_100

Research role:
    This script extends the previous 20-case debug dataset to a larger 100-case debug-plus dataset.
    The purpose is not yet to produce final paper-scale results. The purpose is to test whether:
        1. OpenSeesPy can generate more cases stably;
        2. train/val/test metrics become less random than the 20-case version;
        3. MLP linear and MLP sigmoid baselines behave more consistently;
        4. false positives and damage-location errors can be diagnosed on a larger sample.

中文说明：
    本文件用于生成可配置的数据集。
    与 generate_debug_dataset.py 的区别是：
        1. 样本数不再固定为 20；
        2. 输出文件名可以通过 --output-prefix 控制；
        3. 支持是否保存每个单独 case；
        4. 支持控制损伤层数、损伤程度、激励幅值、激励频率和噪声水平；
        5. 适合生成 100-case debug-plus 数据集。
"""

from __future__ import annotations
# 延迟类型注解解析，提高兼容性。

import argparse
# argparse 用于读取命令行参数，例如 --n-cases、--seed、--output-prefix。

import csv
# csv 用于保存样本索引表，便于人工检查每个 case 的损伤、激励和噪声信息。

from pathlib import Path
# Path 用于处理文件路径，比直接使用字符串更安全。

from typing import Dict, List, Sequence
# Dict、List、Sequence 是类型注解，使函数输入输出更清晰。

import numpy as np
# numpy 用于随机数、数组计算、保存 .npz 数据集。

from opensees_models.frame_base import FrameConfig
# FrameConfig 是前面定义的 4 层 2 跨框架配置类。

from opensees_models.run_dynamic_analysis import run_time_history
# run_time_history 是前面已经验证通过的 OpenSeesPy 动力时程分析函数。


def parse_float_pair(value: str) -> tuple[float, float]:
    """
    Parse a comma-separated pair of floats.

    Example:
        "0.05,0.25" -> (0.05, 0.25)

    中文说明：
        把命令行输入的范围字符串解析成两个浮点数。
    """

    parts = value.split(",")
    # 用逗号切分字符串。

    if len(parts) != 2:
        # 范围必须包含两个数字。
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got: {value}")

    left = float(parts[0])
    # 第一个数。

    right = float(parts[1])
    # 第二个数。

    if left >= right:
        # 左边必须小于右边。
        raise argparse.ArgumentTypeError(f"Range lower bound must be smaller than upper bound, got: {value}")

    return left, right
    # 返回二元组。


def parse_noise_levels(value: str) -> List[float]:
    """
    Parse noise levels from command line.

    Example:
        "0,0.05,0.1,0.2" -> [0.0, 0.05, 0.1, 0.2]

    中文说明：
        解析噪声水平候选值。
    """

    levels = [float(item.strip()) for item in value.split(",") if item.strip() != ""]
    # 把逗号分隔的字符串转换为浮点数列表。

    if len(levels) == 0:
        # 至少需要一个噪声水平。
        raise argparse.ArgumentTypeError("At least one noise level is required.")

    for level in levels:
        # 检查每个噪声水平。

        if level < 0.0:
            # 噪声比例不能为负。
            raise argparse.ArgumentTypeError(f"Noise level must be non-negative, got: {level}")

    return levels
    # 返回噪声候选列表。


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
        当前仍然是 debug-plus 数据集，不是最终论文真实地震波数据集。
    """

    t = np.arange(0.0, duration, dt)
    # 创建时间向量。
    # 例如 duration=20, dt=0.01，则长度为 2000。

    amplitude_g = rng.uniform(amplitude_g_range[0], amplitude_g_range[1])
    # 随机抽取地面加速度幅值，单位为 g。

    frequency_hz = rng.uniform(frequency_hz_range[0], frequency_hz_range[1])
    # 随机抽取正弦激励频率，单位为 Hz。

    phase = rng.uniform(0.0, 2.0 * np.pi)
    # 随机相位，使样本之间不完全对齐。

    accel_g = amplitude_g * np.sin(2.0 * np.pi * frequency_hz * t + phase)
    # 生成正弦加速度，单位为 g。

    return {
        "t": t,
        "accel_g": accel_g,
        "amplitude_g": float(amplitude_g),
        "frequency_hz": float(frequency_hz),
        "phase": float(phase),
    }
    # 返回时程和元信息。


def create_random_damage(
    rng: np.random.Generator,
    n_story: int,
    max_damaged_stories: int,
    damage_range: tuple[float, float],
    healthy_probability: float,
) -> List[float]:
    """
    Create one random story-level stiffness degradation vector.

    中文说明：
        生成随机损伤向量。
        当前定义：
            damage[i] = 第 i+1 层刚度退化比例
        例如：
            [0.0, 0.15, 0.0, 0.25]
        表示第 2 层退化 15%，第 4 层退化 25%。
    """

    if max_damaged_stories < 0:
        # 最大损伤层数不能小于 0。
        raise ValueError("max_damaged_stories must be non-negative.")

    if max_damaged_stories > n_story:
        # 最大损伤层数不能超过总层数。
        raise ValueError("max_damaged_stories cannot exceed n_story.")

    if not (0.0 <= healthy_probability <= 1.0):
        # 健康样本概率必须在 [0, 1]。
        raise ValueError("healthy_probability must be between 0 and 1.")

    damage = [0.0] * n_story
    # 初始化为健康结构。

    if rng.random() < healthy_probability:
        # 按指定概率生成完全健康样本。
        return damage

    if max_damaged_stories == 0:
        # 如果最大损伤层数为 0，则只能返回健康样本。
        return damage

    n_damaged = int(rng.integers(1, max_damaged_stories + 1))
    # 随机选择损伤层数。
    # 这里从 1 开始，因为健康样本已经由 healthy_probability 单独控制。

    damaged_stories = rng.choice(
        np.arange(n_story),
        size=n_damaged,
        replace=False,
    )
    # 随机选择受损楼层。
    # replace=False 表示同一个 case 不重复选择同一层。

    for story_index in damaged_stories:
        # 遍历每个受损楼层。

        damage_value = rng.uniform(damage_range[0], damage_range[1])
        # 随机生成该层刚度退化比例。

        damage[int(story_index)] = float(damage_value)
        # 写入 damage 向量。

    return damage
    # 返回损伤向量。


def add_measurement_noise(
    rng: np.random.Generator,
    signal: np.ndarray,
    noise_level: float,
) -> np.ndarray:
    """
    Add Gaussian measurement noise to response signal.

    中文说明：
        给结构响应加入高斯测量噪声，用于模拟传感器噪声。
    """

    if noise_level <= 0.0:
        # 如果噪声水平为 0，直接返回原信号副本。
        return signal.copy()

    signal_std = np.std(signal)
    # 计算信号标准差。

    noise_std = noise_level * signal_std
    # 噪声标准差 = 噪声比例 × 信号标准差。

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=signal.shape,
    )
    # 生成与信号形状相同的高斯噪声。

    return signal + noise
    # 返回带噪信号。


def save_single_case(
    case_file: Path,
    response: Dict[str, np.ndarray],
    clean_abs_accel: np.ndarray,
    noisy_abs_accel: np.ndarray,
    motion: Dict[str, float | np.ndarray],
    noise_level: float,
) -> None:
    """
    Save one raw case file.

    中文说明：
        保存单个 case 的完整原始数据，便于后续排查异常样本。
    """

    case_file.parent.mkdir(parents=True, exist_ok=True)
    # 创建单样本输出目录。

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
        damping_alpha_m=response["damping_alpha_m"],
        damping_beta_k_init=response["damping_beta_k_init"],
        amplitude_g=np.array(motion["amplitude_g"]),
        frequency_hz=np.array(motion["frequency_hz"]),
        phase=np.array(motion["phase"]),
        noise_level=np.array(noise_level),
    )
    # 保存 .npz 文件。


def write_index_csv(index_csv_path: Path, metadata_rows: List[Dict[str, object]]) -> None:
    """
    Write dataset index CSV.

    中文说明：
        写入样本索引 CSV，方便人工查看每个样本的损伤、激励、噪声、最大响应等信息。
    """

    index_csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。

    fieldnames = [
        "case_id",
        "case_file",
        "damage",
        "n_damaged_stories",
        "amplitude_g",
        "frequency_hz",
        "phase",
        "noise_level",
        "max_abs_accel",
        "modal_periods",
    ]
    # CSV 表头。

    with index_csv_path.open("w", newline="", encoding="utf-8") as f:
        # 打开 CSV 文件。

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 创建字典写入器。

        writer.writeheader()
        # 写入表头。

        writer.writerows(metadata_rows)
        # 写入所有样本元信息。


def generate_dataset(
    n_cases: int,
    seed: int,
    duration: float,
    dt: float,
    output_dir_raw: Path,
    output_path_processed: Path,
    index_csv_path: Path,
    save_raw_cases: bool,
    max_damaged_stories: int,
    healthy_probability: float,
    damage_range: tuple[float, float],
    amplitude_g_range: tuple[float, float],
    frequency_hz_range: tuple[float, float],
    noise_candidates: Sequence[float],
    damping_ratio: float,
) -> None:
    """
    Generate the OpenSeesPy dataset.

    中文说明：
        数据集生成主流程。
    """

    if n_cases <= 0:
        # 样本数必须为正。
        raise ValueError("n_cases must be positive.")

    rng = np.random.default_rng(seed)
    # 创建随机数生成器。
    # 固定 seed 可以保证数据集可复现。

    config = FrameConfig()
    # 使用前面定义的默认 4 层 2 跨框架。

    if save_raw_cases:
        # 如果需要保存每个单独 case。
        output_dir_raw.mkdir(parents=True, exist_ok=True)
        # 创建原始样本目录。

    output_path_processed.parent.mkdir(parents=True, exist_ok=True)
    # 创建合并数据集输出目录。

    all_abs_accel = []
    # 保存带噪绝对加速度，作为第一版神经网络输入。

    all_clean_abs_accel = []
    # 保存无噪绝对加速度。

    all_damage = []
    # 保存损伤标签。

    all_story_drift = []
    # 保存层间位移角。

    all_ground_accel = []
    # 保存地面加速度。

    all_amplitude_g = []
    # 保存激励幅值。

    all_frequency_hz = []
    # 保存激励频率。

    all_phase = []
    # 保存正弦相位。

    all_noise_level = []
    # 保存噪声水平。

    all_case_id = []
    # 保存 case_id。

    metadata_rows = []
    # 保存 CSV 元信息。

    failed_cases = []
    # 保存失败 case 信息。

    for case_id in range(n_cases):
        # 循环生成每一个 case。

        damage = create_random_damage(
            rng=rng,
            n_story=config.n_story,
            max_damaged_stories=max_damaged_stories,
            damage_range=damage_range,
            healthy_probability=healthy_probability,
        )
        # 生成随机损伤向量。

        motion = create_random_sine_motion(
            rng=rng,
            duration=duration,
            dt=dt,
            amplitude_g_range=amplitude_g_range,
            frequency_hz_range=frequency_hz_range,
        )
        # 生成随机正弦激励。

        noise_level = float(rng.choice(noise_candidates))
        # 从候选噪声水平中随机选择一个。

        response = run_time_history(
            damage=damage,
            ground_accel_g=motion["accel_g"],
            dt=dt,
            config=config,
            damping_ratio=damping_ratio,
        )
        # 调用 OpenSeesPy 进行动力时程分析。

        failed_at = int(response["analysis_failed_at"])
        # 读取分析失败步数。
        # -1 表示没有失败。

        if failed_at != -1:
            # 如果分析失败，记录并跳过。
            failed_cases.append((case_id, failed_at, damage))
            print(f"Skip case {case_id:04d}: analysis failed at step {failed_at}")
            continue

        clean_abs_accel = response["abs_accel"]
        # 无噪绝对加速度。

        noisy_abs_accel = add_measurement_noise(
            rng=rng,
            signal=clean_abs_accel,
            noise_level=noise_level,
        )
        # 带噪绝对加速度。

        max_abs_accel = float(np.max(np.abs(noisy_abs_accel)))
        # 计算该样本最大绝对加速度，用于检查是否有异常响应。

        case_file = output_dir_raw / f"case_{case_id:04d}.npz"
        # 单 case 文件路径。

        if save_raw_cases:
            # 如果要求保存每个 case 的完整数据。
            save_single_case(
                case_file=case_file,
                response=response,
                clean_abs_accel=clean_abs_accel,
                noisy_abs_accel=noisy_abs_accel,
                motion=motion,
                noise_level=noise_level,
            )
            # 保存单 case 文件。

        all_abs_accel.append(noisy_abs_accel)
        # 添加带噪输入。

        all_clean_abs_accel.append(clean_abs_accel)
        # 添加无噪输入。

        all_damage.append(response["damage"])
        # 添加损伤标签。

        all_story_drift.append(response["story_drift"])
        # 添加层间位移角。

        all_ground_accel.append(response["ground_accel_mps2"])
        # 添加地面加速度。

        all_amplitude_g.append(float(motion["amplitude_g"]))
        # 添加激励幅值。

        all_frequency_hz.append(float(motion["frequency_hz"]))
        # 添加激励频率。

        all_phase.append(float(motion["phase"]))
        # 添加相位。

        all_noise_level.append(noise_level)
        # 添加噪声水平。

        all_case_id.append(case_id)
        # 添加 case_id。

        metadata_rows.append(
            {
                "case_id": case_id,
                "case_file": str(case_file) if save_raw_cases else "",
                "damage": response["damage"].tolist(),
                "n_damaged_stories": int(np.sum(response["damage"] > 0.0)),
                "amplitude_g": float(motion["amplitude_g"]),
                "frequency_hz": float(motion["frequency_hz"]),
                "phase": float(motion["phase"]),
                "noise_level": noise_level,
                "max_abs_accel": max_abs_accel,
                "modal_periods": response["modal_periods"].tolist(),
            }
        )
        # 添加 CSV 元信息。

        print(
            f"Generated case {case_id:04d}: "
            f"damage={np.round(response['damage'], 3)}, "
            f"amp_g={float(motion['amplitude_g']):.3f}, "
            f"freq={float(motion['frequency_hz']):.3f}, "
            f"noise={noise_level:.2f}, "
            f"max_abs_accel={max_abs_accel:.3f}"
        )
        # 打印当前生成进度。

    if len(all_abs_accel) == 0:
        # 如果全部失败，不能保存数据集。
        raise RuntimeError("No successful cases were generated. Check model and analysis settings.")

    X_abs_accel = np.stack(all_abs_accel, axis=0)
    # 合并带噪加速度。
    # 形状：(n_successful_cases, n_steps, n_story)。

    X_clean_abs_accel = np.stack(all_clean_abs_accel, axis=0)
    # 合并无噪加速度。

    y_damage = np.stack(all_damage, axis=0)
    # 合并损伤标签。
    # 形状：(n_successful_cases, n_story)。

    story_drift = np.stack(all_story_drift, axis=0)
    # 合并层间位移角。

    ground_accel = np.stack(all_ground_accel, axis=0)
    # 合并地面加速度。

    np.savez_compressed(
        output_path_processed,
        X_abs_accel=X_abs_accel,
        X_clean_abs_accel=X_clean_abs_accel,
        y_damage=y_damage,
        story_drift=story_drift,
        ground_accel=ground_accel,
        amplitude_g=np.array(all_amplitude_g, dtype=np.float64),
        frequency_hz=np.array(all_frequency_hz, dtype=np.float64),
        phase=np.array(all_phase, dtype=np.float64),
        noise_level=np.array(all_noise_level, dtype=np.float64),
        case_id=np.array(all_case_id, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        seed=np.array(seed, dtype=np.int64),
        failed_case_count=np.array(len(failed_cases), dtype=np.int64),
    )
    # 保存合并后的数据集。

    write_index_csv(index_csv_path, metadata_rows)
    # 保存 CSV 索引表。

    print("\nDataset generation completed.")
    print(f"Requested cases: {n_cases}")
    print(f"Successful cases: {X_abs_accel.shape[0]}")
    print(f"Failed cases: {len(failed_cases)}")
    print(f"Saved processed dataset to: {output_path_processed}")
    print(f"Saved index CSV to: {index_csv_path}")
    print(f"Saved raw case directory to: {output_dir_raw if save_raw_cases else 'not saved'}")
    print(f"X_abs_accel shape: {X_abs_accel.shape}")
    print(f"y_damage shape: {y_damage.shape}")
    print(f"Max abs acceleration overall: {float(np.max(np.abs(X_abs_accel))):.6f}")
    print(f"Damage mean: {float(np.mean(y_damage)):.6f}")
    print(f"Damage max: {float(np.max(y_damage)):.6f}")
    # 打印结果摘要。


def build_default_paths(output_prefix: str) -> tuple[Path, Path, Path]:
    """
    Build default output paths according to output_prefix.

    中文说明：
        根据输出前缀自动构造数据路径。
    """

    output_dir_raw = Path("data_raw/opensees_outputs") / f"{output_prefix}_cases"
    # 原始单 case 输出目录。

    output_path_processed = Path("data_processed") / f"{output_prefix}_dataset.npz"
    # 合并数据集输出路径。

    index_csv_path = Path("data_processed") / f"{output_prefix}_dataset_index.csv"
    # 索引 CSV 输出路径。

    return output_dir_raw, output_path_processed, index_csv_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    中文说明：
        解析命令行参数。
    """

    parser = argparse.ArgumentParser()
    # 创建参数解析器。

    parser.add_argument("--n-cases", type=int, default=100)
    # 样本数。当前建议先用 100。

    parser.add_argument("--seed", type=int, default=20260625)
    # 随机种子。

    parser.add_argument("--duration", type=float, default=20.0)
    # 每个时程持续时间，单位秒。

    parser.add_argument("--dt", type=float, default=0.01)
    # 时间步长，单位秒。

    parser.add_argument("--output-prefix", type=str, default="debug_plus_100")
    # 输出文件前缀。

    parser.add_argument("--output-raw-dir", type=str, default="")
    # 可选：手动指定原始 case 输出目录。

    parser.add_argument("--output-processed", type=str, default="")
    # 可选：手动指定合并数据集输出路径。

    parser.add_argument("--index-csv", type=str, default="")
    # 可选：手动指定索引 CSV 输出路径。

    parser.add_argument("--save-raw-cases", action="store_true")
    # 是否保存每个单独 case。
    # 如果不加该参数，则只保存合并数据集和索引 CSV。

    parser.add_argument("--max-damaged-stories", type=int, default=2)
    # 每个 case 最多损伤几层。

    parser.add_argument("--healthy-probability", type=float, default=0.20)
    # 完全健康样本概率。
    # 设为 0.20 表示大约 20% case 是健康结构。

    parser.add_argument("--damage-range", type=parse_float_pair, default=(0.05, 0.35))
    # 损伤范围。

    parser.add_argument("--amplitude-g-range", type=parse_float_pair, default=(0.05, 0.25))
    # 输入激励幅值范围，单位 g。

    parser.add_argument("--frequency-hz-range", type=parse_float_pair, default=(0.5, 2.0))
    # 输入正弦频率范围，单位 Hz。

    parser.add_argument("--noise-levels", type=parse_noise_levels, default=[0.0, 0.05, 0.10, 0.20])
    # 噪声水平候选值。

    parser.add_argument("--damping-ratio", type=float, default=0.02)
    # Rayleigh 阻尼目标阻尼比。

    return parser.parse_args()
    # 返回参数。


def main() -> None:
    """
    Command-line entry point.

    中文说明：
        命令行入口。
    """

    args = parse_args()
    # 解析命令行参数。

    default_raw_dir, default_processed, default_index_csv = build_default_paths(args.output_prefix)
    # 根据 output_prefix 构造默认路径。

    output_dir_raw = Path(args.output_raw_dir) if args.output_raw_dir else default_raw_dir
    # 如果命令行指定了原始输出目录，则使用指定目录；否则使用默认路径。

    output_path_processed = Path(args.output_processed) if args.output_processed else default_processed
    # 如果命令行指定了合并数据集路径，则使用指定路径；否则使用默认路径。

    index_csv_path = Path(args.index_csv) if args.index_csv else default_index_csv
    # 如果命令行指定了索引 CSV 路径，则使用指定路径；否则使用默认路径。

    generate_dataset(
        n_cases=args.n_cases,
        seed=args.seed,
        duration=args.duration,
        dt=args.dt,
        output_dir_raw=output_dir_raw,
        output_path_processed=output_path_processed,
        index_csv_path=index_csv_path,
        save_raw_cases=args.save_raw_cases,
        max_damaged_stories=args.max_damaged_stories,
        healthy_probability=args.healthy_probability,
        damage_range=args.damage_range,
        amplitude_g_range=args.amplitude_g_range,
        frequency_hz_range=args.frequency_hz_range,
        noise_candidates=args.noise_levels,
        damping_ratio=args.damping_ratio,
    )
    # 执行数据生成。


if __name__ == "__main__":
    main()
    # 直接运行该文件时执行 main。
