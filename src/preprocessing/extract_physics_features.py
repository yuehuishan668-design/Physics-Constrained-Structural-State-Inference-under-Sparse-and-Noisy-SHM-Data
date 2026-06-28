"""
File location:
    src/preprocessing/extract_physics_features.py

Purpose:
    Extract physics-informed diagnostic features from OpenSeesPy response data.

Why this step:
    Previous MLP/LSTM/two-head LSTM runs indicate that direct sequence learning is confusing
    excitation-induced amplification with damage-induced response changes. This script therefore
    builds features that explicitly describe amplification, interstory response contrast,
    spectral content, and proximity to the healthy first-mode frequency.

中文说明：
    本文件用于从 OpenSeesPy 响应数据中提取物理诊断特征。它不是继续堆神经网络，
    而是先把输入表征做得更符合结构动力学问题本身。
"""

from __future__ import annotations
# English: Postpone type-hint evaluation.
# 中文：延迟类型注解解析，提高兼容性。

import argparse
# English: Parse command-line arguments.
# 中文：解析命令行参数。

import csv
# English: Write feature-name tables.
# 中文：写入特征名称表。

import json
# English: Write summary JSON files.
# 中文：写入 JSON 摘要文件。

from pathlib import Path
# English: Use object-oriented file paths.
# 中文：使用面向对象的路径处理。

from typing import Dict, List, Tuple
# English: Type hints.
# 中文：类型标注。

import numpy as np
# English: Numerical array computation.
# 中文：数值数组计算。


def load_npz_as_dict(path: Path) -> Dict[str, np.ndarray]:
    """Load an npz file as a normal dictionary. / 将 npz 文件读取为普通字典。"""
    if not path.exists():
        # English: Stop early if the input file is missing.
        # 中文：如果输入文件不存在，立即报错。
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path, allow_pickle=False)
    # English: np.load returns an NpzFile object.
    # 中文：np.load 返回 NpzFile 对象。
    return {key: data[key] for key in data.files}
    # English: Convert it to a standard dictionary.
    # 中文：转换为标准字典，后续更容易访问。


def pick_first_available(data: Dict[str, np.ndarray], candidates: List[str]) -> Tuple[str, np.ndarray]:
    """Pick the first existing key from candidates. / 从候选 key 中选择第一个存在的数组。"""
    for key in candidates:
        # English: Try candidate keys in order.
        # 中文：按顺序检查候选 key。
        if key in data:
            return key, data[key]
            # English: Return the first available array.
            # 中文：返回第一个存在的数组。
    raise KeyError(f"None of the candidate keys exist: {candidates}. Available keys: {list(data.keys())}")
    # English: Give a clear error message if all candidates fail.
    # 中文：如果候选项都不存在，给出明确错误。


def ensure_3d_response(X: np.ndarray) -> np.ndarray:
    """Ensure response array shape is (n_cases, n_steps, n_stories). / 确保响应数组为三维。"""
    if X.ndim != 3:
        # English: Response histories must be case × time × story.
        # 中文：结构响应时程必须是 样本 × 时间 × 楼层。
        raise ValueError(f"Response array must be 3D, got shape {X.shape}")
    return X.astype(np.float64)
    # English: Use float64 for stable feature extraction.
    # 中文：使用 float64 提高特征计算稳定性。


def ensure_ground_shape(ground: np.ndarray, n_cases: int, n_steps: int) -> np.ndarray:
    """Ensure ground acceleration shape is (n_cases, n_steps). / 确保地面加速度为二维。"""
    ground = np.asarray(ground)
    # English: Convert input to numpy array.
    # 中文：转换为 NumPy 数组。
    if ground.ndim == 3 and ground.shape[-1] == 1:
        # English: Convert (N, T, 1) to (N, T).
        # 中文：将 (样本, 时间, 1) 压缩为 (样本, 时间)。
        ground = ground[:, :, 0]
    if ground.ndim == 1:
        # English: If one common ground motion is given, tile it to all cases.
        # 中文：如果只有一条地震输入时程，则复制到所有样本。
        if ground.shape[0] != n_steps:
            raise ValueError(f"Ground length mismatch: {ground.shape[0]} vs expected {n_steps}")
        ground = np.tile(ground[None, :], (n_cases, 1))
    if ground.ndim != 2:
        # English: Ground acceleration must now be two-dimensional.
        # 中文：地面加速度现在必须是二维。
        raise ValueError(f"Ground acceleration must be 2D after processing, got shape {ground.shape}")
    if ground.shape != (n_cases, n_steps):
        # English: Check full shape consistency.
        # 中文：检查样本数和时间步数是否一致。
        raise ValueError(f"Ground shape mismatch: got {ground.shape}, expected ({n_cases}, {n_steps})")
    return ground.astype(np.float64)
    # English: Return float64 ground acceleration.
    # 中文：返回 float64 地面加速度。


def get_dt(raw_data: Dict[str, np.ndarray], default_dt: float) -> float:
    """Read time step dt from raw data. / 从原始数据读取时间步长。"""
    if "dt" not in raw_data:
        # English: Fall back to command-line default.
        # 中文：如果没有 dt 字段，则使用命令行默认值。
        return float(default_dt)
    value = raw_data["dt"]
    # English: dt can be scalar or array-like.
    # 中文：dt 可能是标量，也可能是数组。
    return float(np.asarray(value).reshape(-1)[0])
    # English: Use the first value if array-like.
    # 中文：若为数组，取第一个值。


def safe_rms(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute root-mean-square. / 计算均方根。"""
    return np.sqrt(np.mean(x ** 2, axis=axis))
    # English: RMS reflects signal energy.
    # 中文：RMS 反映信号能量。


def dominant_frequency_and_centroid(signal: np.ndarray, dt: float, min_freq: float, max_freq: float, eps: float) -> Tuple[float, float, float]:
    """Compute dominant frequency, spectral centroid, and band energy. / 计算主频、频谱质心和频带能量。"""
    x = np.asarray(signal, dtype=np.float64)
    # English: Convert signal to float64.
    # 中文：转换为 float64。
    x = x - np.mean(x)
    # English: Remove the DC component.
    # 中文：去除直流分量，避免均值影响 FFT。
    n = x.shape[0]
    # English: Number of time steps.
    # 中文：时间步数量。
    if n < 4:
        # English: Too few samples for stable FFT.
        # 中文：点数太少时无法稳定做频域分析。
        return 0.0, 0.0, 0.0
    freqs = np.fft.rfftfreq(n, d=dt)
    # English: One-sided frequency axis.
    # 中文：单边频率轴。
    spectrum = np.abs(np.fft.rfft(x))
    # English: One-sided FFT magnitude.
    # 中文：单边 FFT 幅值。
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    # English: Select the frequency band of interest.
    # 中文：选择关注的频率范围。
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    band_freqs = freqs[mask]
    # English: Frequencies in the selected band.
    # 中文：选定频带内的频率。
    band_mag = spectrum[mask]
    # English: Magnitudes in the selected band.
    # 中文：选定频带内的幅值。
    if float(np.sum(band_mag)) <= eps:
        # English: Nearly zero spectrum.
        # 中文：频谱能量几乎为零。
        return 0.0, 0.0, 0.0
    dominant_frequency = float(band_freqs[int(np.argmax(band_mag))])
    # English: Frequency with maximum magnitude.
    # 中文：幅值最大的频率。
    spectral_centroid = float(np.sum(band_freqs * band_mag) / (np.sum(band_mag) + eps))
    # English: Weighted average frequency.
    # 中文：按幅值加权的平均频率。
    band_energy = float(np.sum(band_mag ** 2))
    # English: Squared magnitude energy in the selected band.
    # 中文：选定频带内的平方幅值能量。
    return dominant_frequency, spectral_centroid, band_energy


def corrcoef_safe(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    """Compute Pearson correlation safely. / 安全计算 Pearson 相关系数。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # English: Convert both signals to float64.
    # 中文：将两个信号转换为 float64。
    if float(np.std(a)) < eps or float(np.std(b)) < eps:
        # English: Avoid NaN correlation for near-constant signals.
        # 中文：避免近似常数信号导致相关系数为 NaN。
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
    # English: Return Pearson correlation coefficient.
    # 中文：返回 Pearson 相关系数。


def extract_one_case_features(
    response: np.ndarray,
    ground: np.ndarray,
    amplitude_g: float,
    frequency_hz: float,
    noise_level: float,
    dt: float,
    healthy_first_frequency_hz: float,
    min_fft_freq: float,
    max_fft_freq: float,
    eps: float,
) -> Tuple[np.ndarray, List[str]]:
    """Extract physics diagnostic features for one case. / 对单个样本提取物理诊断特征。"""
    _, n_stories = response.shape
    # English: response shape is time × story.
    # 中文：response 的形状为 时间 × 楼层。
    values: List[float] = []
    # English: Feature values.
    # 中文：特征值。
    names: List[str] = []
    # English: Feature names.
    # 中文：特征名称。

    story_mean = np.mean(response, axis=0)
    story_std = np.std(response, axis=0)
    story_max_abs = np.max(np.abs(response), axis=0)
    story_rms = safe_rms(response, axis=0)
    story_peak_to_peak = np.max(response, axis=0) - np.min(response, axis=0)
    story_crest = story_max_abs / (story_rms + eps)
    # English: Basic response statistics by story.
    # 中文：逐楼层基本响应统计量。

    for j in range(n_stories):
        story_id = j + 1
        values.extend([story_mean[j], story_std[j], story_max_abs[j], story_rms[j], story_peak_to_peak[j], story_crest[j]])
        names.extend([
            f"story_{story_id}_mean",
            f"story_{story_id}_std",
            f"story_{story_id}_max_abs",
            f"story_{story_id}_rms",
            f"story_{story_id}_peak_to_peak",
            f"story_{story_id}_crest_factor",
        ])
        # English: Add six basic features for each story.
        # 中文：每层加入 6 个基本特征。

    ground_max_abs = float(np.max(np.abs(ground)))
    ground_rms = float(safe_rms(ground, axis=0))
    values.extend([ground_max_abs, ground_rms])
    names.extend(["ground_max_abs", "ground_rms"])
    # English: Ground-motion intensity descriptors.
    # 中文：地面输入强度描述。

    for j in range(n_stories):
        story_id = j + 1
        values.extend([story_max_abs[j] / (ground_max_abs + eps), story_rms[j] / (ground_rms + eps)])
        names.extend([f"story_{story_id}_max_ground_amplification", f"story_{story_id}_rms_ground_amplification"])
        # English: Floor-to-ground amplification ratios.
        # 中文：楼层相对地面输入的放大比。

    previous_signal = ground
    # English: Story 1 is compared against ground input.
    # 中文：第一层与地面输入比较。
    for j in range(n_stories):
        story_id = j + 1
        diff = response[:, j] - previous_signal
        # English: Acceleration contrast relative to lower reference.
        # 中文：相对于下方参考信号的加速度差。
        values.extend([float(np.max(np.abs(diff))), float(safe_rms(diff, axis=0))])
        names.extend([f"story_{story_id}_relative_to_lower_max_abs", f"story_{story_id}_relative_to_lower_rms"])
        previous_signal = response[:, j]
        # English: Next story uses current story as lower reference.
        # 中文：下一层以下一参考为当前层。

    for j in range(1, n_stories):
        upper_id = j + 1
        lower_id = j
        values.extend([story_max_abs[j] / (story_max_abs[j - 1] + eps), story_rms[j] / (story_rms[j - 1] + eps)])
        names.extend([f"story_{upper_id}_to_story_{lower_id}_max_abs_ratio", f"story_{upper_id}_to_story_{lower_id}_rms_ratio"])
        # English: Adjacent-story response ratios.
        # 中文：相邻楼层响应比。

    sum_max_abs = float(np.sum(story_max_abs))
    sum_rms = float(np.sum(story_rms))
    # English: Denominators for spatial response fractions.
    # 中文：空间响应比例的分母。
    for j in range(n_stories):
        story_id = j + 1
        values.extend([story_max_abs[j] / (sum_max_abs + eps), story_rms[j] / (sum_rms + eps)])
        names.extend([f"story_{story_id}_max_abs_spatial_fraction", f"story_{story_id}_rms_spatial_fraction"])
        # English: Spatial response distribution over stories.
        # 中文：各楼层响应在整体响应中的占比。

    g_dom, g_centroid, g_energy = dominant_frequency_and_centroid(ground, dt, min_fft_freq, max_fft_freq, eps)
    values.extend([g_dom, g_centroid, g_energy])
    names.extend(["ground_dominant_frequency", "ground_spectral_centroid", "ground_band_energy"])
    # English: Ground frequency-domain features.
    # 中文：地面输入频域特征。

    for j in range(n_stories):
        story_id = j + 1
        dom, centroid, energy = dominant_frequency_and_centroid(response[:, j], dt, min_fft_freq, max_fft_freq, eps)
        values.extend([dom, centroid, energy, dom / (frequency_hz + eps), centroid / (frequency_hz + eps)])
        names.extend([
            f"story_{story_id}_dominant_frequency",
            f"story_{story_id}_spectral_centroid",
            f"story_{story_id}_band_energy",
            f"story_{story_id}_dominant_frequency_to_input_ratio",
            f"story_{story_id}_centroid_to_input_ratio",
        ])
        # English: Story-level spectral descriptors and their relation to input frequency.
        # 中文：每层响应频域描述及其与输入频率的关系。

    for j in range(n_stories):
        story_id = j + 1
        values.append(corrcoef_safe(response[:, j], ground, eps))
        names.append(f"story_{story_id}_ground_correlation")
        # English: Correlation between floor response and ground input.
        # 中文：楼层响应与地面输入的相关性。

    for j in range(1, n_stories):
        upper_id = j + 1
        lower_id = j
        values.append(corrcoef_safe(response[:, j], response[:, j - 1], eps))
        names.append(f"story_{upper_id}_story_{lower_id}_correlation")
        # English: Correlation between adjacent stories.
        # 中文：相邻楼层响应相关性。

    mode1_distance = abs(float(frequency_hz) - healthy_first_frequency_hz)
    resonance_indicator = 1.0 / (mode1_distance + 0.05)
    # English: Simple resonance-proximity descriptors.
    # 中文：简单近共振特征。
    values.extend([
        float(amplitude_g),
        float(frequency_hz),
        float(noise_level),
        float(frequency_hz) / (healthy_first_frequency_hz + eps),
        mode1_distance,
        resonance_indicator,
    ])
    names.extend([
        "input_amplitude_g",
        "input_frequency_hz",
        "input_noise_level",
        "input_frequency_to_healthy_mode1_ratio",
        "input_frequency_distance_to_healthy_mode1",
        "input_mode1_resonance_indicator",
    ])
    # English: Add input condition features.
    # 中文：加入输入条件特征。

    feature_array = np.asarray(values, dtype=np.float64)
    # English: Convert list of values to a numpy array.
    # 中文：将特征值列表转换为 NumPy 数组。
    if not np.all(np.isfinite(feature_array)):
        # English: Guard against NaN or infinity.
        # 中文：防止出现 NaN 或无穷大。
        raise FloatingPointError("Non-finite physics feature detected.")
    return feature_array, names


def extract_all_features(response: np.ndarray, ground: np.ndarray, amplitude_g: np.ndarray, frequency_hz: np.ndarray, noise_level: np.ndarray, dt: float, healthy_first_frequency_hz: float, min_fft_freq: float, max_fft_freq: float, eps: float) -> Tuple[np.ndarray, List[str]]:
    """Extract features for all cases. / 对全部样本提取特征。"""
    feature_rows: List[np.ndarray] = []
    feature_names: List[str] | None = None
    # English: Store rows and names.
    # 中文：保存特征行和特征名称。
    for i in range(response.shape[0]):
        row, names = extract_one_case_features(response[i], ground[i], amplitude_g[i], frequency_hz[i], noise_level[i], dt, healthy_first_frequency_hz, min_fft_freq, max_fft_freq, eps)
        # English: Extract one sample.
        # 中文：提取一个样本。
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature names changed across cases.")
        feature_rows.append(row)
    if feature_names is None:
        raise RuntimeError("No features extracted.")
    return np.vstack(feature_rows), feature_names


def standardize_by_train(F_train_raw: np.ndarray, F_val_raw: np.ndarray, F_test_raw: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using train-set statistics only. / 只用训练集统计量标准化。"""
    F_mean = np.mean(F_train_raw, axis=0, keepdims=True)
    # English: Training-set feature mean.
    # 中文：训练集特征均值。
    F_std = np.std(F_train_raw, axis=0, keepdims=True)
    # English: Training-set feature standard deviation.
    # 中文：训练集特征标准差。
    F_std = np.where(F_std < eps, 1.0, F_std)
    # English: Avoid division by zero.
    # 中文：避免除零。
    return (F_train_raw - F_mean) / F_std, (F_val_raw - F_mean) / F_std, (F_test_raw - F_mean) / F_std, F_mean, F_std


def save_feature_name_csv(feature_names: List[str], output_path: Path) -> None:
    """Save feature names to CSV. / 保存特征名称表。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "feature_name"])
        for idx, name in enumerate(feature_names):
            writer.writerow([idx, name])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", type=str, default="data_processed/debug_plus_100_dataset.npz")
    parser.add_argument("--split-dataset", type=str, default="data_processed/debug_plus_100_split_normalized.npz")
    parser.add_argument("--output", type=str, default="data_processed/debug_plus_100_physics_features_mlp.npz")
    parser.add_argument("--feature-name-csv", type=str, default="data_processed/debug_plus_100_physics_feature_names.csv")
    parser.add_argument("--summary-json", type=str, default="results/tables/debug_plus_100_physics_feature_summary.json")
    parser.add_argument("--default-dt", type=float, default=0.01)
    parser.add_argument("--healthy-first-frequency-hz", type=float, default=1.0 / 1.2700425053309456)
    parser.add_argument("--min-fft-freq", type=float, default=0.05)
    parser.add_argument("--max-fft-freq", type=float, default=8.0)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    return parser.parse_args()


def main() -> None:
    """Run feature extraction. / 执行特征提取。"""
    args = parse_args()
    raw_data = load_npz_as_dict(Path(args.raw_dataset))
    split_data = load_npz_as_dict(Path(args.split_dataset))
    # English: Load raw response data and split indices.
    # 中文：读取原始响应数据和训练/验证/测试划分索引。

    response_key, response = pick_first_available(raw_data, ["X_abs_accel", "X_clean_abs_accel", "X"])
    response = ensure_3d_response(response)
    n_cases, n_steps, n_stories = response.shape
    # English: Read response histories.
    # 中文：读取结构响应时程。

    ground_key, ground = pick_first_available(raw_data, ["ground_accel", "ground_acceleration", "ground_motion"])
    ground = ensure_ground_shape(ground, n_cases=n_cases, n_steps=n_steps)
    # English: Read ground acceleration histories.
    # 中文：读取地面输入加速度时程。

    y_key, y_damage = pick_first_available(raw_data, ["y_damage", "y", "damage"])
    y_damage = np.asarray(y_damage, dtype=np.float64)
    if y_damage.shape != (n_cases, n_stories):
        raise ValueError(f"Damage label shape mismatch: got {y_damage.shape}, expected ({n_cases}, {n_stories})")
    # English: Read damage labels.
    # 中文：读取损伤标签。

    _, amplitude_g = pick_first_available(raw_data, ["amplitude_g"])
    _, frequency_hz = pick_first_available(raw_data, ["frequency_hz"])
    _, noise_level = pick_first_available(raw_data, ["noise_level"])
    _, case_id = pick_first_available(raw_data, ["case_id"])
    # English: Read input metadata.
    # 中文：读取输入幅值、频率、噪声水平和样本编号。

    amplitude_g = np.asarray(amplitude_g, dtype=np.float64).reshape(-1)
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    noise_level = np.asarray(noise_level, dtype=np.float64).reshape(-1)
    case_id = np.asarray(case_id).reshape(-1).astype(int)
    # English: Flatten metadata arrays.
    # 中文：展平元信息数组。

    dt = get_dt(raw_data, default_dt=args.default_dt)
    # English: Get time step.
    # 中文：获取时间步长。

    for required_key in ["train_idx", "val_idx", "test_idx"]:
        if required_key not in split_data:
            raise KeyError(f"Required split key '{required_key}' not found in {args.split_dataset}")
    train_idx = split_data["train_idx"].astype(int)
    val_idx = split_data["val_idx"].astype(int)
    test_idx = split_data["test_idx"].astype(int)
    # English: Read split indices.
    # 中文：读取划分索引。

    F_all_raw, feature_names = extract_all_features(
        response=response,
        ground=ground,
        amplitude_g=amplitude_g,
        frequency_hz=frequency_hz,
        noise_level=noise_level,
        dt=dt,
        healthy_first_frequency_hz=args.healthy_first_frequency_hz,
        min_fft_freq=args.min_fft_freq,
        max_fft_freq=args.max_fft_freq,
        eps=args.eps,
    )
    # English: Extract physics features for all cases.
    # 中文：对全部样本提取物理特征。

    F_train_raw = F_all_raw[train_idx]
    F_val_raw = F_all_raw[val_idx]
    F_test_raw = F_all_raw[test_idx]
    # English: Split raw features.
    # 中文：按索引划分原始特征。

    F_train, F_val, F_test, F_mean, F_std = standardize_by_train(F_train_raw, F_val_raw, F_test_raw, eps=args.eps)
    # English: Standardize using training-set statistics only.
    # 中文：只使用训练集统计量标准化，避免数据泄漏。

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        F_train=F_train,
        F_val=F_val,
        F_test=F_test,
        F_train_raw=F_train_raw,
        F_val_raw=F_val_raw,
        F_test_raw=F_test_raw,
        y_train=y_damage[train_idx],
        y_val=y_damage[val_idx],
        y_test=y_damage[test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        F_mean=F_mean,
        F_std=F_std,
        feature_names=np.asarray(feature_names),
        case_id_train=case_id[train_idx],
        case_id_val=case_id[val_idx],
        case_id_test=case_id[test_idx],
        amplitude_g_train=amplitude_g[train_idx],
        amplitude_g_val=amplitude_g[val_idx],
        amplitude_g_test=amplitude_g[test_idx],
        frequency_hz_train=frequency_hz[train_idx],
        frequency_hz_val=frequency_hz[val_idx],
        frequency_hz_test=frequency_hz[test_idx],
        noise_level_train=noise_level[train_idx],
        noise_level_val=noise_level[val_idx],
        noise_level_test=noise_level[test_idx],
        dt=np.asarray(dt),
        healthy_first_frequency_hz=np.asarray(args.healthy_first_frequency_hz),
    )
    # English: Save feature dataset in the same format expected by train_mlp.py.
    # 中文：保存为 train_mlp.py 可直接读取的特征数据集格式。

    save_feature_name_csv(feature_names, Path(args.feature_name_csv))
    # English: Save feature-name table.
    # 中文：保存特征名称表。

    summary = {
        "raw_dataset": str(args.raw_dataset),
        "split_dataset": str(args.split_dataset),
        "output": str(args.output),
        "response_key": response_key,
        "ground_key": ground_key,
        "y_key": y_key,
        "n_cases": int(n_cases),
        "n_steps": int(n_steps),
        "n_stories": int(n_stories),
        "n_features": int(len(feature_names)),
        "dt": float(dt),
        "healthy_first_frequency_hz": float(args.healthy_first_frequency_hz),
        "F_train_shape": list(F_train.shape),
        "F_val_shape": list(F_val.shape),
        "F_test_shape": list(F_test.shape),
        "train_feature_mean_after_standardization": float(np.mean(F_train)),
        "train_feature_std_after_standardization": float(np.std(F_train)),
        "feature_name_csv": str(args.feature_name_csv),
    }
    # English: Build summary dictionary.
    # 中文：构造摘要字典。

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # English: Save summary JSON.
    # 中文：保存摘要 JSON。

    print("Physics feature extraction completed.")
    print(f"Raw dataset: {args.raw_dataset}")
    print(f"Split dataset: {args.split_dataset}")
    print(f"Response key: {response_key}")
    print(f"Ground key: {ground_key}")
    print(f"Label key: {y_key}")
    print(f"Output feature dataset: {output_path}")
    print(f"Feature name CSV: {args.feature_name_csv}")
    print(f"Summary JSON: {summary_path}")
    print(f"n_cases: {n_cases}")
    print(f"n_steps: {n_steps}")
    print(f"n_stories: {n_stories}")
    print(f"n_features: {len(feature_names)}")
    print(f"F_train shape: {F_train.shape}")
    print(f"F_val shape: {F_val.shape}")
    print(f"F_test shape: {F_test.shape}")
    print(f"y_train shape: {y_damage[train_idx].shape}")
    print(f"Train standardized mean: {np.mean(F_train):.6f}")
    print(f"Train standardized std: {np.std(F_train):.6f}")
    # English: Print key diagnostics.
    # 中文：打印关键诊断结果。


if __name__ == "__main__":
    main()
    # English: Execute main when called as a module/script.
    # 中文：作为模块或脚本运行时执行 main。
