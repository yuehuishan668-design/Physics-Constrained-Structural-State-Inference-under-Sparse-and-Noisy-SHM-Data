"""
Damage-aware weighted training for structural damage inference.

中文说明：
本脚本用于在已有 500-case physics feature 数据集上进行“损伤感知加权训练”。
目标不是单纯追求整体 MAE 最低，而是重点检查 high-damage 区间是否仍然被系统性低估。

Expected input:
    data_processed/debug_plus_500_physics_features_mlp.npz
    data_processed/debug_plus_500_physics_feature_names.csv

Expected output:
    results/tables/damage_aware_weighting/debug_plus_500/damage_aware_model_comparison.csv
    results/tables/damage_aware_weighting/debug_plus_500/damage_aware_bin_summary.csv
    results/tables/damage_aware_weighting/debug_plus_500/damage_aware_report.md
    results/figures/damage_aware_weighting/debug_plus_500/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Basic utilities
# 基础工具函数
# -----------------------------

def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist. / 如果目录不存在则创建。"""
    path.mkdir(parents=True, exist_ok=True)


def read_feature_names(path: Path) -> List[str]:
    """Read feature names from CSV. / 从 CSV 读取特征名称。"""
    names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "feature_name" not in reader.fieldnames:
            raise ValueError(f"Feature-name CSV must contain a 'feature_name' column: {path}")
        for row in reader:
            names.append(row["feature_name"])
    if not names:
        raise ValueError(f"No feature names were loaded from: {path}")
    return names


def load_feature_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Load feature dataset with flexible key detection.
    弹性读取特征数据集，兼容 F_train/F_val/F_test 或 X_train/X_val/X_test。
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    feature_key_sets = [
        ("F_train", "F_val", "F_test"),
        ("X_train", "X_val", "X_test"),
    ]
    y_key_sets = [
        ("y_train", "y_val", "y_test"),
        ("Y_train", "Y_val", "Y_test"),
    ]

    feature_keys = None
    for candidate in feature_key_sets:
        if all(k in keys for k in candidate):
            feature_keys = candidate
            break

    y_keys = None
    for candidate in y_key_sets:
        if all(k in keys for k in candidate):
            y_keys = candidate
            break

    if feature_keys is None:
        raise KeyError(f"Could not find feature keys in {npz_path}. Available keys: {sorted(keys)}")
    if y_keys is None:
        raise KeyError(f"Could not find label keys in {npz_path}. Available keys: {sorted(keys)}")

    F_train = np.asarray(data[feature_keys[0]], dtype=float)
    F_val = np.asarray(data[feature_keys[1]], dtype=float)
    F_test = np.asarray(data[feature_keys[2]], dtype=float)

    y_train = np.asarray(data[y_keys[0]], dtype=float)
    y_val = np.asarray(data[y_keys[1]], dtype=float)
    y_test = np.asarray(data[y_keys[2]], dtype=float)

    if F_train.ndim != 2 or F_val.ndim != 2 or F_test.ndim != 2:
        raise ValueError("Feature arrays must be 2D: (n_samples, n_features).")
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_val.ndim == 1:
        y_val = y_val[:, None]
    if y_test.ndim == 1:
        y_test = y_test[:, None]

    return {
        "F_train": F_train,
        "F_val": F_val,
        "F_test": F_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


# -----------------------------
# Feature-set selection
# 特征组选择
# -----------------------------

def is_metadata_feature(name: str) -> bool:
    """Identify metadata-like features. / 识别元数据类特征。"""
    lower = name.lower()
    metadata_keywords = [
        "noise",
        "amplitude",
        "frequency_hz",
        "input_frequency",
        "input_amplitude",
        "case_id",
        "seed",
    ]
    return any(k in lower for k in metadata_keywords)


def is_correlation_feature(name: str) -> bool:
    """Identify correlation features. / 识别相关性特征。"""
    return "correlation" in name.lower()


def is_frequency_feature(name: str) -> bool:
    """Identify frequency-domain features. / 识别频域特征。"""
    lower = name.lower()
    return any(k in lower for k in ["dominant_frequency", "spectral_centroid", "band_energy"])


def is_spatial_feature(name: str) -> bool:
    """Identify spatial response features. / 识别空间响应特征。"""
    lower = name.lower()
    spatial_tokens = [
        "spatial_fraction",
        "_to_story_",
        "ground_amplification",
        "to_input_ratio",
    ]
    return any(k in lower for k in spatial_tokens)


def is_basic_response_feature(name: str) -> bool:
    """
    Identify basic time-response statistics.
    识别基本时程响应统计特征。
    """
    lower = name.lower()
    basic_tokens = [
        "_mean",
        "_std",
        "_max_abs",
        "_rms",
        "_peak_to_peak",
        "_crest_factor",
        "ground_max_abs",
        "ground_rms",
    ]
    return any(k in lower for k in basic_tokens)


def select_feature_indices(feature_names: Sequence[str], feature_set: str) -> List[int]:
    """
    Select feature columns by feature-set name.
    根据特征组名称筛选特征列。
    """
    if feature_set == "full":
        return list(range(len(feature_names)))

    if feature_set == "no_meta":
        return [i for i, n in enumerate(feature_names) if not is_metadata_feature(n)]

    if feature_set == "response_spatial":
        return [
            i for i, n in enumerate(feature_names)
            if (is_basic_response_feature(n) or is_spatial_feature(n) or is_correlation_feature(n) or is_frequency_feature(n))
            and not is_metadata_feature(n)
        ]

    if feature_set == "response_basic_only":
        return [
            i for i, n in enumerate(feature_names)
            if is_basic_response_feature(n) and not is_frequency_feature(n) and not is_correlation_feature(n) and not is_metadata_feature(n)
        ]

    if feature_set == "response_frequency":
        return [i for i, n in enumerate(feature_names) if is_frequency_feature(n) and not is_metadata_feature(n)]

    if feature_set == "response_correlation":
        return [i for i, n in enumerate(feature_names) if is_correlation_feature(n) and not is_metadata_feature(n)]

    raise ValueError(f"Unknown feature set: {feature_set}")


# -----------------------------
# Weight schemes
# 加权策略
# -----------------------------

def max_damage_per_case(y: np.ndarray) -> np.ndarray:
    """
    Use the maximum story-level damage in each case as case-level severity.
    使用每个样本中四个楼层的最大损伤作为样本严重程度。
    """
    return np.max(np.asarray(y, dtype=float), axis=1)


def build_sample_weights(
    y: np.ndarray,
    scheme: str,
    low_threshold: float,
    high_threshold: float,
    max_damage: float,
    alpha: float = 8.0,
    gamma: float = 2.0,
) -> np.ndarray:
    """
    Build sample weights for damage-aware training.
    构建损伤感知训练权重。

    Weighting is case-level, not story-level, because scikit-learn sample_weight is per sample.
    权重是样本级而不是楼层级，因为 sklearn 的 sample_weight 是按样本输入的。
    """
    severity = max_damage_per_case(y)
    w = np.ones_like(severity, dtype=float)

    if scheme == "none":
        pass

    elif scheme == "moderate":
        w[(severity > 0.0) & (severity < low_threshold)] = 1.5
        w[(severity >= low_threshold) & (severity < high_threshold)] = 2.5
        w[severity >= high_threshold] = 5.0

    elif scheme == "strong":
        w[(severity > 0.0) & (severity < low_threshold)] = 2.0
        w[(severity >= low_threshold) & (severity < high_threshold)] = 4.0
        w[severity >= high_threshold] = 10.0

    elif scheme == "high_only":
        w[severity >= high_threshold] = 12.0

    elif scheme == "continuous":
        safe_max = max(max_damage, 1e-12)
        scaled = np.clip(severity / safe_max, 0.0, 1.0)
        w = 1.0 + alpha * np.power(scaled, gamma)

    elif scheme == "balanced_bins":
        # Inverse-frequency weighting across zero/low/medium/high bins.
        # 按 zero/low/medium/high 样本数的反比进行加权。
        bins = np.full_like(severity, fill_value=0, dtype=int)
        bins[(severity > 0.0) & (severity < low_threshold)] = 1
        bins[(severity >= low_threshold) & (severity < high_threshold)] = 2
        bins[severity >= high_threshold] = 3
        counts = {b: max(int(np.sum(bins == b)), 1) for b in range(4)}
        total = float(len(severity))
        n_bins = 4.0
        w = np.array([total / (n_bins * counts[int(b)]) for b in bins], dtype=float)

    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")

    # Normalize weights to mean 1 to keep regularization scale more comparable.
    # 将权重均值归一到 1，避免正则强度因权重尺度而不可比。
    mean_w = float(np.mean(w))
    if mean_w > 0:
        w = w / mean_w
    return w


# -----------------------------
# Metrics
# 指标计算
# -----------------------------

@dataclass
class Metrics:
    mae: float
    rmse: float
    bias: float
    underestimation_ratio: float
    overestimation_ratio: float
    mean_true: float
    mean_pred: float
    n_entries: int


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> Metrics:
    """
    Compute flattened multi-output metrics.
    计算多输出展平后的整体指标。
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)

    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        yt = yt[m]
        yp = yp[m]

    if yt.size == 0:
        return Metrics(
            mae=np.nan,
            rmse=np.nan,
            bias=np.nan,
            underestimation_ratio=np.nan,
            overestimation_ratio=np.nan,
            mean_true=np.nan,
            mean_pred=np.nan,
            n_entries=0,
        )

    err = yp - yt
    return Metrics(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        bias=float(np.mean(err)),
        underestimation_ratio=float(np.mean(yp < yt)),
        overestimation_ratio=float(np.mean(yp > yt)),
        mean_true=float(np.mean(yt)),
        mean_pred=float(np.mean(yp)),
        n_entries=int(yt.size),
    )


def make_damage_masks(y_true: np.ndarray, low_threshold: float, high_threshold: float) -> Dict[str, np.ndarray]:
    """
    Build entry-level damage masks.
    构建按条目展开的损伤等级掩码。
    """
    y = np.asarray(y_true, dtype=float)
    return {
        "zero": y == 0.0,
        "low": (y > 0.0) & (y < low_threshold),
        "medium": (y >= low_threshold) & (y < high_threshold),
        "high": y >= high_threshold,
        "damaged": y > 0.0,
        "all": np.ones_like(y, dtype=bool),
    }


def metric_dict(prefix: str, metrics: Metrics) -> Dict[str, float]:
    """Convert metrics to a dictionary. / 将指标对象转成字典。"""
    return {
        f"{prefix}_mae": metrics.mae,
        f"{prefix}_rmse": metrics.rmse,
        f"{prefix}_bias": metrics.bias,
        f"{prefix}_underestimation_ratio": metrics.underestimation_ratio,
        f"{prefix}_overestimation_ratio": metrics.overestimation_ratio,
        f"{prefix}_mean_true": metrics.mean_true,
        f"{prefix}_mean_pred": metrics.mean_pred,
        f"{prefix}_n_entries": metrics.n_entries,
    }


# -----------------------------
# Model creation and tuning
# 模型构建与调参
# -----------------------------

def make_model_candidates(model_name: str, random_seed: int) -> List[Tuple[str, object]]:
    """
    Return candidate estimators.
    返回候选模型列表。
    """
    candidates: List[Tuple[str, object]] = []

    if model_name == "ridge":
        for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha, random_state=random_seed)),
            ])
            candidates.append((f"ridge_alpha_{alpha:g}", model))

    elif model_name == "elasticnet":
        for alpha in [0.001, 0.003, 0.01, 0.03, 0.05, 0.1]:
            for l1_ratio in [0.1, 0.3, 0.5, 0.7]:
                base = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=20000,
                    tol=1e-4,
                    random_state=random_seed,
                    selection="cyclic",
                )
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", MultiOutputRegressor(base)),
                ])
                candidates.append((f"elasticnet_alpha_{alpha:g}_l1_{l1_ratio:g}", model))

    elif model_name == "random_forest":
        for n_estimators in [200, 300]:
            for max_depth in [2, 3, 4, 5]:
                for min_samples_leaf in [1, 2, 4]:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_seed,
                        n_jobs=-1,
                    )
                    candidates.append(
                        (
                            f"random_forest_n_{n_estimators}_depth_{max_depth}_leaf_{min_samples_leaf}",
                            model,
                        )
                    )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return candidates


def fit_with_optional_sample_weight(model: object, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> object:
    """
    Fit estimator with sample_weight where supported.
    在模型支持 sample_weight 时使用加权训练。
    """
    try:
        if isinstance(model, Pipeline):
            last_step_name = list(model.named_steps.keys())[-1]
            model.fit(X, y, **{f"{last_step_name}__sample_weight": sample_weight})
        else:
            model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        # Fallback for old sklearn versions or unsupported wrappers.
        # 兼容旧版本 sklearn 或不支持 sample_weight 的封装器。
        model.fit(X, y)
    return model


def composite_validation_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    high_mae_weight: float,
    high_under_bias_weight: float,
    zero_mae_weight: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute validation score that penalizes high-damage underestimation.
    计算强调高损伤低估惩罚的验证集综合评分。
    """
    overall = compute_metrics(y_true, y_pred)
    masks = make_damage_masks(y_true, low_threshold=low_threshold, high_threshold=high_threshold)
    high = compute_metrics(y_true, y_pred, masks["high"])
    zero = compute_metrics(y_true, y_pred, masks["zero"])

    high_under_bias = max(0.0, -high.bias) if not math.isnan(high.bias) else 0.0

    score = (
        overall.mae
        + high_mae_weight * (0.0 if math.isnan(high.mae) else high.mae)
        + high_under_bias_weight * high_under_bias
        + zero_mae_weight * (0.0 if math.isnan(zero.mae) else zero.mae)
    )

    diagnostics = {
        "val_score": float(score),
        "val_mae": overall.mae,
        "val_rmse": overall.rmse,
        "val_bias": overall.bias,
        "val_high_mae": high.mae,
        "val_high_bias": high.bias,
        "val_high_underestimation_ratio": high.underestimation_ratio,
        "val_zero_mae": zero.mae,
    }
    return float(score), diagnostics


# -----------------------------
# Output writing
# 输出写入
# -----------------------------

def save_predictions_csv(
    output_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    feature_set: str,
    model: str,
    weight_scheme: str,
) -> None:
    """Save flattened prediction table. / 保存展开后的预测结果表。"""
    ensure_dir(output_path.parent)
    rows = []
    n_samples, n_outputs = y_true.shape
    for i in range(n_samples):
        for j in range(n_outputs):
            rows.append({
                "split": split_name,
                "case_index_in_split": i,
                "story": j + 1,
                "feature_set": feature_set,
                "model": model,
                "weight_scheme": weight_scheme,
                "true_damage": float(y_true[i, j]),
                "pred_damage": float(y_pred[i, j]),
                "error": float(y_pred[i, j] - y_true[i, j]),
                "abs_error": float(abs(y_pred[i, j] - y_true[i, j])),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def dataframe_to_markdown(df: pd.DataFrame, columns: Sequence[str], float_digits: int = 6) -> str:
    """
    Convert DataFrame to markdown without requiring tabulate.
    不依赖 tabulate，将 DataFrame 转成 Markdown 表格。
    """
    use_df = df.loc[:, list(columns)].copy()

    def fmt(v):
        if isinstance(v, float) or isinstance(v, np.floating):
            if np.isnan(v):
                return "nan"
            return f"{float(v):.{float_digits}f}"
        return str(v)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in use_df.iterrows():
        body.append("| " + " | ".join(fmt(row[c]) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def write_report(
    report_path: Path,
    comparison_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> None:
    """
    Write markdown report.
    输出 Markdown 总结报告。
    """
    ensure_dir(report_path.parent)
    sorted_df = comparison_df.sort_values(["test_mae", "test_high_mae", "test_high_underestimation_ratio"]).reset_index(drop=True)
    best = sorted_df.iloc[0]

    safety_df = comparison_df.sort_values(["test_high_mae", "test_high_underestimation_ratio", "test_mae"]).reset_index(drop=True)
    best_safety = safety_df.iloc[0]

    top_cols = [
        "rank", "feature_set", "model", "weight_scheme", "n_features",
        "best_candidate", "val_score", "test_mae", "test_rmse",
        "test_bias", "test_zero_mae", "test_high_mae",
        "test_high_bias", "test_high_underestimation_ratio",
    ]

    comparison_for_table = sorted_df.copy()
    comparison_for_table.insert(0, "rank", np.arange(1, len(comparison_for_table) + 1))

    high_cols = [
        "rank", "feature_set", "model", "weight_scheme", "n_features",
        "test_mae", "test_high_mae", "test_high_bias",
        "test_high_underestimation_ratio", "test_zero_mae",
    ]
    safety_for_table = safety_df.copy()
    safety_for_table.insert(0, "rank", np.arange(1, len(safety_for_table) + 1))

    lines = []
    lines.append("# Damage-aware Weighted Training Summary\n")
    lines.append("## 1. Purpose\n")
    lines.append(
        "This experiment tests whether damage-aware sample weighting can reduce severe-damage "
        "underestimation while keeping the overall error within an acceptable range.\n"
    )
    lines.append("中文解释：本实验用于检查损伤感知加权训练是否能够降低严重损伤低估，同时避免整体误差明显恶化。\n")

    lines.append("## 2. Damage bins\n")
    lines.append(f"- zero: damage = 0\n")
    lines.append(f"- low: 0 < damage < {low_threshold}\n")
    lines.append(f"- medium: {low_threshold} <= damage < {high_threshold}\n")
    lines.append(f"- high: damage >= {high_threshold}\n")

    lines.append("## 3. Best overall configuration\n")
    lines.append(f"- Best by overall test MAE: `{best['feature_set']} + {best['model']} + {best['weight_scheme']}`\n")
    lines.append(f"- Test MAE: `{best['test_mae']:.6f}`\n")
    lines.append(f"- Test RMSE: `{best['test_rmse']:.6f}`\n")
    lines.append(f"- Test high-damage MAE: `{best['test_high_mae']:.6f}`\n")
    lines.append(f"- Test high-damage bias: `{best['test_high_bias']:.6f}`\n")
    lines.append(f"- Test high-damage underestimation ratio: `{best['test_high_underestimation_ratio']:.6f}`\n\n")

    lines.append("## 4. Best safety-oriented configuration\n")
    lines.append(f"- Best by high-damage MAE: `{best_safety['feature_set']} + {best_safety['model']} + {best_safety['weight_scheme']}`\n")
    lines.append(f"- Test MAE: `{best_safety['test_mae']:.6f}`\n")
    lines.append(f"- Test high-damage MAE: `{best_safety['test_high_mae']:.6f}`\n")
    lines.append(f"- Test high-damage bias: `{best_safety['test_high_bias']:.6f}`\n")
    lines.append(f"- Test high-damage underestimation ratio: `{best_safety['test_high_underestimation_ratio']:.6f}`\n\n")

    lines.append("## 5. Overall ranking by test MAE\n")
    lines.append(dataframe_to_markdown(comparison_for_table.head(20), top_cols))
    lines.append("\n\n")

    lines.append("## 6. Safety ranking by high-damage MAE\n")
    lines.append(dataframe_to_markdown(safety_for_table.head(20), high_cols))
    lines.append("\n\n")

    lines.append("## 7. Preliminary interpretation\n")
    lines.append(
        "- If weighted configurations reduce high-damage MAE but increase zero-damage MAE sharply, "
        "the model is trading severe-damage sensitivity for false alarms.\n"
    )
    lines.append(
        "- If `full + ridge + weighted` improves high-damage MAE with limited overall MAE increase, "
        "it can become the main methodological improvement over the unweighted baseline.\n"
    )
    lines.append(
        "- If all weighted schemes still underestimate high damage, the next step should move from sample weighting "
        "to either damage-bin-specific calibration or a two-stage damage classifier/regressor.\n"
    )
    lines.append("\n中文解释：\n")
    lines.append("- 如果加权模型降低了高损伤 MAE，但零损伤 MAE 明显上升，说明模型用误报换取高损伤敏感性。\n")
    lines.append("- 如果 `full + ridge + weighted` 能在整体 MAE 小幅增加的前提下降低高损伤 MAE，它就可以作为论文中的主要方法改进。\n")
    lines.append("- 如果所有加权方案仍然明显低估高损伤，下一步应转向损伤等级校准或二阶段分类-回归模型。\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Plotting
# 绘图
# -----------------------------

def config_label(row: pd.Series) -> str:
    return f"{row['feature_set']} + {row['model']} + {row['weight_scheme']}"


def plot_top_bars(
    df: pd.DataFrame,
    y_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
    top_k: int,
) -> None:
    """Plot top-k bar chart. / 绘制前 top-k 柱状图。"""
    ensure_dir(output_path.parent)
    use_df = df.sort_values(y_col).head(top_k).copy()
    labels = [config_label(r) for _, r in use_df.iterrows()]
    values = use_df[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(12, 0.7 * len(labels)), 6))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    """Plot overall-vs-high-damage tradeoff. / 绘制整体误差与高损伤误差权衡图。"""
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 7))

    labels = []
    for _, row in df.iterrows():
        label = config_label(row)
        labels.append(label)
        ax.scatter(row["test_mae"], row["test_high_mae"], s=60, alpha=0.75)

    ax.set_xlabel("Overall test MAE")
    ax.set_ylabel("High-damage test MAE")
    ax.set_title("Overall accuracy versus high-damage sensitivity")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Annotate only the top few to avoid unreadable plot.
    # 只标注少量关键点，避免图像过于拥挤。
    key_df = df.sort_values(["test_mae", "test_high_mae"]).head(5)
    key_df = pd.concat([key_df, df.sort_values(["test_high_mae", "test_mae"]).head(5)]).drop_duplicates()
    for _, row in key_df.iterrows():
        ax.annotate(
            f"{row['feature_set']}\n{row['model']}\n{row['weight_scheme']}",
            (row["test_mae"], row["test_high_mae"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_bias_underestimation(df: pd.DataFrame, output_path: Path, top_k: int) -> None:
    """Plot high-damage bias and underestimation ratio. / 绘制高损伤偏差与低估比例。"""
    ensure_dir(output_path.parent)
    use_df = df.sort_values("test_high_mae").head(top_k).copy()
    labels = [config_label(r) for _, r in use_df.iterrows()]
    x = np.arange(len(use_df))

    fig, ax1 = plt.subplots(figsize=(max(12, 0.7 * len(labels)), 6))
    ax1.bar(x - 0.2, use_df["test_high_bias"].to_numpy(dtype=float), width=0.4, label="High-damage bias")
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.set_ylabel("High-damage bias")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, use_df["test_high_underestimation_ratio"].to_numpy(dtype=float), width=0.4, alpha=0.5, label="Underestimation ratio")
    ax2.set_ylabel("High-damage underestimation ratio")
    ax2.set_ylim(0.0, 1.05)

    ax1.set_title("High-damage bias and underestimation ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_best_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str, max_damage: float) -> None:
    """Plot true-vs-predicted scatter for selected model. / 绘制最佳模型散点图。"""
    ensure_dir(output_path.parent)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    bins = np.full_like(yt, fill_value="high", dtype=object)
    bins[yt == 0.0] = "zero"
    bins[(yt > 0.0) & (yt < 0.1)] = "low"
    bins[(yt >= 0.1) & (yt < 0.2)] = "medium"

    fig, ax = plt.subplots(figsize=(8, 8))
    for name in ["zero", "low", "medium", "high"]:
        mask = bins == name
        if np.any(mask):
            ax.scatter(yt[mask], yp[mask], label=name, alpha=0.65, s=35)

    limit = max(max_damage, float(np.nanmax(yt)), float(np.nanmax(yp)), 0.5)
    ax.plot([0, limit], [0, limit], linestyle="--", label="Ideal y=x")
    ax.set_xlim(-0.02 * limit, limit * 1.02)
    ax.set_ylim(-0.02 * limit, limit * 1.02)
    ax.set_xlabel("True damage")
    ax.set_ylabel("Predicted damage")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Main experiment
# 主实验流程
# -----------------------------

def run_experiment(args: argparse.Namespace) -> None:
    features_path = Path(args.features)
    feature_names_path = Path(args.feature_names)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    dataset = load_feature_dataset(features_path)
    feature_names = read_feature_names(feature_names_path)

    if dataset["F_train"].shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: F_train has {dataset['F_train'].shape[1]} columns, "
            f"but feature-name CSV has {len(feature_names)} names."
        )

    F_train = dataset["F_train"]
    F_val = dataset["F_val"]
    F_test = dataset["F_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]

    print("Damage-aware weighted training started.")
    print(f"Feature dataset: {features_path}")
    print(f"Feature-name CSV: {feature_names_path}")
    print(f"Output directory: {output_dir}")
    print(f"Figures directory: {figures_dir}")
    print(f"Train shape: {F_train.shape}, y_train shape: {y_train.shape}")
    print(f"Val shape: {F_val.shape}, y_val shape: {y_val.shape}")
    print(f"Test shape: {F_test.shape}, y_test shape: {y_test.shape}")
    print(f"Feature sets: {args.feature_sets}")
    print(f"Models: {args.models}")
    print(f"Weight schemes: {args.weight_schemes}")

    comparison_rows: List[Dict[str, object]] = []
    bin_rows: List[Dict[str, object]] = []
    prediction_paths: Dict[str, Path] = {}

    best_overall_tuple = None
    best_safety_tuple = None

    for feature_set in args.feature_sets:
        indices = select_feature_indices(feature_names, feature_set)
        if len(indices) == 0:
            print(f"[SKIP] {feature_set}: no selected features.")
            continue

        X_train = F_train[:, indices]
        X_val = F_val[:, indices]
        X_test = F_test[:, indices]

        for weight_scheme in args.weight_schemes:
            sample_weight = build_sample_weights(
                y_train,
                scheme=weight_scheme,
                low_threshold=args.low_threshold,
                high_threshold=args.high_threshold,
                max_damage=args.max_damage,
                alpha=args.weight_alpha,
                gamma=args.weight_gamma,
            )

            weight_info = {
                "feature_set": feature_set,
                "weight_scheme": weight_scheme,
                "n_features": len(indices),
                "weight_min": float(np.min(sample_weight)),
                "weight_max": float(np.max(sample_weight)),
                "weight_mean": float(np.mean(sample_weight)),
                "weight_std": float(np.std(sample_weight)),
            }

            for model_name in args.models:
                print(f"Running {feature_set} + {model_name} + {weight_scheme} ...")
                candidates = make_model_candidates(model_name, random_seed=args.random_seed)

                best_candidate_name = None
                best_model = None
                best_score = np.inf
                best_val_diag: Dict[str, float] = {}

                for candidate_name, candidate_model in candidates:
                    model = clone(candidate_model)
                    model = fit_with_optional_sample_weight(model, X_train, y_train, sample_weight)
                    val_pred = np.asarray(model.predict(X_val), dtype=float)
                    if args.clip_predictions:
                        val_pred = np.clip(val_pred, 0.0, args.max_damage)

                    score, diag = composite_validation_score(
                        y_val,
                        val_pred,
                        low_threshold=args.low_threshold,
                        high_threshold=args.high_threshold,
                        high_mae_weight=args.high_mae_weight,
                        high_under_bias_weight=args.high_under_bias_weight,
                        zero_mae_weight=args.zero_mae_weight,
                    )

                    if score < best_score:
                        best_score = score
                        best_candidate_name = candidate_name
                        best_model = model
                        best_val_diag = diag

                if best_model is None or best_candidate_name is None:
                    raise RuntimeError(f"No model was selected for {feature_set} + {model_name} + {weight_scheme}")

                test_pred = np.asarray(best_model.predict(X_test), dtype=float)
                if args.clip_predictions:
                    test_pred = np.clip(test_pred, 0.0, args.max_damage)

                overall = compute_metrics(y_test, test_pred)
                masks = make_damage_masks(y_test, low_threshold=args.low_threshold, high_threshold=args.high_threshold)
                zero = compute_metrics(y_test, test_pred, masks["zero"])
                low = compute_metrics(y_test, test_pred, masks["low"])
                medium = compute_metrics(y_test, test_pred, masks["medium"])
                high = compute_metrics(y_test, test_pred, masks["high"])
                damaged = compute_metrics(y_test, test_pred, masks["damaged"])

                row: Dict[str, object] = {
                    "feature_set": feature_set,
                    "model": model_name,
                    "weight_scheme": weight_scheme,
                    "n_features": len(indices),
                    "best_candidate": best_candidate_name,
                    "clip_predictions": bool(args.clip_predictions),
                    **weight_info,
                    **best_val_diag,
                    **metric_dict("test", overall),
                    "test_zero_mae": zero.mae,
                    "test_low_mae": low.mae,
                    "test_medium_mae": medium.mae,
                    "test_high_mae": high.mae,
                    "test_damaged_mae": damaged.mae,
                    "test_zero_bias": zero.bias,
                    "test_low_bias": low.bias,
                    "test_medium_bias": medium.bias,
                    "test_high_bias": high.bias,
                    "test_damaged_bias": damaged.bias,
                    "test_high_underestimation_ratio": high.underestimation_ratio,
                    "test_damaged_underestimation_ratio": damaged.underestimation_ratio,
                }
                comparison_rows.append(row)

                for bin_name, metric in [
                    ("zero", zero),
                    ("low", low),
                    ("medium", medium),
                    ("high", high),
                    ("damaged", damaged),
                    ("all", overall),
                ]:
                    bin_rows.append({
                        "feature_set": feature_set,
                        "model": model_name,
                        "weight_scheme": weight_scheme,
                        "bin": bin_name,
                        "n_features": len(indices),
                        "n_entries": metric.n_entries,
                        "mae": metric.mae,
                        "rmse": metric.rmse,
                        "bias": metric.bias,
                        "underestimation_ratio": metric.underestimation_ratio,
                        "overestimation_ratio": metric.overestimation_ratio,
                        "mean_true": metric.mean_true,
                        "mean_pred": metric.mean_pred,
                    })

                safe_name = f"{feature_set}__{model_name}__{weight_scheme}"
                pred_path = output_dir / "predictions" / safe_name / "predictions_test.csv"
                save_predictions_csv(
                    pred_path,
                    y_true=y_test,
                    y_pred=test_pred,
                    split_name="test",
                    feature_set=feature_set,
                    model=model_name,
                    weight_scheme=weight_scheme,
                )
                prediction_paths[safe_name] = pred_path

                print(
                    f"  selected={best_candidate_name}, "
                    f"val_score={best_score:.6f}, "
                    f"test_mae={overall.mae:.6f}, "
                    f"test_high_mae={high.mae:.6f}, "
                    f"test_high_bias={high.bias:.6f}, "
                    f"high_under={high.underestimation_ratio:.6f}"
                )

                # Keep raw y_pred for best plots.
                # 保存最佳图所需的预测数组。
                if best_overall_tuple is None or overall.mae < best_overall_tuple[0]:
                    best_overall_tuple = (overall.mae, safe_name, y_test.copy(), test_pred.copy())
                if best_safety_tuple is None or high.mae < best_safety_tuple[0]:
                    best_safety_tuple = (high.mae, safe_name, y_test.copy(), test_pred.copy())

    comparison_df = pd.DataFrame(comparison_rows)
    bin_df = pd.DataFrame(bin_rows)

    if comparison_df.empty:
        raise RuntimeError("No experiment result was produced.")

    comparison_df = comparison_df.sort_values(["test_mae", "test_high_mae"]).reset_index(drop=True)
    bin_df = bin_df.sort_values(["feature_set", "model", "weight_scheme", "bin"]).reset_index(drop=True)

    comparison_csv = output_dir / "damage_aware_model_comparison.csv"
    bin_csv = output_dir / "damage_aware_bin_summary.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    bin_df.to_csv(bin_csv, index=False)

    write_report(
        output_dir / "damage_aware_report.md",
        comparison_df=comparison_df,
        bin_df=bin_df,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )

    plot_top_bars(
        comparison_df,
        y_col="test_mae",
        output_path=figures_dir / "top_overall_test_mae.png",
        title="Top configurations by overall test MAE",
        ylabel="Overall test MAE",
        top_k=args.top_k,
    )
    plot_top_bars(
        comparison_df,
        y_col="test_high_mae",
        output_path=figures_dir / "top_high_damage_test_mae.png",
        title="Top configurations by high-damage test MAE",
        ylabel="High-damage test MAE",
        top_k=args.top_k,
    )
    plot_tradeoff(
        comparison_df,
        output_path=figures_dir / "overall_vs_high_damage_tradeoff.png",
    )
    plot_bias_underestimation(
        comparison_df,
        output_path=figures_dir / "high_damage_bias_underestimation.png",
        top_k=args.top_k,
    )

    if best_overall_tuple is not None:
        _, safe_name, yt, yp = best_overall_tuple
        plot_best_scatter(
            yt,
            yp,
            output_path=figures_dir / "best_overall_true_vs_predicted.png",
            title=f"Best overall true vs predicted damage: {safe_name}",
            max_damage=args.max_damage,
        )

    if best_safety_tuple is not None:
        _, safe_name, yt, yp = best_safety_tuple
        plot_best_scatter(
            yt,
            yp,
            output_path=figures_dir / "best_high_damage_true_vs_predicted.png",
            title=f"Best high-damage true vs predicted damage: {safe_name}",
            max_damage=args.max_damage,
        )

    summary_json = {
        "features": str(features_path),
        "feature_names": str(feature_names_path),
        "output_dir": str(output_dir),
        "figures_dir": str(figures_dir),
        "n_train": int(F_train.shape[0]),
        "n_val": int(F_val.shape[0]),
        "n_test": int(F_test.shape[0]),
        "n_outputs": int(y_train.shape[1]),
        "comparison_csv": str(comparison_csv),
        "bin_csv": str(bin_csv),
        "report_md": str(output_dir / "damage_aware_report.md"),
    }
    (output_dir / "damage_aware_run_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDamage-aware weighted training completed.")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Bin summary CSV: {bin_csv}")
    print(f"Report MD: {output_dir / 'damage_aware_report.md'}")
    print(f"Figures directory: {figures_dir}")

    print("\nTop configurations by overall Test MAE:")
    top = comparison_df.sort_values("test_mae").head(args.top_k)
    for _, r in top.iterrows():
        print(
            f"{r['feature_set']} + {r['model']} + {r['weight_scheme']} | "
            f"test_mae={r['test_mae']:.6f} | "
            f"high_mae={r['test_high_mae']:.6f} | "
            f"high_bias={r['test_high_bias']:.6f} | "
            f"high_under={r['test_high_underestimation_ratio']:.6f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run damage-aware weighted sklearn baselines.")

    parser.add_argument("--features", required=True, help="Path to physics feature dataset NPZ.")
    parser.add_argument("--feature-names", required=True, help="Path to feature-name CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for tables and reports.")
    parser.add_argument("--figures-dir", required=True, help="Directory for figures.")

    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["full", "no_meta", "response_spatial", "response_basic_only"],
        choices=[
            "full",
            "no_meta",
            "response_spatial",
            "response_basic_only",
            "response_frequency",
            "response_correlation",
        ],
        help="Feature sets to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "elasticnet", "random_forest"],
        choices=["ridge", "elasticnet", "random_forest"],
        help="Model families to evaluate.",
    )
    parser.add_argument(
        "--weight-schemes",
        nargs="+",
        default=["none", "moderate", "strong", "high_only", "continuous", "balanced_bins"],
        choices=["none", "moderate", "strong", "high_only", "continuous", "balanced_bins"],
        help="Sample-weight schemes to evaluate.",
    )

    parser.add_argument("--low-threshold", type=float, default=0.10, help="Low/medium damage threshold.")
    parser.add_argument("--high-threshold", type=float, default=0.20, help="Medium/high damage threshold.")
    parser.add_argument("--max-damage", type=float, default=0.50, help="Maximum damage value for clipping and plotting.")
    parser.add_argument("--clip-predictions", action="store_true", help="Clip predictions to [0, max_damage].")

    parser.add_argument("--weight-alpha", type=float, default=8.0, help="Alpha for continuous weighting.")
    parser.add_argument("--weight-gamma", type=float, default=2.0, help="Gamma for continuous weighting.")

    parser.add_argument("--high-mae-weight", type=float, default=0.50, help="Validation score weight for high-damage MAE.")
    parser.add_argument("--high-under-bias-weight", type=float, default=0.50, help="Validation score weight for high-damage under-bias.")
    parser.add_argument("--zero-mae-weight", type=float, default=0.10, help="Validation score weight for zero-damage MAE.")

    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top configurations to display/plot.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
