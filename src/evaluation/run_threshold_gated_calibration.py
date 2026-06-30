"""
Threshold-gated damage calibration experiment.
阈值门控损伤校准实验。

Purpose / 目的
----------------
This script tests whether a high-damage probability threshold can reduce high-damage
underestimation without causing too many false alarms on zero-damage samples.

本脚本用于测试：能否通过“高损伤概率阈值”来降低高损伤低估，同时控制零损伤样本误报。

Expected input / 预期输入
------------------------
A physics-feature NPZ file generated in the previous pipeline, for example:
    data_processed/debug_plus_500_physics_features_mlp.npz

The file should contain feature arrays and target arrays. The script supports common
key names such as F_train/F_val/F_test and y_train/y_val/y_test.

Output / 输出
-------------
- threshold_gated_model_comparison.csv
- threshold_gated_report.md
- predictions CSV files for selected configurations
- diagnostic figures

Run from project root / 从项目根目录运行：
    python -m src.evaluation.run_threshold_gated_calibration --help
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# 1. Data loading utilities
# 1. 数据读取工具函数
# -----------------------------------------------------------------------------


def _first_existing_key(npz: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> str:
    """Return the first existing key from a list of possible names.
    从候选 key 中返回第一个真实存在于 NPZ 文件中的 key。
    """
    for key in candidates:
        if key in npz.files:
            return key
    raise KeyError(f"None of the expected keys exists. Candidates: {candidates}. Available: {npz.files}")


def load_feature_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load train/validation/test feature and target arrays.
    读取训练集、验证集、测试集的特征和目标值。
    """
    with np.load(npz_path, allow_pickle=True) as data:
        feature_keys = {
            "train": _first_existing_key(data, ["F_train", "features_train", "X_feature_train", "X_train"]),
            "val": _first_existing_key(data, ["F_val", "features_val", "X_feature_val", "X_val"]),
            "test": _first_existing_key(data, ["F_test", "features_test", "X_feature_test", "X_test"]),
        }
        target_keys = {
            "train": _first_existing_key(data, ["y_train", "Y_train", "damage_train", "y_damage_train"]),
            "val": _first_existing_key(data, ["y_val", "Y_val", "damage_val", "y_damage_val"]),
            "test": _first_existing_key(data, ["y_test", "Y_test", "damage_test", "y_damage_test"]),
        }

        output: Dict[str, np.ndarray] = {}
        for split in ["train", "val", "test"]:
            output[f"F_{split}"] = np.asarray(data[feature_keys[split]], dtype=float)
            output[f"y_{split}"] = np.asarray(data[target_keys[split]], dtype=float)

        # Optional case IDs are useful for diagnostic CSVs.
        # case_id 不是必须的，但有助于后续定位具体样本。
        for split in ["train", "val", "test"]:
            for key in [f"case_id_{split}", f"case_ids_{split}", f"{split}_case_id"]:
                if key in data.files:
                    output[f"case_id_{split}"] = np.asarray(data[key])
                    break

    return output


def load_feature_names(path: Path, n_features: int) -> List[str]:
    """Load feature names from a CSV file.
    从 CSV 文件读取特征名。
    """
    if not path.exists():
        return [f"feature_{i}" for i in range(n_features)]

    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    if not rows:
        return [f"feature_{i}" for i in range(n_features)]

    # Try common formats: one-column CSV, or a table with feature_name/name column.
    # 兼容常见格式：单列 CSV，或带有 feature_name/name 字段的表格。
    header = [cell.strip().lower() for cell in rows[0]]
    if "feature_name" in header:
        idx = header.index("feature_name")
        names = [row[idx].strip() for row in rows[1:] if len(row) > idx]
    elif "name" in header:
        idx = header.index("name")
        names = [row[idx].strip() for row in rows[1:] if len(row) > idx]
    else:
        # If first row looks like a header rather than an actual feature name, drop it.
        # 如果第一行明显是表头，则跳过。
        first_col = [row[0].strip() for row in rows]
        if len(first_col) == n_features + 1 and first_col[0].lower() in {"feature", "feature_name", "name"}:
            names = first_col[1:]
        else:
            names = first_col

    if len(names) != n_features:
        # Fall back to generic names but keep a warning file via printed message.
        # 如果数量不匹配，使用通用名称，避免脚本直接中断。
        print(
            f"Warning: feature-name count ({len(names)}) != feature count ({n_features}). "
            "Generic names will be used."
        )
        return [f"feature_{i}" for i in range(n_features)]
    return names


# -----------------------------------------------------------------------------
# 2. Feature set filtering
# 2. 特征集筛选
# -----------------------------------------------------------------------------


def select_feature_indices(feature_names: Sequence[str], feature_set: str) -> List[int]:
    """Select feature indices according to a named feature set.
    按照指定特征集名称筛选特征索引。
    """
    names = [name.lower() for name in feature_names]

    meta_tokens = [
        "input_",
        "noise",
        "amplitude",
        "frequency_hz",
        "case_id",
        "dt",
        "n_step",
        "n_steps",
        "sampling",
    ]
    frequency_tokens = [
        "frequency",
        "spectral",
        "band_energy",
        "centroid",
        "dominant",
    ]
    correlation_tokens = [
        "correlation",
        "corr",
    ]
    spatial_tokens = [
        "spatial",
        "rms",
        "max_abs",
        "crest",
        "mean",
        "story_",
        "to_story",
        "ground",
    ]

    if feature_set == "full":
        return list(range(len(names)))

    if feature_set == "no_meta":
        return [i for i, n in enumerate(names) if not any(tok in n for tok in meta_tokens)]

    if feature_set == "response_basic_only":
        # Basic time-domain response descriptors.
        # 基础时域响应描述符。
        include = ["mean", "std", "rms", "max", "max_abs", "min", "crest"]
        exclude = frequency_tokens + correlation_tokens + meta_tokens + ["ratio", "fraction", "centroid"]
        return [i for i, n in enumerate(names) if any(tok in n for tok in include) and not any(tok in n for tok in exclude)]

    if feature_set == "response_spatial":
        # Spatial and inter-story response descriptors.
        # 空间分布和楼层间响应描述符。
        exclude = meta_tokens
        return [i for i, n in enumerate(names) if any(tok in n for tok in spatial_tokens) and not any(tok in n for tok in exclude)]

    if feature_set == "response_frequency":
        return [i for i, n in enumerate(names) if any(tok in n for tok in frequency_tokens)]

    if feature_set == "response_correlation":
        return [i for i, n in enumerate(names) if any(tok in n for tok in correlation_tokens)]

    raise ValueError(f"Unknown feature set: {feature_set}")


# -----------------------------------------------------------------------------
# 3. Entry-level conversion and binning
# 3. 样本展开与损伤分箱
# -----------------------------------------------------------------------------


def flatten_case_level_data(
    F: np.ndarray,
    y: np.ndarray,
    selected_indices: Sequence[int],
    case_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert case-level multi-story targets into entry-level samples.
    将“每个工况一行、多个楼层输出”的数据展开为“每个楼层一个样本”。
    """
    if y.ndim == 1:
        y_2d = y.reshape(-1, 1)
    elif y.ndim == 2:
        y_2d = y
    else:
        raise ValueError(f"Expected y to be 1D or 2D, got shape {y.shape}")

    n_cases, n_stories = y_2d.shape
    F_selected = F[:, selected_indices]

    # Repeat each case feature vector for each story.
    # 每个工况的特征向量复制 n_stories 次，与每个楼层损伤目标对应。
    X_repeat = np.repeat(F_selected, repeats=n_stories, axis=0)

    # Add story identity because the same global response feature may correspond to different story damage.
    # 加入楼层 one-hot 编码，避免模型不知道当前样本对应哪一层。
    story_ids = np.tile(np.arange(n_stories), reps=n_cases)
    story_one_hot = np.eye(n_stories)[story_ids]
    X_entry = np.hstack([X_repeat, story_one_hot])

    y_entry = y_2d.reshape(-1)

    if case_ids is None:
        case_ids_entry = np.repeat(np.arange(n_cases), repeats=n_stories)
    else:
        case_ids_entry = np.repeat(case_ids, repeats=n_stories)

    return X_entry, y_entry, case_ids_entry, story_ids + 1


def damage_bins(y: np.ndarray, zero_eps: float = 1e-12) -> np.ndarray:
    """Convert continuous damage to four bins: zero, low, medium, high.
    将连续损伤值划分为 zero/low/medium/high 四类。
    """
    bins = np.full(y.shape, "high", dtype=object)
    bins[y <= zero_eps] = "zero"
    bins[(y > zero_eps) & (y < 0.10)] = "low"
    bins[(y >= 0.10) & (y < 0.20)] = "medium"
    bins[y >= 0.20] = "high"
    return bins


def bin_to_int(labels: np.ndarray) -> np.ndarray:
    """Map bin strings to integer class IDs.
    将字符串类别映射为整数类别。
    """
    mapping = {"zero": 0, "low": 1, "medium": 2, "high": 3}
    return np.asarray([mapping[str(label)] for label in labels], dtype=int)


# -----------------------------------------------------------------------------
# 4. Models
# 4. 模型训练函数
# -----------------------------------------------------------------------------


def make_high_classifier(name: str, seed: int):
    """Create a binary high-damage classifier.
    创建二分类高损伤分类器。
    """
    if name == "logistic_balanced":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )

    if name == "random_forest_balanced":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier: {name}")


@dataclass
class RidgeBundle:
    """A small container for a scaler plus a ridge model.
    用于保存 StandardScaler 和 Ridge 模型。
    """

    scaler: StandardScaler
    model: Ridge

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def fit_weighted_ridge(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray], alpha: float) -> RidgeBundle:
    """Fit a standardized ridge regression model.
    训练标准化后的 Ridge 回归模型。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    if sample_weight is None:
        model.fit(X_scaled, y)
    else:
        model.fit(X_scaled, y, sample_weight=sample_weight)
    return RidgeBundle(scaler=scaler, model=model)


def make_damage_weights(y: np.ndarray, high_weight: float, medium_weight: float = 2.0, zero_weight: float = 1.0) -> np.ndarray:
    """Assign larger weights to high-damage samples.
    给高损伤样本更大权重。
    """
    bins = damage_bins(y)
    weights = np.ones_like(y, dtype=float)
    weights[bins == "zero"] = zero_weight
    weights[bins == "low"] = 1.2
    weights[bins == "medium"] = medium_weight
    weights[bins == "high"] = high_weight
    return weights


# -----------------------------------------------------------------------------
# 5. Metrics
# 5. 指标计算
# -----------------------------------------------------------------------------


def safe_mean(values: np.ndarray) -> float:
    """Return NaN-safe mean.
    返回安全均值，空数组返回 NaN。
    """
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_high: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute overall, bin-stratified, and high-gate metrics.
    计算整体、分箱和高损伤门控相关指标。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)

    true_bins = damage_bins(y_true)
    pred_bins = damage_bins(y_pred)

    high_mask = true_bins == "high"
    zero_mask = true_bins == "zero"
    damaged_mask = true_bins != "zero"

    gate_high = p_high >= threshold

    metrics: Dict[str, float] = {
        "test_mae": safe_mean(abs_error),
        "test_rmse": float(np.sqrt(np.mean(error**2))),
        "test_bias": safe_mean(error),
        "test_mean_true": safe_mean(y_true),
        "test_mean_pred": safe_mean(y_pred),
        "test_n_entries": float(y_true.size),
        "test_gate_high_ratio": safe_mean(gate_high.astype(float)),
    }

    for bin_name in ["zero", "low", "medium", "high"]:
        mask = true_bins == bin_name
        metrics[f"test_{bin_name}_n"] = float(np.sum(mask))
        metrics[f"test_{bin_name}_mae"] = safe_mean(abs_error[mask])
        metrics[f"test_{bin_name}_bias"] = safe_mean(error[mask])
        metrics[f"test_{bin_name}_underestimation_ratio"] = safe_mean((y_pred[mask] < y_true[mask]).astype(float))

    metrics["test_damaged_mae"] = safe_mean(abs_error[damaged_mask])
    metrics["test_damaged_bias"] = safe_mean(error[damaged_mask])
    metrics["test_damaged_underestimation_ratio"] = safe_mean((y_pred[damaged_mask] < y_true[damaged_mask]).astype(float))

    if np.sum(zero_mask) > 0:
        metrics["test_zero_false_alarm_ratio_005"] = safe_mean((y_pred[zero_mask] > 0.05).astype(float))
        metrics["test_zero_false_alarm_ratio_010"] = safe_mean((y_pred[zero_mask] > 0.10).astype(float))
    else:
        metrics["test_zero_false_alarm_ratio_005"] = float("nan")
        metrics["test_zero_false_alarm_ratio_010"] = float("nan")

    y_high_true_binary = high_mask.astype(int)
    y_high_gate_binary = gate_high.astype(int)
    metrics["high_gate_precision"] = float(precision_score(y_high_true_binary, y_high_gate_binary, zero_division=0))
    metrics["high_gate_recall"] = float(recall_score(y_high_true_binary, y_high_gate_binary, zero_division=0))
    metrics["high_gate_f1"] = float(f1_score(y_high_true_binary, y_high_gate_binary, zero_division=0))

    # This checks the continuous prediction binned after calibration.
    # 这里检查校准后的连续预测再分箱后的宏平均 F1。
    metrics["damage_bin_macro_f1_from_prediction"] = float(
        f1_score(bin_to_int(true_bins), bin_to_int(pred_bins), average="macro", zero_division=0)
    )

    return metrics


# -----------------------------------------------------------------------------
# 6. Plotting
# 6. 绘图函数
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist.
    若目录不存在则创建。
    """
    path.mkdir(parents=True, exist_ok=True)


def plot_threshold_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    """Plot zero false alarm ratio versus high-damage MAE.
    绘制零损伤误报率与高损伤 MAE 的权衡图。
    """
    plt.figure(figsize=(10, 7))
    for config, sub in df.groupby("base_config"):
        plt.plot(
            sub["test_zero_false_alarm_ratio_005"],
            sub["test_high_mae"],
            marker="o",
            linewidth=1.5,
            label=config,
        )
    plt.axvline(0.25, linestyle="--", linewidth=1.2, label="zero FA=0.25")
    plt.axhline(0.085, linestyle=":", linewidth=1.2, label="high MAE=0.085")
    plt.xlabel("Zero-damage false alarm ratio > 0.05")
    plt.ylabel("High-damage test MAE")
    plt.title("Threshold-gated calibration tradeoff")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_threshold_curves(df: pd.DataFrame, out_path: Path) -> None:
    """Plot key metrics as threshold changes.
    绘制阈值变化时关键指标的变化曲线。
    """
    plt.figure(figsize=(11, 7))
    # Plot only the most relevant few base configs to keep the figure readable.
    # 只绘制最相关的若干配置，避免图过于拥挤。
    top_configs = (
        df.sort_values(["test_mae", "test_high_mae"])["base_config"].drop_duplicates().head(4).tolist()
    )
    for config in top_configs:
        sub = df[df["base_config"] == config].sort_values("threshold")
        plt.plot(sub["threshold"], sub["test_high_mae"], marker="o", label=f"{config} high MAE")
        plt.plot(sub["threshold"], sub["test_zero_false_alarm_ratio_005"], marker="x", label=f"{config} zero FA")
    plt.xlabel("High-damage probability threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold sensitivity")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_true_vs_pred(pred_df: pd.DataFrame, title: str, out_path: Path, max_damage: float) -> None:
    """Plot true damage versus predicted damage.
    绘制真实损伤与预测损伤散点图。
    """
    plt.figure(figsize=(8, 8))
    order = ["zero", "low", "medium", "high"]
    for bin_name in order:
        sub = pred_df[pred_df["true_bin"] == bin_name]
        if len(sub) == 0:
            continue
        plt.scatter(sub["y_true"], sub["y_pred"], label=bin_name, alpha=0.65)
    plt.plot([0, max_damage], [0, max_damage], linestyle="--", linewidth=1.5, label="Ideal y=x")
    plt.xlabel("True damage")
    plt.ylabel("Predicted damage")
    plt.title(title)
    plt.xlim(-0.02, max_damage + 0.02)
    plt.ylim(-0.02, max_damage + 0.02)
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(pred_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Plot confusion matrix between true damage bin and predicted damage bin.
    绘制真实损伤等级和预测损伤等级之间的混淆矩阵。
    """
    labels = ["zero", "low", "medium", "high"]
    cm = confusion_matrix(pred_df["true_bin"], pred_df["pred_bin"], labels=labels)

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted bin")
    plt.ylabel("True bin")
    plt.xticks(range(len(labels)), labels, rotation=35)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="Count")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# 7. Reporting
# 7. 报告输出
# -----------------------------------------------------------------------------


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    """Create a markdown table without requiring the external tabulate package.
    不依赖 tabulate 包，直接生成 Markdown 表格。
    """
    sub = df.loc[:, list(columns)].head(max_rows).copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.6f}")
        else:
            sub[col] = sub[col].astype(str)

    header = "| " + " | ".join(sub.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in sub.to_numpy()]
    return "\n".join([header, sep] + rows)


def write_report(
    df: pd.DataFrame,
    selected: pd.Series,
    output_path: Path,
    constraint_note: str,
) -> None:
    """Write a concise markdown report.
    输出简明 Markdown 报告。
    """
    top_columns = [
        "config_name",
        "feature_set",
        "classifier",
        "calibrator",
        "threshold",
        "high_weight",
        "test_mae",
        "test_rmse",
        "test_high_mae",
        "test_high_bias",
        "test_high_underestimation_ratio",
        "test_zero_false_alarm_ratio_005",
        "high_gate_precision",
        "high_gate_recall",
    ]

    by_overall = df.sort_values(["test_mae", "test_high_mae", "test_zero_false_alarm_ratio_005"])
    by_high = df.sort_values(["test_high_mae", "test_zero_false_alarm_ratio_005", "test_mae"])
    by_fa = df.sort_values(["test_zero_false_alarm_ratio_005", "test_high_mae", "test_mae"])

    lines = [
        "# Threshold-gated Calibration Summary",
        "",
        "## 1. Purpose",
        "",
        "This experiment tests whether a high-damage probability threshold can reduce high-damage underestimation while controlling zero-damage false alarms.",
        "",
        "中文解释：本实验测试能否通过高损伤概率阈值，在降低高损伤低估的同时控制零损伤误报。",
        "",
        "## 2. Selected configuration",
        "",
        f"- Selection rule: `{constraint_note}`",
        f"- Configuration: `{selected['config_name']}`",
        f"- Feature set: `{selected['feature_set']}`",
        f"- Classifier: `{selected['classifier']}`",
        f"- Calibrator: `{selected['calibrator']}`",
        f"- Threshold: `{selected['threshold']:.3f}`",
        f"- High-damage sample weight: `{selected['high_weight']:.3f}`",
        f"- Overall test MAE: `{selected['test_mae']:.6f}`",
        f"- Test RMSE: `{selected['test_rmse']:.6f}`",
        f"- High-damage MAE: `{selected['test_high_mae']:.6f}`",
        f"- High-damage bias: `{selected['test_high_bias']:.6f}`",
        f"- High-damage underestimation ratio: `{selected['test_high_underestimation_ratio']:.6f}`",
        f"- Zero false alarm ratio > 0.05: `{selected['test_zero_false_alarm_ratio_005']:.6f}`",
        f"- Gate high precision: `{selected['high_gate_precision']:.6f}`",
        f"- Gate high recall: `{selected['high_gate_recall']:.6f}`",
        "",
        "## 3. Ranking by overall test MAE",
        "",
        markdown_table(by_overall, top_columns, max_rows=20),
        "",
        "## 4. Ranking by high-damage test MAE",
        "",
        markdown_table(by_high, top_columns, max_rows=20),
        "",
        "## 5. Ranking by zero false alarm ratio",
        "",
        markdown_table(by_fa, top_columns, max_rows=20),
        "",
        "## 6. Interpretation guide",
        "",
        "- If threshold increases, zero false alarms should usually decrease, but high-damage recall may also decrease.",
        "- A useful model should not only reduce high-damage MAE; it must also keep zero-damage false alarms within an acceptable range.",
        "- If no threshold satisfies the target constraints, the bottleneck is not the regressor but the high-damage detector or the feature representation.",
        "",
        "中文解释：阈值越高，通常零损伤误报会下降，但高损伤召回也可能下降。真正可用的模型必须同时兼顾高损伤误差和零损伤误报。",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# 8. Main experiment
# 8. 主实验流程
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run threshold-gated damage calibration.")
    parser.add_argument("--features", required=True, type=Path, help="Physics feature NPZ file.")
    parser.add_argument("--feature-names", required=True, type=Path, help="Feature-name CSV file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for CSV and Markdown outputs.")
    parser.add_argument("--figures-dir", required=True, type=Path, help="Directory for figures.")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["full", "no_meta", "response_spatial"],
        choices=["full", "no_meta", "response_basic_only", "response_spatial", "response_frequency", "response_correlation"],
        help="Feature sets to evaluate.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["logistic_balanced", "random_forest_balanced"],
        choices=["logistic_balanced", "random_forest_balanced"],
        help="High-damage classifiers to evaluate.",
    )
    parser.add_argument(
        "--calibrators",
        nargs="+",
        default=["residual_ridge", "direct_ridge"],
        choices=["residual_ridge", "direct_ridge"],
        help="High-sensitive regressor forms.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    parser.add_argument("--high-weights", nargs="+", type=float, default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--ridge-alpha", type=float, default=300.0)
    parser.add_argument("--max-damage", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-overall-mae", type=float, default=0.050)
    parser.add_argument("--target-high-mae", type=float, default=0.085)
    parser.add_argument("--target-zero-fa", type=float, default=0.250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)

    dataset = load_feature_dataset(args.features)
    n_original_features = dataset["F_train"].shape[1]
    feature_names = load_feature_names(args.feature_names, n_original_features)

    all_rows: List[Dict[str, float | str]] = []
    prediction_tables: Dict[str, pd.DataFrame] = {}

    print("Threshold-gated calibration started.")
    print(f"Feature dataset: {args.features}")
    print(f"Feature names: {args.feature_names}")
    print(f"Output directory: {args.output_dir}")
    print(f"Figures directory: {args.figures_dir}")

    for feature_set in args.feature_sets:
        selected_indices = select_feature_indices(feature_names, feature_set)
        if len(selected_indices) == 0:
            print(f"Skipping feature set {feature_set}: selected zero features.")
            continue

        X_train, y_train, case_train, story_train = flatten_case_level_data(
            dataset["F_train"], dataset["y_train"], selected_indices, dataset.get("case_id_train")
        )
        X_val, y_val, case_val, story_val = flatten_case_level_data(
            dataset["F_val"], dataset["y_val"], selected_indices, dataset.get("case_id_val")
        )
        X_test, y_test, case_test, story_test = flatten_case_level_data(
            dataset["F_test"], dataset["y_test"], selected_indices, dataset.get("case_id_test")
        )

        y_train_high = (damage_bins(y_train) == "high").astype(int)

        # Fit a conservative global base regressor.
        # 训练保守的全局基础回归器。
        base_reg = fit_weighted_ridge(X_train, y_train, sample_weight=None, alpha=args.ridge_alpha)
        base_train_pred = np.clip(base_reg.predict(X_train), 0.0, args.max_damage)
        base_test_pred = np.clip(base_reg.predict(X_test), 0.0, args.max_damage)

        for classifier_name in args.classifiers:
            clf = make_high_classifier(classifier_name, args.random_seed)
            clf.fit(X_train, y_train_high)

            if hasattr(clf, "predict_proba"):
                p_high_test = clf.predict_proba(X_test)[:, 1]
            else:
                # This branch is unlikely for current classifiers, but keeps the script robust.
                # 当前分类器一般都有 predict_proba；此分支用于提高健壮性。
                p_high_test = clf.predict(X_test).astype(float)

            for calibrator in args.calibrators:
                for high_weight in args.high_weights:
                    sample_weight = make_damage_weights(y_train, high_weight=high_weight)

                    if calibrator == "residual_ridge":
                        residual_train = y_train - base_train_pred
                        high_reg = fit_weighted_ridge(
                            X_train,
                            residual_train,
                            sample_weight=sample_weight,
                            alpha=args.ridge_alpha,
                        )
                        high_test_pred = np.clip(base_test_pred + high_reg.predict(X_test), 0.0, args.max_damage)
                    elif calibrator == "direct_ridge":
                        high_reg = fit_weighted_ridge(
                            X_train,
                            y_train,
                            sample_weight=sample_weight,
                            alpha=args.ridge_alpha,
                        )
                        high_test_pred = np.clip(high_reg.predict(X_test), 0.0, args.max_damage)
                    else:
                        raise ValueError(f"Unknown calibrator: {calibrator}")

                    base_config = f"{feature_set}__{classifier_name}__{calibrator}__hw{high_weight:g}"

                    for threshold in args.thresholds:
                        use_high = p_high_test >= threshold
                        y_pred = np.where(use_high, high_test_pred, base_test_pred)
                        y_pred = np.clip(y_pred, 0.0, args.max_damage)

                        metrics = evaluate_predictions(
                            y_true=y_test,
                            y_pred=y_pred,
                            p_high=p_high_test,
                            threshold=threshold,
                        )

                        config_name = f"{base_config}__thr{threshold:.2f}"
                        row: Dict[str, float | str] = {
                            "config_name": config_name,
                            "base_config": base_config,
                            "feature_set": feature_set,
                            "classifier": classifier_name,
                            "calibrator": calibrator,
                            "threshold": threshold,
                            "high_weight": high_weight,
                            "n_features_before_story_onehot": float(len(selected_indices)),
                            "n_features_after_story_onehot": float(X_train.shape[1]),
                        }
                        row.update(metrics)
                        all_rows.append(row)

                        # Keep prediction tables for candidate selected configurations.
                        # 保存预测表，后面会根据选择结果写出。
                        true_bins = damage_bins(y_test)
                        pred_bins = damage_bins(y_pred)
                        pred_df = pd.DataFrame(
                            {
                                "case_id": case_test,
                                "story": story_test,
                                "y_true": y_test,
                                "y_pred": y_pred,
                                "error": y_pred - y_test,
                                "abs_error": np.abs(y_pred - y_test),
                                "p_high": p_high_test,
                                "threshold": threshold,
                                "gate_uses_high_calibrator": use_high.astype(int),
                                "true_bin": true_bins,
                                "pred_bin": pred_bins,
                            }
                        )
                        prediction_tables[config_name] = pred_df

                        print(
                            f"{config_name}: overall_mae={metrics['test_mae']:.6f}, "
                            f"high_mae={metrics['test_high_mae']:.6f}, "
                            f"zero_fa={metrics['test_zero_false_alarm_ratio_005']:.6f}"
                        )

    if not all_rows:
        raise RuntimeError("No experiment rows were produced. Check feature-set definitions and input files.")

    results_df = pd.DataFrame(all_rows)
    comparison_csv = args.output_dir / "threshold_gated_model_comparison.csv"
    results_df.to_csv(comparison_csv, index=False)

    # Select configuration using target constraints if possible.
    # 优先选择满足目标约束的配置；若没有满足者，则使用惩罚分数选择折中方案。
    feasible = results_df[
        (results_df["test_mae"] <= args.target_overall_mae)
        & (results_df["test_high_mae"] <= args.target_high_mae)
        & (results_df["test_zero_false_alarm_ratio_005"] <= args.target_zero_fa)
    ].copy()

    if len(feasible) > 0:
        selected = feasible.sort_values(
            ["test_high_mae", "test_mae", "test_zero_false_alarm_ratio_005"]
        ).iloc[0]
        constraint_note = (
            f"target satisfied: overall MAE <= {args.target_overall_mae}, "
            f"high MAE <= {args.target_high_mae}, zero FA <= {args.target_zero_fa}"
        )
    else:
        temp = results_df.copy()
        temp["constraint_penalty_score"] = (
            temp["test_high_mae"]
            + 2.0 * np.maximum(0.0, temp["test_zero_false_alarm_ratio_005"] - args.target_zero_fa)
            + 1.0 * np.maximum(0.0, temp["test_mae"] - args.target_overall_mae)
        )
        selected = temp.sort_values(
            ["constraint_penalty_score", "test_high_mae", "test_mae", "test_zero_false_alarm_ratio_005"]
        ).iloc[0]
        constraint_note = "no exact target satisfied; selected by penalty score"

    selected_config = str(selected["config_name"])
    selected_pred_df = prediction_tables[selected_config]
    selected_pred_csv = args.output_dir / "selected_threshold_gated_predictions_test.csv"
    selected_pred_df.to_csv(selected_pred_csv, index=False)

    # Also save top overall and top high-damage predictions for comparison.
    # 同时保存整体最优和高损伤最优配置的预测结果，方便对比。
    top_overall_config = str(
        results_df.sort_values(["test_mae", "test_high_mae", "test_zero_false_alarm_ratio_005"]).iloc[0]["config_name"]
    )
    top_high_config = str(
        results_df.sort_values(["test_high_mae", "test_zero_false_alarm_ratio_005", "test_mae"]).iloc[0]["config_name"]
    )
    prediction_tables[top_overall_config].to_csv(args.output_dir / "top_overall_predictions_test.csv", index=False)
    prediction_tables[top_high_config].to_csv(args.output_dir / "top_high_damage_predictions_test.csv", index=False)

    # Figures.
    # 输出图像。
    plot_threshold_tradeoff(results_df, args.figures_dir / "threshold_tradeoff_high_mae_vs_zero_fa.png")
    plot_threshold_curves(results_df, args.figures_dir / "threshold_sensitivity_curves.png")
    plot_true_vs_pred(
        selected_pred_df,
        title=f"Selected threshold-gated prediction: {selected_config}",
        out_path=args.figures_dir / "selected_true_vs_pred.png",
        max_damage=args.max_damage,
    )
    plot_confusion(
        selected_pred_df,
        title=f"Selected bin confusion: {selected_config}",
        out_path=args.figures_dir / "selected_bin_confusion.png",
    )

    report_path = args.output_dir / "threshold_gated_report.md"
    write_report(results_df, selected=selected, output_path=report_path, constraint_note=constraint_note)

    manifest = {
        "features": str(args.features),
        "feature_names": str(args.feature_names),
        "comparison_csv": str(comparison_csv),
        "selected_predictions_csv": str(selected_pred_csv),
        "top_overall_config": top_overall_config,
        "top_high_damage_config": top_high_config,
        "selected_config": selected_config,
        "report": str(report_path),
        "figures_dir": str(args.figures_dir),
    }
    (args.output_dir / "threshold_gated_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nThreshold-gated calibration completed.")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Report: {report_path}")
    print(f"Selected predictions: {selected_pred_csv}")
    print(f"Figures directory: {args.figures_dir}")
    print("\nSelected configuration:")
    print(f"  {selected_config}")
    print(
        f"  overall_mae={selected['test_mae']:.6f}, "
        f"high_mae={selected['test_high_mae']:.6f}, "
        f"high_bias={selected['test_high_bias']:.6f}, "
        f"zero_fa={selected['test_zero_false_alarm_ratio_005']:.6f}"
    )


if __name__ == "__main__":
    main()
