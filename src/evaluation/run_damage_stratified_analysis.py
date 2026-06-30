"""
Damage-stratified evaluation for physics-informed damage inference.

English:
This script re-runs selected lightweight regression baselines across repeated
random train/validation/test splits, then evaluates prediction error by damage
level. It is intended to diagnose whether a model performs well only on average
or also works for medium/high damage entries.

中文：
本脚本会在多个随机划分下重新训练指定的轻量回归模型，并按照损伤等级统计预测误差。
目的不是继续追求单一 overall MAE，而是诊断模型是否在中高损伤区间存在系统性低估或失效。

Typical command:
python -m src.evaluation.run_damage_stratified_analysis \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --configs full:ridge no_meta:ridge full:elasticnet response_basic_only:ridge \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --output-dir results/tables/damage_stratified/debug_plus_500 \
  --figures-dir results/figures/damage_stratified/debug_plus_500
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------------------
# Data loading
# 数据读取
# --------------------------------------------------------------------------------------


def read_feature_names(path: Path | None, n_features: int) -> list[str]:
    """
    English:
    Read feature names from a CSV file. The expected columns are flexible:
    either `feature_name`, or the last column in the CSV.

    中文：
    从 CSV 文件读取特征名。兼容两种格式：
    1）存在 feature_name 列；
    2）没有 feature_name 列时，默认使用最后一列作为特征名。
    """
    if path is None or not path.exists():
        return [f"feature_{i}" for i in range(n_features)]

    df = pd.read_csv(path)
    if "feature_name" in df.columns:
        names = df["feature_name"].astype(str).tolist()
    else:
        names = df.iloc[:, -1].astype(str).tolist()

    if len(names) != n_features:
        raise ValueError(
            f"Feature name count mismatch: CSV has {len(names)} names, "
            f"but feature array has {n_features} columns."
        )
    return names


def _first_existing_key(npz: np.lib.npyio.NpzFile, candidates: Iterable[str]) -> str | None:
    """
    English:
    Return the first key that exists in the NPZ file.

    中文：
    在 NPZ 文件中按候选列表查找第一个存在的键名。
    """
    keys = set(npz.files)
    for key in candidates:
        if key in keys:
            return key
    return None


def load_feature_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    English:
    Load a feature dataset from several possible NPZ layouts.

    Supported layouts:
    1. F_train/F_val/F_test + y_train/y_val/y_test
    2. X_train/X_val/X_test + y_train/y_val/y_test
    3. F + y
    4. X + y
    5. features + y_damage

    中文：
    读取特征数据集，并兼容几种常见 NPZ 保存格式。
    当前项目里最可能使用的是 F_train/F_val/F_test + y_train/y_val/y_test。
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    split_feature_prefix = None
    if {"F_train", "F_val", "F_test"}.issubset(keys):
        split_feature_prefix = "F"
    elif {"X_train", "X_val", "X_test"}.issubset(keys):
        split_feature_prefix = "X"

    if split_feature_prefix is not None and {"y_train", "y_val", "y_test"}.issubset(keys):
        F = np.concatenate(
            [
                np.asarray(data[f"{split_feature_prefix}_train"]),
                np.asarray(data[f"{split_feature_prefix}_val"]),
                np.asarray(data[f"{split_feature_prefix}_test"]),
            ],
            axis=0,
        )
        y = np.concatenate(
            [
                np.asarray(data["y_train"]),
                np.asarray(data["y_val"]),
                np.asarray(data["y_test"]),
            ],
            axis=0,
        )
        meta = {
            "source_layout": f"{split_feature_prefix}_train/{split_feature_prefix}_val/{split_feature_prefix}_test",
            "n_original_train": int(np.asarray(data[f"{split_feature_prefix}_train"]).shape[0]),
            "n_original_val": int(np.asarray(data[f"{split_feature_prefix}_val"]).shape[0]),
            "n_original_test": int(np.asarray(data[f"{split_feature_prefix}_test"]).shape[0]),
            "npz_keys": sorted(data.files),
        }
        return ensure_2d(F, name="F"), ensure_2d(y, name="y"), meta

    feature_key = _first_existing_key(data, ["F", "X", "features", "feature_matrix"])
    label_key = _first_existing_key(data, ["y", "Y", "y_damage", "damage", "labels"])

    if feature_key is None or label_key is None:
        raise KeyError(
            "Could not infer feature/label keys from NPZ. "
            f"Available keys: {sorted(data.files)}"
        )

    F = np.asarray(data[feature_key])
    y = np.asarray(data[label_key])
    meta = {
        "source_layout": f"{feature_key}/{label_key}",
        "npz_keys": sorted(data.files),
    }
    return ensure_2d(F, name="F"), ensure_2d(y, name="y"), meta


def ensure_2d(array: np.ndarray, name: str) -> np.ndarray:
    """
    English:
    Force array into 2D layout. Labels with shape (n,) become (n, 1).

    中文：
    保证数组为二维格式。若标签是一维 (n,)，则转换为 (n, 1)。
    """
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D after loading, got shape {array.shape}")
    return array.astype(float)


# --------------------------------------------------------------------------------------
# Feature subset selection
# 特征子集选择
# --------------------------------------------------------------------------------------


def select_feature_indices(feature_names: list[str], feature_set: str) -> list[int]:
    """
    English:
    Select feature indices by feature-set name.

    中文：
    根据 feature_set 名称选择特征列。
    注意：这里使用名称规则进行筛选，因此比硬编码特征编号更稳。
    """
    names = [name.lower() for name in feature_names]

    if feature_set == "full":
        return list(range(len(feature_names)))

    if feature_set in {"no_meta", "physics_no_meta_core"}:
        meta_exact_or_suffix = (
            "input_noise_level",
            "noise_level",
            "input_amplitude",
            "amplitude_g",
            "input_frequency",
            "frequency_hz",
            "ground_motion_id",
            "case_id",
        )
        selected = []
        for i, name in enumerate(names):
            is_meta = False
            for token in meta_exact_or_suffix:
                if name == token or name.endswith("_" + token) or name.startswith(token + "_"):
                    is_meta = True
                    break
            # Do not remove physics features such as dominant_frequency_to_input_ratio.
            # 不删除 dominant_frequency_to_input_ratio 这类物理比值特征。
            if not is_meta:
                selected.append(i)
        return selected

    if feature_set == "response_basic_only":
        tokens = ("mean", "std", "max_abs", "rms", "peak_to_peak", "crest_factor")
        return [i for i, name in enumerate(names) if any(token in name for token in tokens)]

    if feature_set == "response_spatial":
        tokens = ("spatial_fraction", "to_story", "story_", "floor")
        selected = [
            i
            for i, name in enumerate(names)
            if any(token in name for token in tokens)
            and not any(meta in name for meta in ("input_noise_level", "amplitude_g", "frequency_hz"))
        ]
        return selected

    if feature_set == "response_frequency":
        tokens = ("spectral", "dominant_frequency", "frequency", "band_energy", "centroid")
        return [i for i, name in enumerate(names) if any(token in name for token in tokens)]

    if feature_set == "response_correlation":
        tokens = ("correlation", "corr")
        return [i for i, name in enumerate(names) if any(token in name for token in tokens)]

    raise ValueError(
        f"Unknown feature_set: {feature_set}. "
        "Supported: full, no_meta, physics_no_meta_core, response_basic_only, "
        "response_spatial, response_frequency, response_correlation."
    )


# --------------------------------------------------------------------------------------
# Model training
# 模型训练
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """
    English:
    One trainable candidate model.

    中文：
    一个可训练的候选模型配置。
    """

    name: str
    estimator: object


def build_candidates(model_name: str, random_seed: int) -> list[Candidate]:
    """
    English:
    Build a small but sufficient hyperparameter grid for each estimator.

    中文：
    为每类模型构建一个小规模但足够使用的超参数候选集合。
    """
    if model_name == "ridge":
        alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
        return [
            Candidate(
                name=f"ridge_alpha_{alpha:g}",
                estimator=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=alpha)),
                    ]
                ),
            )
            for alpha in alphas
        ]

    if model_name == "elasticnet":
        candidates = []
        for alpha in [0.003, 0.01, 0.03, 0.05, 0.1]:
            for l1_ratio in [0.3, 0.5, 0.7, 0.9]:
                candidates.append(
                    Candidate(
                        name=f"elasticnet_alpha_{alpha:g}_l1_{l1_ratio:g}",
                        estimator=Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                (
                                    "model",
                                    MultiOutputRegressor(
                                        ElasticNet(
                                            alpha=alpha,
                                            l1_ratio=l1_ratio,
                                            max_iter=20000,
                                            random_state=random_seed,
                                        )
                                    ),
                                ),
                            ]
                        ),
                    )
                )
        return candidates

    if model_name == "random_forest":
        candidates = []
        for n_estimators in [200, 300]:
            for max_depth in [2, 3, 4, None]:
                for min_samples_leaf in [1, 2, 5]:
                    candidates.append(
                        Candidate(
                            name=(
                                f"random_forest_n_{n_estimators}_"
                                f"depth_{max_depth}_leaf_{min_samples_leaf}"
                            ),
                            estimator=RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                random_state=random_seed,
                                n_jobs=-1,
                            ),
                        )
                    )
        return candidates

    if model_name == "gradient_boosting":
        candidates = []
        for n_estimators in [60, 100]:
            for learning_rate in [0.03, 0.05]:
                for max_depth in [2, 3]:
                    candidates.append(
                        Candidate(
                            name=f"gbr_n_{n_estimators}_lr_{learning_rate:g}_depth_{max_depth}",
                            estimator=MultiOutputRegressor(
                                GradientBoostingRegressor(
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    random_state=random_seed,
                                )
                            ),
                        )
                    )
        return candidates

    raise ValueError(
        f"Unknown model: {model_name}. "
        "Supported: ridge, elasticnet, random_forest, gradient_boosting."
    )


def train_select_evaluate(
    F: np.ndarray,
    y: np.ndarray,
    selected_indices: list[int],
    model_name: str,
    seed: int,
    train_size: float,
    val_size: float,
    max_damage: float,
    clip_predictions: bool,
) -> dict:
    """
    English:
    Split the data, train candidate models, select the best one using validation MSE,
    and evaluate it on the test set.

    中文：
    划分训练/验证/测试集，训练候选模型，用验证集 MSE 选出最佳候选模型，
    最后在测试集上评价。
    """
    F_selected = F[:, selected_indices]

    idx_all = np.arange(F_selected.shape[0])
    idx_train_val, idx_test = train_test_split(
        idx_all,
        test_size=1.0 - train_size - val_size,
        random_state=seed,
        shuffle=True,
    )

    relative_val_size = val_size / (train_size + val_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val_size,
        random_state=seed + 10000,
        shuffle=True,
    )

    X_train, y_train = F_selected[idx_train], y[idx_train]
    X_val, y_val = F_selected[idx_val], y[idx_val]
    X_test, y_test = F_selected[idx_test], y[idx_test]

    best_candidate_name = None
    best_model = None
    best_val_mse = math.inf

    for candidate in build_candidates(model_name, random_seed=seed):
        model = candidate.estimator
        model.fit(X_train, y_train)
        pred_val = np.asarray(model.predict(X_val), dtype=float)
        if pred_val.ndim == 1:
            pred_val = pred_val[:, None]
        if clip_predictions:
            pred_val = np.clip(pred_val, 0.0, max_damage)
        val_mse = float(mean_squared_error(y_val, pred_val))

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_candidate_name = candidate.name
            best_model = model

    if best_model is None or best_candidate_name is None:
        raise RuntimeError("No model candidate was successfully trained.")

    pred_test_raw = np.asarray(best_model.predict(X_test), dtype=float)
    if pred_test_raw.ndim == 1:
        pred_test_raw = pred_test_raw[:, None]

    pred_test = pred_test_raw.copy()
    if clip_predictions:
        pred_test = np.clip(pred_test, 0.0, max_damage)

    return {
        "seed": seed,
        "best_candidate": best_candidate_name,
        "best_val_mse": best_val_mse,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "y_test": y_test,
        "pred_test": pred_test,
        "pred_test_raw": pred_test_raw,
    }


# --------------------------------------------------------------------------------------
# Metrics
# 指标计算
# --------------------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    English:
    Root mean squared error.

    中文：
    均方根误差。
    """
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    English:
    Mean absolute error.

    中文：
    平均绝对误差。
    """
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred - y_true)))


def make_damage_bin_mask(values: np.ndarray, bin_name: str, low_eps: float) -> np.ndarray:
    """
    English:
    Create boolean mask for one damage-level bin.

    中文：
    根据真实损伤值生成某一损伤等级的布尔掩码。
    """
    if bin_name == "zero":
        return values <= low_eps
    if bin_name == "low":
        return (values > low_eps) & (values <= 0.10)
    if bin_name == "medium":
        return (values > 0.10) & (values <= 0.20)
    if bin_name == "high":
        return values > 0.20
    if bin_name == "damaged":
        return values > low_eps
    if bin_name == "all":
        return np.ones_like(values, dtype=bool)
    raise ValueError(f"Unknown damage bin: {bin_name}")


def compute_bin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_name: str,
    low_eps: float,
) -> dict:
    """
    English:
    Compute metrics for one damage bin using all case-story entries.

    中文：
    针对某一个损伤等级，按所有 case-story 条目统计误差指标。
    """
    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    mask = make_damage_bin_mask(true_flat, bin_name=bin_name, low_eps=low_eps)

    true_bin = true_flat[mask]
    pred_bin = pred_flat[mask]

    if true_bin.size == 0:
        return {
            "bin": bin_name,
            "n_entries": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "underestimation_ratio": np.nan,
            "overestimation_ratio": np.nan,
            "mean_true": np.nan,
            "mean_pred": np.nan,
        }

    error = pred_bin - true_bin
    return {
        "bin": bin_name,
        "n_entries": int(true_bin.size),
        "mae": mae(true_bin, pred_bin),
        "rmse": rmse(true_bin, pred_bin),
        "bias": float(np.mean(error)),
        "underestimation_ratio": float(np.mean(error < 0.0)),
        "overestimation_ratio": float(np.mean(error > 0.0)),
        "mean_true": float(np.mean(true_bin)),
        "mean_pred": float(np.mean(pred_bin)),
    }


def compute_story_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """
    English:
    Compute error metrics for each story/output dimension.

    中文：
    按楼层/输出维度统计误差指标。
    """
    rows = []
    n_stories = y_true.shape[1]
    for j in range(n_stories):
        true_j = y_true[:, j]
        pred_j = y_pred[:, j]
        error = pred_j - true_j
        rows.append(
            {
                "story": j + 1,
                "n_entries": int(true_j.size),
                "mae": mae(true_j, pred_j),
                "rmse": rmse(true_j, pred_j),
                "bias": float(np.mean(error)),
                "underestimation_ratio": float(np.mean(error < 0.0)),
                "mean_true": float(np.mean(true_j)),
                "mean_pred": float(np.mean(pred_j)),
            }
        )
    return rows


def collect_high_damage_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
    feature_set: str,
    model: str,
    idx_test: np.ndarray,
    high_threshold: float,
) -> list[dict]:
    """
    English:
    Collect case-story entries whose true damage is high.

    中文：
    收集中高损伤条目的逐项预测结果，便于检查高损伤是否被低估。
    """
    rows = []
    n_cases, n_stories = y_true.shape
    for local_case_idx in range(n_cases):
        for story_idx in range(n_stories):
            true_value = float(y_true[local_case_idx, story_idx])
            if true_value <= high_threshold:
                continue
            pred_value = float(y_pred[local_case_idx, story_idx])
            rows.append(
                {
                    "seed": seed,
                    "feature_set": feature_set,
                    "model": model,
                    "global_case_index": int(idx_test[local_case_idx]),
                    "local_test_case_index": int(local_case_idx),
                    "story": int(story_idx + 1),
                    "true_damage": true_value,
                    "pred_damage": pred_value,
                    "error_pred_minus_true": pred_value - true_value,
                    "abs_error": abs(pred_value - true_value),
                    "is_underestimated": bool(pred_value < true_value),
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# Plotting
# 绘图
# --------------------------------------------------------------------------------------


def safe_label(row: pd.Series) -> str:
    """
    English:
    Compact label for figures.

    中文：
    为图形生成简洁标签。
    """
    return f"{row['feature_set']} + {row['model']}"


def plot_bin_mae(summary: pd.DataFrame, output_path: Path) -> None:
    """
    English:
    Plot mean MAE by damage bin for each model configuration.

    中文：
    绘制不同损伤等级下各模型配置的平均 MAE。
    """
    bins = ["zero", "low", "medium", "high", "damaged", "all"]
    pivot = summary.pivot_table(
        index=["feature_set", "model"], columns="bin", values="mae_mean", aggfunc="first"
    )
    pivot = pivot[[b for b in bins if b in pivot.columns]]

    ax = pivot.plot(kind="bar", figsize=(14, 6))
    ax.set_title("Damage-stratified MAE by configuration")
    ax.set_ylabel("Mean MAE across seeds")
    ax.set_xlabel("Configuration")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bin_bias(summary: pd.DataFrame, output_path: Path) -> None:
    """
    English:
    Plot mean prediction bias by damage bin.

    中文：
    绘制不同损伤等级下的平均预测偏差。
    """
    bins = ["zero", "low", "medium", "high", "damaged", "all"]
    pivot = summary.pivot_table(
        index=["feature_set", "model"], columns="bin", values="bias_mean", aggfunc="first"
    )
    pivot = pivot[[b for b in bins if b in pivot.columns]]

    ax = pivot.plot(kind="bar", figsize=(14, 6))
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Damage-stratified prediction bias")
    ax.set_ylabel("Mean bias: predicted - true")
    ax.set_xlabel("Configuration")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_underestimation(summary: pd.DataFrame, output_path: Path) -> None:
    """
    English:
    Plot underestimation ratio by damage bin.

    中文：
    绘制不同损伤等级下的低估比例。
    """
    bins = ["low", "medium", "high", "damaged", "all"]
    pivot = summary.pivot_table(
        index=["feature_set", "model"],
        columns="bin",
        values="underestimation_ratio_mean",
        aggfunc="first",
    )
    pivot = pivot[[b for b in bins if b in pivot.columns]]

    ax = pivot.plot(kind="bar", figsize=(14, 6))
    ax.set_title("Damage-stratified underestimation ratio")
    ax.set_ylabel("Underestimation ratio")
    ax.set_xlabel("Configuration")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_true_pred_for_best(
    all_predictions: list[dict],
    best_feature_set: str,
    best_model: str,
    output_path: Path,
    low_eps: float,
) -> None:
    """
    English:
    Plot true vs predicted damage entries for the best configuration.

    中文：
    绘制最佳配置的真实损伤 vs 预测损伤散点图，并按损伤区间标记。
    """
    true_values = []
    pred_values = []
    bin_names = []

    for item in all_predictions:
        if item["feature_set"] != best_feature_set or item["model"] != best_model:
            continue
        y_true = item["y_true"].reshape(-1)
        y_pred = item["y_pred"].reshape(-1)
        for true_value, pred_value in zip(y_true, y_pred):
            true_values.append(float(true_value))
            pred_values.append(float(pred_value))
            if true_value <= low_eps:
                bin_names.append("zero")
            elif true_value <= 0.10:
                bin_names.append("low")
            elif true_value <= 0.20:
                bin_names.append("medium")
            else:
                bin_names.append("high")

    if len(true_values) == 0:
        return

    df = pd.DataFrame({"true": true_values, "pred": pred_values, "bin": bin_names})

    fig, ax = plt.subplots(figsize=(7, 7))
    for bin_name, group in df.groupby("bin"):
        ax.scatter(group["true"], group["pred"], alpha=0.65, label=bin_name)

    max_axis = max(float(df["true"].max()), float(df["pred"].max()), 0.01)
    ax.plot([0.0, max_axis], [0.0, max_axis], linestyle="--", linewidth=1.5, label="Ideal y=x")
    ax.set_title(f"True vs predicted damage: {best_feature_set} + {best_model}")
    ax.set_xlabel("True damage")
    ax.set_ylabel("Predicted damage")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------------------
# Reporting
# 报告输出
# --------------------------------------------------------------------------------------


def summarize_bin_metrics(bin_df: pd.DataFrame) -> pd.DataFrame:
    """
    English:
    Aggregate per-seed bin metrics into mean/std summary.

    中文：
    将每个 seed 的分层指标汇总为均值和标准差。
    """
    metric_cols = [
        "n_entries",
        "mae",
        "rmse",
        "bias",
        "underestimation_ratio",
        "overestimation_ratio",
        "mean_true",
        "mean_pred",
    ]

    grouped = bin_df.groupby(["feature_set", "model", "bin"], as_index=False)
    rows = []
    for keys, group in grouped:
        feature_set, model, bin_name = keys
        row = {
            "feature_set": feature_set,
            "model": model,
            "bin": bin_name,
            "n_runs": int(group["seed"].nunique()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)
    bin_order = {"zero": 0, "low": 1, "medium": 2, "high": 3, "damaged": 4, "all": 5}
    summary["bin_order"] = summary["bin"].map(bin_order)
    summary = summary.sort_values(["feature_set", "model", "bin_order"]).drop(columns=["bin_order"])
    return summary


def summarize_story_metrics(story_df: pd.DataFrame) -> pd.DataFrame:
    """
    English:
    Aggregate per-seed story-level metrics.

    中文：
    汇总每个楼层在重复 seed 下的平均误差。
    """
    grouped = story_df.groupby(["feature_set", "model", "story"], as_index=False)
    rows = []
    for keys, group in grouped:
        feature_set, model, story = keys
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "story": int(story),
                "n_runs": int(group["seed"].nunique()),
                "mae_mean": float(group["mae"].mean()),
                "mae_std": float(group["mae"].std(ddof=1)) if len(group) > 1 else 0.0,
                "rmse_mean": float(group["rmse"].mean()),
                "bias_mean": float(group["bias"].mean()),
                "underestimation_ratio_mean": float(group["underestimation_ratio"].mean()),
                "mean_true": float(group["mean_true"].mean()),
                "mean_pred": float(group["mean_pred"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_set", "model", "story"])


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    """
    English:
    Convert a DataFrame to a markdown table without requiring the optional tabulate package.

    中文：
    不依赖 pandas.to_markdown/tabulate，手动生成 Markdown 表格。
    """
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"

    display = df[columns].copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, sep] + rows)


def write_report(
    output_path: Path,
    config_summary: pd.DataFrame,
    bin_summary: pd.DataFrame,
    story_summary: pd.DataFrame,
    high_damage_df: pd.DataFrame,
    metadata: dict,
) -> None:
    """
    English:
    Write a paper-facing markdown report.

    中文：
    输出面向论文结果分析的 Markdown 报告。
    """
    ranked = (
        config_summary.sort_values(["overall_mae_mean", "overall_rmse_mean"])
        .reset_index(drop=True)
        .copy()
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    best = ranked.iloc[0]

    high_bin = bin_summary[bin_summary["bin"] == "high"].copy()
    high_bin = high_bin.sort_values("mae_mean")

    lines = []
    lines.append("# Damage-stratified Evaluation Summary")
    lines.append("")
    lines.append("## 1. Dataset")
    lines.append("")
    lines.append(f"- Source layout: `{metadata.get('source_layout', 'unknown')}`")
    lines.append(f"- Number of samples: `{metadata.get('n_samples', 'unknown')}`")
    lines.append(f"- Number of features: `{metadata.get('n_features', 'unknown')}`")
    lines.append(f"- Number of stories / outputs: `{metadata.get('n_outputs', 'unknown')}`")
    lines.append(f"- Number of seeds: `{metadata.get('n_seeds', 'unknown')}`")
    lines.append("")
    lines.append("## 2. Overall best configuration")
    lines.append("")
    lines.append(f"- Best configuration: `{best['feature_set']} + {best['model']}`")
    lines.append(f"- Overall MAE mean: `{best['overall_mae_mean']:.6f}`")
    lines.append(f"- Overall MAE std: `{best['overall_mae_std']:.6f}`")
    lines.append(f"- Overall RMSE mean: `{best['overall_rmse_mean']:.6f}`")
    lines.append(f"- Mean bias: `{best['overall_bias_mean']:.6f}`")
    lines.append("")
    lines.append("## 3. Overall ranking")
    lines.append("")
    lines.append(
        markdown_table(
            ranked,
            [
                "rank",
                "feature_set",
                "model",
                "n_runs",
                "overall_mae_mean",
                "overall_mae_std",
                "overall_rmse_mean",
                "overall_bias_mean",
                "high_damage_mae_mean",
                "high_damage_underestimation_ratio_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## 4. Damage-bin metrics")
    lines.append("")
    lines.append(
        markdown_table(
            bin_summary.sort_values(["feature_set", "model", "bin"]),
            [
                "feature_set",
                "model",
                "bin",
                "n_entries_mean",
                "mae_mean",
                "rmse_mean",
                "bias_mean",
                "underestimation_ratio_mean",
                "mean_true_mean",
                "mean_pred_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## 5. High-damage ranking")
    lines.append("")
    lines.append(
        markdown_table(
            high_bin,
            [
                "feature_set",
                "model",
                "mae_mean",
                "rmse_mean",
                "bias_mean",
                "underestimation_ratio_mean",
                "mean_true_mean",
                "mean_pred_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## 6. Story-level summary")
    lines.append("")
    lines.append(
        markdown_table(
            story_summary,
            [
                "feature_set",
                "model",
                "story",
                "mae_mean",
                "rmse_mean",
                "bias_mean",
                "underestimation_ratio_mean",
                "mean_true",
                "mean_pred",
            ],
            max_rows=80,
        )
    )
    lines.append("")
    lines.append("## 7. Preliminary interpretation")
    lines.append("")
    lines.append(
        "- If the high-damage MAE is much larger than the overall MAE, the model is not failing on average-case fitting; it is failing mainly on severe-damage estimation."
    )
    lines.append(
        "- If the high-damage bias is negative, the model has systematic high-damage underestimation, which is important for structural safety interpretation."
    )
    lines.append(
        "- If `full + ridge` remains best overall but not best in the high-damage bin, the next paper argument should distinguish average accuracy from safety-critical damage sensitivity."
    )
    lines.append(
        "- If all configurations underestimate high damage, the next methodological step should be weighted training, damage-aware loss, or data rebalancing."
    )
    lines.append("")
    lines.append("中文解释：")
    lines.append("")
    lines.append("- 如果高损伤 MAE 明显大于 overall MAE，问题不是平均拟合失败，而是严重损伤识别不足。")
    lines.append("- 如果高损伤 bias 为负，说明模型存在高损伤低估风险，这对结构安全判断非常关键。")
    lines.append("- 如果 full + ridge 仍是整体最优，但高损伤区间不是最优，论文中需要区分平均精度与安全关键区间敏感性。")
    lines.append("- 如果所有模型都低估高损伤，下一步应考虑加权训练、损伤感知损失函数或数据重平衡。")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Main
# 主程序
# --------------------------------------------------------------------------------------


def parse_configs(values: list[str]) -> list[tuple[str, str]]:
    """
    English:
    Parse config strings such as full:ridge.

    中文：
    解析配置字符串，例如 full:ridge。
    """
    configs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Config must use feature_set:model format, got: {value}")
        feature_set, model = value.split(":", 1)
        configs.append((feature_set.strip(), model.strip()))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Damage-stratified repeated-split evaluation."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, default=None)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "full:ridge",
            "no_meta:ridge",
            "full:elasticnet",
            "response_basic_only:ridge",
        ],
        help="Model configs in feature_set:model format.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--max-damage", type=float, default=0.5)
    parser.add_argument("--high-threshold", type=float, default=0.20)
    parser.add_argument("--zero-eps", type=float, default=1e-12)
    parser.add_argument("--clip-predictions", action="store_true", default=True)
    parser.add_argument("--no-clip-predictions", action="store_false", dest="clip_predictions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    F, y, dataset_meta = load_feature_dataset(args.features)
    feature_names = read_feature_names(args.feature_names, n_features=F.shape[1])
    configs = parse_configs(args.configs)

    print("Damage-stratified evaluation started.")
    print(f"Feature file: {args.features}")
    print(f"Feature shape: {F.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Configs: {configs}")
    print(f"Seeds: {args.seeds}")

    run_rows = []
    bin_rows = []
    story_rows = []
    high_damage_rows = []
    all_predictions = []

    for feature_set, model_name in configs:
        selected_indices = select_feature_indices(feature_names, feature_set)
        if not selected_indices:
            raise ValueError(f"Feature set {feature_set} selected zero features.")

        print(f"\nRunning config: {feature_set} + {model_name}")
        print(f"Selected features: {len(selected_indices)}")

        for seed in args.seeds:
            result = train_select_evaluate(
                F=F,
                y=y,
                selected_indices=selected_indices,
                model_name=model_name,
                seed=seed,
                train_size=args.train_size,
                val_size=args.val_size,
                max_damage=args.max_damage,
                clip_predictions=args.clip_predictions,
            )

            y_test = result["y_test"]
            pred_test = result["pred_test"]
            pred_test_raw = result["pred_test_raw"]

            overall_error = pred_test - y_test
            raw_negative_ratio = float(np.mean(pred_test_raw < 0.0))
            clipped_negative_ratio = float(np.mean(pred_test < 0.0))

            damaged_mask = y_test.reshape(-1) > args.zero_eps
            high_mask = y_test.reshape(-1) > args.high_threshold

            run_row = {
                "seed": seed,
                "feature_set": feature_set,
                "model": model_name,
                "best_candidate": result["best_candidate"],
                "n_total_samples": int(F.shape[0]),
                "n_train": int(len(result["idx_train"])),
                "n_val": int(len(result["idx_val"])),
                "n_test": int(len(result["idx_test"])),
                "n_features": int(len(selected_indices)),
                "best_val_mse": float(result["best_val_mse"]),
                "overall_mse": float(np.mean(overall_error**2)),
                "overall_mae": mae(y_test, pred_test),
                "overall_rmse": rmse(y_test, pred_test),
                "overall_bias": float(np.mean(overall_error)),
                "negative_prediction_ratio_raw": raw_negative_ratio,
                "negative_prediction_ratio": clipped_negative_ratio,
                "mean_true_damage": float(np.mean(y_test)),
                "mean_pred_damage": float(np.mean(pred_test)),
                "mae_damaged": mae(y_test.reshape(-1)[damaged_mask], pred_test.reshape(-1)[damaged_mask]),
                "mae_high_damage": mae(y_test.reshape(-1)[high_mask], pred_test.reshape(-1)[high_mask]),
                "selected_feature_indices": " ".join(str(i) for i in selected_indices),
            }
            run_rows.append(run_row)

            for bin_name in ["zero", "low", "medium", "high", "damaged", "all"]:
                row = compute_bin_metrics(
                    y_true=y_test,
                    y_pred=pred_test,
                    bin_name=bin_name,
                    low_eps=args.zero_eps,
                )
                row.update(
                    {
                        "seed": seed,
                        "feature_set": feature_set,
                        "model": model_name,
                        "best_candidate": result["best_candidate"],
                        "n_features": int(len(selected_indices)),
                    }
                )
                bin_rows.append(row)

            for row in compute_story_metrics(y_test, pred_test):
                row.update(
                    {
                        "seed": seed,
                        "feature_set": feature_set,
                        "model": model_name,
                        "best_candidate": result["best_candidate"],
                        "n_features": int(len(selected_indices)),
                    }
                )
                story_rows.append(row)

            high_damage_rows.extend(
                collect_high_damage_cases(
                    y_true=y_test,
                    y_pred=pred_test,
                    seed=seed,
                    feature_set=feature_set,
                    model=model_name,
                    idx_test=result["idx_test"],
                    high_threshold=args.high_threshold,
                )
            )

            all_predictions.append(
                {
                    "seed": seed,
                    "feature_set": feature_set,
                    "model": model_name,
                    "y_true": y_test,
                    "y_pred": pred_test,
                }
            )

            print(
                f"  seed={seed:02d} | candidate={result['best_candidate']} | "
                f"MAE={run_row['overall_mae']:.6f} | "
                f"high_MAE={run_row['mae_high_damage']:.6f} | "
                f"bias={run_row['overall_bias']:.6f}"
            )

    run_df = pd.DataFrame(run_rows)
    bin_df = pd.DataFrame(bin_rows)
    story_df = pd.DataFrame(story_rows)
    high_damage_df = pd.DataFrame(high_damage_rows)

    config_summary = (
        run_df.groupby(["feature_set", "model"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            n_features=("n_features", "first"),
            overall_mae_mean=("overall_mae", "mean"),
            overall_mae_std=("overall_mae", "std"),
            overall_rmse_mean=("overall_rmse", "mean"),
            overall_rmse_std=("overall_rmse", "std"),
            overall_bias_mean=("overall_bias", "mean"),
            overall_bias_std=("overall_bias", "std"),
            mean_true_damage=("mean_true_damage", "mean"),
            mean_pred_damage=("mean_pred_damage", "mean"),
            mae_damaged_mean=("mae_damaged", "mean"),
            high_damage_mae_mean=("mae_high_damage", "mean"),
            high_damage_mae_std=("mae_high_damage", "std"),
        )
        .sort_values(["overall_mae_mean", "overall_rmse_mean"])
    )

    high_bin_tmp = bin_df[bin_df["bin"] == "high"]
    high_under = (
        high_bin_tmp.groupby(["feature_set", "model"], as_index=False)
        .agg(high_damage_underestimation_ratio_mean=("underestimation_ratio", "mean"))
    )
    config_summary = config_summary.merge(high_under, on=["feature_set", "model"], how="left")

    bin_summary = summarize_bin_metrics(bin_df)
    story_summary = summarize_story_metrics(story_df)

    run_df.to_csv(args.output_dir / "damage_stratified_runs.csv", index=False)
    config_summary.to_csv(args.output_dir / "damage_stratified_config_summary.csv", index=False)
    bin_df.to_csv(args.output_dir / "damage_stratified_metrics_by_seed.csv", index=False)
    bin_summary.to_csv(args.output_dir / "damage_stratified_bin_summary.csv", index=False)
    story_df.to_csv(args.output_dir / "story_level_metrics_by_seed.csv", index=False)
    story_summary.to_csv(args.output_dir / "story_level_metrics_summary.csv", index=False)

    if not high_damage_df.empty:
        high_damage_df = high_damage_df.sort_values(
            ["abs_error", "true_damage"], ascending=[False, False]
        )
    high_damage_df.to_csv(args.output_dir / "high_damage_error_cases.csv", index=False)

    best_row = config_summary.iloc[0]
    metadata = {
        **dataset_meta,
        "feature_file": str(args.features),
        "feature_name_file": str(args.feature_names) if args.feature_names else None,
        "n_samples": int(F.shape[0]),
        "n_features": int(F.shape[1]),
        "n_outputs": int(y.shape[1]),
        "n_seeds": int(len(args.seeds)),
        "configs": args.configs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": 1.0 - args.train_size - args.val_size,
        "high_threshold": args.high_threshold,
        "zero_eps": args.zero_eps,
        "clip_predictions": args.clip_predictions,
    }
    (args.output_dir / "damage_stratified_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(
        output_path=args.output_dir / "damage_stratified_report.md",
        config_summary=config_summary,
        bin_summary=bin_summary,
        story_summary=story_summary,
        high_damage_df=high_damage_df,
        metadata=metadata,
    )

    plot_bin_mae(bin_summary, args.figures_dir / "damage_stratified_mae_by_bin.png")
    plot_bin_bias(bin_summary, args.figures_dir / "damage_stratified_bias_by_bin.png")
    plot_underestimation(
        bin_summary, args.figures_dir / "damage_stratified_underestimation_ratio.png"
    )
    plot_true_pred_for_best(
        all_predictions=all_predictions,
        best_feature_set=str(best_row["feature_set"]),
        best_model=str(best_row["model"]),
        output_path=args.figures_dir / "best_config_true_vs_pred_by_damage_bin.png",
        low_eps=args.zero_eps,
    )

    print("\nDamage-stratified evaluation completed.")
    print(f"Output tables: {args.output_dir}")
    print(f"Output figures: {args.figures_dir}")
    print(f"Best config: {best_row['feature_set']} + {best_row['model']}")
    print(f"Best overall MAE mean: {best_row['overall_mae_mean']:.6f}")
    print(f"Best high-damage MAE mean: {best_row['high_damage_mae_mean']:.6f}")
    print(f"Report: {args.output_dir / 'damage_stratified_report.md'}")


if __name__ == "__main__":
    main()
