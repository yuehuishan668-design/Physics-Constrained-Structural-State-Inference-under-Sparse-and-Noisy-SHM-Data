#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story-local feature alignment diagnosis.

English:
    This script reconstructs story-level samples using story-local features instead
    of simply appending story one-hot identifiers to a full case-level feature vector.

中文：
    本脚本用于重构 story-level 样本，将每一层的预测样本尽量改为由“本层局部特征、
    相邻层特征、地震输入/地面运动特征、局部空间比例特征”等组成，而不是只用
    全局 case-level 特征加 story one-hot。

Purpose:
    Diagnose whether story-local feature-target alignment improves:
        1. story-level stiffness degradation regression;
        2. story-level zero-vs-damaged classification;
        3. high-damage underestimation.

用途：
    诊断 story-local 特征重构是否改善：
        1. 楼层刚度退化率回归；
        2. 楼层 zero-vs-damaged 分类；
        3. 高损伤低估问题。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Basic utilities
# 基础工具
# -----------------------------


def ensure_dir(path: Path) -> None:
    """English: Create directory if missing. 中文：如果目录不存在则创建。"""
    path.mkdir(parents=True, exist_ok=True)


def manual_markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """English: Convert dataframe to markdown without tabulate. 中文：不依赖 tabulate 输出 markdown 表。"""
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_Empty table_"

    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_float_dtype(safe[col]):
            safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else str(x))

    header = "| " + " | ".join(safe.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(safe.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in safe.astype(str).values.tolist()]
    return "\n".join([header, sep] + rows)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """English: Load npz as dict. 中文：读取 npz 为字典。"""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def pick_first_existing_key(data: Dict[str, np.ndarray], candidates: Sequence[str]) -> Optional[str]:
    """English: Find first existing key. 中文：找到第一个存在的键名。"""
    for key in candidates:
        if key in data:
            return key
    return None


def load_feature_names(path: Optional[Path], n_features: int) -> List[str]:
    """English: Read feature names from CSV. 中文：从 CSV 读取特征名。"""
    if path is None or not path.exists():
        return [f"feature_{i}" for i in range(n_features)]

    df = pd.read_csv(path)
    if "feature_name" in df.columns:
        names = df["feature_name"].astype(str).tolist()
    elif "name" in df.columns:
        names = df["name"].astype(str).tolist()
    elif len(df.columns) == 1:
        names = df.iloc[:, 0].astype(str).tolist()
    else:
        possible_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
        names = df[possible_cols[0]].astype(str).tolist() if possible_cols else []

    if len(names) < n_features:
        names += [f"feature_{i}" for i in range(len(names), n_features)]
    elif len(names) > n_features:
        names = names[:n_features]

    return names


def resolve_case_features(features_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """English: Resolve case-level features. 中文：解析 case-level 特征矩阵。"""
    split_to_candidates = {
        "train": ["F_train", "features_train", "X_features_train", "X_train_features", "X_train"],
        "val": ["F_val", "features_val", "X_features_val", "X_val_features", "X_val"],
        "test": ["F_test", "features_test", "X_features_test", "X_test_features", "X_test"],
    }

    out: Dict[str, np.ndarray] = {}
    for split, candidates in split_to_candidates.items():
        key = pick_first_existing_key(features_npz, candidates)
        if key is None:
            raise KeyError(f"Cannot find {split} feature matrix. Available keys: {list(features_npz.keys())}")
        X = np.asarray(features_npz[key], dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{key} should be 2D, got shape {X.shape}")
        out[split] = X
    return out


def resolve_story_damage(split_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """English: Resolve story damage labels. 中文：解析各楼层损伤标签。"""
    split_to_candidates = {
        "train": ["y_train", "damage_train", "Y_train", "target_train"],
        "val": ["y_val", "damage_val", "Y_val", "target_val"],
        "test": ["y_test", "damage_test", "Y_test", "target_test"],
    }
    out: Dict[str, np.ndarray] = {}
    for split, candidates in split_to_candidates.items():
        key = pick_first_existing_key(split_npz, candidates)
        if key is None:
            raise KeyError(f"Cannot find {split} damage label array. Available keys: {list(split_npz.keys())}")
        y = np.asarray(split_npz[key], dtype=float)
        if y.ndim == 1:
            if y.size % 4 != 0:
                raise ValueError(f"{key} is 1D and cannot be grouped into 4 stories: shape={y.shape}")
            y = y.reshape(-1, 4)
        elif y.ndim == 2:
            if y.shape[1] == 1:
                raise ValueError(
                    f"{key} has only one target column. This story-local script requires story-level labels."
                )
        else:
            raise ValueError(f"{key} should be 1D or 2D, got shape {y.shape}")
        out[split] = y
    return out


# -----------------------------
# Story-local feature construction
# story-local 特征构造
# -----------------------------


def story_number_from_name(name: str) -> Optional[int]:
    """
    English: Extract the first story number from a feature name like story_2_rms.
    中文：从 story_2_rms 这类特征名中提取楼层编号。
    """
    match = re.search(r"story[_\- ]?([1-4])", name)
    if match:
        return int(match.group(1))
    return None


def is_global_input_or_ground_feature(name: str) -> bool:
    """
    English:
        Detect global excitation / ground-motion features that are useful for all stories.
    中文：
        识别对所有楼层都可用的输入或地面运动特征。
    """
    lower = name.lower()
    global_tokens = [
        "ground",
        "input",
        "amplitude",
        "frequency_hz",
        "noise",
        "dt",
        "duration",
    ]
    return any(token in lower for token in global_tokens)


def is_story_pair_feature_for_target(name: str, story_id: int) -> bool:
    """
    English:
        Check whether a feature involving two stories is relevant to the target story.
        Examples:
            story_3_to_story_2_rms_ratio is relevant to story 2 and story 3.

    中文：
        判断涉及两个楼层的比值/相关性特征是否与目标楼层相关。
        例如 story_3_to_story_2_rms_ratio 同时与第 2 层和第 3 层相关。
    """
    nums = [int(x) for x in re.findall(r"story[_\- ]?([1-4])", name)]
    return story_id in nums


def select_story_local_columns(
    feature_names: List[str],
    story_id: int,
    include_neighbor: bool,
    include_global: bool,
    include_story_one_hot: bool,
) -> List[int]:
    """
    English:
        Select columns for a target story.
    中文：
        为目标楼层选择局部特征列。
    """
    selected: List[int] = []

    for idx, name in enumerate(feature_names):
        lower = name.lower()
        s = story_number_from_name(lower)

        keep = False

        # English: local story feature.
        # 中文：目标楼层本层特征。
        if s == story_id:
            keep = True

        # English: adjacent story features.
        # 中文：相邻楼层特征。
        if include_neighbor and s is not None and abs(s - story_id) == 1:
            keep = True

        # English: pairwise ratio/correlation involving target story.
        # 中文：包含目标楼层的楼层间比值/相关性特征。
        if is_story_pair_feature_for_target(lower, story_id):
            keep = True

        # English: ground/input metadata features usable by all stories.
        # 中文：输入/地面运动等全局特征，每一层都可以使用。
        if include_global and is_global_input_or_ground_feature(lower):
            keep = True

        # English: optionally include existing story one-hot flags.
        # 中文：可选加入已有 story one-hot 标记。
        if include_story_one_hot and lower in {f"story_is_{story_id}", f"story_{story_id}_is_target"}:
            keep = True

        if keep:
            selected.append(idx)

    return sorted(set(selected))


def build_story_dataset(
    X_case: np.ndarray,
    y_story: np.ndarray,
    feature_names: List[str],
    include_neighbor: bool,
    include_global: bool,
    include_story_one_hot: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    English:
        Convert case-level feature rows and story-level labels into story-local samples.
    中文：
        将 case-level 特征和 story-level 标签转换成 story-local 样本。
    """
    if X_case.shape[0] != y_story.shape[0]:
        # English: if features were already story-expanded, collapse first.
        # 中文：如果特征已经按 story 展开，则先按 4 行还原到 case。
        if X_case.shape[0] == y_story.shape[0] * 4:
            X_case = X_case.reshape(y_story.shape[0], 4, X_case.shape[1]).mean(axis=1)
        else:
            raise ValueError(
                f"Cannot align case features and story labels: X={X_case.shape}, y={y_story.shape}"
            )

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    story_ids: List[int] = []

    # English:
    #   Use a fixed union of possible feature positions by padding missing story-specific
    #   columns with zeros. This keeps one common feature dimension across stories.
    # 中文：
    #   不同楼层可选到的局部列不同，因此这里构造一个“局部槽位”表。
    #   每个样本保留所选原始特征值，并加 story one-hot 与 story_id 数值。
    local_column_sets = {
        s: select_story_local_columns(
            feature_names=feature_names,
            story_id=s,
            include_neighbor=include_neighbor,
            include_global=include_global,
            include_story_one_hot=include_story_one_hot,
        )
        for s in range(1, 5)
    }

    union_cols = sorted(set(col for cols in local_column_sets.values() for col in cols))
    if not union_cols:
        raise ValueError("No story-local feature columns were selected. Check feature names CSV.")

    constructed_feature_names = [f"selected_original::{feature_names[i]}" for i in union_cols]
    constructed_feature_names += [f"target_story_is_{s}" for s in range(1, 5)]
    constructed_feature_names += ["target_story_id_scaled"]

    col_to_pos = {col: pos for pos, col in enumerate(union_cols)}
    n_constructed = len(union_cols) + 4 + 1

    for case_i in range(X_case.shape[0]):
        for story_id in range(1, 5):
            vec = np.zeros(n_constructed, dtype=float)

            cols = local_column_sets[story_id]
            for col in cols:
                pos = col_to_pos[col]
                vec[pos] = X_case[case_i, col]

            # English: add target story one-hot to keep boundary information.
            # 中文：加入目标楼层 one-hot，保留边界/楼层位置差异。
            vec[len(union_cols) + (story_id - 1)] = 1.0

            # English: scaled story index from 0 to 1.
            # 中文：归一化楼层编号，取值 0 到 1。
            vec[-1] = (story_id - 1) / 3.0

            X_rows.append(vec)
            y_rows.append(float(y_story[case_i, story_id - 1]))
            story_ids.append(story_id)

    return np.vstack(X_rows), np.asarray(y_rows), np.asarray(story_ids), constructed_feature_names


# -----------------------------
# Model evaluation
# 模型评估
# -----------------------------


def damage_bin(y: np.ndarray) -> np.ndarray:
    """
    English:
        Convert continuous damage ratio to bins.
    中文：
        将连续损伤率划分为 zero / low / medium / high。
    """
    bins = np.full(y.shape, "zero", dtype=object)
    bins[(y > 1e-12) & (y < 0.10)] = "low"
    bins[(y >= 0.10) & (y < 0.20)] = "medium"
    bins[y >= 0.20] = "high"
    return bins


def regression_metrics(y_true: np.ndarray, pred: np.ndarray, model_name: str, split: str) -> Dict[str, float]:
    """English: Regression metrics. 中文：回归指标。"""
    pred = np.clip(pred, 0.0, 0.5)
    bins = damage_bin(y_true)
    high_mask = bins == "high"
    damaged_mask = y_true > 1e-12
    zero_mask = y_true <= 1e-12

    row = {
        "model": model_name,
        "split": split,
        "n": int(len(y_true)),
        "test_mae": float(mean_absolute_error(y_true, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mean_true_damage": float(np.mean(y_true)),
        "mean_pred_damage": float(np.mean(pred)),
        "overall_bias_pred_minus_true": float(np.mean(pred - y_true)),
        "zero_mean_pred": float(np.mean(pred[zero_mask])) if zero_mask.any() else float("nan"),
        "damaged_mae": float(mean_absolute_error(y_true[damaged_mask], pred[damaged_mask])) if damaged_mask.any() else float("nan"),
        "high_mae": float(mean_absolute_error(y_true[high_mask], pred[high_mask])) if high_mask.any() else float("nan"),
        "high_bias_pred_minus_true": float(np.mean(pred[high_mask] - y_true[high_mask])) if high_mask.any() else float("nan"),
        "high_underestimation_ratio": float(np.mean(pred[high_mask] < y_true[high_mask])) if high_mask.any() else float("nan"),
    }
    return row


def classifier_metrics(y_true_damage: np.ndarray, score: np.ndarray, model_name: str, split: str, threshold: float) -> Dict[str, float]:
    """English: Binary zero-vs-damaged metrics. 中文：zero-vs-damaged 二分类指标。"""
    pred = (score >= threshold).astype(int)
    y_bin = (y_true_damage > 1e-12).astype(int)

    zero_mask = y_bin == 0
    damaged_mask = y_bin == 1
    high_mask = y_true_damage >= 0.20

    if len(np.unique(y_bin)) >= 2:
        roc_auc = float(roc_auc_score(y_bin, score))
        pr_auc = float(average_precision_score(y_bin, score))
        bal_acc = float(balanced_accuracy_score(y_bin, pred))
    else:
        roc_auc = pr_auc = bal_acc = float("nan")

    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "n": int(len(y_bin)),
        "zero_n": int(zero_mask.sum()),
        "damaged_n": int(damaged_mask.sum()),
        "high_n": int(high_mask.sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": bal_acc,
        "precision": float(precision_score(y_bin, pred, zero_division=0)),
        "recall": float(recall_score(y_bin, pred, zero_division=0)),
        "f1": float(f1_score(y_bin, pred, zero_division=0)),
        "zero_false_alarm_ratio": float(pred[zero_mask].mean()) if zero_mask.any() else float("nan"),
        "damaged_miss_ratio": float((1 - pred[damaged_mask]).mean()) if damaged_mask.any() else float("nan"),
        "high_recall": float(pred[high_mask].mean()) if high_mask.any() else float("nan"),
    }


def train_regressors(seed: int) -> Dict[str, object]:
    """English: Build regression baselines. 中文：构建回归基线。"""
    return {
        "local_ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=10.0)),
            ]
        ),
        "local_random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def train_classifiers(seed: int) -> Dict[str, object]:
    """English: Build zero-vs-damaged classifiers. 中文：构建 zero-vs-damaged 分类器。"""
    return {
        "local_logistic_l2_balanced": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "local_extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def predict_proba_positive(model: object, X: np.ndarray) -> np.ndarray:
    """English: Return positive probability. 中文：返回正类概率。"""
    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return np.asarray(proba).reshape(-1)


# -----------------------------
# Plotting
# 绘图
# -----------------------------


def plot_true_vs_pred(y_true: np.ndarray, pred: np.ndarray, title: str, output_path: Path) -> None:
    """English: Plot true-vs-predicted damage. 中文：绘制真实值-预测值散点图。"""
    bins = damage_bin(y_true)
    plt.figure(figsize=(8, 8))
    for b in ["zero", "low", "medium", "high"]:
        mask = bins == b
        if mask.any():
            plt.scatter(y_true[mask], pred[mask], alpha=0.65, label=b)
    plt.plot([0, 0.5], [0, 0.5], "--", label="Ideal y=x")
    plt.xlabel("True damage")
    plt.ylabel("Predicted damage")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_local_pca(X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, output_path: Path) -> None:
    """English: Plot PCA projection. 中文：绘制 PCA 投影。"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_train_s)
    Z = pca.transform(X_test_s)
    bins = damage_bin(y_test)

    plt.figure(figsize=(8, 6))
    for b in ["zero", "low", "medium", "high"]:
        mask = bins == b
        if mask.any():
            plt.scatter(Z[mask, 0], Z[mask, 1], alpha=0.7, label=b)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title("Story-local feature PCA projection by damage bin")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bin_bias(y_true: np.ndarray, pred_by_model: Dict[str, np.ndarray], output_path: Path) -> None:
    """English: Plot bin-level bias. 中文：绘制不同损伤等级下的偏差。"""
    rows = []
    bins = damage_bin(y_true)
    for model_name, pred in pred_by_model.items():
        for b in ["zero", "low", "medium", "high"]:
            mask = bins == b
            if mask.any():
                rows.append(
                    {
                        "model": model_name,
                        "bin": b,
                        "bias": float(np.mean(np.clip(pred[mask], 0, 0.5) - y_true[mask])),
                        "mae": float(mean_absolute_error(y_true[mask], np.clip(pred[mask], 0, 0.5))),
                    }
                )
    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    pivot = df.pivot(index="model", columns="bin", values="bias").reindex(columns=["zero", "low", "medium", "high"])
    x = np.arange(len(pivot.index))
    width = 0.18
    for i, col in enumerate(pivot.columns):
        plt.bar(x + (i - 1.5) * width, pivot[col].values, width=width, label=col)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, pivot.index, rotation=35, ha="right")
    plt.ylabel("Mean bias: predicted - true")
    plt.title("Story-local damage-bin prediction bias")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# 主程序
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--feature-names", required=True, type=Path)
    parser.add_argument("--tag", default="debug_plus_500")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--include-neighbor", action="store_true")
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--include-story-one-hot", action="store_true")
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)

    features_npz = load_npz(args.features)
    split_npz = load_npz(args.split)

    X_case = resolve_case_features(features_npz)
    y_story = resolve_story_damage(split_npz)
    feature_names = load_feature_names(args.feature_names, X_case["train"].shape[1])

    X_local: Dict[str, np.ndarray] = {}
    y_local: Dict[str, np.ndarray] = {}
    story_ids: Dict[str, np.ndarray] = {}

    constructed_names: Optional[List[str]] = None
    for split in ["train", "val", "test"]:
        X_l, y_l, sid_l, names_l = build_story_dataset(
            X_case=X_case[split],
            y_story=y_story[split],
            feature_names=feature_names,
            include_neighbor=args.include_neighbor,
            include_global=not args.no_global,
            include_story_one_hot=args.include_story_one_hot,
        )
        X_local[split] = X_l
        y_local[split] = y_l
        story_ids[split] = sid_l
        constructed_names = names_l

    assert constructed_names is not None
    pd.DataFrame(
        {
            "constructed_feature_index": np.arange(len(constructed_names)),
            "constructed_feature_name": constructed_names,
        }
    ).to_csv(args.output_dir / "story_local_constructed_feature_names.csv", index=False)

    # English: Save label distribution.
    # 中文：保存标签分布。
    dist_rows = []
    for split in ["train", "val", "test"]:
        bins = damage_bin(y_local[split])
        row = {"split": split, "n": int(len(bins))}
        for b in ["zero", "low", "medium", "high"]:
            row[b] = int((bins == b).sum())
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(args.output_dir / "story_local_damage_bin_counts.csv", index=False)

    # Regression
    reg_rows = []
    pred_test_by_reg = {}
    for model_name, model in train_regressors(args.seed).items():
        print(f"Training regressor: {model_name}")
        model.fit(X_local["train"], y_local["train"])
        for split in ["train", "val", "test"]:
            pred = np.asarray(model.predict(X_local[split]), dtype=float)
            reg_rows.append(regression_metrics(y_local[split], pred, model_name, split))
            if split == "test":
                pred_test_by_reg[model_name] = pred
                pd.DataFrame(
                    {
                        "y_true": y_local[split],
                        "y_pred": np.clip(pred, 0, 0.5),
                        "story_id": story_ids[split],
                        "damage_bin": damage_bin(y_local[split]),
                    }
                ).to_csv(args.output_dir / f"{model_name}_predictions_test.csv", index=False)

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(args.output_dir / "story_local_regression_metrics.csv", index=False)

    # Classification
    clf_rows = []
    score_test_by_clf = {}
    y_bin_train = (y_local["train"] > 1e-12).astype(int)
    for model_name, model in train_classifiers(args.seed).items():
        print(f"Training classifier: {model_name}")
        model.fit(X_local["train"], y_bin_train)
        for split in ["train", "val", "test"]:
            score = predict_proba_positive(model, X_local[split])
            clf_rows.append(
                classifier_metrics(
                    y_true_damage=y_local[split],
                    score=score,
                    model_name=model_name,
                    split=split,
                    threshold=args.classification_threshold,
                )
            )
            if split == "test":
                score_test_by_clf[model_name] = score
                pd.DataFrame(
                    {
                        "y_true_damage": y_local[split],
                        "y_binary_true": (y_local[split] > 1e-12).astype(int),
                        "damaged_probability": score,
                        "story_id": story_ids[split],
                        "damage_bin": damage_bin(y_local[split]),
                    }
                ).to_csv(args.output_dir / f"{model_name}_classification_scores_test.csv", index=False)

    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(args.output_dir / "story_local_zero_damaged_classification_metrics.csv", index=False)

    # Figures
    best_reg = (
        reg_df[reg_df["split"] == "test"]
        .sort_values(["test_mae", "high_mae"], ascending=[True, True])
        .iloc[0]["model"]
    )
    plot_true_vs_pred(
        y_true=y_local["test"],
        pred=np.clip(pred_test_by_reg[best_reg], 0, 0.5),
        title=f"Story-local true vs predicted damage: {best_reg}",
        output_path=args.figures_dir / f"story_local_true_vs_pred_{best_reg}.png",
    )
    plot_bin_bias(y_local["test"], pred_test_by_reg, args.figures_dir / "story_local_damage_bin_bias.png")
    plot_local_pca(X_local["train"], X_local["test"], y_local["test"], args.figures_dir / "story_local_pca_projection.png")

    # Write report
    best_reg_row = reg_df[(reg_df["split"] == "test") & (reg_df["model"] == best_reg)].iloc[0]
    best_clf = (
        clf_df[clf_df["split"] == "test"]
        .sort_values(["pr_auc", "roc_auc", "f1"], ascending=[False, False, False])
        .iloc[0]["model"]
    )
    best_clf_row = clf_df[(clf_df["split"] == "test") & (clf_df["model"] == best_clf)].iloc[0]

    interpretation = []
    if best_reg_row["test_mae"] < 0.047:
        interpretation.append(
            "Story-local feature reconstruction improves or matches the previous best overall regression MAE."
        )
    else:
        interpretation.append(
            "Story-local feature reconstruction does not yet beat the previous full-feature ridge baseline."
        )

    if best_reg_row["high_mae"] < 0.10:
        interpretation.append(
            "High-damage regression error is reduced to a potentially useful range."
        )
    else:
        interpretation.append(
            "High-damage regression error remains large; stronger damage-sensitive features or target redesign is still needed."
        )

    if best_clf_row["roc_auc"] > 0.65:
        interpretation.append(
            "Story-local zero-vs-damaged classification is meaningfully better than the previous weak gate."
        )
    else:
        interpretation.append(
            "Story-local zero-vs-damaged classification is still weak; local feature reconstruction alone is insufficient."
        )

    lines = []
    lines.append("# Story-local Feature Alignment Diagnosis")
    lines.append("")
    lines.append("## 1. Configuration")
    lines.append(f"- Features: `{args.features}`")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Feature names: `{args.feature_names}`")
    lines.append(f"- Include neighbor features: `{args.include_neighbor}`")
    lines.append(f"- Include global input/ground features: `{not args.no_global}`")
    lines.append(f"- Include story one-hot: `{args.include_story_one_hot}`")
    lines.append(f"- Constructed feature count: `{len(constructed_names)}`")
    lines.append("")
    lines.append("## 2. Damage-bin counts")
    lines.append(manual_markdown_table(dist_df))
    lines.append("")
    lines.append("## 3. Regression metrics")
    lines.append(manual_markdown_table(reg_df.sort_values(["split", "test_mae"]).reset_index(drop=True)))
    lines.append("")
    lines.append("## 4. Zero-vs-damaged classification metrics")
    lines.append(manual_markdown_table(clf_df.sort_values(["split", "pr_auc"], ascending=[True, False]).reset_index(drop=True)))
    lines.append("")
    lines.append("## 5. Best regression model")
    lines.append(f"- Best test regression model by overall MAE: `{best_reg}`")
    lines.append(f"- Test MAE: `{best_reg_row['test_mae']:.6g}`")
    lines.append(f"- High-damage MAE: `{best_reg_row['high_mae']:.6g}`")
    lines.append(f"- High-damage bias: `{best_reg_row['high_bias_pred_minus_true']:.6g}`")
    lines.append("")
    lines.append("## 6. Best classifier")
    lines.append(f"- Best test classifier by PR-AUC: `{best_clf}`")
    lines.append(f"- Test ROC-AUC: `{best_clf_row['roc_auc']:.6g}`")
    lines.append(f"- Test PR-AUC: `{best_clf_row['pr_auc']:.6g}`")
    lines.append(f"- Test zero false alarm: `{best_clf_row['zero_false_alarm_ratio']:.6g}`")
    lines.append(f"- Test damaged recall: `{best_clf_row['recall']:.6g}`")
    lines.append("")
    lines.append("## 7. Preliminary interpretation")
    for item in interpretation:
        lines.append(f"- {item}")
    lines.append("")

    (args.output_dir / "story_local_feature_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "tag": args.tag,
        "constructed_feature_count": len(constructed_names),
        "best_regression_model": str(best_reg),
        "best_regression_test_mae": float(best_reg_row["test_mae"]),
        "best_regression_high_mae": float(best_reg_row["high_mae"]),
        "best_regression_high_bias": float(best_reg_row["high_bias_pred_minus_true"]),
        "best_classifier": str(best_clf),
        "best_classifier_test_roc_auc": float(best_clf_row["roc_auc"]),
        "best_classifier_test_pr_auc": float(best_clf_row["pr_auc"]),
        "best_classifier_zero_false_alarm": float(best_clf_row["zero_false_alarm_ratio"]),
        "best_classifier_damaged_recall": float(best_clf_row["recall"]),
    }
    (args.output_dir / "story_local_feature_alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nStory-local feature alignment diagnosis completed.")
    print(f"Report: {args.output_dir / 'story_local_feature_alignment_report.md'}")
    print(f"Regression metrics: {args.output_dir / 'story_local_regression_metrics.csv'}")
    print(f"Classification metrics: {args.output_dir / 'story_local_zero_damaged_classification_metrics.csv'}")
    print(f"Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
