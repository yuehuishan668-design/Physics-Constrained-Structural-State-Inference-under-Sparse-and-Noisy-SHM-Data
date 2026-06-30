"""
Paired healthy-baseline feature diagnosis.

中文说明：
本脚本的目的不是继续调模型，而是重新构造特征。

核心思想：
1. 从训练集中找出 healthy / zero-damage 样本；
2. 对每个样本，根据 input / ground / noise 等输入特征，寻找最相似的健康样本；
3. 用该健康样本作为 baseline；
4. 构造：
   - delta = damaged_feature - healthy_baseline_feature
   - abs_delta = abs(delta)
   - relative_delta = delta / abs(healthy_baseline_feature)
   - standardized_delta = delta / std(healthy_baseline_pool)
5. 用这些 paired features 重新做 case-level any-damage 诊断与 damage regression。

Why this matters:
传统 response feature 直接预测 damage，容易把输入强度、噪声、频率影响误认为结构损伤。
paired baseline 特征更接近 SHM 中“相对健康基准变化”的真实逻辑。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist. 如果目录不存在，则自动创建。"""
    path.mkdir(parents=True, exist_ok=True)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load an npz file as a normal dictionary. 将 npz 文件读取为普通字典。"""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def read_feature_names(path: Path, n_features: int) -> List[str]:
    """
    Read feature names from csv.

    中文说明：
    兼容不同 csv 格式：
    - 如果有 feature_name 列，用 feature_name；
    - 如果有 name 列，用 name；
    - 否则取第一列；
    - 如果数量对不上，则自动生成 feature_000, feature_001 ...
    """
    if not path.exists():
        return [f"feature_{i:03d}" for i in range(n_features)]

    df = pd.read_csv(path)

    if "feature_name" in df.columns:
        names = df["feature_name"].astype(str).tolist()
    elif "name" in df.columns:
        names = df["name"].astype(str).tolist()
    else:
        names = df.iloc[:, 0].astype(str).tolist()

    if len(names) != n_features:
        print(
            f"[Warning] feature name count mismatch: "
            f"{len(names)} names for {n_features} features. Use fallback names."
        )
        names = [f"feature_{i:03d}" for i in range(n_features)]

    return names


def get_key(data: Dict[str, np.ndarray], candidates: List[str]) -> np.ndarray:
    """Find the first existing key from candidate names. 从候选 key 中找到第一个存在的 key。"""
    for key in candidates:
        if key in data:
            return data[key]
    raise KeyError(f"None of these keys exist: {candidates}. Existing keys: {list(data.keys())}")


def load_split_features(features_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Load X_train / X_val / X_test. 读取训练、验证、测试特征。"""
    return {
        "train": get_key(features_data, ["X_train", "F_train", "features_train"]),
        "val": get_key(features_data, ["X_val", "F_val", "features_val"]),
        "test": get_key(features_data, ["X_test", "F_test", "features_test"]),
    }


def load_split_targets(
    features_data: Dict[str, np.ndarray],
    split_data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Load y_train / y_val / y_test.

    中文说明：
    优先从 physics_features_mlp.npz 读取 y；
    如果没有，则从 split_normalized.npz 读取。
    """
    result = {}

    for split in ["train", "val", "test"]:
        candidate_keys = [
            f"y_{split}",
            f"damage_{split}",
            f"y_damage_{split}",
            f"target_{split}",
        ]

        found = None
        for key in candidate_keys:
            if key in features_data:
                found = features_data[key]
                break
            if key in split_data:
                found = split_data[key]
                break

        if found is None:
            raise KeyError(
                f"Cannot find target for split={split}. "
                f"Checked keys: {candidate_keys}. "
                f"features_data keys={list(features_data.keys())}. "
                f"split_data keys={list(split_data.keys())}."
            )

        result[split] = found

    return result


def case_damage_from_y(y: np.ndarray) -> np.ndarray:
    """
    Convert story-level damage labels to case-level damage value.

    中文说明：
    如果 y 是二维，例如 shape=(n_case, 4)，说明每个 case 有 4 层损伤；
    此时取 max，表示该结构中最大楼层损伤程度。
    如果 y 已经是一维，则直接使用。
    """
    y = np.asarray(y)

    if y.ndim == 1:
        return y.astype(float)

    if y.ndim == 2:
        return np.max(y.astype(float), axis=1)

    raise ValueError(f"Unsupported y shape: {y.shape}")


def choose_input_matching_columns(feature_names: List[str]) -> List[int]:
    """
    Choose columns used only for matching similar input conditions.

    中文说明：
    这里必须避免使用明显的结构响应损伤特征去找 healthy baseline，
    否则会发生信息泄漏。

    优先使用：
    - input
    - ground
    - amplitude
    - frequency
    - noise
    - excitation

    不使用：
    - story response
    - damage
    - prediction
    - residual
    """
    positive_keywords = [
        "input",
        "ground",
        "amplitude",
        "frequency",
        "freq",
        "noise",
        "excitation",
    ]

    negative_keywords = [
        "damage",
        "pred",
        "target",
        "residual",
        "story_is",
    ]

    selected = []
    for i, name in enumerate(feature_names):
        lower = name.lower()

        has_positive = any(k in lower for k in positive_keywords)
        has_negative = any(k in lower for k in negative_keywords)

        if has_positive and not has_negative:
            selected.append(i)

    return selected


def build_paired_features_for_split(
    X: np.ndarray,
    healthy_X_train: np.ndarray,
    nn_model: NearestNeighbors,
    input_scaler: StandardScaler,
    input_cols: List[int],
    healthy_std: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build paired baseline features for one split.

    返回：
    - paired_features
    - matched_reference_features
    - mean_neighbor_distance
    """
    X_input_scaled = input_scaler.transform(X[:, input_cols])

    distances, neighbor_indices = nn_model.kneighbors(X_input_scaled)

    matched_refs = healthy_X_train[neighbor_indices].mean(axis=1)
    mean_distance = distances.mean(axis=1)

    delta = X - matched_refs
    abs_delta = np.abs(delta)
    relative_delta = delta / (np.abs(matched_refs) + eps)
    standardized_delta = delta / (healthy_std + eps)

    paired = np.concatenate(
        [
            X,
            delta,
            abs_delta,
            relative_delta,
            standardized_delta,
        ],
        axis=1,
    )

    return paired, matched_refs, mean_distance


def make_paired_feature_names(raw_names: List[str]) -> List[str]:
    """Generate paired feature names. 生成 paired feature 名称。"""
    names = []

    names.extend([f"raw__{name}" for name in raw_names])
    names.extend([f"delta__{name}" for name in raw_names])
    names.extend([f"abs_delta__{name}" for name in raw_names])
    names.extend([f"relative_delta__{name}" for name in raw_names])
    names.extend([f"standardized_delta__{name}" for name in raw_names])

    return names


def build_paired_dataset(
    features_npz: Path,
    split_npz: Path,
    feature_names_csv: Path,
    output_npz: Path,
    output_feature_names_csv: Path,
    k_neighbors: int,
    zero_tol: float,
    eps: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], List[int]]:
    """
    Build paired baseline feature dataset.

    中文说明：
    使用 train split 中的 zero-damage 样本作为 healthy reference pool。
    val/test 不使用自身 zero 样本做 baseline，避免测试信息泄漏。
    """
    features_data = load_npz(features_npz)
    split_data = load_npz(split_npz)

    X_split = load_split_features(features_data)
    y_split = load_split_targets(features_data, split_data)

    n_features = X_split["train"].shape[1]
    raw_feature_names = read_feature_names(feature_names_csv, n_features)
    input_cols = choose_input_matching_columns(raw_feature_names)

    if len(input_cols) == 0:
        raise RuntimeError(
            "No input-matching columns found. "
            "Please check feature names. Need columns containing input/ground/amplitude/frequency/noise."
        )

    y_case_train = case_damage_from_y(y_split["train"])
    healthy_mask_train = y_case_train <= zero_tol

    n_healthy = int(healthy_mask_train.sum())
    if n_healthy < 3:
        raise RuntimeError(
            f"Too few healthy training samples: {n_healthy}. "
            f"Cannot build stable paired baseline."
        )

    healthy_X_train = X_split["train"][healthy_mask_train]

    k_eff = min(k_neighbors, n_healthy)

    input_scaler = StandardScaler()
    healthy_input_scaled = input_scaler.fit_transform(healthy_X_train[:, input_cols])

    nn_model = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn_model.fit(healthy_input_scaled)

    healthy_std = np.std(healthy_X_train, axis=0)
    healthy_std = np.where(healthy_std < eps, eps, healthy_std)

    paired_split = {}
    matched_ref_split = {}
    distance_split = {}

    for split in ["train", "val", "test"]:
        paired, refs, mean_dist = build_paired_features_for_split(
            X=X_split[split],
            healthy_X_train=healthy_X_train,
            nn_model=nn_model,
            input_scaler=input_scaler,
            input_cols=input_cols,
            healthy_std=healthy_std,
            eps=eps,
        )

        paired_split[split] = paired
        matched_ref_split[split] = refs
        distance_split[split] = mean_dist

    paired_names = make_paired_feature_names(raw_feature_names)

    ensure_dir(output_npz.parent)
    ensure_dir(output_feature_names_csv.parent)

    np.savez_compressed(
        output_npz,
        X_train=paired_split["train"],
        X_val=paired_split["val"],
        X_test=paired_split["test"],
        y_train=y_split["train"],
        y_val=y_split["val"],
        y_test=y_split["test"],
        y_case_train=case_damage_from_y(y_split["train"]),
        y_case_val=case_damage_from_y(y_split["val"]),
        y_case_test=case_damage_from_y(y_split["test"]),
        matched_reference_train=matched_ref_split["train"],
        matched_reference_val=matched_ref_split["val"],
        matched_reference_test=matched_ref_split["test"],
        baseline_distance_train=distance_split["train"],
        baseline_distance_val=distance_split["val"],
        baseline_distance_test=distance_split["test"],
        input_matching_columns=np.asarray(input_cols, dtype=int),
        input_matching_feature_names=np.asarray([raw_feature_names[i] for i in input_cols]),
    )

    pd.DataFrame({"feature_name": paired_names}).to_csv(output_feature_names_csv, index=False)

    return paired_split, y_split, paired_names, input_cols


def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC with fallback. 安全计算 ROC-AUC。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision with fallback. 安全计算 AP。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def regression_models(random_state: int) -> Dict[str, object]:
    """Define regression models. 定义回归模型。"""
    return {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0])),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def classifier_models(random_state: int) -> Dict[str, object]:
    """Define classification models. 定义分类模型。"""
    return {
        "logistic_l2_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        C=1.0,
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_regressors(
    X_split: Dict[str, np.ndarray],
    y_case_split: Dict[str, np.ndarray],
    output_dir: Path,
    figures_dir: Path,
    random_state: int,
    high_threshold: float,
) -> pd.DataFrame:
    """Train and evaluate regressors. 训练并评价回归模型。"""
    rows = []

    models = regression_models(random_state)

    best_model_name = None
    best_test_mae = float("inf")
    best_pred_test = None

    for model_name, model in models.items():
        print(f"Training paired regressor: {model_name}")

        model.fit(X_split["train"], y_case_split["train"])

        for split in ["train", "val", "test"]:
            y_true = y_case_split[split]
            y_pred = model.predict(X_split[split])
            y_pred = np.clip(y_pred, 0.0, 0.5)

            high_mask = y_true >= high_threshold
            zero_mask = y_true <= 1e-12

            row = {
                "model": model_name,
                "split": split,
                "n": len(y_true),
                "test_mae": mean_absolute_error(y_true, y_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mean_true_damage": float(np.mean(y_true)),
                "mean_pred_damage": float(np.mean(y_pred)),
                "bias_pred_minus_true": float(np.mean(y_pred - y_true)),
                "zero_mean_pred": float(np.mean(y_pred[zero_mask])) if zero_mask.any() else np.nan,
                "high_n": int(high_mask.sum()),
                "high_mae": mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if high_mask.any() else np.nan,
                "high_bias_pred_minus_true": float(np.mean(y_pred[high_mask] - y_true[high_mask])) if high_mask.any() else np.nan,
                "high_underestimation_ratio": float(np.mean(y_pred[high_mask] < y_true[high_mask])) if high_mask.any() else np.nan,
            }
            rows.append(row)

            if split == "test" and row["test_mae"] < best_test_mae:
                best_test_mae = row["test_mae"]
                best_model_name = model_name
                best_pred_test = y_pred.copy()

    df = pd.DataFrame(rows)
    ensure_dir(output_dir)
    df.to_csv(output_dir / "paired_baseline_regression_metrics.csv", index=False)

    if best_pred_test is not None:
        plot_true_vs_pred(
            y_true=y_case_split["test"],
            y_pred=best_pred_test,
            title=f"Paired baseline best regression: {best_model_name}",
            output_path=figures_dir / "paired_baseline_best_regression_true_vs_pred.png",
        )

    return df


def evaluate_classifiers(
    X_split: Dict[str, np.ndarray],
    y_case_split: Dict[str, np.ndarray],
    output_dir: Path,
    figures_dir: Path,
    random_state: int,
    damage_threshold: float,
) -> pd.DataFrame:
    """Train and evaluate zero-vs-damaged classifiers. 训练并评价 zero-vs-damaged 分类模型。"""
    rows = []

    y_bin_split = {
        split: (y_case_split[split] > damage_threshold).astype(int)
        for split in ["train", "val", "test"]
    }

    models = classifier_models(random_state)

    roc_payload = {}
    pr_payload = {}

    best_model_name = None
    best_test_auc = -1.0
    best_test_proba = None

    for model_name, model in models.items():
        print(f"Training paired classifier: {model_name}")

        model.fit(X_split["train"], y_bin_split["train"])

        for split in ["train", "val", "test"]:
            y_true = y_bin_split[split]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_split[split])[:, 1]
            else:
                proba = model.decision_function(X_split[split])

            y_pred = (proba >= 0.5).astype(int)

            zero_mask = y_true == 0
            damaged_mask = y_true == 1

            row = {
                "model": model_name,
                "split": split,
                "threshold": 0.5,
                "n": len(y_true),
                "zero_n": int(zero_mask.sum()),
                "damaged_n": int(damaged_mask.sum()),
                "roc_auc": safe_roc_auc(y_true, proba),
                "pr_auc": safe_average_precision(y_true, proba),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "zero_false_alarm_ratio": float(np.mean(y_pred[zero_mask] == 1)) if zero_mask.any() else np.nan,
                "damaged_miss_ratio": float(np.mean(y_pred[damaged_mask] == 0)) if damaged_mask.any() else np.nan,
                "mean_proba_zero": float(np.mean(proba[zero_mask])) if zero_mask.any() else np.nan,
                "mean_proba_damaged": float(np.mean(proba[damaged_mask])) if damaged_mask.any() else np.nan,
            }
            rows.append(row)

            if split == "test":
                roc_payload[model_name] = (y_true, proba)
                pr_payload[model_name] = (y_true, proba)

                if row["roc_auc"] > best_test_auc:
                    best_test_auc = row["roc_auc"]
                    best_model_name = model_name
                    best_test_proba = proba.copy()

    df = pd.DataFrame(rows)
    ensure_dir(output_dir)
    df.to_csv(output_dir / "paired_baseline_zero_damaged_classification_metrics.csv", index=False)

    plot_roc_curves(
        roc_payload=roc_payload,
        output_path=figures_dir / "paired_baseline_zero_damaged_roc.png",
    )
    plot_pr_curves(
        pr_payload=pr_payload,
        output_path=figures_dir / "paired_baseline_zero_damaged_pr.png",
    )

    if best_test_proba is not None:
        plot_probability_histogram(
            y_true=y_bin_split["test"],
            proba=best_test_proba,
            title=f"Paired baseline probability histogram: {best_model_name}",
            output_path=figures_dir / "paired_baseline_best_classifier_probability_histogram.png",
        )

        cm = confusion_matrix(y_bin_split["test"], (best_test_proba >= 0.5).astype(int), labels=[0, 1])
        plot_confusion_matrix(
            cm=cm,
            title=f"Paired baseline confusion matrix: {best_model_name}",
            output_path=figures_dir / "paired_baseline_best_classifier_confusion_matrix.png",
        )

    return df


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    """Plot true vs predicted damage. 绘制真实值与预测值对比图。"""
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7)

    max_lim = max(float(np.max(y_true)), float(np.max(y_pred)), 0.5)
    plt.plot([0, max_lim], [0, max_lim], linestyle="--", label="Ideal y=x")

    plt.xlabel("True damage")
    plt.ylabel("Predicted damage")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curves(roc_payload: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """Plot ROC curves. 绘制 ROC 曲线。"""
    ensure_dir(output_path.parent)

    plt.figure(figsize=(9, 7))

    for model_name, (y_true, score) in roc_payload.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Paired baseline zero-vs-damaged ROC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curves(pr_payload: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """Plot precision-recall curves. 绘制 PR 曲线。"""
    ensure_dir(output_path.parent)

    plt.figure(figsize=(9, 7))

    for model_name, (y_true, score) in pr_payload.items():
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        plt.plot(recall, precision, label=f"{model_name} AP={ap:.3f}")

    plt.xlabel("Damaged recall")
    plt.ylabel("Damaged precision")
    plt.title("Paired baseline zero-vs-damaged PR")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_probability_histogram(
    y_true: np.ndarray,
    proba: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Plot probability histogram. 绘制预测概率直方图。"""
    ensure_dir(output_path.parent)

    plt.figure(figsize=(9, 7))
    plt.hist(proba[y_true == 0], bins=20, alpha=0.6, label="true zero")
    plt.hist(proba[y_true == 1], bins=20, alpha=0.6, label="true damaged")
    plt.xlabel("Predicted damaged probability")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path) -> None:
    """Plot confusion matrix. 绘制混淆矩阵。"""
    ensure_dir(output_path.parent)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm)
    plt.colorbar(label="Count")

    labels = ["zero", "damaged"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    output_path: Path,
    config: Dict[str, object],
    reg_df: pd.DataFrame,
    cls_df: pd.DataFrame,
    input_feature_names: List[str],
) -> None:
    """Write markdown report manually. 手动写 Markdown 报告，避免依赖 tabulate。"""
    ensure_dir(output_path.parent)

    test_reg = reg_df[reg_df["split"] == "test"].sort_values("test_mae")
    test_cls = cls_df[cls_df["split"] == "test"].sort_values("roc_auc", ascending=False)

    best_reg = test_reg.iloc[0].to_dict()
    best_cls = test_cls.iloc[0].to_dict()

    lines = []
    lines.append("# Paired Healthy-baseline Feature Diagnosis")
    lines.append("")
    lines.append("## 1. Configuration")
    for key, value in config.items():
        lines.append(f"- {key}: `{value}`")

    lines.append("")
    lines.append("## 2. Input-matching features")
    lines.append(f"- Number of input-matching features: `{len(input_feature_names)}`")
    for name in input_feature_names:
        lines.append(f"  - `{name}`")

    lines.append("")
    lines.append("## 3. Best regression result")
    for key in [
        "model",
        "test_mae",
        "test_rmse",
        "mean_true_damage",
        "mean_pred_damage",
        "bias_pred_minus_true",
        "zero_mean_pred",
        "high_n",
        "high_mae",
        "high_bias_pred_minus_true",
        "high_underestimation_ratio",
    ]:
        lines.append(f"- {key}: `{best_reg.get(key)}`")

    lines.append("")
    lines.append("## 4. Best zero-vs-damaged classifier result")
    for key in [
        "model",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "zero_false_alarm_ratio",
        "damaged_miss_ratio",
        "mean_proba_zero",
        "mean_proba_damaged",
    ]:
        lines.append(f"- {key}: `{best_cls.get(key)}`")

    lines.append("")
    lines.append("## 5. Preliminary interpretation")
    lines.append("- If ROC-AUC improves clearly over the previous case-level value around 0.64, paired baseline features are useful.")
    lines.append("- If high-damage MAE and high underestimation remain poor, the bottleneck is not only input normalization but also feature sensitivity.")
    lines.append("- If zero false alarm remains high, the zero/damaged boundary is still not separable enough.")
    lines.append("- If mean_proba_zero and mean_proba_damaged are still close, threshold tuning will not solve the problem.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--feature-names", required=True, type=Path)

    parser.add_argument("--paired-output", required=True, type=Path)
    parser.add_argument("--paired-feature-names-output", required=True, type=Path)

    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)

    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--zero-tol", type=float, default=1e-12)
    parser.add_argument("--damage-threshold", type=float, default=0.05)
    parser.add_argument("--high-threshold", type=float, default=0.20)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)

    print("Step 1: build paired healthy-baseline features")
    paired_split, y_split, paired_names, input_cols = build_paired_dataset(
        features_npz=args.features,
        split_npz=args.split,
        feature_names_csv=args.feature_names,
        output_npz=args.paired_output,
        output_feature_names_csv=args.paired_feature_names_output,
        k_neighbors=args.k_neighbors,
        zero_tol=args.zero_tol,
        eps=args.eps,
    )

    y_case_split = {
        split: case_damage_from_y(y_split[split])
        for split in ["train", "val", "test"]
    }

    feature_names = read_feature_names(args.feature_names, paired_split["train"].shape[1] // 5)
    input_feature_names = [feature_names[i] for i in input_cols]

    print("Step 2: evaluate paired baseline regressors")
    reg_df = evaluate_regressors(
        X_split=paired_split,
        y_case_split=y_case_split,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        random_state=args.seed,
        high_threshold=args.high_threshold,
    )

    print("Step 3: evaluate paired baseline classifiers")
    cls_df = evaluate_classifiers(
        X_split=paired_split,
        y_case_split=y_case_split,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        random_state=args.seed,
        damage_threshold=args.damage_threshold,
    )

    print("Step 4: write report")
    config = {
        "features": str(args.features),
        "split": str(args.split),
        "feature_names": str(args.feature_names),
        "paired_output": str(args.paired_output),
        "k_neighbors": args.k_neighbors,
        "damage_threshold": args.damage_threshold,
        "high_threshold": args.high_threshold,
    }

    write_report(
        output_path=args.output_dir / "paired_baseline_diagnosis_report.md",
        config=config,
        reg_df=reg_df,
        cls_df=cls_df,
        input_feature_names=input_feature_names,
    )

    summary = {
        "paired_feature_shape_train": list(paired_split["train"].shape),
        "paired_feature_shape_val": list(paired_split["val"].shape),
        "paired_feature_shape_test": list(paired_split["test"].shape),
        "n_paired_features": len(paired_names),
        "n_input_matching_features": len(input_feature_names),
        "input_matching_features": input_feature_names,
    }

    with open(args.output_dir / "paired_baseline_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("")
    print("Paired baseline diagnosis completed.")
    print(f"Report: {args.output_dir / 'paired_baseline_diagnosis_report.md'}")
    print(f"Regression metrics: {args.output_dir / 'paired_baseline_regression_metrics.csv'}")
    print(f"Classification metrics: {args.output_dir / 'paired_baseline_zero_damaged_classification_metrics.csv'}")
    print(f"Paired feature dataset: {args.paired_output}")
    print(f"Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
