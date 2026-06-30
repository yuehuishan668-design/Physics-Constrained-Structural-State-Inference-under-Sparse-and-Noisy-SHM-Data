#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case-level any-damage diagnosis for the debug_plus_500 structural state dataset.

English:
    This script tests whether the current physics-informed feature table can detect
    whether a structural case contains any nonzero story damage.

中文：
    本脚本用于诊断当前物理启发特征是否能够判断“一个结构工况中是否存在任意楼层损伤”。
    这一步不再定位具体楼层，而是先判断全结构层面的 zero / damaged 可分性。

Why this script is needed:
    If case-level any-damage classification is still weak, the feature space itself
    lacks damage separability.
    If case-level classification is good but story-level diagnosis is weak, the
    bottleneck is mainly story localization / feature-target alignment.

为什么需要这一步：
    如果 case-level 是否损伤分类仍然很差，说明当前特征本身缺乏损伤区分能力。
    如果 case-level 效果较好但 story-level 很差，说明瓶颈主要是楼层定位或特征-标签对应关系。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Utility functions
# 工具函数
# -----------------------------


def ensure_dir(path: Path) -> None:
    """English: Create a folder if it does not exist. 中文：如果文件夹不存在则创建。"""
    path.mkdir(parents=True, exist_ok=True)


def manual_markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """
    English:
        Convert a dataframe to a markdown table without requiring the optional
        pandas dependency `tabulate`.

    中文：
        将 DataFrame 转换为 markdown 表格，不依赖 pandas 的可选依赖 tabulate。
    """
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
    """English: Load an npz file into a normal dict. 中文：将 npz 文件读取为普通字典。"""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def pick_first_existing_key(data: Dict[str, np.ndarray], candidates: Sequence[str]) -> Optional[str]:
    """English: Return the first key that exists in a dict. 中文：返回候选键名中第一个存在的键。"""
    for key in candidates:
        if key in data:
            return key
    return None


def load_feature_names(path: Optional[Path], n_features: int) -> List[str]:
    """
    English:
        Load feature names from a CSV if possible. Otherwise use generic names.
    中文：
        尽量从 CSV 读取特征名；如果失败，则使用 feature_0, feature_1 等默认名称。
    """
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
        # English: fallback to the first non-index-like column.
        # 中文：如果列名不明确，则优先取第一个不像索引的列。
        possible_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
        if possible_cols:
            names = df[possible_cols[0]].astype(str).tolist()
        else:
            names = [f"feature_{i}" for i in range(n_features)]

    if len(names) < n_features:
        names = names + [f"feature_{i}" for i in range(len(names), n_features)]
    elif len(names) > n_features:
        names = names[:n_features]

    return names


def resolve_case_features(features_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    English:
        Resolve case-level feature matrices from different possible npz key conventions.
    中文：
        兼容不同 npz 键名，解析出 case-level 特征矩阵。

    Expected output:
        train: [n_cases_train, n_features]
        val:   [n_cases_val, n_features]
        test:  [n_cases_test, n_features]
    """
    split_to_candidates = {
        "train": ["F_train", "features_train", "X_features_train", "X_train_features", "X_train"],
        "val": ["F_val", "features_val", "X_features_val", "X_val_features", "X_val"],
        "test": ["F_test", "features_test", "X_features_test", "X_test_features", "X_test"],
    }

    out: Dict[str, np.ndarray] = {}
    for split, candidates in split_to_candidates.items():
        key = pick_first_existing_key(features_npz, candidates)
        if key is None:
            raise KeyError(
                f"Cannot find {split} feature matrix. Tried keys: {candidates}. "
                f"Available keys: {list(features_npz.keys())}"
            )
        arr = np.asarray(features_npz[key])
        if arr.ndim != 2:
            raise ValueError(
                f"Feature key {key} should be a 2D matrix [n_cases, n_features], "
                f"but got shape {arr.shape}."
            )
        out[split] = arr.astype(float)

    return out


def resolve_story_damage(split_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    English:
        Resolve story damage labels from split dataset.
    中文：
        从 split npz 中解析每个 case 的各楼层损伤标签。

    Expected output:
        train: [n_cases_train, n_stories]
        val:   [n_cases_val, n_stories]
        test:  [n_cases_test, n_stories]
    """
    split_to_candidates = {
        "train": ["y_train", "damage_train", "Y_train", "target_train"],
        "val": ["y_val", "damage_val", "Y_val", "target_val"],
        "test": ["y_test", "damage_test", "Y_test", "target_test"],
    }

    out: Dict[str, np.ndarray] = {}
    for split, candidates in split_to_candidates.items():
        key = pick_first_existing_key(split_npz, candidates)
        if key is None:
            raise KeyError(
                f"Cannot find {split} damage label array. Tried keys: {candidates}. "
                f"Available keys: {list(split_npz.keys())}"
            )

        y = np.asarray(split_npz[key]).astype(float)

        # English: If y is flattened story-level labels, infer four stories by grouping.
        # 中文：如果 y 是一维 story-level 标签，则按 4 层楼分组还原为 case-level。
        if y.ndim == 1:
            if y.size % 4 != 0:
                raise ValueError(
                    f"{key} is 1D with length {y.size}, which cannot be grouped into 4-story cases."
                )
            y = y.reshape(-1, 4)
        elif y.ndim == 2:
            if y.shape[1] == 1:
                # English: one output per case already.
                # 中文：如果本来就是每个 case 一个输出，作为单列处理。
                pass
            elif y.shape[1] < 2:
                raise ValueError(f"{key} has unexpected shape {y.shape}.")
        else:
            raise ValueError(f"{key} should be 1D or 2D, got shape {y.shape}.")

        out[split] = y

    return out


def align_features_and_labels(
    X_by_split: Dict[str, np.ndarray],
    y_story_by_split: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    English:
        Make sure feature rows and case-level labels are aligned.
    中文：
        确保特征矩阵行数与 case-level 标签行数一致。
    """
    X_aligned: Dict[str, np.ndarray] = {}
    y_case: Dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        X = X_by_split[split]
        y_story = y_story_by_split[split]

        n_cases = y_story.shape[0]
        if X.shape[0] == n_cases:
            X_case = X
        elif X.shape[0] == n_cases * 4:
            # English:
            #   Some earlier scripts expanded case-level features to story-level rows.
            #   Here we collapse by taking every 4-row block mean and then remove
            #   possible story-one-hot columns if they exist.
            # 中文：
            #   有些早期脚本会将 case-level 特征扩展成 story-level。
            #   这里按每 4 行求均值还原到 case-level，并自动保留普通特征。
            X_case = X.reshape(n_cases, 4, X.shape[1]).mean(axis=1)
        else:
            raise ValueError(
                f"Cannot align {split}: feature rows={X.shape[0]}, label cases={n_cases}. "
                f"Expected {n_cases} or {n_cases * 4}."
            )

        X_aligned[split] = X_case
        y_case[split] = (np.max(y_story, axis=1) > 1e-12).astype(int)

    return X_aligned, y_case


def build_classifiers(seed: int) -> Dict[str, object]:
    """
    English: Build several simple classifiers for diagnosis.
    中文：构建几个简单分类器用于诊断。
    """
    return {
        "logistic_l2_balanced": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def predict_proba_positive(model: object, X: np.ndarray) -> np.ndarray:
    """
    English: Return positive-class probability if available.
    中文：返回正类概率。
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return np.asarray(proba).reshape(-1)

    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-score))

    pred = np.asarray(model.predict(X), dtype=float)
    return pred


def safe_auc(y_true: np.ndarray, score: np.ndarray, kind: str) -> float:
    """English: Compute AUC safely. 中文：安全计算 AUC，避免单类标签报错。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if kind == "roc":
        return float(roc_auc_score(y_true, score))
    if kind == "pr":
        return float(average_precision_score(y_true, score))
    raise ValueError(kind)


def binary_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    split: str,
    model_name: str,
) -> Dict[str, float]:
    """
    English: Compute binary classification metrics.
    中文：计算二分类指标。
    """
    pred = (score >= threshold).astype(int)

    zero_mask = y_true == 0
    damaged_mask = y_true == 1

    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "n": int(len(y_true)),
        "zero_n": int(zero_mask.sum()),
        "damaged_n": int(damaged_mask.sum()),
        "roc_auc": safe_auc(y_true, score, "roc"),
        "pr_auc": safe_auc(y_true, score, "pr"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "zero_false_alarm_ratio": float(pred[zero_mask].mean()) if zero_mask.any() else float("nan"),
        "damaged_miss_ratio": float((1 - pred[damaged_mask]).mean()) if damaged_mask.any() else float("nan"),
        "mean_proba_zero": float(score[zero_mask].mean()) if zero_mask.any() else float("nan"),
        "mean_proba_damaged": float(score[damaged_mask].mean()) if damaged_mask.any() else float("nan"),
    }


def select_threshold_on_val(
    y_val: np.ndarray,
    score_val: np.ndarray,
    thresholds: Sequence[float],
    max_zero_false_alarm: float,
) -> Tuple[float, str]:
    """
    English:
        Select threshold on validation data.
        Priority:
            1. zero false alarm <= max_zero_false_alarm
            2. highest damaged recall
            3. highest F1

    中文：
        在验证集上选择阈值。
        优先级：
            1. 零损伤误报率不超过上限
            2. damaged recall 尽可能高
            3. F1 尽可能高
    """
    rows = []
    for t in thresholds:
        m = binary_metrics(y_val, score_val, t, split="val", model_name="threshold_search")
        rows.append(m)

    df = pd.DataFrame(rows)
    feasible = df[df["zero_false_alarm_ratio"] <= max_zero_false_alarm].copy()

    if not feasible.empty:
        feasible = feasible.sort_values(
            by=["recall", "f1", "balanced_accuracy"],
            ascending=[False, False, False],
        )
        return float(feasible.iloc[0]["threshold"]), "feasible_under_false_alarm_limit"

    # English: If no threshold satisfies the false alarm limit, choose the lowest false alarm.
    # 中文：如果没有任何阈值满足误报限制，则选误报率最低者。
    df = df.sort_values(
        by=["zero_false_alarm_ratio", "recall", "f1"],
        ascending=[True, False, False],
    )
    return float(df.iloc[0]["threshold"]), "no_feasible_threshold_choose_lowest_false_alarm"


def plot_roc_pr(
    y_test: np.ndarray,
    score_by_model: Dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    """English: Plot ROC and PR curves. 中文：绘制 ROC 和 PR 曲线。"""
    plt.figure(figsize=(8, 6))
    for model_name, score in score_by_model.items():
        if len(np.unique(y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test, score)
        auc = roc_auc_score(y_test, score)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Case-level any-damage ROC curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "case_level_any_damage_roc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_name, score in score_by_model.items():
        if len(np.unique(y_test)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_test, score)
        ap = average_precision_score(y_test, score)
        plt.plot(recall, precision, label=f"{model_name} AP={ap:.3f}")
    plt.xlabel("Damaged recall")
    plt.ylabel("Damaged precision")
    plt.title("Case-level any-damage precision-recall curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "case_level_any_damage_pr.png", dpi=200)
    plt.close()


def plot_probability_histogram(
    y_test: np.ndarray,
    score: np.ndarray,
    model_name: str,
    figures_dir: Path,
) -> None:
    """English: Plot probability histogram. 中文：绘制预测概率分布图。"""
    plt.figure(figsize=(8, 6))
    plt.hist(score[y_test == 0], bins=20, alpha=0.6, label="true zero")
    plt.hist(score[y_test == 1], bins=20, alpha=0.6, label="true damaged")
    plt.xlabel("Predicted damaged probability")
    plt.ylabel("Count")
    plt.title(f"Case-level probability histogram: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"case_level_probability_histogram_{model_name}.png", dpi=200)
    plt.close()


def plot_threshold_tradeoff(
    y_test: np.ndarray,
    score: np.ndarray,
    model_name: str,
    thresholds: Sequence[float],
    figures_dir: Path,
) -> None:
    """English: Plot threshold tradeoff. 中文：绘制阈值权衡图。"""
    rows = []
    for t in thresholds:
        rows.append(binary_metrics(y_test, score, t, split="test", model_name=model_name))
    df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 6))
    plt.plot(df["threshold"], df["zero_false_alarm_ratio"], marker="o", label="zero false alarm")
    plt.plot(df["threshold"], df["recall"], marker="o", label="damaged recall")
    plt.plot(df["threshold"], df["f1"], marker="o", label="F1")
    plt.xlabel("Damaged probability threshold")
    plt.ylabel("Metric value")
    plt.title(f"Case-level threshold tradeoff: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"case_level_threshold_tradeoff_{model_name}.png", dpi=200)
    plt.close()


def plot_pca(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    figures_dir: Path,
) -> None:
    """
    English:
        Fit PCA on training features and plot test projection.
    中文：
        在训练集上拟合 PCA，并绘制测试集投影。
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_train_scaled)
    Z = pca.transform(X_test_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[y_test == 0, 0], Z[y_test == 0, 1], alpha=0.7, label="zero")
    plt.scatter(Z[y_test == 1, 0], Z[y_test == 1, 1], alpha=0.7, label="damaged")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title("Case-level PCA projection by any-damage label")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "case_level_pca_projection.png", dpi=200)
    plt.close()


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    feature_shape: Tuple[int, int],
    label_counts: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    best_model: str,
    interpretation: List[str],
) -> None:
    """English: Write markdown report. 中文：写出 markdown 诊断报告。"""
    lines = []
    lines.append("# Case-level Any-damage Diagnosis")
    lines.append("")
    lines.append("## 1. Input files")
    lines.append(f"- Features: `{args.features}`")
    lines.append(f"- Split dataset: `{args.split}`")
    lines.append(f"- Feature names: `{args.feature_names}`")
    lines.append(f"- Feature shape example: `{feature_shape}`")
    lines.append("")
    lines.append("## 2. Case-level any-damage counts")
    lines.append(manual_markdown_table(label_counts))
    lines.append("")
    lines.append("## 3. Metrics at threshold 0.5")
    lines.append(manual_markdown_table(metrics_df.sort_values(["split", "pr_auc"], ascending=[True, False])))
    lines.append("")
    lines.append("## 4. Validation-selected threshold and test result")
    lines.append(manual_markdown_table(selected_df.sort_values("pr_auc", ascending=False)))
    lines.append("")
    lines.append("## 5. Best diagnostic model")
    lines.append(f"- Best model by test PR-AUC: `{best_model}`")
    lines.append("")
    lines.append("## 6. Preliminary interpretation")
    for item in interpretation:
        lines.append(f"- {item}")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path, help="Path to physics feature npz.")
    parser.add_argument("--split", required=True, type=Path, help="Path to split normalized dataset npz.")
    parser.add_argument("--feature-names", default=None, type=Path, help="Optional feature-name CSV.")
    parser.add_argument("--tag", default="debug_plus_500", help="Experiment tag.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output table/report folder.")
    parser.add_argument("--figures-dir", required=True, type=Path, help="Output figure folder.")
    parser.add_argument("--max-zero-false-alarm", type=float, default=0.05)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)

    features_npz = load_npz(args.features)
    split_npz = load_npz(args.split)

    X_raw = resolve_case_features(features_npz)
    y_story = resolve_story_damage(split_npz)
    X, y = align_features_and_labels(X_raw, y_story)

    feature_names = load_feature_names(args.feature_names, X["train"].shape[1])
    pd.DataFrame({"feature_index": np.arange(len(feature_names)), "feature_name": feature_names}).to_csv(
        args.output_dir / "resolved_feature_names.csv", index=False
    )

    label_counts = pd.DataFrame(
        [
            {
                "split": split,
                "zero_cases": int((y[split] == 0).sum()),
                "damaged_cases": int((y[split] == 1).sum()),
                "total_cases": int(len(y[split])),
                "damaged_ratio": float((y[split] == 1).mean()),
            }
            for split in ["train", "val", "test"]
        ]
    )
    label_counts.to_csv(args.output_dir / "case_level_label_counts.csv", index=False)

    models = build_classifiers(args.seed)

    metric_rows = []
    selected_rows = []
    score_test_by_model = {}

    for model_name, model in models.items():
        print(f"Training {model_name} ...")
        model.fit(X["train"], y["train"])

        score_by_split = {split: predict_proba_positive(model, X[split]) for split in ["train", "val", "test"]}
        score_test_by_model[model_name] = score_by_split["test"]

        for split in ["train", "val", "test"]:
            metric_rows.append(binary_metrics(y[split], score_by_split[split], 0.5, split, model_name))

        threshold, note = select_threshold_on_val(
            y_val=y["val"],
            score_val=score_by_split["val"],
            thresholds=args.thresholds,
            max_zero_false_alarm=args.max_zero_false_alarm,
        )
        row = binary_metrics(y["test"], score_by_split["test"], threshold, "test_selected_threshold", model_name)
        row["selected_threshold"] = threshold
        row["selection_note"] = note
        selected_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    selected_df = pd.DataFrame(selected_rows)

    metrics_df.to_csv(args.output_dir / "case_level_any_damage_metrics_threshold_0p5.csv", index=False)
    selected_df.to_csv(args.output_dir / "case_level_any_damage_selected_threshold_metrics.csv", index=False)

    best_model = (
        metrics_df[metrics_df["split"] == "test"]
        .sort_values(["pr_auc", "roc_auc", "f1"], ascending=[False, False, False])
        .iloc[0]["model"]
    )

    plot_roc_pr(y["test"], score_test_by_model, args.figures_dir)
    plot_probability_histogram(y["test"], score_test_by_model[best_model], best_model, args.figures_dir)
    plot_threshold_tradeoff(y["test"], score_test_by_model[best_model], best_model, args.thresholds, args.figures_dir)
    plot_pca(X["train"], y["train"], X["test"], y["test"], args.figures_dir)

    best_test = metrics_df[(metrics_df["split"] == "test") & (metrics_df["model"] == best_model)].iloc[0]
    best_selected = selected_df[selected_df["model"] == best_model].iloc[0]

    interpretation = []
    if best_test["roc_auc"] < 0.65:
        interpretation.append(
            "Case-level any-damage separability is weak: the best test ROC-AUC is below 0.65."
        )
    else:
        interpretation.append(
            "Case-level any-damage separability is meaningfully stronger than the previous story-level gate."
        )

    if best_selected["recall"] < 0.3:
        interpretation.append(
            "Under a low false-alarm operating point, damaged recall is still poor; feature separability is likely insufficient."
        )
    else:
        interpretation.append(
            "Under a low false-alarm operating point, damaged recall is usable; story-local alignment should be investigated next."
        )

    interpretation.append(
        "Compare this result with the previous story-level zero-vs-damaged result to decide whether the bottleneck is global damage detection or story localization."
    )

    write_report(
        output_path=args.output_dir / "case_level_any_damage_report.md",
        args=args,
        feature_shape=X["train"].shape,
        label_counts=label_counts,
        metrics_df=metrics_df,
        selected_df=selected_df,
        best_model=str(best_model),
        interpretation=interpretation,
    )

    summary = {
        "tag": args.tag,
        "best_model_by_test_pr_auc": str(best_model),
        "best_test_roc_auc": float(best_test["roc_auc"]),
        "best_test_pr_auc": float(best_test["pr_auc"]),
        "best_test_zero_false_alarm_at_0p5": float(best_test["zero_false_alarm_ratio"]),
        "best_test_recall_at_0p5": float(best_test["recall"]),
        "selected_threshold_for_best_model": float(best_selected["selected_threshold"]),
        "best_selected_zero_false_alarm": float(best_selected["zero_false_alarm_ratio"]),
        "best_selected_recall": float(best_selected["recall"]),
    }
    (args.output_dir / "case_level_any_damage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nCase-level any-damage diagnosis completed.")
    print(f"Report: {args.output_dir / 'case_level_any_damage_report.md'}")
    print(f"Metrics: {args.output_dir / 'case_level_any_damage_metrics_threshold_0p5.csv'}")
    print(f"Selected-threshold metrics: {args.output_dir / 'case_level_any_damage_selected_threshold_metrics.csv'}")
    print(f"Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
