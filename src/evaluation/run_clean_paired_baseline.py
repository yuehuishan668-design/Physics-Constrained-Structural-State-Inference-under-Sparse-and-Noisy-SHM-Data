"""
Run a clean paired healthy-baseline experiment.

English:
This script rebuilds paired-baseline features using only clean input/ground-motion
features for healthy-neighbor matching. It excludes all story-response features
from the matching stage.

中文：
本脚本重新构建 paired healthy-baseline 特征。
核心区别是：只用纯输入/地震动特征匹配健康基准样本，
不再使用 story_* 结构响应特征做 baseline matching。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)

    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--figures-root", type=Path, required=True)

    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--healthy-threshold", type=float, default=1e-9)
    parser.add_argument("--damage-threshold", type=float, default=0.05)
    parser.add_argument("--high-threshold", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def get_first_existing(data: dict[str, np.ndarray], keys: list[str], name: str) -> np.ndarray:
    for key in keys:
        if key in data:
            return data[key]

    available = ", ".join(data.keys())
    raise KeyError(f"Cannot find {name}. Tried {keys}. Available keys: {available}")


def load_feature_matrix(feature_data: dict[str, np.ndarray], split_name: str) -> np.ndarray:
    keys = [
        f"F_{split_name}",
        f"X_{split_name}",
        f"features_{split_name}",
        f"physics_features_{split_name}",
    ]
    matrix = get_first_existing(feature_data, keys, f"{split_name} feature matrix")
    return np.asarray(matrix, dtype=float)


def load_target(
    feature_data: dict[str, np.ndarray],
    split_data: dict[str, np.ndarray],
    split_name: str,
) -> np.ndarray:
    candidate_keys = [
        f"y_{split_name}",
        f"y_damage_{split_name}",
        f"damage_{split_name}",
        f"Y_{split_name}",
    ]

    for key in candidate_keys:
        if key in feature_data:
            y = feature_data[key]
            return damage_to_scalar(y)

    for key in candidate_keys:
        if key in split_data:
            y = split_data[key]
            return damage_to_scalar(y)

    raise KeyError(f"Cannot find target y for split: {split_name}")


def damage_to_scalar(y: np.ndarray) -> np.ndarray:
    """
    English:
    Convert story-level damage labels into one scalar damage value per case.

    中文：
    如果 y 是多层结构损伤向量，例如每层一个损伤值，
    就取每个样本的最大损伤作为 case-level damage。
    """
    y = np.asarray(y, dtype=float)

    if y.ndim == 1:
        return y

    if y.ndim == 2:
        return np.max(y, axis=1)

    raise ValueError(f"Unsupported y shape: {y.shape}")


def read_feature_names(path: Path, expected_n: int) -> list[str]:
    df = pd.read_csv(path)

    preferred_columns = [
        "feature_name",
        "name",
        "features",
        "feature",
        "column",
        "columns",
    ]

    selected = None

    for col in preferred_columns:
        if col in df.columns:
            selected = df[col]
            break

    if selected is None:
        first_col = df.iloc[:, 0]

        # If first column is only an index, use the second column.
        # 如果第一列只是数字索引，则使用第二列作为特征名。
        if len(df.columns) >= 2:
            numeric_ratio = pd.to_numeric(first_col, errors="coerce").notna().mean()
            if numeric_ratio > 0.8:
                selected = df.iloc[:, 1]
            else:
                selected = first_col
        else:
            selected = first_col

    names = [str(x) for x in selected.tolist()]

    if len(names) != expected_n:
        raise ValueError(
            f"Feature-name length mismatch. "
            f"Expected {expected_n}, got {len(names)} from {path}"
        )

    return names


def find_clean_matching_indices(feature_names: list[str]) -> list[int]:
    """
    English:
    Clean matching means only external input / ground-motion descriptors.
    Any story_* response descriptor is excluded.

    中文：
    clean matching 只允许使用输入激励和地震动相关特征。
    所有 story_* 结构响应特征一律排除。
    """
    clean_indices: list[int] = []

    for idx, name in enumerate(feature_names):
        lower = name.lower()

        is_input_feature = (
            lower.startswith("ground_")
            or lower.startswith("input_")
            or "healthy_mode" in lower
            or "resonance_indicator" in lower
        )

        is_response_feature = lower.startswith("story_")

        if is_input_feature and not is_response_feature:
            clean_indices.append(idx)

    if not clean_indices:
        raise ValueError("No clean input/ground-motion matching features found.")

    return clean_indices


def fit_clean_neighbor_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    clean_indices: list[int],
    healthy_threshold: float,
    damage_threshold: float,
    k_neighbors: int,
) -> tuple[NearestNeighbors, StandardScaler, np.ndarray, list[str]]:
    healthy_mask = y_train <= healthy_threshold
    notes: list[str] = []

    if healthy_mask.sum() < k_neighbors:
        notes.append(
            f"Only {healthy_mask.sum()} strictly healthy samples found. "
            f"Relaxing healthy threshold to damage_threshold={damage_threshold}."
        )
        healthy_mask = y_train <= damage_threshold

    if healthy_mask.sum() < 2:
        raise ValueError(
            "Too few healthy samples for paired baseline matching. "
            "Check damage labels or thresholds."
        )

    healthy_indices = np.where(healthy_mask)[0]

    scaler = StandardScaler()
    scaler.fit(X_train[:, clean_indices])

    X_train_clean_scaled = scaler.transform(X_train[:, clean_indices])
    X_healthy_clean_scaled = X_train_clean_scaled[healthy_indices]

    k_effective = min(k_neighbors + 1, len(healthy_indices))

    neighbor_model = NearestNeighbors(
        n_neighbors=k_effective,
        metric="euclidean",
    )
    neighbor_model.fit(X_healthy_clean_scaled)

    return neighbor_model, scaler, healthy_indices, notes


def build_clean_paired_features(
    X: np.ndarray,
    X_train: np.ndarray,
    clean_indices: list[int],
    scaler: StandardScaler,
    neighbor_model: NearestNeighbors,
    healthy_train_indices: np.ndarray,
    k_neighbors: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    English:
    For each sample, find its nearest healthy training cases using only clean
    input/ground-motion features. Then compute full-feature residuals relative
    to those healthy neighbors.

    中文：
    对每个样本，只用 clean input 特征寻找最相近的健康训练样本；
    然后用当前样本的完整物理特征减去健康邻居的平均完整特征，
    得到 paired residual features。
    """
    X_clean_scaled = scaler.transform(X[:, clean_indices])
    distances, neighbor_pos = neighbor_model.kneighbors(X_clean_scaled)

    baselines = []

    for row_i in range(X.shape[0]):
        selected_healthy_positions = neighbor_pos[row_i]
        selected_train_indices = healthy_train_indices[selected_healthy_positions]

        # For training split, avoid using the exact same healthy sample as its own baseline when possible.
        # 对训练集中的健康样本，尽量避免把自己作为自己的健康基准。
        if split_name == "train":
            selected_train_indices = selected_train_indices[selected_train_indices != row_i]

        if len(selected_train_indices) == 0:
            selected_train_indices = healthy_train_indices[neighbor_pos[row_i][:1]]
        else:
            selected_train_indices = selected_train_indices[:k_neighbors]

        baseline = X_train[selected_train_indices].mean(axis=0)
        baselines.append(baseline)

    baseline_matrix = np.vstack(baselines)

    residual = X - baseline_matrix
    abs_residual = np.abs(residual)

    paired_features = np.concatenate(
        [
            X,
            residual,
            abs_residual,
        ],
        axis=1,
    )

    return paired_features, baseline_matrix


def make_paired_feature_names(original_names: list[str]) -> list[str]:
    names = []
    names.extend([f"raw__{name}" for name in original_names])
    names.extend([f"clean_residual__{name}" for name in original_names])
    names.extend([f"clean_abs_residual__{name}" for name in original_names])
    return names


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def train_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    high_threshold: float,
    random_state: int,
) -> pd.DataFrame:
    models = {
        "ridge_alpha_1": Ridge(alpha=1.0),
        "ridge_alpha_10": Ridge(alpha=10.0),
        "ridge_alpha_100": Ridge(alpha=100.0),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    rows = []

    high_mask = y_test >= high_threshold

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0.0, 0.5)

        row = {
            "model": name,
            "test_mae": mean_absolute_error(y_test, pred),
            "test_rmse": rmse(y_test, pred),
            "mean_true_damage": float(np.mean(y_test)),
            "mean_pred_damage": float(np.mean(pred)),
            "bias_pred_minus_true": float(np.mean(pred - y_test)),
        }

        if high_mask.sum() > 0:
            high_pred = pred[high_mask]
            high_true = y_test[high_mask]
            row["high_n"] = int(high_mask.sum())
            row["high_mae"] = mean_absolute_error(high_true, high_pred)
            row["high_bias_pred_minus_true"] = float(np.mean(high_pred - high_true))
            row["high_underestimation_ratio"] = float(np.mean(high_pred < high_true))
        else:
            row["high_n"] = 0
            row["high_mae"] = np.nan
            row["high_bias_pred_minus_true"] = np.nan
            row["high_underestimation_ratio"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("test_mae")


def train_classifiers(
    X_train: np.ndarray,
    y_train_scalar: np.ndarray,
    X_test: np.ndarray,
    y_test_scalar: np.ndarray,
    damage_threshold: float,
    random_state: int,
) -> pd.DataFrame:
    y_train_bin = (y_train_scalar > damage_threshold).astype(int)
    y_test_bin = (y_test_scalar > damage_threshold).astype(int)

    models = {
        "logistic_l2_balanced": LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
            random_state=random_state,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train_bin)

        proba = model.predict_proba(X_test)[:, 1]
        pred_bin = (proba >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_bin,
            pred_bin,
            average="binary",
            zero_division=0,
        )

        cm = confusion_matrix(y_test_bin, pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        zero_false_alarm_ratio = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        damaged_miss_ratio = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test_bin, proba),
                "pr_auc": average_precision_score(y_test_bin, proba),
                "balanced_accuracy": balanced_accuracy_score(y_test_bin, pred_bin),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "zero_false_alarm_ratio": zero_false_alarm_ratio,
                "damaged_miss_ratio": damaged_miss_ratio,
                "mean_proba_zero": float(np.mean(proba[y_test_bin == 0])) if np.any(y_test_bin == 0) else np.nan,
                "mean_proba_damaged": float(np.mean(proba[y_test_bin == 1])) if np.any(y_test_bin == 1) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["zero", "damaged"])
    ax.set_yticklabels(["zero", "damaged"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.75)

    max_value = max(float(np.max(y_true)), float(np.max(y_pred)), 0.5)
    ax.plot([0, max_value], [0, max_value], linestyle="--", label="Ideal y=x")

    ax.axvline(0.05, linestyle=":", linewidth=1)
    ax.axvline(0.20, linestyle=":", linewidth=1)
    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.axhline(0.20, linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("True damage")
    ax.set_ylabel("Predicted damage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_pr(
    classifier_models: dict[str, object],
    X_test: np.ndarray,
    y_test_bin: np.ndarray,
    output_roc: Path,
    output_pr: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in classifier_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_bin, proba)
        auc = roc_auc_score(y_test_bin, proba)
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], linestyle="--", label="random")
    ax.set_title("Clean paired baseline zero-vs-damaged ROC")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_roc, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in classifier_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test_bin, proba)
        ap = average_precision_score(y_test_bin, proba)
        ax.plot(recall, precision, label=f"{name} AP={ap:.3f}")

    ax.set_title("Clean paired baseline zero-vs-damaged PR")
    ax.set_xlabel("Damaged recall")
    ax.set_ylabel("Damaged precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_pr, dpi=200)
    plt.close(fig)


def plot_probability_histogram(
    proba: np.ndarray,
    y_test_bin: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(proba[y_test_bin == 0], bins=15, alpha=0.6, label="true zero")
    ax.hist(proba[y_test_bin == 1], bins=15, alpha=0.6, label="true damaged")

    ax.set_title(title)
    ax.set_xlabel("Predicted damaged probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pca(
    X_test: np.ndarray,
    y_test_scalar: np.ndarray,
    damage_threshold: float,
    output_path: Path,
) -> None:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X_scaled)

    y_bin = y_test_scalar > damage_threshold

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(z[~y_bin, 0], z[~y_bin, 1], alpha=0.75, label="zero")
    ax.scatter(z[y_bin, 0], z[y_bin, 1], alpha=0.75, label="damaged")

    ax.set_title("Clean paired baseline PCA projection")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    clean_names: list[str],
    notes: list[str],
    regression_df: pd.DataFrame,
    classifier_df: pd.DataFrame,
    clean_feature_count: int,
    paired_feature_count: int,
    healthy_train_count: int,
) -> None:
    best_reg = regression_df.iloc[0]
    best_cls = classifier_df.iloc[0]

    lines = []

    lines.append("# Clean Paired Healthy-baseline Diagnosis")
    lines.append("")
    lines.append("## 1. Configuration")
    lines.append(f"- features: `{args.features}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- feature_names: `{args.feature_names}`")
    lines.append(f"- k_neighbors: `{args.k_neighbors}`")
    lines.append(f"- healthy_threshold: `{args.healthy_threshold}`")
    lines.append(f"- damage_threshold: `{args.damage_threshold}`")
    lines.append(f"- high_threshold: `{args.high_threshold}`")
    lines.append("")
    lines.append("## 2. Clean matching features")
    lines.append(f"- Number of clean matching features: `{clean_feature_count}`")
    lines.append(f"- Number of paired output features: `{paired_feature_count}`")
    lines.append(f"- Number of healthy training samples used for matching: `{healthy_train_count}`")
    lines.append("")

    for name in clean_names:
        lines.append(f"  - `{name}`")

    lines.append("")
    lines.append("## 3. Notes")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No warnings.")
    lines.append("")
    lines.append("## 4. Best regression result")
    for key, value in best_reg.items():
        lines.append(f"- {key}: `{value}`")

    lines.append("")
    lines.append("## 5. Regression table")
    lines.append(regression_df.to_csv(index=False))

    lines.append("")
    lines.append("## 6. Best zero-vs-damaged classifier result")
    for key, value in best_cls.items():
        lines.append(f"- {key}: `{value}`")

    lines.append("")
    lines.append("## 7. Classifier table")
    lines.append(classifier_df.to_csv(index=False))

    lines.append("")
    lines.append("## 8. Preliminary interpretation")
    lines.append(
        "- If clean matching performs similarly to or better than dirty paired matching, "
        "healthy-baseline normalization is genuinely useful."
    )
    lines.append(
        "- If clean matching becomes worse, the previous paired-baseline improvement was partly driven "
        "by response-feature contamination in the matching stage."
    )
    lines.append(
        "- If regression high-damage underestimation remains severe, the bottleneck is still damage-sensitive feature construction."
    )
    lines.append(
        "- If zero false alarm remains high, the zero/damaged decision boundary remains weak and needs better discriminative descriptors."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.figures_root.mkdir(parents=True, exist_ok=True)

    feature_data = load_npz(args.features)
    split_data = load_npz(args.split)

    X_train = load_feature_matrix(feature_data, "train")
    X_val = load_feature_matrix(feature_data, "val")
    X_test = load_feature_matrix(feature_data, "test")

    y_train = load_target(feature_data, split_data, "train")
    y_val = load_target(feature_data, split_data, "val")
    y_test = load_target(feature_data, split_data, "test")

    feature_names = read_feature_names(args.feature_names, expected_n=X_train.shape[1])
    clean_indices = find_clean_matching_indices(feature_names)
    clean_names = [feature_names[i] for i in clean_indices]

    neighbor_model, scaler, healthy_train_indices, notes = fit_clean_neighbor_model(
        X_train=X_train,
        y_train=y_train,
        clean_indices=clean_indices,
        healthy_threshold=args.healthy_threshold,
        damage_threshold=args.damage_threshold,
        k_neighbors=args.k_neighbors,
    )

    F_train, baseline_train = build_clean_paired_features(
        X=X_train,
        X_train=X_train,
        clean_indices=clean_indices,
        scaler=scaler,
        neighbor_model=neighbor_model,
        healthy_train_indices=healthy_train_indices,
        k_neighbors=args.k_neighbors,
        split_name="train",
    )

    F_val, baseline_val = build_clean_paired_features(
        X=X_val,
        X_train=X_train,
        clean_indices=clean_indices,
        scaler=scaler,
        neighbor_model=neighbor_model,
        healthy_train_indices=healthy_train_indices,
        k_neighbors=args.k_neighbors,
        split_name="val",
    )

    F_test, baseline_test = build_clean_paired_features(
        X=X_test,
        X_train=X_train,
        clean_indices=clean_indices,
        scaler=scaler,
        neighbor_model=neighbor_model,
        healthy_train_indices=healthy_train_indices,
        k_neighbors=args.k_neighbors,
        split_name="test",
    )

    paired_names = make_paired_feature_names(feature_names)

    paired_npz_path = args.output_root / "clean_paired_baseline_features.npz"
    np.savez_compressed(
        paired_npz_path,
        F_train=F_train,
        F_val=F_val,
        F_test=F_test,
        X_train=F_train,
        X_val=F_val,
        X_test=F_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        baseline_train=baseline_train,
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        clean_matching_indices=np.asarray(clean_indices),
        healthy_train_indices=healthy_train_indices,
    )

    pd.DataFrame({"feature_name": paired_names}).to_csv(
        args.output_root / "clean_paired_baseline_feature_names.csv",
        index=False,
    )

    pd.DataFrame({"clean_matching_feature": clean_names}).to_csv(
        args.output_root / "clean_matching_features.csv",
        index=False,
    )

    regression_df = train_regressors(
        X_train=F_train,
        y_train=y_train,
        X_test=F_test,
        y_test=y_test,
        high_threshold=args.high_threshold,
        random_state=args.random_state,
    )

    classifier_df = train_classifiers(
        X_train=F_train,
        y_train_scalar=y_train,
        X_test=F_test,
        y_test_scalar=y_test,
        damage_threshold=args.damage_threshold,
        random_state=args.random_state,
    )

    regression_df.to_csv(args.output_root / "clean_paired_regression_results.csv", index=False)
    classifier_df.to_csv(args.output_root / "clean_paired_classifier_results.csv", index=False)

    best_reg_name = regression_df.iloc[0]["model"]

    reg_models = {
        "ridge_alpha_1": Ridge(alpha=1.0),
        "ridge_alpha_10": Ridge(alpha=10.0),
        "ridge_alpha_100": Ridge(alpha=100.0),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=-1,
        ),
    }

    best_reg_model = reg_models[best_reg_name]
    best_reg_model.fit(F_train, y_train)
    best_reg_pred = np.clip(best_reg_model.predict(F_test), 0.0, 0.5)

    plot_regression_scatter(
        y_true=y_test,
        y_pred=best_reg_pred,
        title=f"Clean paired baseline regression: {best_reg_name}",
        output_path=args.figures_root / "clean_paired_best_regression_scatter.png",
    )

    y_test_bin = (y_test > args.damage_threshold).astype(int)

    classifier_models = {
        "logistic_l2_balanced": LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
            random_state=args.random_state,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=-1,
        ),
    }

    for model in classifier_models.values():
        model.fit(F_train, (y_train > args.damage_threshold).astype(int))

    best_cls_name = classifier_df.iloc[0]["model"]
    best_cls_model = classifier_models[best_cls_name]
    best_proba = best_cls_model.predict_proba(F_test)[:, 1]
    best_pred_bin = (best_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test_bin, best_pred_bin, labels=[0, 1])

    plot_confusion_matrix(
        cm=cm,
        title=f"Clean paired baseline confusion matrix: {best_cls_name}",
        output_path=args.figures_root / "clean_paired_best_classifier_confusion.png",
    )

    plot_probability_histogram(
        proba=best_proba,
        y_test_bin=y_test_bin,
        title=f"Clean paired baseline probability histogram: {best_cls_name}",
        output_path=args.figures_root / "clean_paired_best_classifier_probability_histogram.png",
    )

    plot_roc_pr(
        classifier_models=classifier_models,
        X_test=F_test,
        y_test_bin=y_test_bin,
        output_roc=args.figures_root / "clean_paired_zero_vs_damaged_roc.png",
        output_pr=args.figures_root / "clean_paired_zero_vs_damaged_pr.png",
    )

    plot_pca(
        X_test=F_test,
        y_test_scalar=y_test,
        damage_threshold=args.damage_threshold,
        output_path=args.figures_root / "clean_paired_pca_projection.png",
    )

    write_report(
        output_path=args.output_root / "clean_paired_baseline_report.md",
        args=args,
        clean_names=clean_names,
        notes=notes,
        regression_df=regression_df,
        classifier_df=classifier_df,
        clean_feature_count=len(clean_indices),
        paired_feature_count=F_train.shape[1],
        healthy_train_count=len(healthy_train_indices),
    )

    print("Clean paired baseline experiment finished.")
    print(f"Paired features saved to: {paired_npz_path}")
    print(f"Report saved to: {args.output_root / 'clean_paired_baseline_report.md'}")
    print(f"Figures saved to: {args.figures_root}")


if __name__ == "__main__":
    main()
    