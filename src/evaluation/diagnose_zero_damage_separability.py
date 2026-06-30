#!/usr/bin/env python3
"""
Zero-vs-damaged feature separability diagnosis.

Purpose
-------
This script diagnoses whether the current physics-informed feature set can
separate zero-damage samples from nonzero-damage samples.

The script is designed for the current project data structure:
- physics feature NPZ files may store case-level features such as F_train/F_val/F_test.
- damage labels may be story-level arrays such as y_train/y_val/y_test with shape
  (n_cases, n_stories).
- if features are case-level and labels are story-level, the script expands each
  case-level feature vector to story-level entries and appends story one-hot features.

Outputs
-------
Tables:
- classifier_metrics.csv
- threshold_sensitivity.csv
- selected_thresholds.csv
- feature_effect_size_zero_vs_damaged.csv
- top_features_extra_trees.csv
- misclassified_damaged_samples.csv
- zero_damage_separability_report.md

Figures:
- probability_histogram_<model>.png
- roc_curve.png
- pr_curve.png
- threshold_tradeoff_<model>.png
- pca_damage_bins.png
- top_feature_importance.png
- top_effect_size_features.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
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


SPLITS = ("train", "val", "test")


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: object, default: float = float("nan")) -> float:
    """Convert a value to float and return default if conversion fails."""
    try:
        return float(value)
    except Exception:
        return default


def read_feature_names(path: Path, n_base_features: int) -> List[str]:
    """Read feature names from CSV with flexible column handling."""
    if not path.exists():
        return [f"feature_{i:03d}" for i in range(n_base_features)]

    df = pd.read_csv(path)
    if df.empty:
        return [f"feature_{i:03d}" for i in range(n_base_features)]

    preferred_cols = [
        "feature_name",
        "name",
        "feature",
        "features",
        "column",
        "column_name",
    ]
    col = None
    for c in preferred_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        col = df.columns[0]

    names = df[col].astype(str).tolist()
    names = [x for x in names if x and x.lower() != "nan"]

    if len(names) < n_base_features:
        names += [f"feature_{i:03d}" for i in range(len(names), n_base_features)]
    if len(names) > n_base_features:
        names = names[:n_base_features]
    return names


def pick_first_existing_key(npz: np.lib.npyio.NpzFile, candidates: Iterable[str]) -> Optional[str]:
    """Return the first key that exists in an NPZ file."""
    keys = set(npz.files)
    for key in candidates:
        if key in keys:
            return key
    return None


def find_feature_key(npz: np.lib.npyio.NpzFile, split: str) -> Optional[str]:
    """Find a plausible 2D feature matrix key for a split."""
    candidates = [
        f"F_{split}",
        f"features_{split}",
        f"feature_{split}",
        f"X_features_{split}",
        f"X_{split}",
        f"x_{split}",
    ]
    for key in candidates:
        if key in npz.files:
            arr = np.asarray(npz[key])
            if arr.ndim == 2:
                return key
    return None


def find_label_key(npz: np.lib.npyio.NpzFile, split: str) -> Optional[str]:
    """Find a plausible damage-label key for a split."""
    candidates = [
        f"y_{split}",
        f"Y_{split}",
        f"damage_{split}",
        f"damages_{split}",
        f"y_damage_{split}",
        f"target_{split}",
        f"targets_{split}",
    ]
    return pick_first_existing_key(npz, candidates)


def flatten_story_labels(y: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Flatten story-level labels and return optional story indices."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(float), None
    if y.ndim == 2:
        n_cases, n_stories = y.shape
        story_idx = np.tile(np.arange(n_stories), n_cases)
        return y.reshape(-1).astype(float), story_idx
    raise ValueError(f"Unsupported label array ndim={y.ndim}; expected 1D or 2D.")


def repeat_case_features_to_story_entries(
    X_case: np.ndarray,
    y_case_story: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Repeat case-level features to match story-level labels and add story one-hot columns.
    """
    n_cases, n_features = X_case.shape
    if y_case_story.ndim != 2:
        raise ValueError("Story expansion requires a 2D label array.")

    if y_case_story.shape[0] != n_cases:
        raise ValueError(
            f"Cannot expand features: X has {n_cases} cases but y has {y_case_story.shape[0]} cases."
        )

    n_stories = y_case_story.shape[1]
    X_repeated = np.repeat(X_case, repeats=n_stories, axis=0)
    story_idx = np.tile(np.arange(n_stories), n_cases)

    story_one_hot = np.zeros((n_cases * n_stories, n_stories), dtype=float)
    story_one_hot[np.arange(n_cases * n_stories), story_idx] = 1.0

    X_expanded = np.hstack([X_repeated, story_one_hot])
    y_flat = y_case_story.reshape(-1).astype(float)
    return X_expanded, y_flat, story_idx


def align_split(
    npz: np.lib.npyio.NpzFile,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    """
    Align feature rows with damage labels.

    Returns
    -------
    X_entries:
        Entry-level feature matrix.
    y_entries:
        Entry-level damage values.
    case_ids:
        Entry-level case IDs.
    story_idx:
        Optional entry-level story indices.
    expanded:
        Whether case-level features were expanded to story-level entries.
    """
    f_key = find_feature_key(npz, split)
    y_key = find_label_key(npz, split)

    if f_key is None:
        raise KeyError(f"Could not find a 2D feature matrix for split='{split}'.")
    if y_key is None:
        raise KeyError(f"Could not find a damage label array for split='{split}'.")

    X = np.asarray(npz[f_key], dtype=float)
    y_raw = np.asarray(npz[y_key], dtype=float)

    case_key = pick_first_existing_key(
        npz,
        [
            f"case_id_{split}",
            f"case_ids_{split}",
            f"case_{split}",
            f"cases_{split}",
            f"idx_{split}",
            f"indices_{split}",
        ],
    )
    raw_case_ids = np.asarray(npz[case_key]) if case_key is not None else None

    expanded = False
    story_idx = None

    # Case-level features with story-level labels: repeat features and add story one-hot.
    if y_raw.ndim == 2 and X.shape[0] == y_raw.shape[0]:
        X_entries, y_entries, story_idx = repeat_case_features_to_story_entries(X, y_raw)
        expanded = True
        if raw_case_ids is not None and raw_case_ids.shape[0] == y_raw.shape[0]:
            case_ids = np.repeat(raw_case_ids, repeats=y_raw.shape[1])
        else:
            case_ids = np.repeat(np.arange(y_raw.shape[0]), repeats=y_raw.shape[1])

    # Already entry-level features.
    elif y_raw.ndim == 2 and X.shape[0] == y_raw.size:
        y_entries, story_idx = flatten_story_labels(y_raw)
        X_entries = X
        if raw_case_ids is not None:
            if raw_case_ids.shape[0] == y_raw.shape[0]:
                case_ids = np.repeat(raw_case_ids, repeats=y_raw.shape[1])
            elif raw_case_ids.shape[0] == y_raw.size:
                case_ids = raw_case_ids
            else:
                case_ids = np.repeat(np.arange(y_raw.shape[0]), repeats=y_raw.shape[1])
        else:
            case_ids = np.repeat(np.arange(y_raw.shape[0]), repeats=y_raw.shape[1])

    elif y_raw.ndim == 1 and X.shape[0] == y_raw.shape[0]:
        X_entries = X
        y_entries = y_raw.astype(float)
        if raw_case_ids is not None and raw_case_ids.shape[0] == y_raw.shape[0]:
            case_ids = raw_case_ids
        else:
            case_ids = np.arange(y_raw.shape[0])

    else:
        raise ValueError(
            "Cannot align features and labels for split="
            f"{split}. X shape={X.shape}, y shape={y_raw.shape}."
        )

    return X_entries, y_entries, np.asarray(case_ids), story_idx, expanded


def load_aligned_dataset(
    features_path: Path,
    feature_names_path: Path,
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, object]]:
    """Load and align train/val/test data."""
    npz = np.load(features_path, allow_pickle=True)

    data: Dict[str, np.ndarray] = {}
    expanded_any = False
    base_feature_dim = None
    n_stories_used = None

    for split in SPLITS:
        X, y, case_ids, story_idx, expanded = align_split(npz, split)
        data[f"X_{split}"] = X
        data[f"y_{split}"] = y
        data[f"case_id_{split}"] = case_ids
        if story_idx is not None:
            data[f"story_idx_{split}"] = story_idx
            n_stories_used = int(np.max(story_idx)) + 1
        expanded_any = expanded_any or expanded
        if base_feature_dim is None:
            if expanded and n_stories_used is not None:
                base_feature_dim = X.shape[1] - n_stories_used
            else:
                base_feature_dim = X.shape[1]

    if base_feature_dim is None:
        base_feature_dim = data["X_train"].shape[1]

    feature_names = read_feature_names(feature_names_path, base_feature_dim)

    final_dim = data["X_train"].shape[1]
    if len(feature_names) < final_dim:
        extra = final_dim - len(feature_names)
        if expanded_any and n_stories_used == extra:
            feature_names += [f"story_is_{i + 1}" for i in range(extra)]
        else:
            feature_names += [f"added_feature_{i:03d}" for i in range(extra)]
    elif len(feature_names) > final_dim:
        feature_names = feature_names[:final_dim]

    meta = {
        "features_path": str(features_path),
        "feature_names_path": str(feature_names_path),
        "npz_keys": list(npz.files),
        "expanded_case_features_to_story_entries": expanded_any,
        "n_features": final_dim,
        "n_base_features": base_feature_dim,
        "n_stories": n_stories_used,
        "shapes": {
            split: {
                "X": list(data[f"X_{split}"].shape),
                "y": list(data[f"y_{split}"].shape),
            }
            for split in SPLITS
        },
    }

    return data, feature_names, meta


def make_damage_bins(y: np.ndarray, zero_eps: float, low_cut: float, medium_cut: float) -> np.ndarray:
    """Convert continuous damage values into zero/low/medium/high bins."""
    y = np.asarray(y, dtype=float)
    bins = np.empty(y.shape[0], dtype=object)
    bins[y <= zero_eps] = "zero"
    bins[(y > zero_eps) & (y < low_cut)] = "low"
    bins[(y >= low_cut) & (y < medium_cut)] = "medium"
    bins[y >= medium_cut] = "high"
    return bins.astype(str)


def build_models(random_state: int) -> Dict[str, object]:
    """Build zero-vs-damaged classifiers."""
    return {
        "logistic_l2_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=2,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def get_positive_probability(model: object, X: np.ndarray) -> np.ndarray:
    """Return probability for the positive damaged class."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=float)
    # Fallback for models exposing decision_function only.
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError("Model has neither predict_proba nor decision_function.")


def classifier_metric_row(
    model_name: str,
    split: str,
    y_true_binary: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    y_damage: np.ndarray,
    high_cut: float,
) -> Dict[str, object]:
    """Compute classifier metrics for one split."""
    pred = (proba >= threshold).astype(int)

    unique = np.unique(y_true_binary)
    roc_auc = roc_auc_score(y_true_binary, proba) if unique.size == 2 else float("nan")
    pr_auc = average_precision_score(y_true_binary, proba) if unique.size == 2 else float("nan")

    zero_mask = y_true_binary == 0
    damaged_mask = y_true_binary == 1
    high_mask = y_damage >= high_cut

    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "n": int(y_true_binary.size),
        "zero_n": int(np.sum(zero_mask)),
        "damaged_n": int(np.sum(damaged_mask)),
        "high_n": int(np.sum(high_mask)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": balanced_accuracy_score(y_true_binary, pred) if unique.size == 2 else float("nan"),
        "precision": precision_score(y_true_binary, pred, zero_division=0),
        "recall": recall_score(y_true_binary, pred, zero_division=0),
        "f1": f1_score(y_true_binary, pred, zero_division=0),
        "zero_false_alarm_ratio": float(np.mean(pred[zero_mask] == 1)) if np.any(zero_mask) else float("nan"),
        "damaged_miss_ratio": float(np.mean(pred[damaged_mask] == 0)) if np.any(damaged_mask) else float("nan"),
        "high_recall": float(np.mean(pred[high_mask] == 1)) if np.any(high_mask) else float("nan"),
        "mean_proba_zero": float(np.mean(proba[zero_mask])) if np.any(zero_mask) else float("nan"),
        "mean_proba_damaged": float(np.mean(proba[damaged_mask])) if np.any(damaged_mask) else float("nan"),
        "mean_proba_high": float(np.mean(proba[high_mask])) if np.any(high_mask) else float("nan"),
    }


def threshold_table(
    model_name: str,
    split: str,
    y_true_binary: np.ndarray,
    proba: np.ndarray,
    y_damage: np.ndarray,
    high_cut: float,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Create threshold-sensitivity table for one model and split."""
    rows = []
    for threshold in thresholds:
        rows.append(
            classifier_metric_row(
                model_name=model_name,
                split=split,
                y_true_binary=y_true_binary,
                proba=proba,
                threshold=float(threshold),
                y_damage=y_damage,
                high_cut=high_cut,
            )
        )
    return pd.DataFrame(rows)


def select_thresholds(
    threshold_df: pd.DataFrame,
    split_for_selection: str,
    max_zero_false_alarm: float,
) -> pd.DataFrame:
    """
    Select one operating threshold per model using validation data.

    Rule:
    1. zero_false_alarm_ratio <= max_zero_false_alarm
    2. maximize high_recall
    3. maximize damaged recall
    4. maximize PR-AUC
    5. choose lower threshold if still tied
    """
    rows = []
    val_df = threshold_df[threshold_df["split"] == split_for_selection].copy()

    for model_name, g in val_df.groupby("model"):
        feasible = g[g["zero_false_alarm_ratio"] <= max_zero_false_alarm].copy()
        if feasible.empty:
            candidate = g.sort_values(
                by=["zero_false_alarm_ratio", "high_recall", "recall", "pr_auc", "threshold"],
                ascending=[True, False, False, False, True],
            ).iloc[0].copy()
            candidate["selection_note"] = "no_feasible_threshold_under_false_alarm_limit"
        else:
            candidate = feasible.sort_values(
                by=["high_recall", "recall", "pr_auc", "threshold"],
                ascending=[False, False, False, True],
            ).iloc[0].copy()
            candidate["selection_note"] = "feasible_under_false_alarm_limit"
        rows.append(candidate)

    return pd.DataFrame(rows)


def calculate_feature_effects(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_damage: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute univariate zero-vs-damaged feature separability statistics.

    effect_size_d = abs(mean_damaged - mean_zero) / pooled_std
    corr_abs = absolute Pearson correlation with continuous damage
    """
    rows = []
    zero_mask = y_binary == 0
    damaged_mask = y_binary == 1

    for j, name in enumerate(feature_names):
        x = X[:, j].astype(float)
        x_zero = x[zero_mask]
        x_damaged = x[damaged_mask]

        mean_zero = np.nanmean(x_zero) if x_zero.size else float("nan")
        mean_damaged = np.nanmean(x_damaged) if x_damaged.size else float("nan")
        std_zero = np.nanstd(x_zero) if x_zero.size else float("nan")
        std_damaged = np.nanstd(x_damaged) if x_damaged.size else float("nan")

        pooled = math.sqrt((std_zero**2 + std_damaged**2) / 2.0) if np.isfinite(std_zero) and np.isfinite(std_damaged) else float("nan")
        if pooled > 1e-12:
            effect = abs(mean_damaged - mean_zero) / pooled
            signed_effect = (mean_damaged - mean_zero) / pooled
        else:
            effect = 0.0
            signed_effect = 0.0

        corr = float("nan")
        valid = np.isfinite(x) & np.isfinite(y_damage)
        if np.sum(valid) >= 3 and np.nanstd(x[valid]) > 1e-12 and np.nanstd(y_damage[valid]) > 1e-12:
            corr = float(np.corrcoef(x[valid], y_damage[valid])[0, 1])

        rows.append(
            {
                "feature_index": j,
                "feature_name": name,
                "mean_zero": mean_zero,
                "mean_damaged": mean_damaged,
                "std_zero": std_zero,
                "std_damaged": std_damaged,
                "signed_effect_size_d": signed_effect,
                "abs_effect_size_d": effect,
                "pearson_corr_with_damage": corr,
                "abs_pearson_corr_with_damage": abs(corr) if np.isfinite(corr) else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(
        by=["abs_effect_size_d", "abs_pearson_corr_with_damage"],
        ascending=[False, False],
    )


def extract_tree_importance(model: object, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Extract feature importance from a fitted tree-based pipeline."""
    clf = model.named_steps.get("clf") if isinstance(model, Pipeline) else model
    if not hasattr(clf, "feature_importances_"):
        return None
    imp = np.asarray(clf.feature_importances_, dtype=float)
    df = pd.DataFrame(
        {
            "feature_index": np.arange(len(imp)),
            "feature_name": feature_names[: len(imp)],
            "importance": imp,
        }
    )
    return df.sort_values("importance", ascending=False)


def write_markdown_table(df: pd.DataFrame, max_rows: int = 20, float_digits: int = 6) -> str:
    """Write a small DataFrame as Markdown without requiring tabulate."""
    if df is None or df.empty:
        return "_No rows._\n"

    d = df.head(max_rows).copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")

    columns = list(d.columns)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def plot_probability_histogram(
    y_binary: np.ndarray,
    proba: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot probability distributions for zero and damaged classes."""
    zero = proba[y_binary == 0]
    damaged = proba[y_binary == 1]
    plt.figure(figsize=(9, 6))
    bins = np.linspace(0.0, 1.0, 31)
    plt.hist(zero, bins=bins, alpha=0.55, label="true zero", density=False)
    plt.hist(damaged, bins=bins, alpha=0.55, label="true damaged", density=False)
    plt.xlabel("Predicted damaged probability")
    plt.ylabel("Count")
    plt.title(f"Zero-vs-damaged probability histogram: {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_pr_curves(
    probabilities: Dict[str, np.ndarray],
    y_binary: np.ndarray,
    roc_path: Path,
    pr_path: Path,
) -> None:
    """Plot ROC and PR curves for all models."""
    plt.figure(figsize=(8, 6))
    for model_name, proba in probabilities.items():
        if np.unique(y_binary).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_binary, proba)
        auc = roc_auc_score(y_binary, proba)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Zero-vs-damaged ROC curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_name, proba in probabilities.items():
        if np.unique(y_binary).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_binary, proba)
        ap = average_precision_score(y_binary, proba)
        plt.plot(recall, precision, label=f"{model_name} AP={ap:.3f}")
    plt.xlabel("Damaged recall")
    plt.ylabel("Damaged precision")
    plt.title("Zero-vs-damaged precision-recall curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()


def plot_threshold_tradeoff(
    threshold_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot zero false alarm and damaged/high recall across thresholds."""
    g = threshold_df[
        (threshold_df["model"] == model_name) & (threshold_df["split"] == "test")
    ].sort_values("threshold")

    plt.figure(figsize=(9, 6))
    plt.plot(g["threshold"], g["zero_false_alarm_ratio"], marker="o", label="zero false alarm")
    plt.plot(g["threshold"], g["recall"], marker="o", label="damaged recall")
    plt.plot(g["threshold"], g["high_recall"], marker="o", label="high-damage recall")
    plt.xlabel("Damaged probability threshold")
    plt.ylabel("Ratio")
    plt.title(f"Threshold tradeoff: {model_name}")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pca_bins(
    X_train: np.ndarray,
    X_test: np.ndarray,
    bins_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot PCA projection colored by damage bin."""
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=0)),
        ]
    )
    pipe.fit(X_train)
    Z = pipe.transform(X_test)

    plt.figure(figsize=(8, 7))
    order = ["zero", "low", "medium", "high"]
    for b in order:
        mask = bins_test == b
        if np.any(mask):
            plt.scatter(Z[mask, 0], Z[mask, 1], alpha=0.65, label=b)
    pca = pipe.named_steps["pca"]
    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    plt.title("PCA projection of physics features by damage bin")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bar(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    top_k: int,
) -> None:
    """Plot a horizontal bar chart for top-ranked features."""
    d = df.head(top_k).copy()
    d = d.iloc[::-1]
    plt.figure(figsize=(10, max(5, 0.32 * len(d))))
    plt.barh(d[label_col].astype(str), d[value_col].astype(float))
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_failure_table(
    split: str,
    y_damage: np.ndarray,
    y_binary: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    case_ids: np.ndarray,
    story_idx: Optional[np.ndarray],
    bins: np.ndarray,
) -> pd.DataFrame:
    """List damaged entries missed by the zero-vs-damaged classifier."""
    pred = (proba >= threshold).astype(int)
    miss = (y_binary == 1) & (pred == 0)

    df = pd.DataFrame(
        {
            "split": split,
            "entry_index": np.arange(y_damage.size),
            "case_id": case_ids,
            "story_index_zero_based": story_idx if story_idx is not None else np.full(y_damage.size, -1),
            "story": (story_idx + 1) if story_idx is not None else np.full(y_damage.size, -1),
            "true_damage": y_damage,
            "damage_bin": bins,
            "damaged_probability": proba,
            "threshold": threshold,
            "predicted_damaged": pred,
        }
    )
    return df[miss].sort_values(["damage_bin", "true_damage"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose zero-vs-damaged feature separability.")
    parser.add_argument("--features", required=True, type=Path, help="Physics feature NPZ file.")
    parser.add_argument("--feature-names", required=True, type=Path, help="Feature names CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for CSV/MD outputs.")
    parser.add_argument("--figures-dir", required=True, type=Path, help="Directory for figures.")
    parser.add_argument("--zero-eps", type=float, default=1e-8, help="Damage <= zero_eps is treated as zero.")
    parser.add_argument("--low-cut", type=float, default=0.1, help="Low/medium damage boundary.")
    parser.add_argument("--medium-cut", type=float, default=0.2, help="Medium/high damage boundary.")
    parser.add_argument("--max-zero-false-alarm", type=float, default=0.05, help="False alarm limit for selection.")
    parser.add_argument("--top-k", type=int, default=30, help="Top features/configurations to report.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)

    data, feature_names, meta = load_aligned_dataset(args.features, args.feature_names)

    X_train = data["X_train"]
    y_train_damage = data["y_train"]
    X_val = data["X_val"]
    y_val_damage = data["y_val"]
    X_test = data["X_test"]
    y_test_damage = data["y_test"]

    y_train_binary = (y_train_damage > args.zero_eps).astype(int)
    y_val_binary = (y_val_damage > args.zero_eps).astype(int)
    y_test_binary = (y_test_damage > args.zero_eps).astype(int)

    bins_train = make_damage_bins(y_train_damage, args.zero_eps, args.low_cut, args.medium_cut)
    bins_val = make_damage_bins(y_val_damage, args.zero_eps, args.low_cut, args.medium_cut)
    bins_test = make_damage_bins(y_test_damage, args.zero_eps, args.low_cut, args.medium_cut)

    models = build_models(args.random_state)
    thresholds = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])

    metric_rows = []
    threshold_tables = []
    test_probabilities = {}
    fitted_models = {}

    print("Loaded dataset:")
    print(json.dumps(meta["shapes"], indent=2))
    print(f"Number of features: {meta['n_features']}")

    for model_name, model in models.items():
        print(f"Training zero-vs-damaged classifier: {model_name}")
        model.fit(X_train, y_train_binary)
        fitted_models[model_name] = model

        split_data = {
            "train": (X_train, y_train_binary, y_train_damage),
            "val": (X_val, y_val_binary, y_val_damage),
            "test": (X_test, y_test_binary, y_test_damage),
        }

        for split, (X_split, y_binary, y_damage) in split_data.items():
            proba = get_positive_probability(model, X_split)
            if split == "test":
                test_probabilities[model_name] = proba
            metric_rows.append(
                classifier_metric_row(
                    model_name=model_name,
                    split=split,
                    y_true_binary=y_binary,
                    proba=proba,
                    threshold=0.5,
                    y_damage=y_damage,
                    high_cut=args.medium_cut,
                )
            )
            threshold_tables.append(
                threshold_table(
                    model_name=model_name,
                    split=split,
                    y_true_binary=y_binary,
                    proba=proba,
                    y_damage=y_damage,
                    high_cut=args.medium_cut,
                    thresholds=thresholds,
                )
            )

    metrics_df = pd.DataFrame(metric_rows)
    threshold_df = pd.concat(threshold_tables, ignore_index=True)

    selected_df = select_thresholds(
        threshold_df=threshold_df,
        split_for_selection="val",
        max_zero_false_alarm=args.max_zero_false_alarm,
    )

    # Add corresponding test rows for selected thresholds.
    selected_test_rows = []
    for _, row in selected_df.iterrows():
        model_name = row["model"]
        threshold = float(row["threshold"])
        proba = test_probabilities[model_name]
        selected_test_rows.append(
            classifier_metric_row(
                model_name=model_name,
                split="test_selected_threshold",
                y_true_binary=y_test_binary,
                proba=proba,
                threshold=threshold,
                y_damage=y_test_damage,
                high_cut=args.medium_cut,
            )
            | {
                "selected_from_val_threshold": threshold,
                "selection_note": row["selection_note"],
            }
        )
    selected_test_df = pd.DataFrame(selected_test_rows)

    metrics_df.to_csv(args.output_dir / "classifier_metrics.csv", index=False)
    threshold_df.to_csv(args.output_dir / "threshold_sensitivity.csv", index=False)
    selected_df.to_csv(args.output_dir / "selected_thresholds_from_validation.csv", index=False)
    selected_test_df.to_csv(args.output_dir / "selected_thresholds_test_metrics.csv", index=False)

    # Feature diagnostics.
    effect_df = calculate_feature_effects(X_train, y_train_binary, y_train_damage, feature_names)
    effect_df.to_csv(args.output_dir / "feature_effect_size_zero_vs_damaged.csv", index=False)

    et_model_name = "extra_trees_balanced"
    importance_df = extract_tree_importance(fitted_models[et_model_name], feature_names)
    if importance_df is not None:
        importance_df.to_csv(args.output_dir / "top_features_extra_trees.csv", index=False)

    # Select a main diagnostic model:
    # prefer highest test PR-AUC; this checks separability rather than operating-point conservatism.
    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    main_model_name = (
        test_metrics.sort_values(["pr_auc", "roc_auc"], ascending=False)["model"].iloc[0]
        if not test_metrics.empty
        else et_model_name
    )
    main_test_proba = test_probabilities[main_model_name]

    # Failure table under the validation-selected threshold of the main model.
    main_selection = selected_df[selected_df["model"] == main_model_name]
    if main_selection.empty:
        main_threshold = 0.5
    else:
        main_threshold = float(main_selection.iloc[0]["threshold"])

    failure_df = build_failure_table(
        split="test",
        y_damage=y_test_damage,
        y_binary=y_test_binary,
        proba=main_test_proba,
        threshold=main_threshold,
        case_ids=data["case_id_test"],
        story_idx=data.get("story_idx_test", None),
        bins=bins_test,
    )
    failure_df.to_csv(args.output_dir / "misclassified_damaged_samples.csv", index=False)

    # Figures.
    plot_probability_histogram(
        y_binary=y_test_binary,
        proba=main_test_proba,
        model_name=main_model_name,
        output_path=args.figures_dir / f"probability_histogram_{main_model_name}.png",
    )
    plot_roc_pr_curves(
        probabilities=test_probabilities,
        y_binary=y_test_binary,
        roc_path=args.figures_dir / "roc_curve.png",
        pr_path=args.figures_dir / "pr_curve.png",
    )
    plot_threshold_tradeoff(
        threshold_df=threshold_df,
        model_name=main_model_name,
        output_path=args.figures_dir / f"threshold_tradeoff_{main_model_name}.png",
    )
    plot_pca_bins(
        X_train=X_train,
        X_test=X_test,
        bins_test=bins_test,
        output_path=args.figures_dir / "pca_damage_bins.png",
    )
    if importance_df is not None:
        plot_bar(
            df=importance_df,
            value_col="importance",
            label_col="feature_name",
            title="Top ExtraTrees feature importance for zero-vs-damaged classification",
            xlabel="Feature importance",
            output_path=args.figures_dir / "top_feature_importance.png",
            top_k=min(args.top_k, 30),
        )
    plot_bar(
        df=effect_df,
        value_col="abs_effect_size_d",
        label_col="feature_name",
        title="Top feature effect sizes: zero vs damaged",
        xlabel="Absolute standardized mean difference",
        output_path=args.figures_dir / "top_effect_size_features.png",
        top_k=min(args.top_k, 30),
    )

    # Compact report.
    bin_counts = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "zero": [
                int(np.sum(bins_train == "zero")),
                int(np.sum(bins_val == "zero")),
                int(np.sum(bins_test == "zero")),
            ],
            "low": [
                int(np.sum(bins_train == "low")),
                int(np.sum(bins_val == "low")),
                int(np.sum(bins_test == "low")),
            ],
            "medium": [
                int(np.sum(bins_train == "medium")),
                int(np.sum(bins_val == "medium")),
                int(np.sum(bins_test == "medium")),
            ],
            "high": [
                int(np.sum(bins_train == "high")),
                int(np.sum(bins_val == "high")),
                int(np.sum(bins_test == "high")),
            ],
        }
    )

    main_metric = test_metrics[test_metrics["model"] == main_model_name].iloc[0].to_dict()
    selected_main_test = selected_test_df[selected_test_df["model"] == main_model_name]
    selected_main_test_row = selected_main_test.iloc[0].to_dict() if not selected_main_test.empty else {}

    report = []
    report.append("# Zero-vs-Damaged Feature Separability Diagnosis\n")
    report.append("## 1. Dataset alignment\n")
    report.append(f"- Features: `{args.features}`\n")
    report.append(f"- Feature names: `{args.feature_names}`\n")
    report.append(f"- Expanded case-level features to story-level entries: `{meta['expanded_case_features_to_story_entries']}`\n")
    report.append(f"- Number of final features: `{meta['n_features']}`\n")
    report.append(f"- Shapes: `{json.dumps(meta['shapes'])}`\n\n")

    report.append("## 2. Damage-bin counts\n")
    report.append(write_markdown_table(bin_counts, max_rows=10))
    report.append("\n")

    report.append("## 3. Classifier separability metrics at threshold 0.5\n")
    show_cols = [
        "model",
        "split",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "zero_false_alarm_ratio",
        "high_recall",
        "mean_proba_zero",
        "mean_proba_damaged",
        "mean_proba_high",
    ]
    report.append(write_markdown_table(metrics_df[show_cols].sort_values(["split", "pr_auc"], ascending=[True, False]), max_rows=30))
    report.append("\n")

    report.append("## 4. Main diagnostic model\n")
    report.append(f"- Main model selected by highest test PR-AUC: `{main_model_name}`\n")
    report.append(f"- Test ROC-AUC: `{safe_float(main_metric.get('roc_auc')):.6f}`\n")
    report.append(f"- Test PR-AUC: `{safe_float(main_metric.get('pr_auc')):.6f}`\n")
    report.append(f"- Test zero false alarm ratio at 0.5: `{safe_float(main_metric.get('zero_false_alarm_ratio')):.6f}`\n")
    report.append(f"- Test damaged recall at 0.5: `{safe_float(main_metric.get('recall')):.6f}`\n")
    report.append(f"- Test high-damage recall at 0.5: `{safe_float(main_metric.get('high_recall')):.6f}`\n\n")

    if selected_main_test_row:
        report.append("## 5. Validation-selected threshold for the main model\n")
        report.append(f"- Selected threshold: `{safe_float(selected_main_test_row.get('selected_from_val_threshold')):.6f}`\n")
        report.append(f"- Selection note: `{selected_main_test_row.get('selection_note')}`\n")
        report.append(f"- Test zero false alarm ratio: `{safe_float(selected_main_test_row.get('zero_false_alarm_ratio')):.6f}`\n")
        report.append(f"- Test damaged recall: `{safe_float(selected_main_test_row.get('recall')):.6f}`\n")
        report.append(f"- Test high-damage recall: `{safe_float(selected_main_test_row.get('high_recall')):.6f}`\n")
        report.append(f"- Missed damaged samples under selected threshold: `{len(failure_df)}`\n\n")

    report.append("## 6. Validation-selected thresholds for all models\n")
    report.append(write_markdown_table(selected_df, max_rows=20))
    report.append("\n")

    report.append("## 7. Test metrics under validation-selected thresholds\n")
    report.append(write_markdown_table(selected_test_df, max_rows=20))
    report.append("\n")

    report.append("## 8. Top feature effects: zero vs damaged\n")
    report.append(write_markdown_table(effect_df.head(args.top_k), max_rows=args.top_k))
    report.append("\n")

    if importance_df is not None:
        report.append("## 9. Top ExtraTrees feature importance\n")
        report.append(write_markdown_table(importance_df.head(args.top_k), max_rows=args.top_k))
        report.append("\n")

    report.append("## 10. Interpretation template\n")
    report.append(
        "- If ROC-AUC/PR-AUC is high but no operating threshold gives both low zero false alarm and high damaged recall, the bottleneck is threshold calibration or objective design.\n"
    )
    report.append(
        "- If ROC-AUC/PR-AUC is weak and probability histograms overlap heavily, the bottleneck is feature separability rather than model tuning.\n"
    )
    report.append(
        "- If PCA shows zero, low, medium and high samples heavily mixed, the current physics feature space is insufficient for robust zero/damaged transition detection.\n"
    )
    report.append(
        "- If high-damage samples have low damaged probability, the previous high-damage underestimation is caused partly by damaged-gate failure, not only by continuous regression bias.\n"
    )

    report_path = args.output_dir / "zero_damage_separability_report.md"
    report_path.write_text("".join(report), encoding="utf-8")

    print("\nDiagnosis completed.")
    print(f"Main diagnostic model: {main_model_name}")
    print(f"Report: {report_path}")
    print(f"Tables directory: {args.output_dir}")
    print(f"Figures directory: {args.figures_dir}")


if __name__ == "__main__":
    main()
