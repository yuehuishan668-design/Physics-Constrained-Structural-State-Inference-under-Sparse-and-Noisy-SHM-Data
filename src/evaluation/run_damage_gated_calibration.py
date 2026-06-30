"""
Two-stage damage-gated calibration experiment.

中文说明：
本文件执行“两阶段损伤门控校准”实验：
1) Stage 1：用物理特征预测 damage bin：zero / low / medium / high。
2) Stage 2：用基准 Ridge 预测值 + damage-bin 概率做回归校准。
目标是降低高损伤低估，同时控制零损伤误报。
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BIN_NAMES = ["zero", "low", "medium", "high"]
BIN_TO_ID = {name: i for i, name in enumerate(BIN_NAMES)}


@dataclass
class SplitData:
    """One dataset split. 中文：保存一个 train/val/test 划分。"""
    F: np.ndarray
    y: np.ndarray
    case_ids: Optional[np.ndarray] = None


@dataclass
class Result:
    """One experiment result. 中文：保存一个实验配置的结果。"""
    config_name: str
    feature_set: str
    classifier: str
    calibrator: str
    n_features: int
    y_true: np.ndarray
    y_base: np.ndarray
    y_pred: np.ndarray
    y_bin_true: np.ndarray
    y_bin_pred: np.ndarray
    metrics: Dict[str, float]


def first_key(npz, keys: Sequence[str]) -> Optional[str]:
    """Return first existing NPZ key. 中文：返回第一个存在的 npz 字段名。"""
    for key in keys:
        if key in npz:
            return key
    return None


def load_dataset(path: Path) -> Dict[str, SplitData]:
    """Load feature NPZ. 中文：读取物理特征 npz 数据集。"""
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    npz = np.load(path, allow_pickle=True)
    out: Dict[str, SplitData] = {}
    for split in ["train", "val", "test"]:
        f_key = first_key(npz, [f"F_{split}", f"features_{split}", f"X_features_{split}", f"X_{split}"])
        y_key = first_key(npz, [f"y_{split}", f"Y_{split}", f"damage_{split}", f"y_damage_{split}"])
        c_key = first_key(npz, [f"case_id_{split}", f"case_ids_{split}", f"{split}_case_ids"])
        if f_key is None or y_key is None:
            raise KeyError(f"Cannot locate {split} features/targets. Available keys: {list(npz.keys())}")
        F = np.asarray(npz[f_key], dtype=float)
        y = np.asarray(npz[y_key], dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if F.shape[0] != y.shape[0]:
            raise ValueError(f"Sample mismatch in {split}: F={F.shape}, y={y.shape}")
        case_ids = np.asarray(npz[c_key]) if c_key is not None else None
        out[split] = SplitData(F=F, y=y, case_ids=case_ids)
    return out


def load_feature_names(path: Optional[Path], n_features: int) -> List[str]:
    """Load feature names. 中文：读取特征名；失败则自动生成。"""
    if path is None or not path.exists():
        return [f"feature_{i}" for i in range(n_features)]
    df = pd.read_csv(path)
    if "feature_name" in df.columns:
        names = df["feature_name"].astype(str).tolist()
    elif df.shape[1] >= 2:
        names = df.iloc[:, 1].astype(str).tolist()
    else:
        names = df.iloc[:, 0].astype(str).tolist()
    if len(names) != n_features:
        print(f"[Warning] feature names {len(names)} != n_features {n_features}; fallback names used.")
        return [f"feature_{i}" for i in range(n_features)]
    return names


def build_feature_masks(names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Build feature-set masks. 中文：根据特征名构造不同特征组。"""
    n = len(names)
    names = list(names)
    full = np.ones(n, dtype=bool)
    meta_keys = ["input_", "ground_", "noise", "amplitude", "frequency_hz", "dt"]
    no_meta = np.array([not any(k in name for k in meta_keys) for name in names], dtype=bool)
    spatial_keys = ["story_", "spatial", "ratio", "correlation", "dominant_frequency", "spectral", "band_energy", "centroid"]
    response_spatial = np.array([any(k in name for k in spatial_keys) for name in names], dtype=bool) & no_meta
    basic_keys = ["_mean", "_std", "_max_abs", "_rms", "_peak_to_peak", "_crest_factor"]
    response_basic_only = np.array([("story_" in name) and any(k in name for k in basic_keys) for name in names], dtype=bool)
    response_correlation = np.array(["correlation" in name for name in names], dtype=bool)
    masks = {
        "full": full,
        "no_meta": no_meta,
        "response_spatial": response_spatial,
        "response_basic_only": response_basic_only,
        "response_correlation": response_correlation,
    }
    return {k: v for k, v in masks.items() if int(v.sum()) > 0}


def damage_to_bin(y: np.ndarray) -> np.ndarray:
    """Convert damage to bins. 中文：将连续损伤转为 zero/low/medium/high 编号。"""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=int)
    out[(y > 0.0) & (y < 0.1)] = BIN_TO_ID["low"]
    out[(y >= 0.1) & (y < 0.2)] = BIN_TO_ID["medium"]
    out[y >= 0.2] = BIN_TO_ID["high"]
    return out


def make_entry_features(F: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand case-level features into story-level entries. 中文：把 case 级特征展开为楼层级样本。"""
    Fs = F[:, mask]
    n_case = Fs.shape[0]
    n_story = y.shape[1]
    rep_F = np.repeat(Fs, repeats=n_story, axis=0)
    story_ids = np.tile(np.arange(n_story), reps=n_case)
    story_onehot = np.eye(n_story)[story_ids]
    X = np.concatenate([rep_F, story_onehot], axis=1)
    yy = y.reshape(-1)
    bb = damage_to_bin(yy)
    return X, yy, bb


def base_model(alpha: float = 0.1) -> Pipeline:
    """Baseline Ridge. 中文：基准 Ridge 回归模型。"""
    return Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=alpha))])


def classifier_model(name: str, seed: int):
    """Build bin classifier. 中文：构造损伤等级分类器。"""
    if name == "logistic_balanced":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed)),
        ])
    if name == "random_forest_balanced":
        return RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {name}")


def calibrator_model() -> Pipeline:
    """Second-stage calibrator. 中文：第二阶段 Ridge 校准器。"""
    return Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=0.1))])


def get_classes(model) -> np.ndarray:
    """Get class labels from classifier or pipeline. 中文：读取分类器类别标签。"""
    if hasattr(model, "classes_"):
        return model.classes_
    if hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return last.classes_
    raise AttributeError("Classifier has no classes_.")


def fixed_proba(model, X: np.ndarray, n_class: int = 4) -> np.ndarray:
    """Predict probabilities with fixed 4 columns. 中文：固定输出四类概率列。"""
    raw = model.predict_proba(X)
    classes = get_classes(model)
    out = np.zeros((X.shape[0], n_class), dtype=float)
    for j, c in enumerate(classes):
        if int(c) < n_class:
            out[:, int(c)] = raw[:, j]
    return out


def make_Z(X: np.ndarray, base_pred: np.ndarray, proba: np.ndarray, include_raw: bool) -> np.ndarray:
    """Build calibrator input. 中文：构造校准器输入。"""
    parts = [base_pred.reshape(-1, 1), proba]
    if include_raw:
        parts.append(X)
    return np.concatenate(parts, axis=1)


def residual_weights(y: np.ndarray, calibrator: str) -> Optional[np.ndarray]:
    """Sample weights for high-sensitive residual calibration. 中文：高损伤敏感残差校准权重。"""
    if calibrator != "high_sensitive_residual_ridge":
        return None
    w = np.ones_like(y, dtype=float)
    w[(y > 0.0) & (y < 0.1)] = 1.2
    w[(y >= 0.1) & (y < 0.2)] = 1.8
    w[y >= 0.2] = 3.0
    return w


def safe_mean(x: np.ndarray) -> float:
    """Safe mean. 中文：空数组返回 NaN。"""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Regression metrics. 中文：计算整体和分层回归指标。"""
    yt = y_true.reshape(-1).astype(float)
    yp = y_pred.reshape(-1).astype(float)
    err = yp - yt
    ae = np.abs(err)
    bins = damage_to_bin(yt)
    m: Dict[str, float] = {
        "test_mae": safe_mean(ae),
        "test_rmse": float(np.sqrt(np.mean(err ** 2))),
        "test_bias": safe_mean(err),
        "test_mean_true": safe_mean(yt),
        "test_mean_pred": safe_mean(yp),
        "test_n_entries": int(yt.size),
    }
    damaged = yt > 0
    zero = yt == 0
    m["test_damaged_mae"] = safe_mean(ae[damaged])
    m["test_damaged_bias"] = safe_mean(err[damaged])
    m["test_damaged_underestimation_ratio"] = safe_mean((err[damaged] < 0).astype(float))
    m["test_zero_mae"] = safe_mean(ae[zero])
    m["test_zero_false_alarm_ratio_005"] = safe_mean((yp[zero] > 0.05).astype(float))
    m["test_zero_false_alarm_ratio_010"] = safe_mean((yp[zero] > 0.10).astype(float))
    for name, idx in BIN_TO_ID.items():
        mask = bins == idx
        m[f"test_{name}_mae"] = safe_mean(ae[mask])
        m[f"test_{name}_bias"] = safe_mean(err[mask])
        m[f"test_{name}_underestimation_ratio"] = safe_mean((err[mask] < 0).astype(float))
        m[f"test_{name}_n"] = int(mask.sum())
    return m


def classification_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    """Classification metrics. 中文：计算分类指标。"""
    high = BIN_TO_ID["high"]
    true_high = y_true_bin == high
    pred_high = y_pred_bin == high
    tp = int(np.sum(true_high & pred_high))
    fp = int(np.sum(~true_high & pred_high))
    fn = int(np.sum(true_high & ~pred_high))
    return {
        "bin_accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "bin_macro_f1": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "high_detection_precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "high_detection_recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "high_detection_tp": tp,
        "high_detection_fp": fp,
        "high_detection_fn": fn,
    }


def fit_base_predict(train: SplitData, val: SplitData, test: SplitData, mask: np.ndarray, max_damage: float, clip: bool):
    """Fit base model and predict. 中文：训练基准模型并预测。"""
    model = base_model(alpha=0.1)
    model.fit(train.F[:, mask], train.y)
    p_train = model.predict(train.F[:, mask])
    p_val = model.predict(val.F[:, mask])
    p_test = model.predict(test.F[:, mask])
    if clip:
        p_train = np.clip(p_train, 0, max_damage)
        p_val = np.clip(p_val, 0, max_damage)
        p_test = np.clip(p_test, 0, max_damage)
    return p_train, p_val, p_test


def run_one(train: SplitData, val: SplitData, test: SplitData, mask: np.ndarray, feature_set: str,
            clf_name: str, cal_name: str, seed: int, max_damage: float, clip: bool, include_raw: bool) -> Result:
    """Run one configuration. 中文：运行一个具体配置。"""
    base_train, base_val, base_test = fit_base_predict(train, val, test, mask, max_damage, clip)
    Xtr, ytr, btr = make_entry_features(train.F, train.y, mask)
    Xv, yv, bv = make_entry_features(val.F, val.y, mask)
    Xte, yte, bte = make_entry_features(test.F, test.y, mask)
    base_tr_e = base_train.reshape(-1)
    base_v_e = base_val.reshape(-1)
    base_te_e = base_test.reshape(-1)

    clf = classifier_model(clf_name, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(Xtr, btr)
    ptr = fixed_proba(clf, Xtr)
    pv = fixed_proba(clf, Xv)
    pte = fixed_proba(clf, Xte)
    bte_pred = np.argmax(pte, axis=1)

    Ztr = make_Z(Xtr, base_tr_e, ptr, include_raw)
    Zv = make_Z(Xv, base_v_e, pv, include_raw)
    Zte = make_Z(Xte, base_te_e, pte, include_raw)

    cal = calibrator_model()
    if cal_name in ["residual_ridge", "high_sensitive_residual_ridge"]:
        target = ytr - base_tr_e
        weights = residual_weights(ytr, cal_name)
        kwargs = {"regressor__sample_weight": weights} if weights is not None else {}
        cal.fit(Ztr, target, **kwargs)
        pred_e = base_te_e + cal.predict(Zte)
        val_pred_e = base_v_e + cal.predict(Zv)
    elif cal_name == "direct_ridge":
        cal.fit(Ztr, ytr)
        pred_e = cal.predict(Zte)
        val_pred_e = cal.predict(Zv)
    else:
        raise ValueError(f"Unknown calibrator: {cal_name}")

    if clip:
        pred_e = np.clip(pred_e, 0, max_damage)
        val_pred_e = np.clip(val_pred_e, 0, max_damage)
    y_pred = pred_e.reshape(test.y.shape)
    val_pred = val_pred_e.reshape(val.y.shape)

    metrics = regression_metrics(test.y, y_pred)
    metrics.update(classification_metrics(bte, bte_pred))
    val_metrics = regression_metrics(val.y, val_pred)
    for k, v in val_metrics.items():
        metrics["val_" + k.replace("test_", "")] = v

    name = f"{feature_set}__{clf_name}__{cal_name}"
    return Result(name, feature_set, clf_name, cal_name, int(mask.sum()), test.y, base_test, y_pred, bte, bte_pred, metrics)


def save_result(result: Result, out_dir: Path) -> None:
    """Save metrics and predictions. 中文：保存指标和预测表。"""
    d = out_dir / result.config_name
    d.mkdir(parents=True, exist_ok=True)
    payload = {"config_name": result.config_name, "feature_set": result.feature_set,
               "classifier": result.classifier, "calibrator": result.calibrator,
               "n_features": result.n_features, **result.metrics}
    (d / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    yt = result.y_true.reshape(-1)
    yp = result.y_pred.reshape(-1)
    yb = result.y_base.reshape(-1)
    n_case, n_story = result.y_true.shape
    df = pd.DataFrame({
        "case_index_in_test": np.repeat(np.arange(n_case), n_story),
        "story": np.tile(np.arange(1, n_story + 1), n_case),
        "true_damage": yt,
        "base_pred_damage": yb,
        "calibrated_pred_damage": yp,
        "error": yp - yt,
        "abs_error": np.abs(yp - yt),
        "true_bin": [BIN_NAMES[i] for i in damage_to_bin(yt)],
        "predicted_bin": [BIN_NAMES[i] for i in result.y_bin_pred],
    })
    df.to_csv(d / "predictions_test.csv", index=False)


def result_df(results: Sequence[Result]) -> pd.DataFrame:
    """Build comparison table. 中文：生成对比总表。"""
    rows = []
    for r in results:
        row = {"config_name": r.config_name, "feature_set": r.feature_set, "classifier": r.classifier,
               "calibrator": r.calibrator, "n_features": r.n_features}
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["test_high_mae", "test_mae"]).reset_index(drop=True)


def md_table(df: pd.DataFrame, cols: Sequence[str], n: int) -> str:
    """Markdown table without tabulate. 中文：不依赖 tabulate 的 Markdown 表格。"""
    sub = df[[c for c in cols if c in df.columns]].head(n)
    def fmt(x):
        if isinstance(x, float):
            return "nan" if math.isnan(x) else f"{x:.6f}"
        return str(x)
    lines = ["| " + " | ".join(sub.columns) + " |", "| " + " | ".join(["---"] * len(sub.columns)) + " |"]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in sub.columns) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, out_dir: Path, top_k: int) -> Path:
    """Write summary markdown. 中文：输出总结报告。"""
    best_overall = df.sort_values("test_mae").iloc[0]
    best_high = df.sort_values("test_high_mae").iloc[0]
    cols = ["config_name", "test_mae", "test_rmse", "test_high_mae", "test_high_bias",
            "test_high_underestimation_ratio", "test_zero_mae", "test_zero_false_alarm_ratio_005",
            "bin_macro_f1", "high_detection_recall"]
    lines = [
        "# Damage-gated Calibration Summary\n",
        "## 1. Purpose\n",
        "This experiment tests whether a damage-bin classifier plus calibration regressor can reduce high-damage underestimation while controlling zero-damage false alarms.\n",
        "中文解释：本实验检查‘损伤等级分类器 + 校准回归器’是否能够降低高损伤低估，同时控制零损伤误报。\n",
        "## 2. Best overall configuration\n",
        f"- Configuration: `{best_overall['config_name']}`",
        f"- Overall test MAE: `{best_overall['test_mae']:.6f}`",
        f"- High-damage MAE: `{best_overall['test_high_mae']:.6f}`",
        f"- High-damage bias: `{best_overall['test_high_bias']:.6f}`",
        f"- High-damage underestimation ratio: `{best_overall['test_high_underestimation_ratio']:.6f}`",
        f"- Zero false alarm ratio > 0.05: `{best_overall['test_zero_false_alarm_ratio_005']:.6f}`\n",
        "## 3. Best high-damage configuration\n",
        f"- Configuration: `{best_high['config_name']}`",
        f"- Overall test MAE: `{best_high['test_mae']:.6f}`",
        f"- High-damage MAE: `{best_high['test_high_mae']:.6f}`",
        f"- High-damage bias: `{best_high['test_high_bias']:.6f}`",
        f"- High-damage underestimation ratio: `{best_high['test_high_underestimation_ratio']:.6f}`",
        f"- Zero false alarm ratio > 0.05: `{best_high['test_zero_false_alarm_ratio_005']:.6f}`\n",
        "## 4. Ranking by high-damage MAE\n",
        md_table(df.sort_values("test_high_mae"), cols, top_k),
        "\n## 5. Ranking by overall MAE\n",
        md_table(df.sort_values("test_mae"), cols, top_k),
        "\n## 6. Interpretation guide\n",
        "- If high-damage MAE decreases and zero false-alarm ratio stays moderate, gated calibration is stronger than global sample weighting.",
        "- If high-damage MAE decreases but zero false alarms become excessive, the gate is too aggressive.",
        "- If bin_macro_f1 or high_detection_recall is poor, the bottleneck is the classifier, not the regressor.\n",
        "中文解释：如果门控校准降低高损伤 MAE 且没有明显提高零损伤误报，它就可以作为论文主方法候选。若分类指标差，则下一步应改分类器。\n",
    ]
    path = out_dir / "gated_calibration_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_true_pred(r: Result, path: Path, title: str, max_damage: float) -> None:
    """True-predicted scatter. 中文：真实值-预测值散点图。"""
    yt = r.y_true.reshape(-1)
    yp = r.y_pred.reshape(-1)
    bins = damage_to_bin(yt)
    plt.figure(figsize=(8, 8))
    for name, idx in BIN_TO_ID.items():
        m = bins == idx
        if np.any(m):
            plt.scatter(yt[m], yp[m], alpha=0.65, label=name)
    plt.plot([0, max_damage], [0, max_damage], linestyle="--", label="Ideal y=x")
    plt.xlabel("True damage")
    plt.ylabel("Predicted damage")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion(r: Result, path: Path, title: str) -> None:
    """Confusion matrix. 中文：损伤等级混淆矩阵。"""
    cm = confusion_matrix(r.y_bin_true, r.y_bin_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.xticks(range(4), BIN_NAMES, rotation=45)
    plt.yticks(range(4), BIN_NAMES)
    plt.xlabel("Predicted bin")
    plt.ylabel("True bin")
    plt.title(title)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_bars(df: pd.DataFrame, path: Path, metric: str, title: str, top_k: int) -> None:
    """Top-k bar plot. 中文：Top-K 柱状图。"""
    sub = df.sort_values(metric).head(top_k)
    labels = sub["config_name"].astype(str).tolist()
    vals = sub[metric].astype(float).to_numpy()
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_tradeoff(df: pd.DataFrame, path: Path) -> None:
    """Tradeoff plot. 中文：整体 MAE 与高损伤 MAE 权衡图。"""
    plt.figure(figsize=(8, 6))
    plt.scatter(df["test_mae"], df["test_high_mae"], alpha=0.75)
    label_df = pd.concat([df.sort_values("test_mae").head(3), df.sort_values("test_high_mae").head(3)]).drop_duplicates("config_name")
    for _, row in label_df.iterrows():
        plt.text(row["test_mae"], row["test_high_mae"], row["config_name"], fontsize=8)
    plt.xlabel("Overall test MAE")
    plt.ylabel("High-damage test MAE")
    plt.title("Overall accuracy versus high-damage sensitivity")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_plots(results: Sequence[Result], df: pd.DataFrame, fig_dir: Path, max_damage: float, top_k: int) -> None:
    """Write all plots. 中文：输出全部关键图像。"""
    fig_dir.mkdir(parents=True, exist_ok=True)
    by_name = {r.config_name: r for r in results}
    best_overall = by_name[df.sort_values("test_mae").iloc[0]["config_name"]]
    best_high = by_name[df.sort_values("test_high_mae").iloc[0]["config_name"]]
    plot_true_pred(best_overall, fig_dir / "best_overall_true_vs_predicted.png", f"Best overall: {best_overall.config_name}", max_damage)
    plot_true_pred(best_high, fig_dir / "best_high_damage_true_vs_predicted.png", f"Best high-damage: {best_high.config_name}", max_damage)
    plot_confusion(best_overall, fig_dir / "best_overall_bin_confusion_matrix.png", f"Bin confusion: {best_overall.config_name}")
    plot_confusion(best_high, fig_dir / "best_high_damage_bin_confusion_matrix.png", f"Bin confusion: {best_high.config_name}")
    plot_bars(df, fig_dir / "top_overall_mae.png", "test_mae", "Top configurations by overall test MAE", top_k)
    plot_bars(df, fig_dir / "top_high_damage_mae.png", "test_high_mae", "Top configurations by high-damage test MAE", top_k)
    plot_tradeoff(df, fig_dir / "overall_vs_high_damage_tradeoff.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run damage-gated calibration experiment.")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--feature-names", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--figures-dir", type=Path, required=True)
    p.add_argument("--dataset-tag", type=str, default="debug_plus_500")
    p.add_argument("--max-damage", type=float, default=0.5)
    p.add_argument("--clip-predictions", action="store_true")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--feature-sets", nargs="+", default=["full", "no_meta", "response_spatial"])
    p.add_argument("--classifiers", nargs="+", default=["logistic_balanced", "random_forest_balanced"])
    p.add_argument("--calibrators", nargs="+", default=["direct_ridge", "residual_ridge", "high_sensitive_residual_ridge"])
    p.add_argument("--include-raw-features-in-calibrator", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    print("Damage-gated calibration started.")
    print(f"Feature dataset: {args.features}")
    data = load_dataset(args.features)
    n_features = data["train"].F.shape[1]
    names = load_feature_names(args.feature_names, n_features)
    masks = build_feature_masks(names)
    results: List[Result] = []
    for fs in args.feature_sets:
        if fs not in masks:
            print(f"[Skip] feature set unavailable or empty: {fs}")
            continue
        for clf in args.classifiers:
            for cal in args.calibrators:
                print(f"Running {fs} + {clf} + {cal} ...")
                r = run_one(data["train"], data["val"], data["test"], masks[fs], fs, clf, cal,
                            args.random_seed, args.max_damage, args.clip_predictions,
                            args.include_raw_features_in_calibrator)
                save_result(r, args.output_dir)
                results.append(r)
                print(f"  test_mae={r.metrics['test_mae']:.6f}, high_mae={r.metrics['test_high_mae']:.6f}, "
                      f"high_under={r.metrics['test_high_underestimation_ratio']:.6f}, "
                      f"zero_fa_005={r.metrics['test_zero_false_alarm_ratio_005']:.6f}, "
                      f"bin_f1={r.metrics['bin_macro_f1']:.6f}")
    if not results:
        raise RuntimeError("No experiment completed.")
    df = result_df(results)
    comp = args.output_dir / "gated_calibration_model_comparison.csv"
    df.to_csv(comp, index=False)
    report = write_report(df, args.output_dir, args.top_k)
    write_plots(results, df, args.figures_dir, args.max_damage, args.top_k)
    print("\nDamage-gated calibration completed.")
    print(f"Comparison CSV: {comp}")
    print(f"Report MD: {report}")
    print(f"Figures dir: {args.figures_dir}")
    cols = ["config_name", "test_mae", "test_high_mae", "test_high_bias", "test_high_underestimation_ratio", "test_zero_false_alarm_ratio_005", "bin_macro_f1", "high_detection_recall"]
    print("\nTop configurations by high-damage MAE:")
    print(df[cols].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
