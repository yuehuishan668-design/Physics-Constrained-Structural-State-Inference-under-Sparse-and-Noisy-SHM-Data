"""
Run repeated-split robustness experiments for physics-informed feature models.

中文说明：
本文件用于对当前最优路线（physics-informed features + regularized regression）
进行多随机种子稳健性验证。它不会重新生成 OpenSees 数据，只会读取已经提取好的
physics feature npz 文件，然后在不同随机划分下重复训练/验证/测试。

Typical command:
python -m src.evaluation.run_seed_robustness \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_500_physics_feature_names.csv \
  --output-dir results/tables/seed_robustness/debug_plus_500 \
  --figures-dir results/figures/seed_robustness/debug_plus_500 \
  --models ridge elasticnet random_forest \
  --feature-sets full no_meta response_basic_only response_frequency response_correlation response_spatial \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --clip-predictions \
  --max-damage 0.5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Data loading
# 数据读取
# -----------------------------


def load_feature_dataset(npz_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load feature and target arrays from an npz file.

    中文说明：
    优先读取 F_train/F_val/F_test 和 y_train/y_val/y_test。
    如果文件中存在 F/y，也兼容读取。最终统一拼接成完整 X, y，
    便于后续按不同 seed 重新划分训练集、验证集和测试集。
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    meta = {"npz_path": str(npz_path), "keys": sorted(data.files)}

    if {"F_train", "F_val", "F_test", "y_train", "y_val", "y_test"}.issubset(keys):
        X = np.concatenate([data["F_train"], data["F_val"], data["F_test"]], axis=0)
        y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)
        meta["source_layout"] = "F_train/F_val/F_test"
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float), meta

    if {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}.issubset(keys):
        X = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
        y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)
        meta["source_layout"] = "X_train/X_val/X_test"
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float), meta

    if {"F", "y"}.issubset(keys):
        meta["source_layout"] = "F/y"
        return np.asarray(data["F"], dtype=float), np.asarray(data["y"], dtype=float), meta

    if {"X", "y"}.issubset(keys):
        meta["source_layout"] = "X/y"
        return np.asarray(data["X"], dtype=float), np.asarray(data["y"], dtype=float), meta

    raise KeyError(
        "Cannot find supported feature/target keys in npz file. "
        "Expected F_train/F_val/F_test + y_train/y_val/y_test, "
        "or X_train/X_val/X_test + y_train/y_val/y_test, "
        "or F/y, or X/y."
    )


def load_feature_names(csv_path: Path, n_features: int) -> list[str]:
    """
    Load feature names from CSV.

    中文说明：
    支持两种常见格式：
    1. feature_index,feature_name
    2. 只有一列 feature_name
    """
    df = pd.read_csv(csv_path)

    if "feature_name" in df.columns:
        names = df["feature_name"].astype(str).tolist()
    elif df.shape[1] >= 2:
        names = df.iloc[:, 1].astype(str).tolist()
    elif df.shape[1] == 1:
        names = df.iloc[:, 0].astype(str).tolist()
    else:
        raise ValueError(f"No feature names found in {csv_path}")

    if len(names) != n_features:
        raise ValueError(
            f"Feature name count mismatch: csv has {len(names)} names, "
            f"but feature matrix has {n_features} columns."
        )
    return names


# -----------------------------
# Feature subset rules
# 特征子集规则
# -----------------------------


def _is_meta_feature(name: str) -> bool:
    """
    Identify external metadata features.

    中文说明：
    这里的 meta 指输入地震动、噪声水平等外部工况元数据。
    注意不要把 response_frequency 中的 dominant_frequency 等响应频域特征误删。
    """
    lower = name.lower()
    return (
        lower.startswith("input_")
        or lower in {"noise_level", "amplitude_g", "frequency_hz"}
        or lower.startswith("meta_")
    )


def _is_basic_response_feature(name: str) -> bool:
    """
    Basic time-domain response descriptors.

    中文说明：
    基础响应统计量通常包括均值、标准差、最大绝对值、RMS、峰峰值、峰值因子。
    """
    lower = name.lower()
    basic_tokens = [
        "_mean",
        "_std",
        "_max_abs",
        "_rms",
        "_peak_to_peak",
        "_crest_factor",
    ]
    if lower in {"ground_max_abs", "ground_rms"}:
        return True
    return any(token in lower for token in basic_tokens) and not _is_meta_feature(lower)


def _is_frequency_feature(name: str) -> bool:
    """
    Frequency-domain response descriptors.

    中文说明：
    频域特征用于表征结构动力特性变化，例如主频、谱质心、频带能量等。
    """
    lower = name.lower()
    return any(
        token in lower
        for token in [
            "dominant_frequency",
            "spectral_centroid",
            "band_energy",
            "frequency_to_input_ratio",
            "centroid_to_input_ratio",
        ]
    )


def _is_correlation_feature(name: str) -> bool:
    """
    Inter-story or ground-response correlation descriptors.

    中文说明：
    相关性特征用于描述楼层之间或楼层与输入地震动之间的耦合关系。
    """
    return "correlation" in name.lower()


def _is_spatial_feature(name: str) -> bool:
    """
    Spatial distribution descriptors across stories.

    中文说明：
    空间特征用于描述响应在不同楼层之间的分布、放大和相对比例。
    """
    lower = name.lower()
    return any(
        token in lower
        for token in [
            "spatial_fraction",
            "ground_amplification",
            "_to_story_",
            "to_input_ratio",
            "_to_ground_",
        ]
    )


def get_feature_indices(feature_names: list[str], feature_set: str) -> np.ndarray:
    """
    Return column indices for a named feature subset.

    中文说明：
    这里的 full/no_meta/response_* 用于做稳健性验证和消融复核。
    如果某个特征组没有匹配到任何特征，会直接报错，避免静默运行错误实验。
    """
    feature_set = feature_set.strip().lower()

    if feature_set == "full":
        mask = [True for _ in feature_names]
    elif feature_set in {"no_meta", "physics_no_meta_core"}:
        mask = [not _is_meta_feature(name) for name in feature_names]
    elif feature_set == "response_basic_only":
        mask = [_is_basic_response_feature(name) for name in feature_names]
    elif feature_set == "response_frequency":
        mask = [_is_frequency_feature(name) for name in feature_names]
    elif feature_set == "response_correlation":
        mask = [_is_correlation_feature(name) for name in feature_names]
    elif feature_set == "response_spatial":
        mask = [_is_spatial_feature(name) for name in feature_names]
    else:
        raise ValueError(
            f"Unknown feature set: {feature_set}. "
            "Supported: full, no_meta, physics_no_meta_core, response_basic_only, "
            "response_frequency, response_correlation, response_spatial."
        )

    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        raise ValueError(f"Feature set '{feature_set}' selected zero features.")
    return idx


# -----------------------------
# Models and metrics
# 模型与评价指标
# -----------------------------


@dataclass
class CandidateResult:
    model_name: str
    candidate_name: str
    val_mse: float
    estimator: object


def make_model_candidates(model_name: str, random_seed: int) -> list[tuple[str, object]]:
    """
    Build candidate estimators for validation-set model selection.

    中文说明：
    Ridge/ElasticNet 通过验证集选择正则化强度；
    RandomForest 只使用少量保守参数，避免在小数据上过拟合。
    """
    model_name = model_name.lower()

    if model_name == "ridge":
        alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
        return [(f"ridge_alpha_{alpha}", Ridge(alpha=alpha)) for alpha in alphas]

    if model_name == "elasticnet":
        candidates: list[tuple[str, object]] = []
        alphas = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1]
        l1_ratios = [0.2, 0.5, 0.7, 0.9]
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                base = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=30000,
                    tol=1e-4,
                    random_state=random_seed,
                )
                candidates.append(
                    (
                        f"elasticnet_alpha_{alpha}_l1_{l1_ratio}",
                        MultiOutputRegressor(base),
                    )
                )
        return candidates

    if model_name == "random_forest":
        configs = [
            {"n_estimators": 200, "max_depth": 2, "min_samples_leaf": 1},
            {"n_estimators": 300, "max_depth": 2, "min_samples_leaf": 2},
            {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 2},
            {"n_estimators": 500, "max_depth": 3, "min_samples_leaf": 3},
        ]
        candidates = []
        for cfg in configs:
            name = (
                f"random_forest_n_{cfg['n_estimators']}"
                f"_depth_{cfg['max_depth']}_leaf_{cfg['min_samples_leaf']}"
            )
            estimator = RandomForestRegressor(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                random_state=random_seed,
                n_jobs=-1,
            )
            candidates.append((name, estimator))
        return candidates

    raise ValueError(f"Unsupported model: {model_name}")


def split_indices(n_samples: int, seed: int, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/validation/test indices.

    中文说明：
    每个 seed 都会产生一组新的 train/val/test 划分。
    这一步用于验证模型结论是否依赖某一次偶然划分。
    """
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Invalid split ratios. Need train_ratio > 0, val_ratio > 0, train_ratio + val_ratio < 1.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)

    n_train = int(round(n_samples * train_ratio))
    n_val = int(round(n_samples * val_ratio))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return train_idx, val_idx, test_idx


def fit_select_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_set: str,
    model_name: str,
    seed: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    clip_predictions: bool,
    max_damage: float,
) -> dict:
    """
    Fit candidates, select by validation MSE, and evaluate on test split.

    中文说明：
    训练集用于拟合模型，验证集用于选择超参数，测试集只用于最终评价。
    """
    selected_idx = get_feature_indices(feature_names, feature_set)

    X_train = X[train_idx][:, selected_idx]
    X_val = X[val_idx][:, selected_idx]
    X_test = X[test_idx][:, selected_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # Impute and standardize only from the training split.
    # 只用训练集拟合缺失值填充器和标准化器，避免测试集信息泄露。
    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_p = preprocess.fit_transform(X_train)
    X_val_p = preprocess.transform(X_val)
    X_test_p = preprocess.transform(X_test)

    candidates = make_model_candidates(model_name, random_seed=seed)
    best: CandidateResult | None = None

    for candidate_name, estimator in candidates:
        estimator.fit(X_train_p, y_train)
        val_pred = estimator.predict(X_val_p)
        if clip_predictions:
            val_pred = np.clip(val_pred, 0.0, max_damage)
        val_mse = float(mean_squared_error(y_val, val_pred))

        if best is None or val_mse < best.val_mse:
            best = CandidateResult(
                model_name=model_name,
                candidate_name=candidate_name,
                val_mse=val_mse,
                estimator=estimator,
            )

    if best is None:
        raise RuntimeError("No candidate model was trained.")

    test_pred_raw = best.estimator.predict(X_test_p)
    test_pred = np.clip(test_pred_raw, 0.0, max_damage) if clip_predictions else test_pred_raw

    error = test_pred - y_test
    damaged_mask = y_test > 0.0
    zero_mask = y_test == 0.0
    high_damage_mask = y_test >= 0.15

    test_mse = float(mean_squared_error(y_test, test_pred))
    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_rmse = float(np.sqrt(test_mse))

    result = {
        "seed": seed,
        "feature_set": feature_set,
        "model": model_name,
        "best_candidate": best.candidate_name,
        "n_total_samples": int(X.shape[0]),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "n_features": int(selected_idx.size),
        "best_val_mse": float(best.val_mse),
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "negative_prediction_ratio_raw": float(np.mean(test_pred_raw < 0.0)),
        "negative_prediction_ratio": float(np.mean(test_pred < 0.0)),
        "mean_true_damage": float(np.mean(y_test)),
        "mean_pred_damage": float(np.mean(test_pred)),
        "bias_mean_pred_minus_true": float(np.mean(test_pred - y_test)),
        "mean_prediction_on_zero_entries": float(np.mean(test_pred[zero_mask])) if np.any(zero_mask) else np.nan,
        "mae_on_damaged_entries": float(np.mean(np.abs(error[damaged_mask]))) if np.any(damaged_mask) else np.nan,
        "mae_on_high_damage_entries": float(np.mean(np.abs(error[high_damage_mask]))) if np.any(high_damage_mask) else np.nan,
        "selected_feature_indices": " ".join(map(str, selected_idx.tolist())),
    }
    return result


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize repeated-split results by feature set and model.

    中文说明：
    输出均值、标准差、最好值、最差值和排名次数，用于论文级稳定性分析。
    """
    group_cols = ["feature_set", "model"]

    summary = (
        results_df.groupby(group_cols)
        .agg(
            n_runs=("test_mae", "count"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_mae_min=("test_mae", "min"),
            test_mae_max=("test_mae", "max"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            best_val_mse_mean=("best_val_mse", "mean"),
            mean_true_damage=("mean_true_damage", "mean"),
            mean_pred_damage=("mean_pred_damage", "mean"),
            bias_mean=("bias_mean_pred_minus_true", "mean"),
            mae_damaged_mean=("mae_on_damaged_entries", "mean"),
            mae_high_damage_mean=("mae_on_high_damage_entries", "mean"),
            mean_zero_pred=("mean_prediction_on_zero_entries", "mean"),
        )
        .reset_index()
    )

    summary["test_mae_cv_pct"] = 100.0 * summary["test_mae_std"] / summary["test_mae_mean"].replace(0, np.nan)
    summary = summary.sort_values(["test_mae_mean", "test_rmse_mean"], ascending=True).reset_index(drop=True)
    summary.insert(0, "rank_by_mean_test_mae", np.arange(1, len(summary) + 1))
    return summary


def write_markdown_report(
    output_path: Path,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    meta: dict,
) -> None:
    """
    Write a plain Markdown report without requiring the tabulate package.

    中文说明：
    不使用 pandas.to_markdown，因此不会再触发 tabulate 缺失问题。
    """
    def df_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
        view = df.head(max_rows).copy()
        # Keep important numeric columns readable.
        # 控制小数位，便于直接放入论文实验记录。
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
        headers = list(view.columns)
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    best = summary_df.iloc[0]

    lines = [
        "# Seed Robustness Summary",
        "",
        "## 1. Dataset",
        "",
        f"- Feature file: `{meta.get('npz_path', '')}`",
        f"- Source layout: `{meta.get('source_layout', '')}`",
        f"- Number of samples: `{int(results_df['n_total_samples'].iloc[0])}`",
        f"- Train/val/test per seed: `{int(results_df['n_train'].iloc[0])}` / `{int(results_df['n_val'].iloc[0])}` / `{int(results_df['n_test'].iloc[0])}`",
        f"- Number of repeated seeds: `{results_df['seed'].nunique()}`",
        "",
        "## 2. Overall best configuration",
        "",
        f"- Best feature set: `{best['feature_set']}`",
        f"- Best model: `{best['model']}`",
        f"- Mean Test MAE: `{best['test_mae_mean']:.6f}`",
        f"- Std Test MAE: `{best['test_mae_std']:.6f}`",
        f"- Mean Test RMSE: `{best['test_rmse_mean']:.6f}`",
        f"- Mean prediction bias: `{best['bias_mean']:.6f}`",
        "",
        "## 3. Summary ranked by mean Test MAE",
        "",
        df_to_md(
            summary_df[
                [
                    "rank_by_mean_test_mae",
                    "feature_set",
                    "model",
                    "n_runs",
                    "test_mae_mean",
                    "test_mae_std",
                    "test_mae_cv_pct",
                    "test_rmse_mean",
                    "bias_mean",
                    "mae_damaged_mean",
                    "mae_high_damage_mean",
                ]
            ],
            max_rows=30,
        ),
        "",
        "## 4. Per-seed best model",
        "",
    ]

    per_seed_best = (
        results_df.sort_values(["seed", "test_mae", "test_rmse"])
        .groupby("seed", as_index=False)
        .first()
    )
    lines.append(
        df_to_md(
            per_seed_best[
                [
                    "seed",
                    "feature_set",
                    "model",
                    "best_candidate",
                    "test_mae",
                    "test_rmse",
                    "bias_mean_pred_minus_true",
                    "mae_on_high_damage_entries",
                ]
            ],
            max_rows=100,
        )
    )

    lines.extend(
        [
            "",
            "## 5. Preliminary interpretation template",
            "",
            "- If the same feature set and model ranks first across most seeds, the conclusion is stable rather than split-specific.",
            "- If the full feature set is only marginally better than no_meta, external metadata contributes limited incremental information.",
            "- If high-damage MAE remains high, the next methodological problem is large-damage underestimation rather than average-case fitting.",
            "- If Ridge remains competitive against nonlinear models, this supports using regularized physics-informed descriptors under limited-data SHM settings.",
            "",
            "中文解释：",
            "",
            "- 如果同一模型在多数 seed 下排名第一，说明结论不是某一次数据划分造成的偶然结果。",
            "- 如果 full 和 no_meta 差距很小，说明外部元数据贡献有限，主体信息来自结构响应特征。",
            "- 如果大损伤样本误差仍高，下一步问题不是继续堆模型，而是处理大损伤低估偏差。",
            "- 如果 Ridge 稳定优于非线性模型，可以支撑“正则化物理特征模型适合小样本 noisy SHM”的论文主张。",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")



def safe_boxplot(data, tick_labels) -> None:
    """
    Draw a boxplot with Matplotlib-version compatibility.

    中文说明：
    不同 Matplotlib 版本的箱线图标签参数名称不同：
    - 新版本使用 tick_labels
    - 旧版本使用 labels
    因此这里先尝试 tick_labels，失败后回退 labels。
    """
    try:
        plt.boxplot(data, tick_labels=tick_labels, showmeans=True)
    except TypeError:
        plt.boxplot(data, labels=tick_labels, showmeans=True)

def plot_mae_boxplot(results_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot Test MAE distribution for each feature/model configuration.

    中文说明：
    箱线图用于查看不同随机种子下的误差稳定性。
    """
    results_df = results_df.copy()
    results_df["config"] = results_df["feature_set"] + " + " + results_df["model"]
    ordered = (
        results_df.groupby("config")["test_mae"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    data = [results_df.loc[results_df["config"] == cfg, "test_mae"].values for cfg in ordered]

    plt.figure(figsize=(max(10, 0.65 * len(ordered)), 6))
    safe_boxplot(data, ordered)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test MAE")
    plt.title("Repeated-split Test MAE distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_summary_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot mean Test MAE with standard deviation error bars.

    中文说明：
    误差棒图用于展示 mean ± std，是论文结果表之外的直观补充。
    """
    df = summary_df.copy()
    df["config"] = df["feature_set"] + " + " + df["model"]

    x = np.arange(len(df))
    y = df["test_mae_mean"].values
    yerr = df["test_mae_std"].fillna(0.0).values

    plt.figure(figsize=(max(10, 0.65 * len(df)), 6))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, df["config"], rotation=45, ha="right")
    plt.ylabel("Mean Test MAE")
    plt.title("Mean Test MAE across repeated splits")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bias_vs_mae(results_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot bias versus test MAE for all repeated runs.

    中文说明：
    用于判断某些模型是否虽然 MAE 不高，但存在系统性低估或高估。
    """
    plt.figure(figsize=(8, 6))
    for (feature_set, model), group in results_df.groupby(["feature_set", "model"]):
        label = f"{feature_set} + {model}"
        plt.scatter(group["bias_mean_pred_minus_true"], group["test_mae"], label=label, alpha=0.75)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Mean prediction bias: predicted - true")
    plt.ylabel("Test MAE")
    plt.title("Bias versus Test MAE across repeated splits")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# 主函数
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated-split robustness evaluation for physics-informed feature models."
    )
    parser.add_argument("--features", type=Path, required=True, help="Path to physics feature npz file.")
    parser.add_argument("--feature-names", type=Path, required=True, help="Path to feature names CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON/Markdown outputs.")
    parser.add_argument("--figures-dir", type=Path, required=True, help="Directory for figure outputs.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "elasticnet", "random_forest"],
        help="Models to evaluate: ridge elasticnet random_forest.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["full", "no_meta"],
        help=(
            "Feature sets to evaluate: full no_meta physics_no_meta_core "
            "response_basic_only response_frequency response_correlation response_spatial."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)), help="Random split seeds.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training ratio for each seed.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio for each seed.")
    parser.add_argument("--clip-predictions", action="store_true", help="Clip predictions to [0, max_damage].")
    parser.add_argument("--max-damage", type=float, default=0.5, help="Upper bound when clipping damage predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_feature_dataset(args.features)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    feature_names = load_feature_names(args.feature_names, n_features=X.shape[1])

    print("Seed robustness experiment started.")
    print(f"Feature file: {args.features}")
    print(f"Feature names: {args.feature_names}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Models: {args.models}")
    print(f"Feature sets: {args.feature_sets}")
    print(f"Seeds: {args.seeds}")

    all_results: list[dict] = []

    for seed in args.seeds:
        train_idx, val_idx, test_idx = split_indices(
            n_samples=X.shape[0],
            seed=seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        print(f"\nSeed {seed}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        for feature_set in args.feature_sets:
            for model_name in args.models:
                print(f"  Running {feature_set} + {model_name} ...", flush=True)
                result = fit_select_evaluate(
                    X=X,
                    y=y,
                    feature_names=feature_names,
                    feature_set=feature_set,
                    model_name=model_name,
                    seed=seed,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    clip_predictions=args.clip_predictions,
                    max_damage=args.max_damage,
                )
                all_results.append(result)
                print(
                    f"    test_mae={result['test_mae']:.6f}, "
                    f"test_rmse={result['test_rmse']:.6f}, "
                    f"candidate={result['best_candidate']}"
                )

    results_df = pd.DataFrame(all_results)
    summary_df = summarize_results(results_df)

    raw_csv = args.output_dir / "seed_robustness_all_runs.csv"
    summary_csv = args.output_dir / "seed_robustness_summary.csv"
    best_csv = args.output_dir / "seed_robustness_per_seed_best.csv"
    meta_json = args.output_dir / "seed_robustness_meta.json"
    report_md = args.output_dir / "seed_robustness_report.md"

    results_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    (
        results_df.sort_values(["seed", "test_mae", "test_rmse"])
        .groupby("seed", as_index=False)
        .first()
        .to_csv(best_csv, index=False)
    )

    meta_out = {
        **meta,
        "feature_file": str(args.features),
        "feature_names_file": str(args.feature_names),
        "output_dir": str(args.output_dir),
        "figures_dir": str(args.figures_dir),
        "models": args.models,
        "feature_sets": args.feature_sets,
        "seeds": args.seeds,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "clip_predictions": args.clip_predictions,
        "max_damage": args.max_damage,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets": int(y.shape[1]),
    }
    meta_json.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    write_markdown_report(
        output_path=report_md,
        results_df=results_df,
        summary_df=summary_df,
        meta=meta_out,
    )

    plot_mae_boxplot(results_df, args.figures_dir / "seed_robustness_test_mae_boxplot.png")
    plot_summary_bar(summary_df, args.figures_dir / "seed_robustness_test_mae_mean_std.png")
    plot_bias_vs_mae(results_df, args.figures_dir / "seed_robustness_bias_vs_mae.png")

    print("\nSeed robustness experiment completed.")
    print(f"All runs CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-seed best CSV: {best_csv}")
    print(f"Markdown report: {report_md}")
    print(f"Figures directory: {args.figures_dir}")
    print("\nTop configurations by mean Test MAE:")
    print(
        summary_df[
            [
                "rank_by_mean_test_mae",
                "feature_set",
                "model",
                "n_runs",
                "test_mae_mean",
                "test_mae_std",
                "test_rmse_mean",
                "bias_mean",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
