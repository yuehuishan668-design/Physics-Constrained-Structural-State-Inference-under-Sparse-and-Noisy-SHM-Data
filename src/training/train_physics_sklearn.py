"""
Train regularized small-sample models on physics-informed diagnostic features.

中文说明：
当前 debug_plus_100 数据集只有 100 个样本，训练集只有 70 个样本；
使用 92 维 physics features 训练大 MLP 容易过拟合。
本脚本训练 Ridge / ElasticNet / RandomForest / GradientBoosting，
用验证集选择超参数，并输出 metrics、predictions、feature importance。

Recommended command:
python -m src.training.train_physics_sklearn \
  --features data_processed/debug_plus_100_physics_features_mlp.npz \
  --feature-names data_processed/debug_plus_100_physics_feature_names.csv \
  --models ridge elasticnet random_forest gradient_boosting \
  --clip-predictions \
  --max-damage 0.5 \
  --output-dir results/tables/physics_sklearn_100 \
  --random-seed 42
"""

from __future__ import annotations

import argparse  # 读取命令行参数 / Read command-line arguments.
import csv       # 写 CSV 表格 / Write CSV tables.
import json      # 写 JSON 指标 / Write JSON metrics.
from pathlib import Path  # 处理路径 / Handle filesystem paths.
from typing import Dict, List, Tuple

import numpy as np  # 数组计算 / Numerical array operations.

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.multioutput import MultiOutputRegressor
except ImportError as exc:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from exc


ArrayDict = Dict[str, np.ndarray]


def load_feature_dataset(path: Path) -> ArrayDict:
    """Load the feature npz file and check required arrays. / 读取特征数据并检查必要数组。"""
    data = np.load(path, allow_pickle=False)
    required = ["F_train", "F_val", "F_test", "y_train", "y_val", "y_test"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing required arrays in {path}: {missing}")
    return {key: data[key] for key in data.files}


def load_feature_names(path: Path | None, n_features: int) -> List[str]:
    """Load feature names; fall back to feature_0... if unavailable. / 读取特征名。"""
    if path is None or not path.exists():
        return [f"feature_{i}" for i in range(n_features)]

    names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row.get("feature_name", f"feature_{len(names)}"))

    if len(names) != n_features:
        print(f"Warning: feature names {len(names)} != n_features {n_features}; default names used.")
        return [f"feature_{i}" for i in range(n_features)]
    return names


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error. / 均方误差。"""
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. / 平均绝对误差。"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error. / 均方根误差。"""
    return float(np.sqrt(mse(y_true, y_pred)))


def clip_predictions(y_pred: np.ndarray, max_damage: float) -> np.ndarray:
    """Apply physical output range [0, max_damage]. / 将预测限制在物理可行区间。"""
    return np.clip(y_pred, 0.0, max_damage)


def compute_metrics(y_true: np.ndarray, y_raw: np.ndarray, y_eval: np.ndarray) -> Dict[str, object]:
    """Compute diagnostic metrics. / 计算回归与物理有效性指标。"""
    true_pos = y_true > 1.0e-8
    true_zero = ~true_pos
    return {
        "mse_overall": mse(y_true, y_eval),
        "mae_overall": mae(y_true, y_eval),
        "rmse_overall": rmse(y_true, y_eval),
        "mae_per_story": np.mean(np.abs(y_true - y_eval), axis=0).tolist(),
        "rmse_per_story": np.sqrt(np.mean((y_true - y_eval) ** 2, axis=0)).tolist(),
        "negative_prediction_ratio_raw": float(np.mean(y_raw < 0.0)),
        "negative_prediction_ratio": float(np.mean(y_eval < 0.0)),
        "over_max_prediction_ratio_raw": float(np.mean(y_raw > 0.5)),
        "mean_true_damage": float(np.mean(y_true)),
        "mean_pred_damage": float(np.mean(y_eval)),
        "max_true_damage": float(np.max(y_true)),
        "max_pred_damage": float(np.max(y_eval)),
        "mae_on_damaged_entries": float(np.mean(np.abs(y_true[true_pos] - y_eval[true_pos]))) if np.any(true_pos) else None,
        "mean_prediction_on_zero_entries": float(np.mean(y_eval[true_zero])) if np.any(true_zero) else None,
    }


def make_candidates(model_name: str, random_seed: int) -> List[Tuple[str, object]]:
    """Create hyperparameter candidates. / 生成超参数候选。"""
    if model_name == "ridge":
        return [(f"ridge_alpha_{a}", Ridge(alpha=a)) for a in [0.01, 0.1, 1.0, 10.0, 100.0, 300.0]]

    if model_name == "elasticnet":
        out: List[Tuple[str, object]] = []
        for alpha in [0.0005, 0.001, 0.005, 0.01, 0.05]:
            for l1_ratio in [0.1, 0.3, 0.5, 0.7]:
                base = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=random_seed)
                out.append((f"elasticnet_alpha_{alpha}_l1_{l1_ratio}", MultiOutputRegressor(base)))
        return out

    if model_name == "random_forest":
        return [
            (f"random_forest_depth_{depth}_leaf_{leaf}", RandomForestRegressor(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=random_seed,
                n_jobs=-1,
            ))
            for depth in [2, 3, 4, None]
            for leaf in [1, 3, 5]
        ]

    if model_name == "gradient_boosting":
        out = []
        for n_estimators in [80, 150, 250]:
            for learning_rate in [0.03, 0.05, 0.1]:
                for max_depth in [2, 3]:
                    base = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_seed,
                    )
                    out.append((f"gbr_n_{n_estimators}_lr_{learning_rate}_depth_{max_depth}", MultiOutputRegressor(base)))
        return out

    raise ValueError(f"Unknown model name: {model_name}")


def fit_select_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
    clip: bool,
    max_damage: float,
) -> Tuple[str, object, List[Dict[str, object]]]:
    """Fit candidates and select by validation MSE. / 训练候选模型并用验证集 MSE 选择最优。"""
    best_name = ""
    best_model = None
    best_val_mse = float("inf")
    records: List[Dict[str, object]] = []

    for candidate_name, model in make_candidates(model_name, random_seed):
        model.fit(X_train, y_train)
        pred_raw = model.predict(X_val)
        pred_eval = clip_predictions(pred_raw, max_damage) if clip else pred_raw
        val_mse = mse(y_val, pred_eval)
        val_mae = mae(y_val, pred_eval)
        records.append({"candidate": candidate_name, "val_mse": val_mse, "val_mae": val_mae})

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_name = candidate_name
            best_model = model

    if best_model is None:
        raise RuntimeError(f"No model was trained for {model_name}.")
    return best_name, best_model, records


def write_predictions_csv(path: Path, split: str, y_true: np.ndarray, y_pred: np.ndarray, case_ids: np.ndarray | None) -> None:
    """Write predictions compatible with plot_mlp_predictions.py. / 写出兼容既有绘图脚本的预测表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples, n_stories = y_true.shape
    header = ["split", "sample_id", "case_id"]
    header += [f"y_true_story_{i + 1}" for i in range(n_stories)]
    header += [f"y_pred_story_{i + 1}" for i in range(n_stories)]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_samples):
            case_id = int(case_ids[i]) if case_ids is not None else i
            row = [split, i, case_id] + [float(v) for v in y_true[i]] + [float(v) for v in y_pred[i]]
            writer.writerow(row)


def write_dict_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write list of dictionaries to CSV. / 将字典列表写成 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_feature_importance(path: Path, model_name: str, model: object, feature_names: List[str]) -> None:
    """Save feature importance or coefficient magnitude. / 保存特征重要性或系数绝对值。"""
    scores = None
    score_type = "not_available"

    if model_name == "ridge" and hasattr(model, "coef_"):
        scores = np.mean(np.abs(model.coef_), axis=0)
        score_type = "mean_abs_coefficient"
    elif model_name == "elasticnet" and hasattr(model, "estimators_"):
        scores = np.mean(np.vstack([np.abs(est.coef_) for est in model.estimators_]), axis=0)
        score_type = "mean_abs_coefficient"
    elif model_name == "random_forest" and hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
        score_type = "tree_feature_importance"
    elif model_name == "gradient_boosting" and hasattr(model, "estimators_"):
        scores = np.mean(np.vstack([est.feature_importances_ for est in model.estimators_]), axis=0)
        score_type = "tree_feature_importance"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature_index", "feature_name", "importance_type", "importance"])
        if scores is None:
            writer.writerow([1, -1, "not_available", score_type, 0.0])
            return
        for rank, idx in enumerate(np.argsort(scores)[::-1], start=1):
            writer.writerow([rank, int(idx), feature_names[int(idx)], score_type, float(scores[int(idx)])])


def run_one_model(model_name: str, arrays: ArrayDict, feature_names: List[str], output_dir: Path, random_seed: int, clip: bool, max_damage: float) -> Dict[str, object]:
    """Train/evaluate one model family. / 训练并评估一个模型族。"""
    X_train, X_val, X_test = arrays["F_train"], arrays["F_val"], arrays["F_test"]
    y_train, y_val, y_test = arrays["y_train"], arrays["y_val"], arrays["y_test"]
    case_id_train = arrays.get("case_id_train")
    case_id_val = arrays.get("case_id_val")
    case_id_test = arrays.get("case_id_test")

    best_name, model, records = fit_select_model(model_name, X_train, y_train, X_val, y_val, random_seed, clip, max_damage)
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    raw_train, raw_val, raw_test = model.predict(X_train), model.predict(X_val), model.predict(X_test)
    pred_train = clip_predictions(raw_train, max_damage) if clip else raw_train
    pred_val = clip_predictions(raw_val, max_damage) if clip else raw_val
    pred_test = clip_predictions(raw_test, max_damage) if clip else raw_test

    train_metrics = compute_metrics(y_train, raw_train, pred_train)
    val_metrics = compute_metrics(y_val, raw_val, pred_val)
    test_metrics = compute_metrics(y_test, raw_test, pred_test)

    write_predictions_csv(model_dir / "predictions_train.csv", "train", y_train, pred_train, case_id_train)
    write_predictions_csv(model_dir / "predictions_val.csv", "val", y_val, pred_val, case_id_val)
    write_predictions_csv(model_dir / "predictions_test.csv", "test", y_test, pred_test, case_id_test)
    write_dict_rows(model_dir / "candidate_validation_records.csv", records)
    write_feature_importance(model_dir / "feature_importance.csv", model_name, model, feature_names)

    metrics = {
        "model": "PhysicsSklearnRegressor",
        "estimator": model_name,
        "best_candidate": best_name,
        "clip_predictions": clip,
        "max_damage": max_damage,
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "best_val_loss_mse": min(float(r["val_mse"]) for r in records),
        "train_loss_mse": train_metrics["mse_overall"],
        "val_loss_mse": val_metrics["mse_overall"],
        "test_loss_mse": test_metrics["mse_overall"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "output_mode": "clipped" if clip else "raw",
        "negative_prediction_ratio": test_metrics["negative_prediction_ratio"],
    }

    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{model_name} training completed.")
    print(f"Best candidate: {best_name}")
    print(f"Best validation MSE: {metrics['best_val_loss_mse']:.6f}")
    print(f"Test MSE: {test_metrics['mse_overall']:.6f}")
    print(f"Test MAE overall: {test_metrics['mae_overall']:.6f}")
    print(f"Test RMSE overall: {test_metrics['rmse_overall']:.6f}")
    print(f"Negative prediction ratio: {test_metrics['negative_prediction_ratio']:.6f}")
    print(f"Mean prediction on zero entries: {test_metrics['mean_prediction_on_zero_entries']}")
    print(f"Metrics saved to: {model_dir / 'metrics.json'}")
    return metrics


def write_comparison(path: Path, all_metrics: List[Dict[str, object]]) -> None:
    """Write comparison table sorted by test MAE. / 按测试 MAE 排序写出对比表。"""
    rows = []
    for m in all_metrics:
        test = m["test_metrics"]
        rows.append({
            "estimator": m["estimator"],
            "best_candidate": m["best_candidate"],
            "clip_predictions": m["clip_predictions"],
            "n_features": m["n_features"],
            "best_val_mse": m["best_val_loss_mse"],
            "test_mse": test["mse_overall"],
            "test_mae": test["mae_overall"],
            "test_rmse": test["rmse_overall"],
            "negative_prediction_ratio": test["negative_prediction_ratio"],
            "mean_prediction_on_zero_entries": test["mean_prediction_on_zero_entries"],
            "mae_on_damaged_entries": test["mae_on_damaged_entries"],
            "mean_true_damage": test["mean_true_damage"],
            "mean_pred_damage": test["mean_pred_damage"],
        })
    rows.sort(key=lambda r: float(r["test_mae"]))
    write_dict_rows(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sklearn models on physics-informed features.")
    parser.add_argument("--features", type=Path, required=True, help="Physics feature npz path.")
    parser.add_argument("--feature-names", type=Path, default=None, help="Feature-name CSV path.")
    parser.add_argument("--models", nargs="+", default=["ridge", "elasticnet", "random_forest", "gradient_boosting"], choices=["ridge", "elasticnet", "random_forest", "gradient_boosting"])
    parser.add_argument("--clip-predictions", action="store_true", help="Clip predictions to [0, max_damage].")
    parser.add_argument("--max-damage", type=float, default=0.5, help="Maximum damage ratio for clipping.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_seed)

    arrays = load_feature_dataset(args.features)
    feature_names = load_feature_names(args.feature_names, int(arrays["F_train"].shape[1]))

    print("Physics sklearn training started.")
    print(f"Feature dataset: {args.features}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {args.models}")
    print(f"Clip predictions: {args.clip_predictions}")
    print(f"F_train shape: {arrays['F_train'].shape}")
    print(f"F_val shape: {arrays['F_val'].shape}")
    print(f"F_test shape: {arrays['F_test'].shape}")

    all_metrics = []
    for model_name in args.models:
        all_metrics.append(run_one_model(model_name, arrays, feature_names, args.output_dir, args.random_seed, args.clip_predictions, args.max_damage))

    comparison_path = args.output_dir / "model_comparison.csv"
    write_comparison(comparison_path, all_metrics)
    print("\nAll requested sklearn models completed.")
    print(f"Comparison CSV saved to: {comparison_path}")
    print("\nSorted comparison table:")
    print(comparison_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
