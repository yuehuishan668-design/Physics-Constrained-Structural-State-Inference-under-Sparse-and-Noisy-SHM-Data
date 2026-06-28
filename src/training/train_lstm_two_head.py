"""
File: src/training/train_lstm_two_head.py

Purpose / 作用:
    Train a two-head LSTM model.

Core loss / 核心损失:
    total_loss =
        lambda_cls  * BCEWithLogitsLoss(damage_logits, damage_mask)
      + lambda_reg  * MaskedMSE(magnitude, y_true on damaged stories)
      + lambda_zero * MSE(magnitude, 0 on healthy stories)

Final prediction / 最终预测:
    threshold:
        y_pred = 1[p_damage >= threshold] * magnitude

    expectation:
        y_pred = p_damage * magnitude
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_two_head import LSTMTwoHeadDamageModel, count_trainable_parameters


def set_seed(seed: int) -> None:
    """Fix random seeds. 固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    """Select device. 选择训练设备。"""
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load .npz file. 读取 npz 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.keys()}


def downsample_sequence(X: np.ndarray, stride: int) -> np.ndarray:
    """
    Downsample time axis.
    中文：沿时间轴降采样；2000 点在 stride=10 时变为 200 点。
    """
    if stride <= 0:
        raise ValueError("sequence_stride must be positive.")
    return X[:, ::stride, :]


def make_damage_mask(y: np.ndarray, damage_threshold: float) -> np.ndarray:
    """
    Convert continuous damage labels to binary damage masks.
    中文：把连续损伤标签转成 0/1 分类标签。
    """
    return (y > damage_threshold).astype(np.float32)


def build_meta_features(
    dataset: Dict[str, np.ndarray],
    split: str,
    meta_mean: np.ndarray | None = None,
    meta_std: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build normalized metadata: amplitude_g, frequency_hz, noise_level.
    中文：构建并标准化元信息特征。
    """
    required = [f"amplitude_g_{split}", f"frequency_hz_{split}", f"noise_level_{split}"]
    for key in required:
        if key not in dataset:
            raise KeyError(f"Required metadata key not found: {key}")

    meta = np.stack(
        [
            dataset[f"amplitude_g_{split}"],
            dataset[f"frequency_hz_{split}"],
            dataset[f"noise_level_{split}"],
        ],
        axis=1,
    ).astype(np.float32)

    if meta_mean is None or meta_std is None:
        meta_mean = np.mean(meta, axis=0, keepdims=True)
        meta_std = np.std(meta, axis=0, keepdims=True)
        meta_std = np.where(meta_std < 1.0e-8, 1.0, meta_std)

    meta_norm = (meta - meta_mean) / meta_std
    return meta_norm.astype(np.float32), meta_mean.astype(np.float32), meta_std.astype(np.float32)


def get_case_ids(dataset: Dict[str, np.ndarray], split: str, n_samples: int) -> np.ndarray:
    """Get case IDs. 获取 case_id。"""
    key = f"case_id_{split}"
    if key in dataset:
        return dataset[key].astype(int)
    return np.arange(n_samples, dtype=int)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    damage_mask: np.ndarray,
    meta: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create DataLoader. 创建 PyTorch DataLoader。"""
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(damage_mask, dtype=torch.float32, device=device)

    if meta is None:
        dataset = TensorDataset(x_tensor, y_tensor, mask_tensor)
    else:
        meta_tensor = torch.tensor(meta, dtype=torch.float32, device=device)
        dataset = TensorDataset(x_tensor, meta_tensor, y_tensor, mask_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def unpack_batch(
    batch: tuple[torch.Tensor, ...],
    condition_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Unpack batch. 解析 batch。"""
    if condition_mode == "none":
        x_batch, y_batch, mask_batch = batch
        return x_batch, None, y_batch, mask_batch
    if condition_mode == "meta":
        x_batch, meta_batch, y_batch, mask_batch = batch
        return x_batch, meta_batch, y_batch, mask_batch
    raise ValueError(f"Unknown condition_mode: {condition_mode}")


def magnitude_from_raw(magnitude_raw: torch.Tensor, max_damage: float) -> torch.Tensor:
    """
    Bound magnitude to (0, max_damage).
    中文：用 sigmoid 将损伤幅值约束到 0 到 max_damage。
    """
    return max_damage * torch.sigmoid(magnitude_raw)


def final_prediction(
    probability: torch.Tensor,
    magnitude: torch.Tensor,
    threshold: float,
    final_mode: str,
) -> torch.Tensor:
    """
    Compute final damage prediction.
    中文：根据分类概率和幅值头输出得到最终损伤预测。
    """
    if final_mode == "threshold":
        return (probability >= threshold).float() * magnitude
    if final_mode == "expectation":
        return probability * magnitude
    raise ValueError(f"Unknown final_mode: {final_mode}")


def compute_losses(
    logits: torch.Tensor,
    magnitude: torch.Tensor,
    y_true: torch.Tensor,
    damage_mask: torch.Tensor,
    bce_loss_fn: nn.Module,
    lambda_cls: float,
    lambda_reg: float,
    lambda_zero: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute classification + masked regression + zero regularization loss.
    中文：计算分类损失、损伤楼层回归损失、健康楼层幅值正则。
    """
    cls_loss = bce_loss_fn(logits, damage_mask)

    positive_mask = damage_mask
    positive_count = torch.sum(positive_mask)
    if positive_count > 0:
        reg_loss = torch.sum(((magnitude - y_true) ** 2) * positive_mask) / positive_count
    else:
        reg_loss = torch.zeros((), dtype=y_true.dtype, device=y_true.device)

    zero_mask = 1.0 - damage_mask
    zero_count = torch.sum(zero_mask)
    if zero_count > 0:
        zero_loss = torch.sum((magnitude ** 2) * zero_mask) / zero_count
    else:
        zero_loss = torch.zeros((), dtype=y_true.dtype, device=y_true.device)

    total_loss = lambda_cls * cls_loss + lambda_reg * reg_loss + lambda_zero * zero_loss

    return total_loss, {
        "total_loss": float(total_loss.detach().cpu().item()),
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "reg_loss": float(reg_loss.detach().cpu().item()),
        "zero_loss": float(zero_loss.detach().cpu().item()),
    }


def build_bce_loss(mask_train: np.ndarray, use_pos_weight: bool, device: torch.device) -> nn.Module:
    """
    Build BCE loss.
    中文：构建二分类损失；可使用 pos_weight 缓解类别不平衡。
    """
    if not use_pos_weight:
        return nn.BCEWithLogitsLoss()

    pos = np.sum(mask_train, axis=0)
    neg = mask_train.shape[0] - pos
    pos_weight = neg / np.maximum(pos, 1.0)

    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    bce_loss_fn: nn.Module,
    condition_mode: str,
    max_damage: float,
    lambda_cls: float,
    lambda_reg: float,
    lambda_zero: float,
    grad_clip: float,
) -> Dict[str, float]:
    """
    Run train/eval epoch.
    中文：运行一个训练或验证 epoch；optimizer=None 表示验证。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    totals = {"total_loss": 0.0, "cls_loss": 0.0, "reg_loss": 0.0, "zero_loss": 0.0}
    total_samples = 0

    for batch in loader:
        x_batch, meta_batch, y_batch, mask_batch = unpack_batch(batch, condition_mode)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits, magnitude_raw = model(x_batch, meta_batch)
            magnitude = magnitude_from_raw(magnitude_raw, max_damage)
            loss, loss_items = compute_losses(
                logits=logits,
                magnitude=magnitude,
                y_true=y_batch,
                damage_mask=mask_batch,
                bce_loss_fn=bce_loss_fn,
                lambda_cls=lambda_cls,
                lambda_reg=lambda_reg,
                lambda_zero=lambda_zero,
            )

            if is_train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        batch_size = y_batch.shape[0]
        for key in totals:
            totals[key] += loss_items[key] * batch_size
        total_samples += batch_size

    return {key: val / max(total_samples, 1) for key, val in totals.items()}


@torch.no_grad()
def predict_all(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    meta: np.ndarray | None,
    device: torch.device,
    condition_mode: str,
    max_damage: float,
    threshold: float,
    final_mode: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Predict all samples. 预测一个 split 的全部样本。"""
    model.eval()
    loader = make_loader(X, y, mask, meta, batch_size, shuffle=False, device=device)

    probs, mags, preds = [], [], []
    for batch in loader:
        x_batch, meta_batch, _, _ = unpack_batch(batch, condition_mode)
        logits, magnitude_raw = model(x_batch, meta_batch)
        probability = torch.sigmoid(logits)
        magnitude = magnitude_from_raw(magnitude_raw, max_damage)
        pred = final_prediction(probability, magnitude, threshold, final_mode)

        probs.append(probability.detach().cpu().numpy())
        mags.append(magnitude.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    return {
        "probability": np.concatenate(probs, axis=0),
        "magnitude": np.concatenate(mags, axis=0),
        "prediction": np.concatenate(preds, axis=0),
    }


def compute_metrics(
    y_true: np.ndarray,
    mask_true: np.ndarray,
    probability: np.ndarray,
    magnitude: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    """Compute regression and classification metrics. 计算回归和分类指标。"""
    error = y_pred - y_true
    pred_mask = (probability >= threshold).astype(np.float32)

    tp = np.sum((pred_mask == 1) & (mask_true == 1), axis=0)
    fp = np.sum((pred_mask == 1) & (mask_true == 0), axis=0)
    fn = np.sum((pred_mask == 0) & (mask_true == 1), axis=0)
    tn = np.sum((pred_mask == 0) & (mask_true == 0), axis=0)

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1.0e-8)

    return {
        "mae_overall": float(np.mean(np.abs(error))),
        "rmse_overall": float(np.sqrt(np.mean(error ** 2))),
        "mae_per_story": np.mean(np.abs(error), axis=0).tolist(),
        "rmse_per_story": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "negative_prediction_ratio": float(np.mean(y_pred < 0)),
        "probability_mean": float(np.mean(probability)),
        "magnitude_mean": float(np.mean(magnitude)),
        "true_positive_ratio": float(np.mean(mask_true)),
        "predicted_positive_ratio": float(np.mean(pred_mask)),
        "exact_damage_mask_match_ratio": float(np.mean(np.all(pred_mask == mask_true, axis=1))),
        "precision_per_story": precision.tolist(),
        "recall_per_story": recall.tolist(),
        "f1_per_story": f1.tolist(),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "tp_per_story": tp.astype(int).tolist(),
        "fp_per_story": fp.astype(int).tolist(),
        "fn_per_story": fn.astype(int).tolist(),
        "tn_per_story": tn.astype(int).tolist(),
    }


def save_predictions_csv(
    output_path: Path,
    split_name: str,
    case_ids: np.ndarray,
    y_true: np.ndarray,
    mask_true: np.ndarray,
    probability: np.ndarray,
    magnitude: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> None:
    """
    Save prediction CSV compatible with plot_mlp_predictions.py.
    中文：保存预测 CSV；前面的 y_true/y_pred 列兼容已有绘图脚本。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_targets = y_true.shape[1]

    fieldnames = ["split", "sample_id", "case_id"]
    fieldnames += [f"y_true_story_{i+1}" for i in range(n_targets)]
    fieldnames += [f"y_pred_story_{i+1}" for i in range(n_targets)]
    fieldnames += [f"p_damage_story_{i+1}" for i in range(n_targets)]
    fieldnames += [f"magnitude_story_{i+1}" for i in range(n_targets)]
    fieldnames += [f"mask_true_story_{i+1}" for i in range(n_targets)]
    fieldnames += [f"mask_pred_story_{i+1}" for i in range(n_targets)]
    fieldnames += [f"error_story_{i+1}" for i in range(n_targets)]
    fieldnames += ["sample_mae", "sample_rmse", "has_negative_prediction"]

    mask_pred = (probability >= threshold).astype(np.float32)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample_id in range(y_true.shape[0]):
            error = y_pred[sample_id] - y_true[sample_id]
            row = {
                "split": split_name,
                "sample_id": sample_id,
                "case_id": int(case_ids[sample_id]),
                "sample_mae": float(np.mean(np.abs(error))),
                "sample_rmse": float(np.sqrt(np.mean(error ** 2))),
                "has_negative_prediction": bool(np.any(y_pred[sample_id] < 0)),
            }
            for i in range(n_targets):
                row[f"y_true_story_{i+1}"] = float(y_true[sample_id, i])
                row[f"y_pred_story_{i+1}"] = float(y_pred[sample_id, i])
                row[f"p_damage_story_{i+1}"] = float(probability[sample_id, i])
                row[f"magnitude_story_{i+1}"] = float(magnitude[sample_id, i])
                row[f"mask_true_story_{i+1}"] = int(mask_true[sample_id, i])
                row[f"mask_pred_story_{i+1}"] = int(mask_pred[sample_id, i])
                row[f"error_story_{i+1}"] = float(error[i])

            writer.writerow(row)


def save_loss_curve(history: Dict[str, list[float]], output_path: Path) -> None:
    """Save loss curve. 保存损失曲线。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_total"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_total"], label="Train total")
    plt.plot(epochs, history["val_total"], label="Val total")
    plt.plot(epochs, history["train_cls"], label="Train cls", alpha=0.7)
    plt.plot(epochs, history["val_cls"], label="Val cls", alpha=0.7)
    plt.plot(epochs, history["train_reg"], label="Train reg", alpha=0.7)
    plt.plot(epochs, history["val_reg"], label="Val reg", alpha=0.7)
    plt.plot(epochs, history["train_zero"], label="Train zero", alpha=0.7)
    plt.plot(epochs, history["val_zero"], label="Val zero", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Two-head LSTM loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_two_head(args: argparse.Namespace) -> None:
    """Main training pipeline. 主训练流程。"""
    set_seed(args.seed)
    device = select_device(args.device)

    dataset = load_npz(Path(args.data))
    for key in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        if key not in dataset:
            raise KeyError(f"Required key '{key}' not found.")

    X_train = downsample_sequence(dataset["X_train"].astype(np.float32), args.sequence_stride)
    X_val = downsample_sequence(dataset["X_val"].astype(np.float32), args.sequence_stride)
    X_test = downsample_sequence(dataset["X_test"].astype(np.float32), args.sequence_stride)

    y_train = dataset["y_train"].astype(np.float32)
    y_val = dataset["y_val"].astype(np.float32)
    y_test = dataset["y_test"].astype(np.float32)

    mask_train = make_damage_mask(y_train, args.damage_threshold)
    mask_val = make_damage_mask(y_val, args.damage_threshold)
    mask_test = make_damage_mask(y_test, args.damage_threshold)

    case_id_train = get_case_ids(dataset, "train", X_train.shape[0])
    case_id_val = get_case_ids(dataset, "val", X_val.shape[0])
    case_id_test = get_case_ids(dataset, "test", X_test.shape[0])

    if args.condition_mode == "meta":
        meta_train, meta_mean, meta_std = build_meta_features(dataset, "train")
        meta_val, _, _ = build_meta_features(dataset, "val", meta_mean, meta_std)
        meta_test, _, _ = build_meta_features(dataset, "test", meta_mean, meta_std)
        meta_dim = meta_train.shape[1]
    else:
        meta_train = meta_val = meta_test = None
        meta_mean = meta_std = None
        meta_dim = 0

    train_loader = make_loader(X_train, y_train, mask_train, meta_train, args.batch_size, True, device)
    val_loader = make_loader(X_val, y_val, mask_val, meta_val, args.batch_size, False, device)

    shared_dims = tuple(int(v.strip()) for v in args.shared_head_hidden_dims.split(",") if v.strip())

    model = LSTMTwoHeadDamageModel(
        input_dim=X_train.shape[2],
        output_dim=y_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        meta_dim=meta_dim,
        meta_hidden_dim=args.meta_hidden_dim,
        shared_head_hidden_dims=shared_dims,
    ).to(device)

    bce_loss_fn = build_bce_loss(mask_train, args.use_pos_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_total": [], "val_total": [],
        "train_cls": [], "val_cls": [],
        "train_reg": [], "val_reg": [],
        "train_zero": [], "val_zero": [],
    }

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_items = run_one_epoch(
            model, train_loader, optimizer, bce_loss_fn, args.condition_mode,
            args.max_damage, args.lambda_cls, args.lambda_reg, args.lambda_zero, args.grad_clip
        )
        val_items = run_one_epoch(
            model, val_loader, None, bce_loss_fn, args.condition_mode,
            args.max_damage, args.lambda_cls, args.lambda_reg, args.lambda_zero, args.grad_clip
        )

        history["train_total"].append(train_items["total_loss"])
        history["val_total"].append(val_items["total_loss"])
        history["train_cls"].append(train_items["cls_loss"])
        history["val_cls"].append(val_items["cls_loss"])
        history["train_reg"].append(train_items["reg_loss"])
        history["val_reg"].append(val_items["reg_loss"])
        history["train_zero"].append(train_items["zero_loss"])
        history["val_zero"].append(val_items["zero_loss"])

        if val_items["total_loss"] < best_val - args.early_stop_min_delta:
            best_val = val_items["total_loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d} | "
                f"train_total={train_items['total_loss']:.6f} | "
                f"val_total={val_items['total_loss']:.6f} | "
                f"train_cls={train_items['cls_loss']:.6f} | "
                f"val_cls={val_items['cls_loss']:.6f} | "
                f"train_reg={train_items['reg_loss']:.6f} | "
                f"val_reg={val_items['reg_loss']:.6f} | "
                f"train_zero={train_items['zero_loss']:.6f} | "
                f"val_zero={val_items['zero_loss']:.6f}"
            )

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred = predict_all(model, X_train, y_train, mask_train, meta_train, device, args.condition_mode,
                             args.max_damage, args.prob_threshold, args.final_mode, args.batch_size)
    val_pred = predict_all(model, X_val, y_val, mask_val, meta_val, device, args.condition_mode,
                           args.max_damage, args.prob_threshold, args.final_mode, args.batch_size)
    test_pred = predict_all(model, X_test, y_test, mask_test, meta_test, device, args.condition_mode,
                            args.max_damage, args.prob_threshold, args.final_mode, args.batch_size)

    train_metrics = compute_metrics(y_train, mask_train, train_pred["probability"], train_pred["magnitude"],
                                    train_pred["prediction"], args.prob_threshold)
    val_metrics = compute_metrics(y_val, mask_val, val_pred["probability"], val_pred["magnitude"],
                                  val_pred["prediction"], args.prob_threshold)
    test_metrics = compute_metrics(y_test, mask_test, test_pred["probability"], test_pred["magnitude"],
                                   test_pred["prediction"], args.prob_threshold)

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "LSTMTwoHeadDamageModel",
        "data_path": args.data,
        "device": str(device),
        "condition_mode": args.condition_mode,
        "final_mode": args.final_mode,
        "prob_threshold": args.prob_threshold,
        "damage_threshold": args.damage_threshold,
        "max_damage": args.max_damage,
        "sequence_stride": args.sequence_stride,
        "original_sequence_length": int(dataset["X_train"].shape[1]),
        "used_sequence_length": int(X_train.shape[1]),
        "input_dim": int(X_train.shape[2]),
        "output_dim": int(y_train.shape[1]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "bidirectional": args.bidirectional,
        "dropout": args.dropout,
        "shared_head_hidden_dims": list(shared_dims),
        "meta_dim": meta_dim,
        "meta_mean": None if meta_mean is None else meta_mean.reshape(-1).tolist(),
        "meta_std": None if meta_std is None else meta_std.reshape(-1).tolist(),
        "n_parameters": count_trainable_parameters(model),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history["train_total"]),
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val,
        "lambda_cls": args.lambda_cls,
        "lambda_reg": args.lambda_reg,
        "lambda_zero": args.lambda_zero,
        "use_pos_weight": args.use_pos_weight,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "loss_history": history,
    }

    metrics_path = output_dir / "two_head_lstm_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_predictions_csv(output_dir / "two_head_lstm_predictions_train.csv", "train", case_id_train,
                         y_train, mask_train, train_pred["probability"], train_pred["magnitude"],
                         train_pred["prediction"], args.prob_threshold)
    save_predictions_csv(output_dir / "two_head_lstm_predictions_val.csv", "val", case_id_val,
                         y_val, mask_val, val_pred["probability"], val_pred["magnitude"],
                         val_pred["prediction"], args.prob_threshold)
    save_predictions_csv(output_dir / "two_head_lstm_predictions_test.csv", "test", case_id_test,
                         y_test, mask_test, test_pred["probability"], test_pred["magnitude"],
                         test_pred["prediction"], args.prob_threshold)

    save_loss_curve(history, figures_dir / "two_head_lstm_loss_curve.png")
    torch.save(model.state_dict(), output_dir / "two_head_lstm_model.pt")

    print("\nTwo-head LSTM training completed.")
    print(f"Device: {device}")
    print(f"Condition mode: {args.condition_mode}")
    print(f"Final mode: {args.final_mode}")
    print(f"Probability threshold: {args.prob_threshold}")
    print(f"Used sequence length: {X_train.shape[1]}")
    print(f"Parameters: {count_trainable_parameters(model)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation total loss: {best_val:.6f}")
    print(f"Test MAE overall: {test_metrics['mae_overall']:.6f}")
    print(f"Test RMSE overall: {test_metrics['rmse_overall']:.6f}")
    print(f"Test precision macro: {test_metrics['precision_macro']:.6f}")
    print(f"Test recall macro: {test_metrics['recall_macro']:.6f}")
    print(f"Test F1 macro: {test_metrics['f1_macro']:.6f}")
    print(f"Test exact mask match ratio: {test_metrics['exact_damage_mask_match_ratio']:.6f}")
    print(f"Test predicted positive ratio: {test_metrics['predicted_positive_ratio']:.6f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {output_dir}")
    print(f"Loss curve saved to: {figures_dir / 'two_head_lstm_loss_curve.png'}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. 解析命令行参数。"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="data_processed/debug_plus_100_split_normalized.npz")
    parser.add_argument("--output-dir", type=str, default="results/tables/lstm_two_head_response_100")
    parser.add_argument("--figures-dir", type=str, default="results/figures/lstm_two_head_response_100")

    parser.add_argument("--condition-mode", type=str, default="none", choices=["none", "meta"])
    parser.add_argument("--final-mode", type=str, default="threshold", choices=["threshold", "expectation"])
    parser.add_argument("--prob-threshold", type=float, default=0.5)

    parser.add_argument("--damage-threshold", type=float, default=1.0e-8)
    parser.add_argument("--max-damage", type=float, default=0.5)

    parser.add_argument("--sequence-stride", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--meta-hidden-dim", type=int, default=16)
    parser.add_argument("--shared-head-hidden-dims", type=str, default="64")

    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=5.0)
    parser.add_argument("--lambda-zero", type=float, default=0.5)
    parser.add_argument("--use-pos-weight", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--early-stop-patience", type=int, default=40)
    parser.add_argument("--early-stop-min-delta", type=float, default=1.0e-5)

    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--print-every", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    """Entry point. 程序入口。"""
    args = parse_args()
    train_two_head(args)


if __name__ == "__main__":
    main()
