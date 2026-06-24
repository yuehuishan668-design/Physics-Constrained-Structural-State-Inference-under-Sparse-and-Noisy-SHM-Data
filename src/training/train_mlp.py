"""
File location:
    src/training/train_mlp.py

Purpose:
    Train the MLP baseline for story-level stiffness degradation regression,
    while saving richer diagnostic outputs.

Input:
    data_processed/debug_features_mlp.npz

Outputs:
    results/tables/mlp_debug_metrics.json
    results/tables/mlp_debug_predictions_train.csv
    results/tables/mlp_debug_predictions_val.csv
    results/tables/mlp_debug_predictions_test.csv
    results/tables/mlp_debug_model.pt
    results/figures/mlp_debug_loss_curve.png

中文说明：
    这是 MLP baseline 的改进版训练脚本。相比上一版，它增加了三点：
    1. 预测 CSV 中保存原始 OpenSees case_id，方便回查具体样本；
    2. 预测 CSV 中保存每层误差、样本级 MAE/RMSE、是否存在负值预测；
    3. 支持 output_mode：linear / sigmoid / clamp，用于检查物理边界约束的影响。
"""

from __future__ import annotations
# 延迟类型注解解析，提高兼容性。

import argparse
# argparse：解析命令行参数。

import csv
# csv：保存预测结果表格。

import json
# json：保存评价指标。

import random
# random：设置 Python 随机种子。

from pathlib import Path
# Path：处理文件路径。

from typing import Dict
# Dict：类型注解。

import matplotlib.pyplot as plt
# matplotlib：绘制 loss 曲线。

import numpy as np
# numpy：读取数据、计算指标。

import torch
# torch：PyTorch 主模块。

from torch import nn
# nn：神经网络模块，例如 MSELoss。

from torch.utils.data import DataLoader, TensorDataset
# TensorDataset：把输入和标签配对；DataLoader：按 batch 加载数据。

from src.models.mlp import MLPDamageRegressor, count_trainable_parameters
# 导入项目中已经定义好的 MLP 模型。


def set_seed(seed: int) -> None:
    """Set random seeds. 中文说明：固定随机种子，使结果尽量可复现。"""
    random.seed(seed)
    # 设置 Python random 随机种子。
    np.random.seed(seed)
    # 设置 NumPy 随机种子。
    torch.manual_seed(seed)
    # 设置 PyTorch CPU 随机种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 设置 CUDA 随机种子。


def select_device(device_arg: str) -> torch.device:
    """Select cpu/cuda/mps. 中文说明：自动或手动选择训练设备。"""
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_feature_dataset(path: Path) -> Dict[str, np.ndarray]:
    """Load MLP feature dataset. 中文说明：读取 debug_features_mlp.npz。"""
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    data = np.load(path, allow_pickle=False)
    # allow_pickle=False：避免加载任意 Python 对象，更安全。
    dataset = {key: data[key] for key in data.keys()}
    # 转换为普通字典。
    required = ["F_train", "F_val", "F_test", "y_train", "y_val", "y_test"]
    for key in required:
        if key not in dataset:
            raise KeyError(f"Required key '{key}' not found. Available keys: {list(dataset.keys())}")
    return dataset


def get_case_ids(dataset: Dict[str, np.ndarray], split: str, n_samples: int) -> np.ndarray:
    """Get original case_id. 中文说明：读取原始 OpenSees 样本编号。"""
    key = f"case_id_{split}"
    # 例如 split='test'，则读取 case_id_test。
    if key in dataset:
        return dataset[key].astype(int)
    return np.arange(n_samples, dtype=int)
    # 如果数据里没有 case_id，则用内部编号代替。


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    """Create DataLoader. 中文说明：把 NumPy 数组封装成 PyTorch DataLoader。"""
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    # 输入特征张量，形状为 (样本数, 特征数)。
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    # 标签张量，形状为 (样本数, 4)。
    dataset = TensorDataset(x_tensor, y_tensor)
    # TensorDataset 会把 x 和 y 按样本一一配对。
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # DataLoader 负责 batch 读取；训练集通常 shuffle=True。


def apply_output_mode(raw_pred: torch.Tensor, output_mode: str, max_damage: float) -> torch.Tensor:
    """
    Apply optional physical constraint to predictions.

    output_mode:
        linear  : no constraint.
        sigmoid : max_damage * sigmoid(raw_pred), output in (0, max_damage).
        clamp   : clamp(raw_pred, 0, max_damage).

    中文说明：
        损伤比例理论上不应为负，也不应大于 1。
        这里提供三种输出模式，用于比较“无约束 baseline”和“物理边界约束 baseline”。
    """
    if output_mode == "linear":
        return raw_pred
    if output_mode == "sigmoid":
        return max_damage * torch.sigmoid(raw_pred)
    if output_mode == "clamp":
        return torch.clamp(raw_pred, min=0.0, max=max_damage)
    raise ValueError(f"Unknown output_mode: {output_mode}")


def forward_model(model: nn.Module, x: torch.Tensor, output_mode: str, max_damage: float) -> torch.Tensor:
    """Forward pass with optional output constraint. 中文说明：统一处理前向传播和输出约束。"""
    raw_pred = model(x)
    # raw_pred：模型原始线性输出。
    pred = apply_output_mode(raw_pred, output_mode, max_damage)
    # 根据 output_mode 进行约束或不约束。
    return pred


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """Compute MAE/RMSE and invalid-output ratios. 中文说明：计算回归指标和非法预测比例。"""
    error = y_pred - y_true
    # 预测误差。
    mae_per_story = np.mean(np.abs(error), axis=0)
    # 每层 MAE。
    rmse_per_story = np.sqrt(np.mean(error ** 2, axis=0))
    # 每层 RMSE。
    return {
        "mae_overall": float(np.mean(np.abs(error))),
        "rmse_overall": float(np.sqrt(np.mean(error ** 2))),
        "mae_per_story": mae_per_story.tolist(),
        "rmse_per_story": rmse_per_story.tolist(),
        "negative_prediction_ratio": float(np.mean(y_pred < 0.0)),
        "over_one_prediction_ratio": float(np.mean(y_pred > 1.0)),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, output_mode: str, max_damage: float) -> float:
    """Train one epoch. 中文说明：训练一个 epoch，返回平均训练 loss。"""
    model.train()
    # 切换到训练模式，Dropout 等层会启用。
    total_loss = 0.0
    total_n = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        # 清空上一 batch 的梯度。
        pred = forward_model(model, x_batch, output_mode, max_damage)
        # 前向预测。
        loss = loss_fn(pred, y_batch)
        # 计算 MSE loss。
        loss.backward()
        # 反向传播，计算梯度。
        optimizer.step()
        # 更新模型参数。
        n = x_batch.shape[0]
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, output_mode: str, max_damage: float) -> float:
    """Evaluate loss. 中文说明：验证/测试平均 loss。"""
    model.eval()
    # 切换到评估模式，Dropout 关闭。
    total_loss = 0.0
    total_n = 0
    for x_batch, y_batch in loader:
        pred = forward_model(model, x_batch, output_mode, max_damage)
        loss = loss_fn(pred, y_batch)
        n = x_batch.shape[0]
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device, output_mode: str, max_damage: float) -> np.ndarray:
    """Predict and return NumPy array. 中文说明：使用模型预测并转回 NumPy。"""
    model.eval()
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    pred = forward_model(model, x_tensor, output_mode, max_damage)
    return pred.detach().cpu().numpy()


def save_loss_curve(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    """Save loss curve. 中文说明：保存训练和验证 loss 曲线。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("MLP baseline loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_predictions_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray, case_ids: np.ndarray, split: str) -> None:
    """Save prediction table. 中文说明：保存真实值、预测值、误差和 case_id。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_targets = y_true.shape[1]
    columns = ["split", "sample_id", "case_id"]
    columns += [f"y_true_story_{i + 1}" for i in range(n_targets)]
    columns += [f"y_pred_story_{i + 1}" for i in range(n_targets)]
    columns += [f"error_story_{i + 1}" for i in range(n_targets)]
    columns += ["sample_mae", "sample_rmse", "has_negative_prediction"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(y_true.shape[0]):
            error = y_pred[i] - y_true[i]
            row = {
                "split": split,
                "sample_id": i,
                "case_id": int(case_ids[i]),
                "sample_mae": float(np.mean(np.abs(error))),
                "sample_rmse": float(np.sqrt(np.mean(error ** 2))),
                "has_negative_prediction": bool(np.any(y_pred[i] < 0.0)),
            }
            for j in range(n_targets):
                row[f"y_true_story_{j + 1}"] = float(y_true[i, j])
                row[f"y_pred_story_{j + 1}"] = float(y_pred[i, j])
                row[f"error_story_{j + 1}"] = float(error[j])
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    """Run full training pipeline. 中文说明：执行完整训练流程。"""
    set_seed(args.seed)
    device = select_device(args.device)
    data = load_feature_dataset(Path(args.features))

    F_train = data["F_train"].astype(np.float32)
    F_val = data["F_val"].astype(np.float32)
    F_test = data["F_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    case_id_train = get_case_ids(data, "train", F_train.shape[0])
    case_id_val = get_case_ids(data, "val", F_val.shape[0])
    case_id_test = get_case_ids(data, "test", F_test.shape[0])

    train_loader = make_loader(F_train, y_train, args.batch_size, True, device)
    val_loader = make_loader(F_val, y_val, args.batch_size, False, device)
    test_loader = make_loader(F_test, y_test, args.batch_size, False, device)

    model = MLPDamageRegressor(
        input_dim=F_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, args.output_mode, args.max_damage)
        val_loss = evaluate_loss(model, val_loader, loss_fn, args.output_mode, args.max_damage)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate_loss(model, test_loader, loss_fn, args.output_mode, args.max_damage)
    pred_train = predict(model, F_train, device, args.output_mode, args.max_damage)
    pred_val = predict(model, F_val, device, args.output_mode, args.max_damage)
    pred_test = predict(model, F_test, device, args.output_mode, args.max_damage)

    train_metrics = regression_metrics(y_train, pred_train)
    val_metrics = regression_metrics(y_val, pred_val)
    test_metrics = regression_metrics(y_test, pred_test)

    tables_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "device": str(device),
        "n_parameters": count_trainable_parameters(model),
        "input_dim": int(F_train.shape[1]),
        "output_dim": int(y_train.shape[1]),
        "epochs": int(args.epochs),
        "output_mode": args.output_mode,
        "max_damage": float(args.max_damage),
        "best_val_loss_mse": float(best_val),
        "test_loss_mse": float(test_loss),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = tables_dir / "mlp_debug_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_predictions_csv(tables_dir / "mlp_debug_predictions_train.csv", y_train, pred_train, case_id_train, "train")
    save_predictions_csv(tables_dir / "mlp_debug_predictions_val.csv", y_val, pred_val, case_id_val, "val")
    save_predictions_csv(tables_dir / "mlp_debug_predictions_test.csv", y_test, pred_test, case_id_test, "test")
    save_loss_curve(train_losses, val_losses, figures_dir / "mlp_debug_loss_curve.png")
    torch.save(model.state_dict(), tables_dir / "mlp_debug_model.pt")

    print("\nMLP baseline training completed.")
    print(f"Device: {device}")
    print(f"Output mode: {args.output_mode}")
    print(f"Parameters: {count_trainable_parameters(model)}")
    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test MAE overall: {test_metrics['mae_overall']:.6f}")
    print(f"Test RMSE overall: {test_metrics['rmse_overall']:.6f}")
    print(f"Negative prediction ratio: {test_metrics['negative_prediction_ratio']:.6f}")
    print(f"Metrics saved to: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data_processed/debug_features_mlp.npz")
    parser.add_argument("--output-dir", type=str, default="results/tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dims", type=str, default="64,64")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--output-mode", type=str, default="linear", choices=["linear", "sigmoid", "clamp"])
    parser.add_argument("--max-damage", type=float, default=0.50)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
