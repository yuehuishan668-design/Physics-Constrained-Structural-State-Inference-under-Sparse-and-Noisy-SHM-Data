"""
File location:
    src/training/train_mlp.py

Purpose:
    Train the first MLP baseline.

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
    本文件用于训练第一个 PyTorch baseline。
    当前目标不是论文级精度，而是打通：
        feature dataset → DataLoader → MLP → loss → optimizer → metrics → saved results
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLPDamageRegressor, count_trainable_parameters


def set_seed(seed: int) -> None:
    """Set random seeds. 中文说明：设置随机种子，尽量保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    """Select device. 中文说明：选择 CPU / CUDA / MPS。"""
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_feature_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load feature dataset. 中文说明：读取 MLP 特征数据集。"""
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    data = np.load(path, allow_pickle=False)
    out = {key: data[key] for key in data.keys()}
    for key in ["F_train", "F_val", "F_test", "y_train", "y_val", "y_test"]:
        if key not in out:
            raise KeyError(f"Missing key: {key}. Available keys: {list(out.keys())}")
    return out


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    """Create DataLoader. 中文说明：把 NumPy 数组转换成 PyTorch DataLoader。"""
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    # x_tensor：输入特征张量。

    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    # y_tensor：标签张量。

    dataset = TensorDataset(x_tensor, y_tensor)
    # TensorDataset：把输入和标签按样本配对。

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # DataLoader：按 batch 返回数据；训练集通常 shuffle=True。


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Compute MAE/RMSE. 中文说明：计算回归指标。"""
    error = y_pred - y_true
    mae_per_story = np.mean(np.abs(error), axis=0)
    rmse_per_story = np.sqrt(np.mean(error ** 2, axis=0))
    return {
        "mae_overall": float(np.mean(np.abs(error))),
        "rmse_overall": float(np.sqrt(np.mean(error ** 2))),
        "mae_per_story": mae_per_story.tolist(),
        "rmse_per_story": rmse_per_story.tolist(),
        "negative_prediction_ratio": float(np.mean(y_pred < 0.0)),
        "over_one_prediction_ratio": float(np.mean(y_pred > 1.0)),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    """Train one epoch. 中文说明：训练一个 epoch，返回平均训练损失。"""
    model.train()
    total_loss = 0.0
    total_n = 0

    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        # 清空上一 batch 的梯度。

        pred = model(x_batch)
        # 前向传播。

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
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    """Evaluate loss. 中文说明：计算验证/测试平均损失。"""
    model.eval()
    total_loss = 0.0
    total_n = 0

    for x_batch, y_batch in loader:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        n = x_batch.shape[0]
        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict. 中文说明：使用模型预测，并返回 NumPy 数组。"""
    model.eval()
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    pred = model(x_tensor)
    return pred.detach().cpu().numpy()


def save_loss_curve(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    """Save loss curve. 中文说明：保存训练/验证 loss 曲线。"""
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


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, split: str) -> None:
    """Save predictions to CSV. 中文说明：保存预测值和真实值，便于人工检查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_targets = y_true.shape[1]
    columns = ["split", "sample_id"]
    columns += [f"y_true_story_{i+1}" for i in range(n_targets)]
    columns += [f"y_pred_story_{i+1}" for i in range(n_targets)]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for i in range(y_true.shape[0]):
            row = {"split": split, "sample_id": i}
            for j in range(n_targets):
                row[f"y_true_story_{j+1}"] = float(y_true[i, j])
                row[f"y_pred_story_{j+1}"] = float(y_pred[i, j])
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
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate_loss(model, test_loader, loss_fn)

    pred_train = predict(model, F_train, device)
    pred_val = predict(model, F_val, device)
    pred_test = predict(model, F_test, device)

    metrics = {
        "device": str(device),
        "n_parameters": count_trainable_parameters(model),
        "input_dim": int(F_train.shape[1]),
        "output_dim": int(y_train.shape[1]),
        "best_val_loss_mse": float(best_val),
        "test_loss_mse": float(test_loss),
        "train": regression_metrics(y_train, pred_train),
        "val": regression_metrics(y_val, pred_val),
        "test": regression_metrics(y_test, pred_test),
    }

    tables_dir = Path(args.output_dir)
    figs_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / "mlp_debug_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_predictions(tables_dir / "mlp_debug_predictions_train.csv", y_train, pred_train, "train")
    save_predictions(tables_dir / "mlp_debug_predictions_val.csv", y_val, pred_val, "val")
    save_predictions(tables_dir / "mlp_debug_predictions_test.csv", y_test, pred_test, "test")
    save_loss_curve(train_losses, val_losses, figs_dir / "mlp_debug_loss_curve.png")
    torch.save(model.state_dict(), tables_dir / "mlp_debug_model.pt")

    print("\nMLP baseline training completed.")
    print(f"Device: {device}")
    print(f"Parameters: {count_trainable_parameters(model)}")
    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test MAE overall: {metrics['test']['mae_overall']:.6f}")
    print(f"Test RMSE overall: {metrics['test']['rmse_overall']:.6f}")
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
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
