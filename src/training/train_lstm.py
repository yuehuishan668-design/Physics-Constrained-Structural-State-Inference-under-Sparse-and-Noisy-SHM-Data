"""
File location:
    src/training/train_lstm.py

Purpose:
    Train LSTM sequence baselines for story-level stiffness degradation regression.

Input:
    data_processed/debug_plus_100_split_normalized.npz

Outputs:
    results/tables/<experiment_name>/lstm_metrics.json
    results/tables/<experiment_name>/lstm_predictions_train.csv
    results/tables/<experiment_name>/lstm_predictions_val.csv
    results/tables/<experiment_name>/lstm_predictions_test.csv
    results/tables/<experiment_name>/lstm_model.pt
    results/figures/<experiment_name>/lstm_loss_curve.png

中文说明：
    本文件用于训练 LSTM baseline。
    相比 MLP，LSTM 直接读取结构响应时程 X_train / X_val / X_test。

    支持两种训练版本：
        1. condition-mode none：只输入结构响应时程；
        2. condition-mode meta：输入结构响应时程 + amplitude_g / frequency_hz / noise_level。

    当前 100-case 阶段仍是 debug-plus，不是论文最终实验。
"""

from __future__ import annotations
# 延迟类型注解解析。

import argparse
# 解析命令行参数。

import csv
# 保存预测表。

import json
# 保存指标。

import random
# 固定随机种子。

from pathlib import Path
# 路径处理。

from typing import Dict
# 类型注解。

import matplotlib.pyplot as plt
# 绘制 loss 曲线。

import numpy as np
# 数组处理。

import torch
# PyTorch 主模块。

from torch import nn
# 神经网络模块。

from torch.utils.data import DataLoader, TensorDataset
# DataLoader 用于 batch 训练；TensorDataset 用于封装张量数据。

from src.models.lstm import LSTMDamageRegressor, count_trainable_parameters
# 导入 LSTM 模型。


def set_seed(seed: int) -> None:
    """Set random seeds. 中文：固定随机种子。"""

    random.seed(seed)
    # 固定 Python 随机数。

    np.random.seed(seed)
    # 固定 NumPy 随机数。

    torch.manual_seed(seed)
    # 固定 PyTorch CPU 随机数。

    if torch.cuda.is_available():
        # 如果有 CUDA GPU。
        torch.cuda.manual_seed_all(seed)
        # 固定所有 CUDA 设备随机数。


def select_device(device_arg: str) -> torch.device:
    """Select training device. 中文：选择训练设备。"""

    if device_arg != "auto":
        # 用户显式指定 cpu / mps / cuda。
        return torch.device(device_arg)

    if torch.cuda.is_available():
        # NVIDIA GPU 优先。
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Mac Apple Silicon 使用 MPS。
        return torch.device("mps")

    return torch.device("cpu")
    # 默认 CPU。


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load .npz as dict. 中文：读取 npz 文件。"""

    if not path.exists():
        # 文件不存在则报错。
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path, allow_pickle=False)
    # 读取 npz。

    return {key: data[key] for key in data.keys()}
    # 转为普通字典。


def downsample_sequence(X: np.ndarray, stride: int) -> np.ndarray:
    """
    Downsample along time axis.

    中文说明：
        原始序列长度为 2000。stride=10 时，序列长度变为 200。
        这样训练更快，也能避免 100-case 阶段 LSTM 过慢。
    """

    if stride <= 0:
        # stride 必须为正数。
        raise ValueError("sequence_stride must be positive.")

    return X[:, ::stride, :]
    # 对时间维进行间隔采样。


def build_meta_features(
    dataset: Dict[str, np.ndarray],
    split: str,
    meta_mean: np.ndarray | None = None,
    meta_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build metadata features: amplitude_g, frequency_hz, noise_level.

    中文说明：
        构造并标准化元信息特征。
        标准化参数只应从训练集计算，验证集和测试集使用训练集统计量。
    """

    amp_key = f"amplitude_g_{split}"
    # 当前 split 的激励幅值键名。

    freq_key = f"frequency_hz_{split}"
    # 当前 split 的激励频率键名。

    noise_key = f"noise_level_{split}"
    # 当前 split 的噪声水平键名。

    for key in [amp_key, freq_key, noise_key]:
        # 检查必要键名。
        if key not in dataset:
            raise KeyError(f"Required metadata key not found: {key}")

    meta = np.stack(
        [dataset[amp_key], dataset[freq_key], dataset[noise_key]],
        axis=1,
    ).astype(np.float32)
    # meta 形状为 (n_samples, 3)。

    if meta_mean is None or meta_std is None:
        # 训练集第一次调用时，计算均值和标准差。

        meta_mean = np.mean(meta, axis=0, keepdims=True)
        # 训练集元信息均值。

        meta_std = np.std(meta, axis=0, keepdims=True)
        # 训练集元信息标准差。

        meta_std = np.where(meta_std < 1.0e-8, 1.0, meta_std)
        # 防止标准差为 0。

    meta_norm = (meta - meta_mean) / meta_std
    # 标准化元信息。

    return meta_norm.astype(np.float32), meta_mean.astype(np.float32), meta_std.astype(np.float32)
    # 返回标准化后的 meta 以及训练集均值/标准差。


def get_case_ids(dataset: Dict[str, np.ndarray], split: str, n_samples: int) -> np.ndarray:
    """Get original case ids. 中文：读取原始 case_id。"""

    key = f"case_id_{split}"
    # 例如 case_id_test。

    if key in dataset:
        # 如果存在原始 case_id。
        return dataset[key].astype(int)

    return np.arange(n_samples, dtype=int)
    # 如果不存在，则用内部编号代替。


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create DataLoader. 中文：创建 PyTorch 数据加载器。"""

    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    # 响应时程张量，形状 (n_samples, seq_len, n_channels)。

    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    # 标签张量，形状 (n_samples, 4)。

    if meta is None:
        # response-only 版本。
        tensor_dataset = TensorDataset(x_tensor, y_tensor)
        # 数据集中只有 x 和 y。

    else:
        # response + meta 版本。
        meta_tensor = torch.tensor(meta, dtype=torch.float32, device=device)
        # 元信息张量，形状 (n_samples, 3)。

        tensor_dataset = TensorDataset(x_tensor, meta_tensor, y_tensor)
        # 数据集中包含 x、meta、y。

    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
    # 返回 DataLoader。


def apply_output_mode(raw_pred: torch.Tensor, output_mode: str, max_damage: float) -> torch.Tensor:
    """Apply output constraint. 中文：施加输出物理边界约束。"""

    if output_mode == "linear":
        # 无约束输出。
        return raw_pred

    if output_mode == "sigmoid":
        # 输出限制在 (0, max_damage)。
        return max_damage * torch.sigmoid(raw_pred)

    if output_mode == "clamp":
        # 直接截断到 [0, max_damage]。
        return torch.clamp(raw_pred, min=0.0, max=max_damage)

    raise ValueError(f"Unknown output_mode: {output_mode}")
    # 未知模式报错。


def forward_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    condition_mode: str,
    output_mode: str,
    max_damage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward one batch. 中文：统一处理两种输入模式。"""

    if condition_mode == "none":
        # 只输入响应时程。

        x_batch, y_batch = batch
        # batch 包含 x 和 y。

        raw_pred = model(x_batch)
        # 模型前向传播。

    elif condition_mode == "meta":
        # 输入响应时程 + 元信息。

        x_batch, meta_batch, y_batch = batch
        # batch 包含 x、meta、y。

        raw_pred = model(x_batch, meta_batch)
        # 模型前向传播。

    else:
        # 未知模式。
        raise ValueError(f"Unknown condition_mode: {condition_mode}")

    pred = apply_output_mode(raw_pred, output_mode=output_mode, max_damage=max_damage)
    # 对原始输出施加 linear / sigmoid / clamp 处理。

    return pred, y_batch
    # 返回预测值和真实值。


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    condition_mode: str,
    output_mode: str,
    max_damage: float,
    grad_clip: float,
) -> float:
    """Train one epoch. 中文：训练一个 epoch。"""

    model.train()
    # 训练模式。

    total_loss = 0.0
    # 累计损失。

    total_samples = 0
    # 样本数。

    for batch in loader:
        # 遍历 batch。

        optimizer.zero_grad()
        # 清空上一轮梯度。

        pred, y_batch = forward_batch(model, batch, condition_mode, output_mode, max_damage)
        # 前向传播。

        loss = loss_fn(pred, y_batch)
        # 计算 MSE loss。

        loss.backward()
        # 反向传播。

        if grad_clip > 0:
            # 如果启用梯度裁剪。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            # 防止 LSTM 梯度爆炸。

        optimizer.step()
        # 更新模型参数。

        batch_size = y_batch.shape[0]
        # 当前 batch 样本数。

        total_loss += float(loss.item()) * batch_size
        # 累计总损失。

        total_samples += batch_size
        # 累计样本数。

    return total_loss / max(total_samples, 1)
    # 返回样本平均损失。


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    condition_mode: str,
    output_mode: str,
    max_damage: float,
) -> float:
    """Evaluate average loss. 中文：计算验证/测试平均损失。"""

    model.eval()
    # 评估模式。

    total_loss = 0.0
    # 累计损失。

    total_samples = 0
    # 样本数。

    for batch in loader:
        # 遍历 batch。

        pred, y_batch = forward_batch(model, batch, condition_mode, output_mode, max_damage)
        # 前向传播。

        loss = loss_fn(pred, y_batch)
        # 计算 loss。

        batch_size = y_batch.shape[0]
        # batch 样本数。

        total_loss += float(loss.item()) * batch_size
        # 累计总损失。

        total_samples += batch_size
        # 累计样本数。

    return total_loss / max(total_samples, 1)
    # 返回平均损失。


@torch.no_grad()
def predict_in_batches(
    model: nn.Module,
    X: np.ndarray,
    meta: np.ndarray | None,
    device: torch.device,
    condition_mode: str,
    output_mode: str,
    max_damage: float,
    batch_size: int,
    output_dim: int,
) -> np.ndarray:
    """Predict split data. 中文：分 batch 预测，避免一次性占用过多内存。"""

    model.eval()
    # 评估模式。

    dummy_y = np.zeros((X.shape[0], output_dim), dtype=np.float32)
    # 占位标签，只是为了复用 make_loader。

    loader = make_loader(X, dummy_y, meta, batch_size=batch_size, shuffle=False, device=device)
    # 创建预测 DataLoader。

    pred_list = []
    # 保存每个 batch 的预测。

    for batch in loader:
        # 遍历 batch。

        pred, _ = forward_batch(model, batch, condition_mode, output_mode, max_damage)
        # 前向预测。

        pred_list.append(pred.detach().cpu().numpy())
        # 转到 CPU 并保存为 NumPy。

    return np.concatenate(pred_list, axis=0)
    # 拼接所有 batch。


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """Compute regression metrics. 中文：计算回归指标。"""

    error = y_pred - y_true
    # 误差。

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
    # 返回指标字典。


def save_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    """Save loss curve. 中文：保存 loss 曲线。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。

    epochs = np.arange(1, len(train_losses) + 1)
    # epoch 序号。

    plt.figure(figsize=(7, 4))
    # 创建图像。

    plt.plot(epochs, train_losses, label="Train loss")
    # 训练损失。

    plt.plot(epochs, val_losses, label="Validation loss")
    # 验证损失。

    plt.xlabel("Epoch")
    # 横轴。

    plt.ylabel("MSE loss")
    # 纵轴。

    plt.title("LSTM baseline loss curve")
    # 标题。

    plt.legend()
    # 图例。

    plt.tight_layout()
    # 布局调整。

    plt.savefig(output_path, dpi=200)
    # 保存图片。

    plt.close()
    # 关闭图像。


def save_predictions_csv(
    output_path: Path,
    split_name: str,
    case_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Save prediction CSV.

    中文说明：
        输出列名与 MLP 预测文件保持兼容，因此可以复用 plot_mlp_predictions.py。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。

    n_targets = y_true.shape[1]
    # 目标数量，当前为 4 层。

    fieldnames = ["split", "sample_id", "case_id"]
    # 基础列。

    for i in range(n_targets):
        fieldnames.append(f"y_true_story_{i + 1}")
        # 真实值列。

    for i in range(n_targets):
        fieldnames.append(f"y_pred_story_{i + 1}")
        # 预测值列。

    for i in range(n_targets):
        fieldnames.append(f"error_story_{i + 1}")
        # 误差列。

    fieldnames.extend(["sample_mae", "sample_rmse", "has_negative_prediction"])
    # 样本级指标列。

    with output_path.open("w", newline="", encoding="utf-8") as f:
        # 打开 CSV。

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # CSV 写入器。

        writer.writeheader()
        # 写入表头。

        for sample_id in range(y_true.shape[0]):
            # 遍历样本。

            error = y_pred[sample_id] - y_true[sample_id]
            # 当前样本误差。

            row = {
                "split": split_name,
                "sample_id": sample_id,
                "case_id": int(case_ids[sample_id]),
                "sample_mae": float(np.mean(np.abs(error))),
                "sample_rmse": float(np.sqrt(np.mean(error ** 2))),
                "has_negative_prediction": bool(np.any(y_pred[sample_id] < 0.0)),
            }
            # 基础字段。

            for i in range(n_targets):
                row[f"y_true_story_{i + 1}"] = float(y_true[sample_id, i])
                # 写真实值。

            for i in range(n_targets):
                row[f"y_pred_story_{i + 1}"] = float(y_pred[sample_id, i])
                # 写预测值。

            for i in range(n_targets):
                row[f"error_story_{i + 1}"] = float(error[i])
                # 写误差。

            writer.writerow(row)
            # 写入一行。


def train_lstm(args: argparse.Namespace) -> None:
    """Main training pipeline. 中文：LSTM 训练主流程。"""

    set_seed(args.seed)
    # 固定随机种子。

    device = select_device(args.device)
    # 选择训练设备。

    data = load_npz(Path(args.data))
    # 读取 split normalized 数据。

    for key in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        # 检查必要字段。
        if key not in data:
            raise KeyError(f"Required key not found: {key}")

    X_train = downsample_sequence(data["X_train"].astype(np.float32), args.sequence_stride)
    # 训练响应时程，降采样。

    X_val = downsample_sequence(data["X_val"].astype(np.float32), args.sequence_stride)
    # 验证响应时程，降采样。

    X_test = downsample_sequence(data["X_test"].astype(np.float32), args.sequence_stride)
    # 测试响应时程，降采样。

    y_train = data["y_train"].astype(np.float32)
    # 训练标签。

    y_val = data["y_val"].astype(np.float32)
    # 验证标签。

    y_test = data["y_test"].astype(np.float32)
    # 测试标签。

    case_id_train = get_case_ids(data, "train", X_train.shape[0])
    # 训练集 case_id。

    case_id_val = get_case_ids(data, "val", X_val.shape[0])
    # 验证集 case_id。

    case_id_test = get_case_ids(data, "test", X_test.shape[0])
    # 测试集 case_id。

    if args.condition_mode == "meta":
        # 使用元信息条件变量。

        meta_train, meta_mean, meta_std = build_meta_features(data, "train")
        # 用训练集计算标准化参数。

        meta_val, _, _ = build_meta_features(data, "val", meta_mean=meta_mean, meta_std=meta_std)
        # 验证集用训练集统计量标准化。

        meta_test, _, _ = build_meta_features(data, "test", meta_mean=meta_mean, meta_std=meta_std)
        # 测试集用训练集统计量标准化。

        meta_dim = meta_train.shape[1]
        # 当前为 3。

    elif args.condition_mode == "none":
        # 不使用元信息。

        meta_train = None
        meta_val = None
        meta_test = None
        meta_mean = None
        meta_std = None
        meta_dim = 0

    else:
        # 未知模式。
        raise ValueError(f"Unknown condition_mode: {args.condition_mode}")

    train_loader = make_loader(X_train, y_train, meta_train, args.batch_size, True, device)
    # 训练 DataLoader。

    val_loader = make_loader(X_val, y_val, meta_val, args.batch_size, False, device)
    # 验证 DataLoader。

    test_loader = make_loader(X_test, y_test, meta_test, args.batch_size, False, device)
    # 测试 DataLoader。

    head_hidden_dims = tuple(int(v.strip()) for v in args.head_hidden_dims.split(",") if v.strip())
    # 解析回归头隐藏层。

    model = LSTMDamageRegressor(
        input_dim=X_train.shape[2],
        output_dim=y_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        meta_dim=meta_dim,
        meta_hidden_dim=args.meta_hidden_dim,
        head_hidden_dims=head_hidden_dims,
    ).to(device)
    # 创建模型。

    loss_fn = nn.MSELoss()
    # MSE 损失。

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Adam 优化器。

    train_losses: list[float] = []
    # 训练 loss。

    val_losses: list[float] = []
    # 验证 loss。

    best_val_loss = float("inf")
    # 最佳验证损失。

    best_state_dict = None
    # 最佳模型参数。

    for epoch in range(1, args.epochs + 1):
        # epoch 循环。

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            condition_mode=args.condition_mode,
            output_mode=args.output_mode,
            max_damage=args.max_damage,
            grad_clip=args.grad_clip,
        )
        # 训练一个 epoch。

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            condition_mode=args.condition_mode,
            output_mode=args.output_mode,
            max_damage=args.max_damage,
        )
        # 验证。

        train_losses.append(train_loss)
        # 记录 train loss。

        val_losses.append(val_loss)
        # 记录 val loss。

        if val_loss < best_val_loss:
            # 如果验证集更好。

            best_val_loss = val_loss
            # 更新最佳验证损失。

            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            # 保存最佳模型参数到 CPU。

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            # 控制打印频率。
            print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if best_state_dict is not None:
        # 如果保存了最佳模型。
        model.load_state_dict(best_state_dict)
        # 恢复最佳模型。

    test_loss = evaluate_loss(model, test_loader, loss_fn, args.condition_mode, args.output_mode, args.max_damage)
    # 测试集 loss。

    y_train_pred = predict_in_batches(model, X_train, meta_train, device, args.condition_mode, args.output_mode, args.max_damage, args.batch_size, y_train.shape[1])
    # 训练集预测。

    y_val_pred = predict_in_batches(model, X_val, meta_val, device, args.condition_mode, args.output_mode, args.max_damage, args.batch_size, y_val.shape[1])
    # 验证集预测。

    y_test_pred = predict_in_batches(model, X_test, meta_test, device, args.condition_mode, args.output_mode, args.max_damage, args.batch_size, y_test.shape[1])
    # 测试集预测。

    train_metrics = compute_metrics(y_train, y_train_pred)
    # 训练指标。

    val_metrics = compute_metrics(y_val, y_val_pred)
    # 验证指标。

    test_metrics = compute_metrics(y_test, y_test_pred)
    # 测试指标。

    output_dir = Path(args.output_dir)
    # 表格输出目录。

    figures_dir = Path(args.figures_dir)
    # 图片输出目录。

    output_dir.mkdir(parents=True, exist_ok=True)
    # 创建表格目录。

    figures_dir.mkdir(parents=True, exist_ok=True)
    # 创建图片目录。

    metrics = {
        "model": "LSTMDamageRegressor",
        "device": str(device),
        "n_parameters": count_trainable_parameters(model),
        "data_path": args.data,
        "condition_mode": args.condition_mode,
        "output_mode": args.output_mode,
        "max_damage": float(args.max_damage),
        "sequence_stride": int(args.sequence_stride),
        "original_sequence_length": int(data["X_train"].shape[1]),
        "used_sequence_length": int(X_train.shape[1]),
        "input_dim": int(X_train.shape[2]),
        "output_dim": int(y_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "bidirectional": bool(args.bidirectional),
        "dropout": float(args.dropout),
        "head_hidden_dims": list(head_hidden_dims),
        "meta_dim": int(meta_dim),
        "meta_mean": None if meta_mean is None else meta_mean.reshape(-1).tolist(),
        "meta_std": None if meta_std is None else meta_std.reshape(-1).tolist(),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "best_val_loss_mse": float(best_val_loss),
        "test_loss_mse": float(test_loss),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    # 汇总指标。

    metrics_path = output_dir / "lstm_metrics.json"
    # 指标路径。

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        # 保存指标。

    save_predictions_csv(output_dir / "lstm_predictions_train.csv", "train", case_id_train, y_train, y_train_pred)
    # 保存训练集预测。

    save_predictions_csv(output_dir / "lstm_predictions_val.csv", "val", case_id_val, y_val, y_val_pred)
    # 保存验证集预测。

    save_predictions_csv(output_dir / "lstm_predictions_test.csv", "test", case_id_test, y_test, y_test_pred)
    # 保存测试集预测。

    save_loss_curve(train_losses, val_losses, figures_dir / "lstm_loss_curve.png")
    # 保存 loss 曲线。

    model_path = output_dir / "lstm_model.pt"
    # 模型保存路径。

    torch.save(model.state_dict(), model_path)
    # 保存模型权重。

    print("\nLSTM baseline training completed.")
    print(f"Device: {device}")
    print(f"Condition mode: {args.condition_mode}")
    print(f"Output mode: {args.output_mode}")
    print(f"Sequence stride: {args.sequence_stride}")
    print(f"Used sequence length: {X_train.shape[1]}")
    print(f"Parameters: {count_trainable_parameters(model)}")
    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test MAE overall: {test_metrics['mae_overall']:.6f}")
    print(f"Test RMSE overall: {test_metrics['rmse_overall']:.6f}")
    print(f"Negative prediction ratio: {test_metrics['negative_prediction_ratio']:.6f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to: {model_path}")
    print(f"Loss curve saved to: {figures_dir / 'lstm_loss_curve.png'}")
    # 打印摘要。


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. 中文：解析命令行参数。"""

    parser = argparse.ArgumentParser()
    # 创建参数解析器。

    parser.add_argument("--data", type=str, default="data_processed/debug_plus_100_split_normalized.npz")
    # 输入数据。

    parser.add_argument("--output-dir", type=str, default="results/tables/lstm_response_100")
    # 表格输出目录。

    parser.add_argument("--figures-dir", type=str, default="results/figures/lstm_response_100")
    # 图片输出目录。

    parser.add_argument("--condition-mode", type=str, default="none", choices=["none", "meta"])
    # none：仅响应；meta：响应 + amplitude/frequency/noise。

    parser.add_argument("--output-mode", type=str, default="linear", choices=["linear", "sigmoid", "clamp"])
    # 输出模式。

    parser.add_argument("--max-damage", type=float, default=0.5)
    # 最大损伤约束。

    parser.add_argument("--sequence-stride", type=int, default=10)
    # 序列降采样步长，2000 点变为 200 点。

    parser.add_argument("--hidden-dim", type=int, default=64)
    # LSTM 隐藏层维度。

    parser.add_argument("--num-layers", type=int, default=1)
    # LSTM 层数。

    parser.add_argument("--dropout", type=float, default=0.10)
    # Dropout。

    parser.add_argument("--bidirectional", action="store_true")
    # 是否双向 LSTM。

    parser.add_argument("--meta-hidden-dim", type=int, default=16)
    # 元信息编码维度。

    parser.add_argument("--head-hidden-dims", type=str, default="64")
    # 回归头隐藏层。

    parser.add_argument("--epochs", type=int, default=200)
    # 训练轮数。

    parser.add_argument("--batch-size", type=int, default=8)
    # batch size。

    parser.add_argument("--lr", type=float, default=1.0e-3)
    # 学习率。

    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    # 权重衰减。

    parser.add_argument("--grad-clip", type=float, default=1.0)
    # 梯度裁剪。

    parser.add_argument("--seed", type=int, default=20260625)
    # 随机种子。

    parser.add_argument("--device", type=str, default="auto")
    # 设备。

    parser.add_argument("--print-every", type=int, default=20)
    # 打印间隔。

    return parser.parse_args()
    # 返回参数。


def main() -> None:
    """Command-line entry point. 中文：命令行入口。"""

    args = parse_args()
    # 解析参数。

    train_lstm(args)
    # 执行训练。


if __name__ == "__main__":
    main()
    # 直接运行时执行 main。
