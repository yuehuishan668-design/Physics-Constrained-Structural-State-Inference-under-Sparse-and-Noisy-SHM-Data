"""
File: src/models/lstm_two_head.py

Purpose / 作用:
    Two-head LSTM for sparse structural damage identification.
    双头 LSTM，用于稀疏结构损伤识别。

Why / 为什么:
    Direct regression with MLP/LSTM tended to output small average damage values.
    之前的 MLP/LSTM 直接回归会产生“均值化预测”：健康楼层被误报小损伤，严重损伤被低估。

    This model separates:
        1) damage existence: is this story damaged?
           损伤存在性分类：该层是否损伤？
        2) damage magnitude: how large is the damage if damaged?
           损伤幅值回归：如果损伤，损伤有多大？
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class LSTMTwoHeadDamageModel(nn.Module):
    """
    Two-head LSTM damage model.
    双头 LSTM 损伤识别模型。

    Inputs / 输入:
        x_seq:  (batch, time, channels)
                当前项目中 channels=4，对应 4 层楼面加速度响应。
        x_meta: (batch, meta_dim), optional
                可选条件变量，如 amplitude_g, frequency_hz, noise_level。

    Outputs / 输出:
        damage_logits:
            classification logits, not probabilities.
            分类头 logits，不是概率；训练脚本会再做 sigmoid。
        magnitude_raw:
            raw magnitude output.
            幅值头原始输出；训练脚本会用 sigmoid * max_damage 限制范围。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.10,
        bidirectional: bool = False,
        meta_dim: int = 0,
        meta_hidden_dim: int = 16,
        shared_head_hidden_dims: Sequence[int] = (64,),
    ) -> None:
        super().__init__()
        # super().__init__(): initialize PyTorch module.
        # 中文：初始化 PyTorch 模型父类。

        self.input_dim = input_dim
        # Number of response channels per time step.
        # 中文：每个时间步的输入通道数；当前为 4 层响应。

        self.output_dim = output_dim
        # Number of story-level outputs.
        # 中文：输出目标数量；当前为 4 层损伤。

        self.hidden_dim = hidden_dim
        # LSTM hidden state size.
        # 中文：LSTM 隐藏状态维度。

        self.num_layers = num_layers
        # Number of stacked LSTM layers.
        # 中文：LSTM 堆叠层数。

        self.bidirectional = bidirectional
        # Whether to use bidirectional LSTM.
        # 中文：是否使用双向 LSTM。

        self.meta_dim = meta_dim
        # Metadata dimension. 0 means response-only.
        # 中文：元信息维度；0 表示只输入结构响应。

        self.num_directions = 2 if bidirectional else 1
        # Bidirectional LSTM has two directions.
        # 中文：双向 LSTM 有两个方向，单向只有一个方向。

        lstm_dropout = dropout if num_layers > 1 else 0.0
        # PyTorch LSTM internal dropout works only for num_layers > 1.
        # 中文：PyTorch 的 LSTM 内部 dropout 只有在多层时生效。

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        # nn.LSTM: sequence encoder.
        # 中文：LSTM 时序编码器，输入形状为 (batch, time, channel)。

        if meta_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_dim, meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # Metadata encoder.
            # 中文：元信息编码器，将 amplitude/frequency/noise 映射成隐藏特征。

            feature_dim = hidden_dim * self.num_directions + meta_hidden_dim
            # Concatenated feature dimension.
            # 中文：时序特征和元信息特征拼接后的维度。
        else:
            self.meta_encoder = None
            # No metadata encoder.
            # 中文：不使用元信息时，不创建编码器。

            feature_dim = hidden_dim * self.num_directions
            # Sequence feature dimension only.
            # 中文：只使用 LSTM 时序特征。

        shared_layers: list[nn.Module] = []
        # Store shared MLP layers after LSTM.
        # 中文：保存 LSTM 后面的共享 MLP 层。

        current_dim = feature_dim
        # Current feature dimension.
        # 中文：当前特征维度。

        for layer_dim in shared_head_hidden_dims:
            shared_layers.append(nn.Linear(current_dim, layer_dim))
            # Linear layer.
            # 中文：线性层。

            shared_layers.append(nn.ReLU())
            # Nonlinear activation.
            # 中文：非线性激活函数。

            shared_layers.append(nn.Dropout(dropout))
            # Dropout regularization.
            # 中文：Dropout 正则化，降低过拟合。

            current_dim = layer_dim
            # Update current dimension.
            # 中文：更新下一层输入维度。

        self.shared_head = nn.Sequential(*shared_layers)
        # Shared feature extractor before the two heads.
        # 中文：两个输出头之前的共享特征层。

        self.classification_head = nn.Linear(current_dim, output_dim)
        # Damage existence head.
        # 中文：损伤存在性分类头，输出每层是否损伤的 logits。

        self.magnitude_head = nn.Linear(current_dim, output_dim)
        # Damage magnitude head.
        # 中文：损伤幅值回归头，输出每层损伤幅值的原始值。

    def _encode_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence by LSTM.
        中文：用 LSTM 编码结构响应时程。
        """
        _, (h_n, _) = self.lstm(x_seq)
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        # 中文：h_n 是各层/各方向最后时间步的隐藏状态。

        if self.bidirectional:
            forward_last = h_n[-2]
            # Last forward hidden state.
            # 中文：最后一层正向隐藏状态。

            backward_last = h_n[-1]
            # Last backward hidden state.
            # 中文：最后一层反向隐藏状态。

            return torch.cat([forward_last, backward_last], dim=1)
            # Concatenate forward and backward features.
            # 中文：拼接正向和反向特征。

        return h_n[-1]
        # Last hidden state for unidirectional LSTM.
        # 中文：单向 LSTM 直接取最后一层隐藏状态。

    def forward(
        self,
        x_seq: torch.Tensor,
        x_meta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        中文：前向传播。
        """
        seq_feature = self._encode_sequence(x_seq)
        # Extract sequence feature.
        # 中文：提取结构响应时序特征。

        if self.meta_dim > 0:
            if x_meta is None:
                raise ValueError("x_meta must be provided when meta_dim > 0.")
                # 中文：如果模型设置为使用元信息，则必须输入 x_meta。

            meta_feature = self.meta_encoder(x_meta)
            # Encode metadata.
            # 中文：编码 amplitude/frequency/noise 等元信息。

            feature = torch.cat([seq_feature, meta_feature], dim=1)
            # Concatenate sequence and metadata features.
            # 中文：拼接时序特征和元信息特征。
        else:
            feature = seq_feature
            # Response-only feature.
            # 中文：只使用结构响应特征。

        shared_feature = self.shared_head(feature)
        # Shared representation.
        # 中文：共享特征表示。

        damage_logits = self.classification_head(shared_feature)
        # Classification logits for damage existence.
        # 中文：分类头输出每层损伤存在性的 logits。

        magnitude_raw = self.magnitude_head(shared_feature)
        # Raw magnitude output.
        # 中文：幅值头输出原始损伤幅值。

        return damage_logits, magnitude_raw
        # Return both heads.
        # 中文：返回两个输出头。


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    中文：统计可训练参数数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
