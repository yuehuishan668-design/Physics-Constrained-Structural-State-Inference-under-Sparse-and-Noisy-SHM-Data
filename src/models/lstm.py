"""
File location:
    src/models/lstm.py

Purpose:
    Define an LSTM baseline for story-level stiffness degradation regression.

Inputs:
    x_seq:  shape = (batch_size, sequence_length, n_response_channels)
    x_meta: shape = (batch_size, meta_dim), optional

Output:
    raw_pred: shape = (batch_size, n_stories)

中文说明：
    本文件定义 LSTM 损伤识别模型。
    相比 MLP 只使用 mean/std/max/min/rms 统计特征，LSTM 直接读取结构响应时程，
    因而可以利用响应的时间顺序、相位关系和动态演化特征。

    支持两种版本：
        Version A: 仅输入结构响应时程；
        Version B: 输入结构响应时程 + 元信息条件变量，例如 amplitude_g、frequency_hz、noise_level。
"""

from __future__ import annotations
# 延迟解析类型注解，降低 Python 版本兼容问题。

from typing import Sequence
# Sequence 用于表示回归头隐藏层维度序列。

import torch
# PyTorch 主模块。

from torch import nn
# nn 是 PyTorch 的神经网络模块。


class LSTMDamageRegressor(nn.Module):
    """
    LSTM damage regressor.

    中文说明：
        从多层结构加速度响应时程中预测各楼层刚度退化比例。
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
        head_hidden_dims: Sequence[int] = (64,),
    ) -> None:
        """
        Initialize LSTM model.

        Args:
            input_dim:
                每个时间步的响应通道数。当前 4 层结构对应 input_dim=4。
            output_dim:
                输出损伤变量数量。当前 4 层结构对应 output_dim=4。
            hidden_dim:
                LSTM 隐藏状态维度。
            num_layers:
                LSTM 堆叠层数。
            dropout:
                Dropout 比例，用于缓解过拟合。
            bidirectional:
                是否使用双向 LSTM。离线诊断可用，在线监测通常不应使用。
            meta_dim:
                元信息维度。0 表示不使用元信息。
            meta_hidden_dim:
                元信息编码后的维度。
            head_hidden_dims:
                最终回归头的隐藏层维度。
        """

        super().__init__()
        # 初始化 nn.Module。所有自定义 PyTorch 模型都需要调用。

        self.input_dim = input_dim
        # 保存输入通道数。

        self.output_dim = output_dim
        # 保存输出目标数。

        self.hidden_dim = hidden_dim
        # 保存隐藏层维度。

        self.num_layers = num_layers
        # 保存 LSTM 层数。

        self.bidirectional = bidirectional
        # 保存是否双向。

        self.meta_dim = meta_dim
        # 保存元信息维度。

        self.num_directions = 2 if bidirectional else 1
        # 双向 LSTM 有两个方向；单向 LSTM 只有一个方向。

        lstm_dropout = dropout if num_layers > 1 else 0.0
        # PyTorch LSTM 的内部 dropout 只在 num_layers > 1 时生效。
        # 如果 num_layers=1，必须设为 0，否则会出现警告。

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        # LSTM 主体。
        # batch_first=True 表示输入为 (batch, time, channels)。

        if meta_dim > 0:
            # 如果使用 amplitude_g / frequency_hz / noise_level 等元信息。

            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_dim, meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # 元信息编码器：把低维条件变量映射为可与 LSTM 特征拼接的向量。

            head_input_dim = hidden_dim * self.num_directions + meta_hidden_dim
            # 回归头输入 = LSTM 特征 + 元信息特征。

        else:
            # 不使用元信息。

            self.meta_encoder = None
            # 没有元信息编码器。

            head_input_dim = hidden_dim * self.num_directions
            # 回归头只接收 LSTM 特征。

        layers: list[nn.Module] = []
        # 保存回归头各层。

        current_dim = head_input_dim
        # 当前输入维度。

        for hidden in head_hidden_dims:
            # 构建多层 MLP 回归头。

            layers.append(nn.Linear(current_dim, hidden))
            # 线性层。

            layers.append(nn.ReLU())
            # 非线性激活。

            layers.append(nn.Dropout(dropout))
            # Dropout。

            current_dim = hidden
            # 更新下一层输入维度。

        layers.append(nn.Linear(current_dim, output_dim))
        # 最后一层输出各楼层损伤预测。
        # 不在这里加 sigmoid；输出约束由训练脚本统一控制。

        self.head = nn.Sequential(*layers)
        # 回归头。

    def forward(self, x_seq: torch.Tensor, x_meta: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_seq:
                响应时程序列，形状 (batch, sequence_length, input_dim)。
            x_meta:
                可选元信息，形状 (batch, meta_dim)。

        Returns:
            raw_pred:
                原始损伤预测，形状 (batch, output_dim)。
        """

        _, (h_n, _) = self.lstm(x_seq)
        # h_n 形状为 (num_layers * num_directions, batch, hidden_dim)。
        # 这里使用最后隐藏状态作为整个时程的压缩表示。

        if self.bidirectional:
            # 双向 LSTM 时，最后一层有 forward/backward 两个隐藏状态。

            forward_last = h_n[-2]
            # 最后一层 forward 方向隐藏状态。

            backward_last = h_n[-1]
            # 最后一层 backward 方向隐藏状态。

            seq_feature = torch.cat([forward_last, backward_last], dim=1)
            # 拼接两个方向的特征。

        else:
            # 单向 LSTM。

            seq_feature = h_n[-1]
            # 最后一层最后时刻隐藏状态。

        if self.meta_dim > 0:
            # 如果模型设置为使用元信息。

            if x_meta is None:
                # 没传入元信息则报错。
                raise ValueError("x_meta is required when meta_dim > 0.")

            meta_feature = self.meta_encoder(x_meta)
            # 编码元信息。

            feature = torch.cat([seq_feature, meta_feature], dim=1)
            # 拼接时序特征与元信息特征。

        else:
            # 不使用元信息。

            feature = seq_feature
            # 只使用 LSTM 特征。

        return self.head(feature)
        # 输出原始预测。


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.

    中文说明：
        统计模型可训练参数数量。
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # p.numel() 是参数元素数量；requires_grad=True 表示该参数参与训练。
