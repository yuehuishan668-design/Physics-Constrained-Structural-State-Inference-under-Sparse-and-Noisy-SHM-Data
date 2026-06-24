"""
File location:
    src/models/mlp.py

Purpose:
    Define the first MLP baseline for story-level stiffness degradation regression.

中文说明：
    MLP = Multi-Layer Perceptron，多层感知机。
    当前 MLP 输入是统计特征，不是完整时程序列。
    输出是 4 层刚度退化向量：
        y_pred = [d1, d2, d3, d4]
"""

from __future__ import annotations

from typing import Iterable, List
import torch
from torch import nn


def parse_hidden_dims(hidden_dims: str | Iterable[int]) -> List[int]:
    """
    Parse hidden layer sizes.

    中文说明：
        把 "64,64" 这样的字符串转换成 [64, 64]。
    """
    if isinstance(hidden_dims, str):
        if hidden_dims.strip() == "":
            return []
        return [int(item.strip()) for item in hidden_dims.split(",")]
    return [int(item) for item in hidden_dims]


class MLPDamageRegressor(nn.Module):
    """
    MLP model for damage regression.

    中文说明：
        nn.Module 是所有 PyTorch 模型的基类。
        继承 nn.Module 后，模型才能使用 forward、parameters、to(device) 等 PyTorch 功能。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: str | Iterable[int] = "64,64",
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        # super().__init__()：初始化 PyTorch 模型基类，必须调用。

        dims = parse_hidden_dims(hidden_dims)
        # dims：隐藏层维度列表，例如 [64, 64]。

        layers: list[nn.Module] = []
        # layers：保存网络层的列表，最后交给 nn.Sequential 串联。

        prev_dim = input_dim
        # prev_dim：当前层输入维度。第一层输入维度就是 input_dim。

        for hidden_dim in dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # nn.Linear：全连接层，执行 y = xW^T + b。

            layers.append(nn.ReLU())
            # ReLU：非线性激活函数，执行 max(0, x)。

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
                # Dropout：训练时随机屏蔽部分神经元，降低过拟合风险。

            prev_dim = hidden_dim
            # 更新下一层的输入维度。

        layers.append(nn.Linear(prev_dim, output_dim))
        # 最后一层把隐藏特征映射到 4 个损伤变量。
        # 当前不加 Sigmoid，是为了保持最基础 baseline；后续可再加入物理边界约束。

        self.network = nn.Sequential(*layers)
        # nn.Sequential：按顺序执行 layers 中的所有层。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        中文说明：
            前向传播：输入特征 x，输出损伤预测。
        """
        return self.network(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.

    中文说明：
        统计模型中需要训练的参数数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
