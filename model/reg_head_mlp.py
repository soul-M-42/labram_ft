import torch
import torch.nn as nn
from typing import Sequence


class MLPRegHead(nn.Module):
    def __init__(
        self,
        fea_dim: int,
        n_tar_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ):
        """
        Args:
            fea_dim: 输入特征维度
            n_tar_dim: 回归目标维度
            hidden_dims: MLP 中间隐藏层维度列表
            activation: 激活函数类（如 nn.ReLU, nn.GELU）
            dropout: dropout 概率
        """
        super().__init__()

        layers = []
        in_dim = fea_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, n_tar_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [bs, fea_dim]
        Returns:
            y: [bs, n_tar_dim]
        """
        return self.mlp(x)
