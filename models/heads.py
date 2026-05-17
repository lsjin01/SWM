# models/heads.py
"""
Prediction Heads
----------------
Graph Node Position Head → Ĝ_t  (B, max_nodes, 3)
Depth Head 제거 (GT depth 없음, Graph Head의 z좌표가 depth 역할)
"""

from __future__ import annotations
import torch
import torch.nn as nn


class GraphNodePositionHead(nn.Module):
    """
    Predicts (x, y, z) position for each graph node from latent z.
    Output: (B, max_nodes, 3)
    """

    def __init__(
        self,
        latent_dim: int = 2176,
        max_nodes: int = 8,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512]

        layers = []
        in_d = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.GELU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, max_nodes * 3))
        self.mlp = nn.Sequential(*layers)
        self.max_nodes = max_nodes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, max_nodes, 3)"""
        return self.mlp(z).view(-1, self.max_nodes, 3)


# SWMHeads = GraphNodePositionHead alias (이전 코드 호환)
class SWMHeads(nn.Module):
    """Graph Node Position Head만 포함."""

    def __init__(
        self,
        latent_dim: int = 2176,
        max_nodes: int = 8,
        grid_size: int = 32,   # 호환성 유지 (사용 안 함)
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph_head = GraphNodePositionHead(
            latent_dim, max_nodes, hidden_dims, dropout
        )

    def forward(self, z: torch.Tensor):
        """
        Returns (graph_pred, None)
        graph_pred: (B, max_nodes, 3)
        """
        return self.graph_head(z), None