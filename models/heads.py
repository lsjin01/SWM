# models/heads.py
"""
Prediction Heads
----------------
Graph Node Position Head  →  Ĝ_t  (B, max_nodes, 3)
Sparse Depth Head         →  D̂_t  (B, grid_size, grid_size)
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
        latent_dim: int = 1024,
        max_nodes: int = 8,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_d = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.GELU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, max_nodes * 3))
        self.mlp = nn.Sequential(*layers)
        self.max_nodes = max_nodes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:  (B, latent_dim)
        →   (B, max_nodes, 3)
        """
        out = self.mlp(z)                             # (B, max_nodes*3)
        return out.view(-1, self.max_nodes, 3)


class SparseDepthHead(nn.Module):
    """
    Predicts a sparse depth grid from latent z.
    Output: (B, grid_size, grid_size)
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        grid_size: int = 32,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_d = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.GELU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, grid_size * grid_size))
        self.mlp = nn.Sequential(*layers)
        self.grid_size = grid_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z:  (B, latent_dim)
        →   (B, grid_size, grid_size)
        """
        out = self.mlp(z)   # (B, grid_size^2)
        return out.view(-1, self.grid_size, self.grid_size)


class SWMHeads(nn.Module):
    """Convenience wrapper: both heads together."""

    def __init__(
        self,
        latent_dim: int = 1024,
        max_nodes: int = 8,
        grid_size: int = 32,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph_head = GraphNodePositionHead(
            latent_dim, max_nodes, hidden_dims, dropout
        )
        self.depth_head = SparseDepthHead(
            latent_dim, grid_size, hidden_dims, dropout
        )

    def forward(self, z: torch.Tensor):
        """
        Returns (graph_pred, depth_pred)
            graph_pred: (B, max_nodes, 3)
            depth_pred: (B, grid_size, grid_size)
        """
        return self.graph_head(z), self.depth_head(z)
