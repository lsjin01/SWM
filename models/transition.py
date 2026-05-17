# models/transition.py
"""
SWM Transition — Small Transformer (~10M params)
-------------------------------------------------
DINOv2+SigLIP feature space 기반 작은 Transition 모델.

구조:
  hidden_dim  = 512
  num_layers  = 6
  num_heads   = 8
  input_dim   = 2176  (DINOv2+SigLIP concat)
  action_dim  = 7
  → ~10M params

z_t (B, 2176) + a_t (B, 7) → ẑ_t+1 (B, 2176)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class SWMTransition(nn.Module):
    """
    Small action-conditioned Transformer.
    V-JEPA-AC의 구조를 따르되 DINOv2+SigLIP feature space에 맞게 조정.

    params: ~10M (데이터 315K에 적합)
    """

    def __init__(
        self,
        latent_dim: int = 2176,
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads:  int = 8,
        dropout:    float = 0.1,
        noise_std:  float = 0.01,
        freeze:     bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.noise_std  = noise_std

        # Input projections
        self.latent_embed = nn.Linear(latent_dim, hidden_dim)
        self.action_embed  = nn.Linear(action_dim, hidden_dim)

        # Transformer blocks
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj   = nn.Linear(hidden_dim, latent_dim)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        n = sum(p.numel() for p in self.parameters())
        print(f"[SWMTransition] params: {n:,}  "
              f"(latent={latent_dim}, hidden={hidden_dim}, "
              f"layers={num_layers}, heads={num_heads})")

    def forward(
        self,
        z_t: torch.Tensor,       # (B, 2176)
        action: torch.Tensor,    # (B, 7)
        add_noise: bool = False,
    ) -> torch.Tensor:
        """z_t + a_t → ẑ_t+1  (B, 2176)"""
        if add_noise and self.noise_std > 0:
            z_t = z_t + torch.randn_like(z_t) * self.noise_std

        z_emb = self.latent_embed(z_t)    # (B, hidden)
        a_emb = self.action_embed(action)  # (B, hidden)
        x = (z_emb + a_emb).unsqueeze(1)  # (B, 1, hidden)

        x = self.blocks(x)                 # (B, 1, hidden)
        x = self.norm(x.squeeze(1))        # (B, hidden)
        return self.proj(x)                # (B, 2176)

    def rollout(
        self,
        z0: torch.Tensor,          # (B, 2176)
        actions: torch.Tensor,     # (B, T, 7)
        tf_ratio: float = 0.0,
        z_gt_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-step rollout → (B, T, 2176)"""
        B, T, _ = actions.shape
        z = z0
        latents = []

        for t in range(T):
            z_next = self.forward(z, actions[:, t], add_noise=False)
            latents.append(z_next)
            if z_gt_seq is not None and torch.rand(1).item() < tf_ratio:
                z = z_gt_seq[:, t]
            else:
                z = z_next

        return torch.stack(latents, dim=1)