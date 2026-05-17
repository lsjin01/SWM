# models/reward_model.py
"""
SWM Reward Model (Stage 2.5 v3 — Temporal Transformer)
=======================================================
WMPO LatentRewardModel 방식: trajectory latents → temporal attention → P(success)

MLP + goal comparison 방식의 문제:
  - z_T 하나만 보면 temporal dynamics 없음 → discriminative하지 않음
  - goal 비교는 불필요 (성공 궤적 vs 실패 궤적 자체가 다름)

새 구조 (WMPO LatentRewardModel 동일):
  z_history[-T:] (T, latent_dim) → proj → CLS + pos_emb → Temporal Transformer → sigmoid
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SWMRewardModel(nn.Module):
    """
    Temporal Transformer reward model over SWM WM latent trajectory.

    Input:  z_traj (B, T, latent_dim)  — T timesteps of global WM latents
    Output: P(success) ∈ [0, 1]

    Pipeline (WMPO LatentRewardModel style):
      1. LayerNorm + Linear projection  → (B, T, hidden_dim)
      2. Prepend CLS token              → (B, T+1, hidden_dim)
      3. Add positional embedding
      4. Temporal TransformerEncoder    (Pre-LN, stable)
      5. CLS token → LayerNorm → Linear → sigmoid
    """

    def __init__(
        self,
        latent_dim: int = 2176,
        hidden_dim: int = 256,
        n_frames:   int = 8,
        n_heads:    int = 4,
        n_layers:   int = 2,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.n_frames   = n_frames
        self.latent_dim = latent_dim

        self.input_norm = nn.LayerNorm(latent_dim)
        self.proj       = nn.Linear(latent_dim, hidden_dim)
        self.proj_drop  = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_frames + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_norm  = nn.LayerNorm(hidden_dim)
        self.head_drop = nn.Dropout(dropout)
        self.head      = nn.Linear(hidden_dim, 1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z_traj: torch.Tensor) -> torch.Tensor:
        """
        z_traj: (B, T, latent_dim)
        returns: (B,) P(success) logit-free sigmoid output
        """
        B, T, D = z_traj.shape

        x = self.proj_drop(self.proj(self.input_norm(z_traj)))  # (B, T, hidden_dim)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)          # (B, T+1, hidden_dim)
        x   = x + self.pos_embed[:, :T+1]

        x = self.transformer(x)                   # (B, T+1, hidden_dim)
        return torch.sigmoid(
            self.head(self.head_drop(self.cls_norm(x[:, 0])))
        ).squeeze(-1)                             # (B,)

    @torch.no_grad()
    def reward(self, z_traj: torch.Tensor) -> float:
        """Inference-time scalar reward."""
        return self.forward(z_traj).mean().item()


def sample_traj(z_history: list, n_frames: int = 8) -> torch.Tensor:
    """
    z_history: list of (1, latent_dim) tensors (action-step level)
    → uniformly sample n_frames → (1, n_frames, latent_dim)
    """
    n = len(z_history)
    indices = [int(round(i * (n - 1) / (n_frames - 1))) for i in range(n_frames)]
    sampled = [z_history[idx] for idx in indices]
    return torch.cat(sampled, dim=0).unsqueeze(0)   # (1, n_frames, latent_dim)
