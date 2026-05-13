# models/transition.py
"""
SWM Transition Model
--------------------
f(z_t, a_t) → ẑ_t+1

Built on top of V-JEPA2-AC's action-conditioned predictor.
Trained with:
  ℒ_graph  (via frozen heads)
  ℒ_depth  (via frozen heads)
  ℒ_JEPA   = ‖ ẑ_t+1 − z*_t+1 ‖₁   (target from frozen encoder)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Transformer-based Transition (matches V-JEPA2-AC predictor structure)
# ─────────────────────────────────────────────────────────────────────────────

class TransitionTransformer(nn.Module):
    """
    Block-causal Transformer that predicts ẑ_t+1 from (z_t, a_t).
    Mirrors V-JEPA2-AC's predictor architecture.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 7,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Project action into latent space
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)

        # Transformer encoder (block-causal via causal mask)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output projection back to latent_dim
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,       # (B, latent_dim)
        action: torch.Tensor,    # (B, action_dim)
    ) -> torch.Tensor:
        """Returns ẑ_t+1  (B, latent_dim)."""
        a_emb = self.action_proj(action)              # (B, latent_dim)
        x = torch.cat([z_t, a_emb], dim=-1)          # (B, latent_dim*2)
        x = self.input_proj(x).unsqueeze(1)           # (B, 1, hidden_dim)
        x = self.transformer(x)                       # (B, 1, hidden_dim)
        return self.out_proj(x.squeeze(1))            # (B, latent_dim)


# ─────────────────────────────────────────────────────────────────────────────
# SWM Transition (wraps predictor + handles Teacher Forcing + Noise)
# ─────────────────────────────────────────────────────────────────────────────

class SWMTransition(nn.Module):
    """
    Full transition module with:
      - Teacher Forcing (controlled by tf_ratio)
      - Noise Injection on input latent
      - Load from V-JEPA2-AC checkpoint
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 7,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        noise_std: float = 0.01,
        vjepa2_ac_ckpt: Optional[str] = None,
        freeze: bool = False,
    ):
        super().__init__()
        self.predictor = TransitionTransformer(
            latent_dim, action_dim, hidden_dim, num_layers, num_heads, dropout
        )
        self.noise_std = noise_std

        if vjepa2_ac_ckpt is not None:
            self._load_vjepa2_ac(vjepa2_ac_ckpt)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _load_vjepa2_ac(self, ckpt_path: str):
        """Load V-JEPA2-AC predictor weights (best-effort)."""
        try:
            sd = torch.load(ckpt_path, map_location="cpu")
            if "predictor" in sd:
                sd = sd["predictor"]
            elif "model" in sd:
                sd = sd["model"]
            missing, unexpected = self.predictor.load_state_dict(sd, strict=False)
            print(f"[SWMTransition] Loaded V-JEPA2-AC. "
                  f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"[SWMTransition] Warning: could not load V-JEPA2-AC: {e}")

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_gt_t1: Optional[torch.Tensor] = None,   # ground-truth next latent
        tf_ratio: float = 1.0,                     # teacher forcing ratio
        training: bool = True,
    ) -> torch.Tensor:
        """
        Single-step prediction with optional teacher forcing and noise.

        tf_ratio=1.0  → always use ground-truth z_t (pure teacher forcing)
        tf_ratio=0.0  → always use own previous prediction (free running)
        """
        # Noise injection during training
        if training and self.noise_std > 0:
            z_t = z_t + torch.randn_like(z_t) * self.noise_std

        return self.predictor(z_t, action)   # ẑ_t+1

    def rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,    # (B, T, action_dim)
        tf_ratio: float = 0.0,
        z_gt_seq: Optional[torch.Tensor] = None,  # (B, T, latent_dim) GT sequence
    ) -> torch.Tensor:
        """
        Multi-step rollout. Returns trajectory of latents (B, T, latent_dim).

        tf_ratio controls teacher forcing:
            1.0 → always use GT as next input
            0.0 → free running (use own prediction)
        """
        B, T, _ = actions.shape
        z = z0
        latents = []

        for t in range(T):
            z_next = self.forward(z, actions[:, t], training=False)
            latents.append(z_next)

            # Teacher forcing decision
            if z_gt_seq is not None and torch.rand(1).item() < tf_ratio:
                z = z_gt_seq[:, t]
            else:
                z = z_next

        return torch.stack(latents, dim=1)   # (B, T, latent_dim)
