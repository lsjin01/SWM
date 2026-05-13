# models/transition.py
"""
SWM Transition — V-JEPA-2-AC Predictor
----------------------------------------
checkpoint['predictor'] 구조:
  predictor_embed  : (1024, 1408)  latent → hidden
  action_encoder   : (1024, 7)     action → hidden
  state_encoder    : (1024, 7)     (unused in SWM)
  extrinsics_encoder: (1024, 6)   (unused in SWM)
  predictor_blocks.0~23: 24 Transformer blocks, hidden=1024
  predictor_norm   : (1024,)
  predictor_proj   : (1408, 1024)  hidden → latent

z_t (B,1408) + a_t (B,7) → ẑ_t+1 (B,1408)
"""

from __future__ import annotations
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn


def _strip_module(sd: dict) -> OrderedDict:
    return OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in sd.items()
    )


class SWMTransition(nn.Module):
    """
    V-JEPA-2-AC predictor를 그대로 재현한 Transition model.

    구조:
      predictor_embed  : Linear(1408 → 1024)
      action_encoder   : Linear(7 → 1024)
      predictor_blocks : 24 × TransformerEncoderLayer(1024, heads=16)
      predictor_norm   : LayerNorm(1024)
      predictor_proj   : Linear(1024 → 1408)
    """

    HIDDEN_DIM  = 1024
    NUM_LAYERS  = 24
    NUM_HEADS   = 16
    LATENT_DIM  = 1408   # V-JEPA-2 ViT-Giant embed dim

    def __init__(
        self,
        latent_dim: int = 1408,    # V-JEPA-2 output dim
        action_dim: int = 7,
        noise_std: float = 0.01,
        vjepa2_ac_ckpt: Optional[str] = None,
        freeze: bool = False,
    ):
        super().__init__()
        H = self.HIDDEN_DIM
        self.latent_dim = latent_dim
        self.noise_std  = noise_std

        # ── Predictor 구조 (checkpoint 키와 1:1 매핑) ─────────────────────
        self.predictor_embed = nn.Linear(latent_dim, H)
        self.action_encoder  = nn.Linear(action_dim, H)

        self.predictor_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=H,
                nhead=self.NUM_HEADS,
                dim_feedforward=H * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.NUM_LAYERS)
        ])
        self.predictor_norm = nn.LayerNorm(H)
        self.predictor_proj = nn.Linear(H, latent_dim)

        if vjepa2_ac_ckpt is not None:
            self._load(vjepa2_ac_ckpt)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _load(self, ckpt_path: str):
        raw = torch.load(ckpt_path, map_location="cpu")
        sd  = _strip_module(raw.get("predictor", raw))

        mapped = OrderedDict()
        for k, v in sd.items():
            if k.startswith("predictor_blocks."):
                # "predictor_blocks.N.xxx" → "predictor_blocks.N.xxx"
                # TransformerEncoderLayer 키 변환
                parts = k.split(".", 2)   # ['predictor_blocks', 'N', 'rest']
                idx, rest = parts[1], parts[2]
                new_k = self._remap_block_key(idx, rest)
                if new_k:
                    mapped[new_k] = v
            else:
                # predictor_embed, action_encoder, predictor_norm, predictor_proj
                mapped[k] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        print(f"[SWMTransition] Loaded  missing={len(missing)}  "
              f"unexpected={len(unexpected)}")
        if missing:
            print(f"  missing sample: {missing[:3]}")

    @staticmethod
    def _remap_block_key(idx: str, rest: str) -> Optional[str]:
        """
        V-JEPA-2-AC block key → nn.TransformerEncoderLayer key
        """
        prefix = f"predictor_blocks.{idx}."
        mapping = {
            "norm1.weight":     f"{prefix}norm1.weight",
            "norm1.bias":       f"{prefix}norm1.bias",
            "norm2.weight":     f"{prefix}norm2.weight",
            "norm2.bias":       f"{prefix}norm2.bias",
            "attn.proj.weight": f"{prefix}self_attn.out_proj.weight",
            "attn.proj.bias":   f"{prefix}self_attn.out_proj.bias",
            "mlp.fc1.weight":   f"{prefix}linear1.weight",
            "mlp.fc1.bias":     f"{prefix}linear1.bias",
            "mlp.fc2.weight":   f"{prefix}linear2.weight",
            "mlp.fc2.bias":     f"{prefix}linear2.bias",
        }
        return mapping.get(rest, None)

    def forward(
        self,
        z_t: torch.Tensor,        # (B, 1408)
        action: torch.Tensor,     # (B, 7)
        add_noise: bool = False,
    ) -> torch.Tensor:
        """z_t + a_t → ẑ_t+1  (B, 1408)"""
        if add_noise and self.noise_std > 0:
            z_t = z_t + torch.randn_like(z_t) * self.noise_std

        # Embed
        z_emb = self.predictor_embed(z_t)    # (B, 1024)
        a_emb = self.action_encoder(action)  # (B, 1024)
        x = (z_emb + a_emb).unsqueeze(1)    # (B, 1, 1024)

        # Transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)

        x = self.predictor_norm(x.squeeze(1))  # (B, 1024)
        return self.predictor_proj(x)           # (B, 1408)

    def rollout(
        self,
        z0: torch.Tensor,           # (B, 1408)
        actions: torch.Tensor,      # (B, T, 7)
        tf_ratio: float = 0.0,
        z_gt_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-step rollout → (B, T, 1408)"""
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
