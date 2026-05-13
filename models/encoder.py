# models/encoder.py
"""
SWM Encoder — V-JEPA-2 ViT-Giant
----------------------------------
정확한 아키텍처 (checkpoint에서 측정):
  embed_dim  : 1408
  num_layers : 40
  num_heads  : 22  (head_dim=64)
  ffn_dim    : 6144
  patch_embed: 3D Conv (1408, 3, 2, 16, 16)

단일 이미지 처리:
  image (B,3,H,W) → unsqueeze(T=2) → 3D patch embed → ViT → mean pool → z_t (B,1408)
"""

from __future__ import annotations
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn


def _strip_module(sd: dict) -> OrderedDict:
    """'module.xxx' → 'xxx'"""
    return OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in sd.items()
    )


# ─────────────────────────────────────────────────────────────────────────────
# V-JEPA-2 정확한 아키텍처
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention (V-JEPA-2 스타일: qkv fused)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm Transformer block."""
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    """3D Conv patch embedding for video input."""
    def __init__(self, embed_dim=1408, in_chans=3,
                 t_patch=2, h_patch=16, w_patch=16):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(t_patch, h_patch, w_patch),
            stride=(t_patch, h_patch, w_patch),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)   # (B, D, t', h', w')
        B, D, t, h, w = x.shape
        return x.flatten(2).transpose(1, 2)   # (B, t*h*w, D)


class VJepa2Encoder(nn.Module):
    """
    V-JEPA-2 ViT-Giant encoder.
    image (B,3,H,W) → z_t (B, 1408)
    """

    EMBED_DIM  = 1408
    NUM_LAYERS = 40
    NUM_HEADS  = 22
    FFN_DIM    = 6144

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        img_size: int = 224,
        t_patch: int = 2,
        h_patch: int = 16,
        w_patch: int = 16,
        freeze: bool = False,
    ):
        super().__init__()
        D = self.EMBED_DIM
        self.t_patch = t_patch
        self.embed_dim = D

        # Patch embed (3D)
        self.patch_embed = PatchEmbed3D(D, 3, t_patch, h_patch, w_patch)

        # Positional embedding
        num_h = img_size // h_patch   # 14
        num_w = img_size // w_patch   # 14
        num_t = 1                      # T=2 → t'=1
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_t * num_h * num_w, D)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(D, self.NUM_HEADS, self.FFN_DIM)
            for _ in range(self.NUM_LAYERS)
        ])
        self.norm = nn.LayerNorm(D)

        if ckpt_path is not None:
            self._load(ckpt_path)
        else:
            print("[VJepa2Encoder] No ckpt — random init")

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _load(self, ckpt_path: str):
        raw = torch.load(ckpt_path, map_location="cpu")
        sd  = _strip_module(raw.get("encoder", raw))

        # 키 매핑: checkpoint → 현재 모델
        # checkpoint keys: patch_embed.proj.*, blocks.N.norm1/norm2/attn.qkv/attn.proj/mlp.fc1/mlp.fc2, norm.*
        # model keys:      patch_embed.proj.*, blocks.N.norm1/norm2/attn.qkv/attn.proj/mlp.fc1/mlp.fc2, norm.*
        # → 키 구조가 동일하므로 그대로 로드 가능 (pos_embed 제외)
        filtered = OrderedDict()
        for k, v in sd.items():
            if "pos_embed" in k or "cls_token" in k:
                # shape 불일치 가능 → skip
                continue
            filtered[k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"[VJepa2Encoder] Loaded from {ckpt_path}")
        print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print(f"  missing sample: {missing[:3]}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image (B,3,H,W) → z_t (B, 1408)"""
        B = image.shape[0]
        # T=2로 복제 (video encoder 입력)
        x = image.unsqueeze(2).expand(-1, -1, self.t_patch, -1, -1)  # (B,3,2,H,W)
        x = self.patch_embed(x)   # (B, L, 1408)

        if self.pos_embed.shape[1] == x.shape[1]:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.mean(dim=1)   # (B, 1408)  mean pool


# ─────────────────────────────────────────────────────────────────────────────
# SWM Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SWMEncoder(nn.Module):
    def __init__(
        self,
        vjepa2_ckpt: Optional[str] = None,
        freeze_backbone: bool = False,
        latent_dim: int = 1408,
        img_size: int = 224,
    ):
        super().__init__()
        self.backbone = VJepa2Encoder(
            ckpt_path=vjepa2_ckpt,
            img_size=img_size,
            freeze=freeze_backbone,
        )
        # latent_dim == embed_dim이면 proj 불필요하지만 유연성을 위해 유지
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(image))