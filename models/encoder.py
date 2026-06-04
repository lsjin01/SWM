# models/encoder.py
"""
SWM Encoder — DINOv2 + SigLIP (OpenVLA compatible)
----------------------------------------------------
OpenVLA vision backbone에서 DINOv2 + SigLIP을 그대로 사용.

구조:
  image → DINOv2 (frozen) → (256, 1024)
  image → SigLIP (frozen) → (256, 1152)
  concat → (256, 2176) → mean pool → z_t (2176)

장점:
  - OpenVLA projector와 완벽 호환 (2176 → projector input)
  - 언어 정렬된 feature space (SigLIP)
  - 강한 visual representation (DINOv2)
  - Stage 3에서 z_t → projector → LLM 바로 연결 가능
"""

from __future__ import annotations
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn


class DINOSigLIPEncoder(nn.Module):
    """
    OpenVLA의 vision backbone (DINOv2 + SigLIP)을 그대로 사용.
    두 encoder를 frozen으로 유지하고 feature를 concat.

    image (B, 3, H, W) → z_t (B, 2176)
    """

    DINO_DIM   = 1024
    SIGLIP_DIM = 1152
    EMBED_DIM  = 2176   # concat dim = projector input dim

    def __init__(
        self,
        vla_path: str,
        freeze: bool = True,
    ):
        super().__init__()
        self._load_from_vla(vla_path, freeze)
        self.embed_dim = self.EMBED_DIM

    def _load_from_vla(self, vla_path: str, freeze: bool):
        """OpenVLA에서 vision backbone만 추출."""
        import sys
        sys.path.insert(0, '/home/mipstu/jiPark/openvla-oft/experiments/robot')
        from transformers import AutoModelForVision2Seq

        print(f"[DINOSigLIPEncoder] Loading vision backbone from {vla_path} ...")
        vla = AutoModelForVision2Seq.from_pretrained(
            vla_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # DINOv2 + SigLIP 추출
        self.dino   = vla.vision_backbone.featurizer       # DINOv2
        self.siglip = vla.vision_backbone.fused_featurizer # SigLIP

        # 메모리 해제 (VLA 나머지 부분)
        del vla
        torch.cuda.empty_cache()

        if freeze:
            for p in self.dino.parameters():
                p.requires_grad = False
            for p in self.siglip.parameters():
                p.requires_grad = False

        print(f"[DINOSigLIPEncoder] Loaded  "
              f"DINOv2({self.DINO_DIM}) + SigLIP({self.SIGLIP_DIM}) "
              f"= {self.EMBED_DIM}  freeze={freeze}")

    # per-backbone normalization constants
    _DINO_MEAN   = [0.485, 0.456, 0.406]
    _DINO_STD    = [0.229, 0.224, 0.225]
    _SIGLIP_MEAN = [0.5,   0.5,   0.5  ]
    _SIGLIP_STD  = [0.5,   0.5,   0.5  ]

    def _split_normalize(self, image_01: torch.Tensor, dtype):
        """
        image_01: (B, 3, H, W) in [0, 1] range
        → dino_img:   ImageNet normalized
        → siglip_img: SigLIP normalized
        """
        def _norm(img, mean, std):
            m = torch.tensor(mean, device=img.device, dtype=dtype).view(1, 3, 1, 1)
            s = torch.tensor(std,  device=img.device, dtype=dtype).view(1, 3, 1, 1)
            return (img - m) / s

        return _norm(image_01, self._DINO_MEAN, self._DINO_STD), \
               _norm(image_01, self._SIGLIP_MEAN, self._SIGLIP_STD)

    def forward_spatial(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, 3, H, W) in [0, 1] range (ToTensor만, Normalize 없음)
        → spatial patch features: (B, 256, 2176)  [공간 정보 보존]
        DINOv2 / SigLIP 각자 올바른 normalization 적용.
        """
        dtype = next(self.dino.parameters()).dtype
        image_01 = image.to(dtype).clamp(0, 1)

        dino_img, siglip_img = self._split_normalize(image_01, dtype)

        dino_feat = self.dino.forward_features(dino_img)
        if dino_feat.ndim == 3:
            dino_feat = dino_feat[:, -256:]        # (B, 256, 1024)

        siglip_feat = self.siglip.forward_features(siglip_img)
        if siglip_feat.ndim == 3:
            siglip_feat = siglip_feat[:, -256:]    # (B, 256, 1152)

        return torch.cat([dino_feat, siglip_feat], dim=-1)  # (B, 256, 2176)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, 3, H, W) in [0, 1] range
        → global z_t: (B, 2176)  [transition / reward 전용]
        """
        return self.forward_spatial(image).mean(dim=1)


class SWMEncoder(nn.Module):
    """
    Stage 1 Encoder:
      DINOSigLIPEncoder(image) → z_t (B, 2176)

    latent_dim=2176로 고정 (projector 호환)
    """

    def __init__(
        self,
        vla_path: str,
        freeze_backbone: bool = True,
        latent_dim: int = 2176,
        spatial_dim: int = 256,
    ):
        super().__init__()
        self.backbone = DINOSigLIPEncoder(
            vla_path=vla_path,
            freeze=freeze_backbone,
        )
        # latent_dim == embed_dim이면 proj 생략 가능하지만 유연성을 위해 유지
        if latent_dim == self.backbone.embed_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.backbone.embed_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        self.latent_dim = latent_dim
        if spatial_dim == self.backbone.embed_dim:
            self.spatial_proj = nn.Identity()
        else:
            self.spatial_proj = nn.Sequential(
                nn.Linear(self.backbone.embed_dim, spatial_dim),
                nn.LayerNorm(spatial_dim),
            )
        self.spatial_dim = spatial_dim

    def encode_spatial(self, image: torch.Tensor) -> torch.Tensor:
        """image (B,3,H,W) → spatial patch features (B, 256, latent_dim) [VLA projector용]"""
        spatial = self.backbone.forward_spatial(image)  # (B, 256, 2176)
        # Identity proj인 경우 그대로 반환, 아니면 per-token projection 필요
        if isinstance(self.proj, nn.Identity):
            return spatial.float()
        # Linear + LayerNorm projection: (B, 256, 2176) → (B, 256, latent_dim)
        return self.proj(spatial.float())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image (B,3,H,W) → global z_t (B, latent_dim) [transition / reward 전용]"""
        z = self.proj(self.backbone(image))
        return z.float()

    def encode_weighted(self, image: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
        """
        token_weights: (256,) normalized weights, task-relevant tokens에 높은 값
        → weighted z: (B, latent_dim)
        """
        spatial = self.backbone.forward_spatial(image)  # (B, 256, 2176)
        w = token_weights.to(spatial.device, dtype=spatial.dtype)
        w = w / w.sum().clamp(min=1e-8)
        z = (spatial * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, 2176)
        return self.proj(z.float())

    def encode_spatial_projected(self, image: torch.Tensor) -> torch.Tensor:
        """image (B,3,H,W) → projected spatial tokens (B, 256, spatial_dim)"""
        spatial = self.backbone.forward_spatial(image)   # (B, 256, 2176)
        return self.spatial_proj(spatial.float())        # (B, 256, spatial_dim)