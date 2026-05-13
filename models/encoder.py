# models/encoder.py
"""
SWM Encoder
-----------
Visual backbone (V-JEPA2 / DINOv2) + optional GNN graph branch.
Outputs z_t: the structured latent state.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Simple GNN for graph branch
# ─────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x:          (B, N, D)
        edge_index: not used (fully-connected mean aggregation for simplicity)
        """
        # Mean aggregation (all neighbours)
        agg = x.mean(dim=1, keepdim=True).expand_as(x)   # (B, N, D)
        out = self.lin(torch.cat([x, agg], dim=-1))        # (B, N, out_dim)
        return self.norm(F.relu(out))


class GraphEncoder(nn.Module):
    """Lightweight GNN that encodes graph node features into a fixed-dim vector."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256, output_dim: int = 512,
                 num_layers: int = 2):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(GraphConvLayer(d, hidden_dim))
            d = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.pool = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_feat: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        node_feat:  (B, N, input_dim)
        node_mask:  (B, N) bool
        Returns:    (B, output_dim)
        """
        x = node_feat
        for layer in self.layers:
            x = layer(x, None)
        # Masked mean pooling
        mask = node_mask.unsqueeze(-1).float()   # (B, N, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.pool(x)   # (B, output_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Backbone wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VisualBackbone(nn.Module):
    """
    Wraps V-JEPA2 or DINOv2.
    Returns CLS token (B, embed_dim) or patch tokens (B, L, embed_dim).
    """

    def __init__(self, backbone: str = "vjepa2", ckpt_path: Optional[str] = None,
                 freeze: bool = False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "vjepa2":
            self.model = self._load_vjepa2(ckpt_path)
            self.embed_dim = 1408   # ViT-Giant
        elif backbone == "dinov2":
            import timm
            self.model = timm.create_model("vit_giant_patch14_dinov2", pretrained=True,
                                           num_classes=0)
            self.embed_dim = 1536
        elif backbone == "resnet50":
            import torchvision.models as tv
            m = tv.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(m.children())[:-1])
            self.embed_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _load_vjepa2(self, ckpt_path: Optional[str]) -> nn.Module:
        """
        Load V-JEPA2 ViT encoder.
        Falls back to timm ViT-Giant if checkpoint not found.
        """
        try:
            import timm
            model = timm.create_model(
                "vit_giant_patch14_224", pretrained=(ckpt_path is None), num_classes=0
            )
            if ckpt_path is not None:
                sd = torch.load(ckpt_path, map_location="cpu")
                # Handle various checkpoint formats
                if "model" in sd:
                    sd = sd["model"]
                elif "encoder" in sd:
                    sd = sd["encoder"]
                model.load_state_dict(sd, strict=False)
            return model
        except Exception as e:
            print(f"[VisualBackbone] V-JEPA2 load warning: {e}. Using timm ViT.")
            import timm
            return timm.create_model("vit_large_patch14_224", pretrained=True, num_classes=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)  →  (B, embed_dim)"""
        if self.backbone_name in ("vjepa2", "dinov2"):
            feat = self.model.forward_features(x)
            if isinstance(feat, dict):
                feat = feat["x_norm_clstoken"]
            elif feat.ndim == 3:
                feat = feat[:, 0]   # CLS token
        else:
            feat = self.model(x).squeeze(-1).squeeze(-1)
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# SWM Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SWMEncoder(nn.Module):
    """
    Full SWM Encoder:
        visual_feat = VisualBackbone(O_t)          (B, V)
        graph_feat  = GraphEncoder(G_t)            (B, G)
        z_t         = MLP(concat(visual, graph))   (B, Z)
    """

    def __init__(
        self,
        backbone: str = "vjepa2",
        backbone_ckpt: Optional[str] = None,
        freeze_backbone: bool = False,
        visual_embed_dim: int = 1408,
        use_graph_branch: bool = True,
        graph_input_dim: int = 5,
        graph_hidden_dim: int = 256,
        graph_embed_dim: int = 512,
        latent_dim: int = 1024,
    ):
        super().__init__()

        self.visual = VisualBackbone(backbone, backbone_ckpt, freeze=freeze_backbone)
        visual_out = self.visual.embed_dim

        self.use_graph = use_graph_branch
        if use_graph_branch:
            self.graph_enc = GraphEncoder(graph_input_dim, graph_hidden_dim, graph_embed_dim)
            fusion_dim = visual_out + graph_embed_dim
        else:
            fusion_dim = visual_out

        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(
        self,
        image: torch.Tensor,
        graph_feat: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        image:      (B, 3, H, W)
        graph_feat: (B, N, 5)   — topology only (no position)
        node_mask:  (B, N) bool
        Returns:    z_t  (B, latent_dim)
        """
        v = self.visual(image)   # (B, V)

        if self.use_graph and graph_feat is not None:
            g = self.graph_enc(graph_feat, node_mask)   # (B, G)
            feat = torch.cat([v, g], dim=-1)
        else:
            feat = v

        return self.proj(feat)   # (B, latent_dim)
