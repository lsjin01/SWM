# models/policy.py
"""
SWM Policy — OpenVLA-OFT with V-JEPA-2 spatial encoder
--------------------------------------------------------

OpenVLA-OFT 원래 구조:
  image → SigLIP (semantic) ─┐
  image → DINOv2 (spatial)  ─┴→ concat → MLP projector → LLM → action

SWM 수정 구조 (방법 C):
  image → SigLIP (frozen)           ─┐
  image → V-JEPA-2 encoder (z_t)    ─┴→ concat → MLP projector → LLM → action chunk

변경점:
  - DINOv2 → V-JEPA-2 encoder (z_t와 동일한 latent space)
  - WM rollout: z_t + a_t → Transition → z_t+1
  - SigLIP은 frozen 유지 (language alignment 보존)
  - V-JEPA-2는 fine-tune (manipulation domain 적응)
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.encoder import VJepa2Encoder


class SWMPolicy(nn.Module):
    """
    OpenVLA-OFT + V-JEPA-2 spatial encoder.

    핵심 인터페이스:
      encode_image(image) → z_t          [V-JEPA-2 encoder]
      forward(image, instruction) → action_chunk, log_probs
      sample(z_t, instruction) → action_chunk, log_probs
      log_prob(z_t, instruction, actions) → log_probs
    """

    def __init__(
        self,
        openvla_ckpt: Optional[str] = None,
        vjepa2_ckpt: Optional[str] = None,
        freeze_siglip: bool = True,
        freeze_vjepa2: bool = False,
        latent_dim: int = 1024,
        action_dim: int = 7,
        action_chunk_size: int = 4,
        use_peft: bool = True,
        peft_r: int = 16,
        peft_alpha: int = 32,
    ):
        super().__init__()
        self.action_dim        = action_dim
        self.action_chunk_size = action_chunk_size
        self.latent_dim        = latent_dim

        # ── V-JEPA-2 spatial encoder (DINOv2 replacement) ──
        self.vjepa2 = VJepa2Encoder(
            ckpt_path=vjepa2_ckpt,
            freeze=freeze_vjepa2,
        )

        # ── OpenVLA-OFT backbone ──
        self.openvla = self._load_openvla(
            openvla_ckpt, freeze_siglip, use_peft, peft_r, peft_alpha
        )

        # ── Projection: V-JEPA-2 feat → OpenVLA vision token dim ──
        # OpenVLA vision token dim은 SigLIP embed dim과 같음 (1152 for SigLIP-SO400M)
        openvla_vision_dim = self._get_openvla_vision_dim()
        self.vjepa2_proj = nn.Linear(self.vjepa2.embed_dim, openvla_vision_dim)

    def _load_openvla(self, ckpt, freeze_siglip, use_peft, peft_r, peft_alpha):
        """OpenVLA-OFT 로드 및 DINOv2 → V-JEPA-2 교체."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            model = AutoModelForVision2Seq.from_pretrained(
                ckpt or "openvla/openvla-7b-oft",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            # SigLIP freeze
            if freeze_siglip:
                for name, param in model.named_parameters():
                    if "vision_backbone.featurizer" in name:
                        # SigLIP만 freeze, DINOv2(fused_featurizer) 제외
                        if "fused" not in name:
                            param.requires_grad = False

            # PEFT (LoRA)
            if use_peft:
                from peft import get_peft_model, LoraConfig, TaskType
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=peft_r,
                    lora_alpha=peft_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

            print(f"[SWMPolicy] OpenVLA-OFT loaded from {ckpt}")
            return model

        except Exception as e:
            print(f"[SWMPolicy] OpenVLA load failed: {e}\n"
                  f"           Using stub policy (for testing only)")
            return None

    def _get_openvla_vision_dim(self) -> int:
        """OpenVLA의 vision token dimension 반환."""
        if self.openvla is None:
            return 1024
        try:
            # OpenVLA Prismatic: SigLIP-SO400M embed dim = 1152
            return self.openvla.config.vision_config.hidden_size
        except Exception:
            return 1152

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        image (B,3,H,W) → z_t (B, latent_dim)
        Stage 1/2 Encoder와 동일한 역할.
        WM rollout의 초기 latent 생성에 사용.
        """
        return self.vjepa2(image)   # (B, embed_dim) — proj 없이 raw feature

    def _replace_dino_with_vjepa(
        self,
        image: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        OpenVLA 내부에서 DINOv2 feature를 V-JEPA-2 feature로 교체.
        OpenVLA Prismatic vision backbone의 fused_featurizer 자리에 주입.
        """
        vjepa_feat = self.vjepa2(image)              # (B, vjepa_embed_dim)
        vjepa_proj = self.vjepa2_proj(vjepa_feat)    # (B, openvla_vision_dim)
        return vjepa_proj

    def forward(
        self,
        image: torch.Tensor,
        instruction: str,
        actions_gt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.
        Returns (action_logits, log_probs).
        """
        if self.openvla is None:
            # Stub: random actions
            B = image.shape[0]
            actions = torch.randn(B, self.action_chunk_size, self.action_dim,
                                  device=image.device)
            log_probs = torch.zeros_like(actions) - 1.0
            return actions, log_probs

        # V-JEPA-2 feature를 DINOv2 자리에 주입하는 방식은
        # OpenVLA-OFT 코드의 vision backbone 구조에 따라 달라짐.
        # 아래는 forward hook을 이용한 주입 방식.
        vjepa_feat = self._replace_dino_with_vjepa(image, image)

        # OpenVLA-OFT forward (실제 구현은 OpenVLA-OFT API에 맞게 조정)
        outputs = self.openvla(
            pixel_values=image,
            input_ids=self._encode_instruction(instruction, image.device),
            vjepa_features=vjepa_feat,   # custom injection
        )
        return outputs

    def sample(
        self,
        z_t: torch.Tensor,          # (B, latent_dim)  from WM encoder
        instruction: str,
        chunk_size: int = 4,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GRPO rollout용 샘플링.
        z_t: WM latent (V-JEPA-2 encoder output)
        Returns: actions (B, chunk_size, action_dim), log_probs (B, chunk_size, action_dim)
        """
        if self.openvla is None:
            B = z_t.shape[0]
            actions   = torch.randn(B, chunk_size, self.action_dim,
                                    device=z_t.device) * temperature
            log_probs = torch.full_like(actions, -1.0)
            return actions, log_probs

        # TODO: OpenVLA-OFT의 실제 sample API로 교체
        # openvla_oft.predict_action(z_t, instruction, ...)
        raise NotImplementedError(
            "OpenVLA-OFT sample API 연결 필요. "
            "기존 WMPO 코드의 policy.predict_action()을 여기에 이식하세요."
        )

    def log_prob(
        self,
        z_t: torch.Tensor,
        instruction: str,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        GRPO ratio 계산용 log π_θ(a|s).
        Returns: (B, chunk_size, action_dim)
        """
        if self.openvla is None:
            return torch.full_like(actions, -1.0)

        raise NotImplementedError(
            "OpenVLA-OFT log_prob API 연결 필요."
        )

    def _encode_instruction(self, instruction: str, device) -> torch.Tensor:
        """instruction string → token ids."""
        # TODO: tokenizer 연결
        return torch.zeros(1, 32, dtype=torch.long, device=device)
