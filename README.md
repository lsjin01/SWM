# SWM — Structured World Model

> **Pixel-free World Modeling via DINOv2+SigLIP Encoder for Robot Manipulation**

## Overview

SWM은 pixel-level reconstruction 없이 DINOv2+SigLIP 기반 latent world model을 학습하고,
GRPO(Group Relative Policy Optimization)로 VLA(OpenVLA-OFT)를 fine-tuning하는 프레임워크.

```
Stage 1   │ Encoder    : image → z_t  (DINOv2 1024 + SigLIP 1152 = 2176-dim)
Stage 2   │ Transition : z_t + a_t → ẑ_t+1  (JEPA style)
Stage 2.5 │ Reward Model: z_trajectory → P(success)  (optional, temporal transformer)
Stage 3   │ Policy     : GRPO with latent reward
```

## Key Differences from WMPO

| | WMPO | SWM |
|---|---|---|
| World Model | V-JEPA2 + pixel generation | DINOv2+SigLIP latent prediction |
| Reward | VideoMAE sparse binary | Latent cosine similarity (continuous) |
| Rollout | Dedicated rollout GPU | WM transition only (lightweight) |
| VLA input | Native vision backbone | Native vision backbone (freeze_projector) |

---

## Repo Structure

```
swm/
├── configs/
│   ├── stage1_multitask_dinosiglip.yaml
│   ├── stage2_multitask_dinosiglip.yaml
│   ├── stage2_5_v3.yaml               # Temporal Reward Model
│   ├── stage3_latent_cos.yaml         # Normalized cosine reward
│   ├── stage3_latent_cos_temporal.yaml    # Temporal progress reward
│   └── stage3_latent_cos_temporal_v2.yaml # + n_states×4, iter 300
├── models/
│   ├── encoder.py       # SWMEncoder (DINOv2 + SigLIP, freeze_backbone=True)
│   ├── transition.py    # SWMTransition (latent=2176, hidden=512, layers=6)
│   ├── heads.py         # Graph/Depth prediction heads
│   └── reward_model.py  # SWMRewardModel (Temporal Transformer)
├── scripts/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── train_stage2_5.py  # Temporal RM training
│   └── train_stage3.py    # GRPO fine-tuning
└── eval/
    └── eval_swm_mimicgen.py  # MimicGen simulator evaluation
```

---

## Quick Start

```bash
# Stage 1: Encoder
python scripts/train_stage1.py --config configs/stage1_multitask_dinosiglip.yaml

# Stage 2: Transition
python scripts/train_stage2.py --config configs/stage2_multitask_dinosiglip.yaml

# Stage 3: GRPO (temporal progress reward)
export CUDA_VISIBLE_DEVICES=2,3,7
torchrun --nproc_per_node=3 --master_port=29507 \
    scripts/train_stage3.py \
    --config configs/stage3_latent_cos_temporal_v2.yaml

# Eval
USE_Z_BYPASS=0 CUDA_VISIBLE_DEVICES=0 \
TASK=square CKPT_PATH=outputs/stage3/latent_cos_temporal_v2/square/best.pt \
bash eval/run_eval.sh
```

---

## Experiment History (square task, 50 episodes)

### Baselines

| 모델 | SR |
|------|-----|
| SFT (OpenVLA-OFT) | 16% |
| WMPO P128 | 24% |
| WMPO P1280 | 36% |

### SWM 실험 결과

| # | 실험 | reward 설계 | SR | 비고 |
|---|------|------------|-----|------|
| 1 | Stage 3 초기 | graph distance | 10% | projector 학습됨 → feature drift |
| 2 | Stage 3 freeze_proj | graph distance | 10% | projector 고정. reward 설계 문제 |
| 3 | Stage 3 temporal_rm | temporal RM | —  | reward=0.0003 고정, GRPO 동작 안 함 |
| 4 | Stage 3 latent_cos (절대값) | cos(z_T, z_goal) | — | reward=0.97 고정, variance 없음 |
| 5 | **Stage 3 latent_cos (정규화)** | (cos_T − cos_0)/(1−cos_0) | **24%** | SFT 대비 +8%p, WMPO P128 동등 |
| 6 | Stage 3 latent_cos_temporal | trajectory 전체 평균 progress | TBD | 학습 완료 후 eval 예정 |
| 7 | Stage 3 latent_cos_temporal_v2 | 위와 동일 + n_states×4 | TBD | 다음 실험 예정 |

---

## Reward 설계 변천

### 실패 케이스

**Temporal RM (Stage 2.5 v3)**
- demo WM trajectory(+) vs random Gaussian trajectory(−) 로 학습 → val acc=100%
- 문제: VLA 생성 action이 전부 "random" bucket에 분류 → reward≈0, GRPO gradient=0

**Absolute cosine similarity**
- `reward = cos_sim(z_T, z_goal)`
- 문제: z_init과 z_goal이 이미 cosine sim≈0.97 → 모든 rollout reward 동일

### 현재 방식 (정규화된 temporal progress)

```python
# 각 timestep t에서 goal을 향한 progress 측정
progress_t = (cos(z_t, z_goal) - cos(z_init, z_goal)) / (1 - cos(z_init, z_goal))

# 궤적 전체 평균 (최종 state만 보지 않음)
reward = mean_t(progress_t)
```

- `reward=0`: 제자리 (아무 progress 없음)
- `reward=1`: goal에 완전 도달
- 실제 범위: −0.05 ~ 0.24 (GRPO에 충분한 variance)

---

## 주요 발견 및 교훈

1. **z_bypass는 학습 전용** — eval에서 USE_Z_BYPASS=True 사용 시 항상 SR=0%. eval은 반드시 native VLA pipeline 사용.

2. **freeze_projector=True 필수** — projector를 학습시키면 visual feature가 drift되어 성능 저하.

3. **reward variance가 핵심** — GRPO는 그룹 내 reward variance로 advantage를 계산하므로, 모든 rollout이 동일한 reward를 받으면 gradient=0.

4. **z_init 정규화** — cosine similarity의 절대값이 아닌 초기 상태 대비 상대적 progress를 측정해야 의미있는 학습 signal 생성.

5. **Temporal reward** — 최종 state만 보는 것보다 궤적 전체 평균이 일관적인 학습 signal 제공.

---

## Hardware

- GPU: 3× (183 GB VRAM)
- 학습: torchrun --nproc_per_node=3
- 평가: single GPU

## Citation

```bibtex
@misc{swm2026,
  title  = {SWM: Structured World Model for Robot Manipulation},
  author = {sjLee},
  year   = {2026}
}
```
