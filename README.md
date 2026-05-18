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
│   ├── stage2_5_v3.yaml                    # Temporal Reward Model
│   ├── stage3_latent_cos.yaml              # Normalized cosine, endpoint (best: 24%)
│   ├── stage3_latent_cos_temporal.yaml     # Temporal progress reward (10%)
│   ├── stage3_multi_goal.yaml              # Uniform 4-goal endpoint (20%)
│   ├── stage3_pca_goal.yaml                # PCA cosine, n_goals=4 (20%)
│   ├── stage3_pca_goal_single.yaml         # PCA cosine, n_goals=1 (16%)
│   ├── stage3_pca_delta.yaml               # PCA delta cosine, n_goals=1 (12%)
│   ├── stage3_action_goal.yaml             # Action-velocity based phase detection (16%)
│   ├── stage3_pca_binary.yaml              # PCA binary phase reward (dir.1) → 26% last
│   ├── stage3_diversity.yaml               # High-temperature rollout diversity (dir.4, 24%)
│   ├── stage3_pca_max.yaml                 # PCA max-progress aggregation (dir.2, 8%)
│   ├── stage3_action_pca_delta.yaml        # Action goal + PCA delta (dir.3, TBD)
│   ├── stage3_var_pool.yaml                # Variance-weighted spatial pooling
│   ├── stage3_corr_pool.yaml               # Correlation-weighted spatial pooling
│   ├── stage3_act_pool.yaml                # Activation-weighted spatial pooling
│   └── stage3_slot_attn.yaml               # Slot attention spatial pooling
├── models/
│   ├── encoder.py       # SWMEncoder (DINOv2 + SigLIP, freeze_backbone=True)
│   ├── transition.py    # SWMTransition (latent=2176, hidden=512, layers=6)
│   ├── heads.py         # Graph/Depth prediction heads
│   └── reward_model.py  # SWMRewardModel (Temporal Transformer)
├── scripts/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── train_stage2_5.py  # Temporal RM training
│   ├── train_stage3.py    # GRPO fine-tuning
│   ├── analyze_latent_inflection.py    # Δz / curvature 분석
│   ├── analyze_pca_latent.py           # PCA fit & goal separation 분석
│   └── analyze_pca_generalization.py   # train/test PCA 일반화 분석
└── eval/
    └── eval_swm_mimicgen.py  # MimicGen simulator evaluation (pixel mode only)
```

---

## Quick Start

```bash
# Stage 1: Encoder
python scripts/train_stage1.py --config configs/stage1_multitask_dinosiglip.yaml

# Stage 2: Transition
python scripts/train_stage2.py --config configs/stage2_multitask_dinosiglip.yaml

# Stage 3: GRPO (예: latent_cos endpoint)
export CUDA_VISIBLE_DEVICES=2,3,7
torchrun --nproc_per_node=3 --master_port=29507 \
    scripts/train_stage3.py \
    --config configs/stage3_latent_cos.yaml

# Eval (pixel mode — z_bypass 완전 제거됨)
CUDA_VISIBLE_DEVICES=2 TASK=square \
  CKPT_PATH=outputs/stage3/latent_cos/square/best.pt \
  STAGE1_CKPT=outputs/stage1/multitask_dinosiglip/best.pt \
  STAGE2_CKPT=outputs/stage2/multitask_dinosiglip/best.pt \
  VLA_BASE=<SFT_model_path> VLA_DEVICE=cuda N_EPISODES=50 \
  python eval/eval_swm_mimicgen.py
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

| # | 실험 | reward 설계 | SR (best) | SR (last) | 비고 |
|---|------|------------|-----------|-----------|------|
| 1 | Stage 3 초기 | graph distance | 10% | — | projector 학습됨 → feature drift |
| 2 | freeze_proj | graph distance | 10% | — | projector 고정. reward 설계 문제 |
| 3 | temporal_rm | temporal RM | — | — | reward=0.0003 고정, GRPO 동작 안 함 |
| 4 | latent_cos (절대값) | cos(z_T, z_goal) | — | — | reward=0.97 고정, variance 없음 |
| 5 | **latent_cos (정규화)** | (cos_T−cos_0)/(1−cos_0), n_goals=1 | **24%** | **24%** | SFT +8%p, WMPO P128 동등 |
| 6 | latent_cos_temporal | trajectory 전체 평균 progress | 10% | 10% | variance 희석, endpoint보다 열등 |
| 7 | multi_goal (uniform) | endpoint, K=4 균등 goal 평균 | 20% | 18% | n_goals=4 (25/50/75/100%) |
| 8 | pca_goal (n_goals=4) | PCA-16 투영 후 cosine progress, K=4 | 20% | 12% | goal 분리도 0.97→0.05 |
| 9 | pca_goal_single (n_goals=1) | PCA-16 투영 후 cosine progress, K=1 | 16% | 16% | PCA 단일 goal = SFT 동등 |
| 10 | action_goal | action velocity 기반 phase 탐지, K=4 | 16% | 16% | |
| 11 | pca_delta | PCA 공간 delta 방향 비교 | 12% | 16% | |
| 12 | **pca_binary (dir.1)** | PCA phase binary threshold=0.3, K=4 | 20% | **26%** | 현재 최고 (last) |
| 13 | diversity (dir.4) | temperature=2.0, endpoint cosine | 24% | 18% | baseline 동급 |
| 14 | pca_max (dir.2) | PCA max-progress aggregation, K=4 | 8% | 12% | max 집계 역효과 |
| 15 | action_pca_delta (dir.3) | action goal + PCA delta cosine | TBD | TBD | 실험 진행 중 |
| 16 | var_pool | variance-weighted spatial pooling | TBD | TBD | 실험 예정 |
| 17 | corr_pool | correlation-weighted spatial pooling | TBD | TBD | 실험 예정 |
| 18 | act_pool | activation-weighted spatial pooling | TBD | TBD | 실험 예정 |
| 19 | slot_attn | slot attention spatial pooling (PCA init) | TBD | TBD | 실험 예정 |

---

## Reward 설계 변천

### 실패 케이스

**Temporal RM (Stage 2.5 v3)**
- demo WM trajectory(+) vs random Gaussian trajectory(−) 로 학습 → val acc=100%
- 문제: VLA 생성 action이 전부 "random" bucket에 분류 → reward≈0, GRPO gradient=0

**Absolute cosine similarity**
- `reward = cos_sim(z_T, z_goal)`
- 문제: z_init과 z_goal이 이미 cosine sim≈0.97 → 모든 rollout reward 동일

**Multi-goal (uniform) — latent 공간 goal 분리 실패**
- 2176-dim 전체 latent 공간에서 25/50/75/100% 프레임의 goal inter-similarity = 0.97+
- 4개 goal이 사실상 동일 → reward variance=0 → GRPO 학습 실패
- 재평가(z_bypass 버그 수정 후): best 20%, last 18%

### Latent Indistinguishability 분석

```
DINOv2+SigLIP 2176-dim 공간의 문제:
  - goal 간 cosine similarity: 0.97+ (goal 4개가 거의 동일)
  - Δz norm: ~0.002 (매우 작음)
  - 곡률(curvature) mean=0.85~0.93 (random orthogonality에 가까움)

PCA top-16 적용 시:
  - explained variance: 73.2% (task-relevant 차원만 보존)
  - goal 간 cosine similarity: 0.05~0.08 (잘 분리됨)
  - train/test 일반화: held-out demo에서도 동일한 separation
```

### 현재 방식 (Endpoint Progress + PCA)

```python
# PCA 투영 후 cosine progress (pca_cosine)
z_pca = pca.transform(z)           # 2176-dim → 16-dim
progress_k = (cos(z_T_pca, goal_k_pca) - cos(z_0_pca, goal_k_pca)) /
             (1 - cos(z_0_pca, goal_k_pca))
reward = mean_k(progress_k)        # or max_k for pca_max

# PCA delta cosine (pca_delta_cosine)
dz_T    = z_T_pca - z_0_pca
dz_goal = z_goal_pca - z_0_pca
reward  = cos(dz_T, dz_goal)       # 변화 방향 유사도

# Binary phase reward (pca_binary, phase_threshold=0.3)
binary_k = 1.0 if progress_k > 0.3 else 0.0
reward   = mean_k(binary_k)        # → {0, 0.25, 0.5, 0.75, 1.0}
```

### 방식 비교 (확정된 결과)

| 방식 | 핵심 아이디어 | SR (best) |
|------|------------|-----------|
| endpoint (n_goals=1) | `(cos_T − cos_0)/(1−cos_0)` | **24%** |
| temporal average | `mean_t[progress_t]` | 10% ↓ |
| multi-goal uniform (n_goals=4) | `mean_k[progress_k]`, 2176-dim | 20% |
| pca_goal (n_goals=4) | `mean_k[progress_k]`, PCA-16 | 20% |
| pca_goal_single (n_goals=1) | `progress`, PCA-16 | 16% |

### 개선 방향 및 결과 (chain7~14)

GRPO 핵심 문제 = **within-group reward variance 부족** (std≈0.07~0.08, 8 rollout이 거의 동일한 reward 수령)

#### Reward Variance 개선 (chain7~10)

| 방향 | 아이디어 | config | SR (best) | SR (last) |
|------|---------|--------|-----------|-----------|
| Dir.1 | Binary phase reward: progress_k > 0.3 → 1.0 | `pca_binary` | 20% | **26%** ← 최고 |
| Dir.2 | Max-progress: mean → max aggregation | `pca_max` | 8% ↓ | 12% |
| Dir.3 | Action-goal + PCA delta 조합 | `action_pca_delta` | TBD | TBD |
| Dir.4 | Temperature 2.0 → rollout 다양성 증가 | `diversity` | 24% | 18% |

**분석:** Binary reward (Dir.1)가 가장 효과적 — 이산적 reward {0, 0.25, 0.5, 0.75, 1.0}로 within-group variance 증가. Max aggregation (Dir.2)은 역효과.

#### Mean Pool 문제 해결 (chain11~14)

**근본 문제:** 256 spatial token mean-pool → 배경 토큰이 지배 → goal latent 간 구분 불가 (sim=0.97+)

| 방향 | 아이디어 | config |
|------|---------|--------|
| Method 1 | Variance-weighted: 시간적으로 많이 변하는 토큰 | `var_pool` |
| Method 2 | Correlation-weighted: task progress와 상관 높은 토큰 | `corr_pool` |
| Method 3 | Activation-weighted: activation norm이 큰 토큰 | `act_pool` |
| Method 4 | Slot attention: PCA init + iterative competition | `slot_attn` |

---

## 주요 발견 및 교훈

1. **z_bypass는 eval에서 완전 제거** — eval script에서 `use_z_bypass` 관련 코드 삭제. 항상 pixel mode(native VLA pipeline) 사용.

2. **freeze_projector=True 필수** — projector를 학습시키면 visual feature가 drift되어 성능 저하.

3. **reward variance가 핵심** — GRPO는 그룹 내 reward variance로 advantage를 계산하므로, 모든 rollout이 동일한 reward를 받으면 gradient=0. latent_cos std≈0.07, 학습 전반에 걸쳐 reward 거의 flat.

4. **z_init 정규화** — cosine similarity의 절대값이 아닌 초기 상태 대비 상대적 progress를 측정해야 의미있는 학습 signal 생성.

5. **Temporal averaging HURTS** — 궤적 전체 평균은 per-rollout variance를 희석 → GRPO signal 약화(10%). endpoint(z_T only)가 더 좋음(24%).

6. **PCA top-16으로 latent 분리도 개선** — 2176-dim 공간에서 goal sim=0.97+이던 것이 PCA-16에서 0.05~0.08로 감소. 그러나 SR 개선은 미미 (pca_goal: 20% = multi_goal 동등). reward variance 문제가 근본 원인.

7. **GRPO learning signal 미약** — 200 iter 학습에서 reward mean이 초반~후반 거의 동일 (latent_cos: 0.12→0.11, pca_goal: -0.02→+0.005). loss도 flat (~0.09). 7B 모델 fine-tuning에 더 강한 signal 필요.

---

## Hardware

- GPU: 3× B200 (183 GB VRAM each)
- 학습: torchrun --nproc_per_node=3, GPU 2,3,7
- 평가: single GPU (GPU 2 or 3)

## Citation

```bibtex
@misc{swm2026,
  title  = {SWM: Structured World Model for Robot Manipulation},
  author = {sjLee},
  year   = {2026}
}
```
