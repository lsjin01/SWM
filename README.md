# SWM Phase 1 — Latent Reward Design for GRPO-based VLA Fine-tuning

> **상태: COMPLETED** | 최고 성능: **26% SR** (pca_binary, last ckpt) | 한계 확인 후 Phase 2로 이행

→ Phase 2 설계: [README2.md](README2.md)

---

## Abstract

SWM(Structured World Model)은 전문가 데모에서 학습한 latent world model을 dense reward signal로 삼아 GRPO로 VLA를 fine-tuning하는 프레임워크다. Phase 1에서는 **mean-pool 기반 scalar latent 공간**에서 다양한 reward 설계를 실험했다.

핵심 발견:
1. 256 spatial token을 mean-pool하면 배경(95%+)이 지배 → 서로 다른 task phase의 latent가 cosine sim ≥ 0.97로 구별 불가
2. PCA 투영으로 goal 분리는 가능하지만 reward variance는 여전히 낮음
3. Binary thresholding으로 reward를 이산화하는 것이 GRPO에 가장 효과적

**Phase 1 최고 결과: pca_binary 26% SR** — SFT(16%) +10%p, WMPO-P128(24%) 초과.
**Phase 1 핵심 한계:** mean-pool 공간의 근본적인 indistinguishability는 해결 안 됨 → Phase 2 (SpatialSWMTransition) 이행.

---

## 1. 시스템 구성

```
Stage 1  │ Encoder    : image → z_t  (DINOv2-1024 ⊕ SigLIP-1152 concat → 2176-dim, frozen)
Stage 2  │ Transition : (z_t, a_t) → ẑ_{t+1}  (6-layer Transformer, 21.2M params, scalar latent)
Stage 3  │ Policy     : GRPO, latent reward signal
```

평가: MimicGen square-assembly, 50 episodes, native pixel-mode VLA inference (no world model at test time)

---

## 2. 핵심 문제: Reward Variance Collapse

GRPO advantage = `(r_i - μ_group) / σ_group`

σ_group ≈ 0이면 gradient ≈ 0. Phase 1 전반에 걸쳐 σ ≈ 0.07–0.08로 고정되어 학습 신호가 매우 약하다.

**근본 원인: Mean-pool indistinguishability**

```
task object 크기:     ~1–5% of image (256 tokens 중 3–13개)
배경 비율:            ~95% of tokens → mean-pool 지배
goal cosine sim:      서로 다른 phase 간에도 ≥ 0.97 (전부 같아 보임)
→ 어떤 rollout이든 비슷한 reward → σ ≈ 0 → GRPO gradient ≈ 0
```

---

## 3. Phase 1 실험 전체 결과

### Baselines

| 모델 | SR | 비고 |
|---|---|---|
| SFT (OpenVLA-OFT) | 16% | Supervised fine-tuning only |
| WMPO-P128 | 24% | Pixel-level world model, 128 rollout steps |
| WMPO-P1280 | 36% | Pixel-level, 1280 rollout steps (target) |

### Phase 1 실험 결과

| # | 실험명 | Reward 설계 | SR (best) | SR (last) | 판정 |
|---|---|---|---|---|---|
| 0 | latent_cos_abs | cos(z_T, z_goal) 절댓값 | 0% | 0% | 실패: reward≈0.97 상수, σ≈0 |
| 1 | temporal_rm | Temporal RM 분류기 | 0% | 0% | 실패: covariate shift, reward≈0 |
| 2 | freeze_proj | Graph distance | ~10% | — | 실패: reward 설계 미흡 |
| 3 | latent_cos (ZBP) | 정규화 progress, K=1 | 0% | 0% | 실패: ZBP 인코딩 버그 |
| 4 | **latent_cos** | 정규화 progress, K=1 | **24%** | **24%** | SFT+8%p, WMPO-P128 동급 |
| 5 | latent_cos_temporal | 궤적 평균 progress | 10% | 10% | 실패: variance 희석 |
| 6 | multi_goal | 정규화 progress, K=4 (2176-dim) | 20% | 18% | goal sim≥0.97 |
| 7 | pca_goal | PCA-16, K=4 (fixed temporal) | 20% | 12% | goal 분리↑ but variance-limited |
| 8 | pca_goal_single | PCA-16, K=1 | 16% | 16% | multi-goal 없으면 SFT 수준 |
| 9 | action_goal | Action velocity peak, K=4 | 16% | 16% | phase 감지 부정확 |
| 10 | pca_delta | PCA delta cosine, K=1 | 12% | 16% | 연속값, flat signal |
| 11 | **pca_binary** | PCA binary, τ=0.3, K=4 (fixed) | 20% | **26%** | **Phase 1 최고** ★ |
| 12 | diversity | 정규화 progress, K=1, T=2.0 | 24% | 18% | temp↑ = 탐색↑ but 한계 동일 |
| 13 | pca_max | PCA max-progress, K=4 | 8% | 12% | max aggregation 역효과 |
| 14 | action_pca_delta | Action phase + PCA delta | 20% | 16% | 유망하나 한계 동일 |
| 15–18 | var/corr/act_pool + slot_attn (binary/t2) | pca_binary + 공간 가중치 | **INVALID** | **INVALID** | **distribution mismatch 버그로 전체 kill** |

### 실험 15–18 폐기 사유

```
bug: z_history ← SWMTransition (mean-pool scalar 공간)
     z_goal    ← encode_weighted (weighted spatial 공간)
     → 다른 분포 벡터끼리 cosine similarity 계산 → reward 무의미
```

이 발견이 Phase 2 설계의 직접적인 동기가 되었다.

---

## 4. 핵심 발견

### 이산 reward > 연속 reward (GRPO에서)

| reward 유형 | reward값 범위 | SR |
|---|---|---|
| 연속 cosine | [0, 1] 실수 | 10–24% |
| **이산 binary {0, 0.25, 0.5, 0.75, 1.0}** | 5개 이산값 | **26%** |

### Multi-goal이 single-goal보다 효과적

| 설정 | SR (best) |
|---|---|
| K=1 (endpoint only, PCA) | 16% |
| K=4 (temporal 25/50/75/100%) | 20% |
| K=4 + binary τ=0.3 | 26% |

### Aggregation 비교 (K=4, PCA)

| 방식 | SR (last) |
|---|---|
| Mean | 12–26% |
| **Binary mean** | **26%** |
| Max | 8–12% |

Max는 8개 rollout 중 가장 쉬운 phase만 달성해도 높은 reward → variance 감소 → 역효과.

### PCA 투영의 필요성

| 공간 | goal 간 cosine sim | 학습 효과 |
|---|---|---|
| 2176-dim 원본 | ≥ 0.97 (사실상 동일) | 제한적 |
| PCA top-16 | 0.05–0.08 (명확히 분리) | 개선 |

PCA는 goal 분리를 가능하게 하지만 reward variance 자체는 여전히 낮아 silver bullet이 아님.

### Freeze projector 필수

GRPO 중 projector 업데이트 시 reward hacking 발생: encoder가 reward geometry를 재정의 → transition model과 불일치 → SR 하락.

---

## 5. Phase 1 한계 및 Phase 2 이행 근거

Phase 1 최고 성능 26%는 WMPO-P1280(36%)에 10%p 뒤처진다. 근본 문제는 해결되지 않았다:

```
mean-pool latent: 256 토큰 → scalar 벡터 1개
                  배경 패치(95%+)가 task 패치(5%)를 압도
                  → reward가 실제 task 진행과 약하게만 연결됨
                  → GRPO gradient 신호 약함
```

Phase 2에서는 SpatialSWMTransition 도입으로 이 문제를 구조적으로 해결한다:
- Transition이 256 패치 각각을 독립적으로 예측
- task 관련 패치(물체가 있는 위치)는 action에 따라 크게 변동
- 배경 패치는 거의 변하지 않음 (학습에서 자동 무시됨)

→ 자세한 설계: [README2.md](README2.md)

---

## 6. 공통 학습 설정

```yaml
# Stage 3 GRPO (Phase 1)
iterations: 200
max_demos: 300
n_states_per_iter: 4
g_rollouts: 8
mini_g: 2
n_rollout_chunks: 20       # 20 chunks × 8 actions = 160 action steps
temperature: 1.2
freeze_projector: true
lr: 5.0e-6
kl_coef: 0.05
clip_eps: 0.2
```

**하드웨어**: NVIDIA B200 (183 GB VRAM) × 3 for training, × 1 for eval

---

## 7. 파일 구조

```
SWM/
├── configs/
│   ├── stage2_multitask_dinosiglip.yaml       # Phase 1 scalar transition
│   ├── stage3_latent_cos.yaml                  → 24% SR ✓
│   ├── stage3_multi_goal.yaml                  → 20% SR ✓
│   ├── stage3_pca_goal.yaml                    → 20% SR ✓
│   ├── stage3_pca_goal_single.yaml             → 16% SR ✓
│   ├── stage3_pca_delta.yaml                   → 16% SR ✓
│   ├── stage3_pca_binary.yaml                  → 26% SR ✓ ★ Phase 1 Best
│   ├── stage3_diversity.yaml                   → 24% SR ✓
│   ├── stage3_pca_max.yaml                     → 12% SR ✓
│   ├── stage3_action_goal.yaml                 → 16% SR ✓
│   ├── stage3_action_pca_delta.yaml            → 20% SR ✓
│   └── stage3_{var,corr,act}_pool*.yaml        → INVALID (distribution mismatch)
├── models/
│   ├── encoder.py       # SWMEncoder: DINOv2+SigLIP, encode_weighted()
│   ├── transition.py    # SWMTransition (scalar) + SpatialSWMTransition (Phase 2)
│   └── reward_model.py  # Temporal RM (deprecated)
├── scripts/
│   ├── train_stage2.py  # scalar + spatial 모드 지원
│   └── train_stage3.py  # GRPO, pca_adaptive goal 지원 (Phase 2)
└── eval/
    └── eval_swm_mimicgen.py
```
