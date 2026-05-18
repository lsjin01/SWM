# SWM Phase 2 — Spatial World Model for GRPO Reward Shaping

> **Status**: Chain A (GPU 2,3,4) + Chain B (GPU 5,6,7) running in parallel

→ Phase 1 결과 및 한계: [README.md](README.md)

---

## Abstract

Phase 1의 핵심 실패 원인은 **mean-pool scalar latent의 구조적 정보 손실**이다. DINOv2+SigLIP 인코더가 생성한 256개 patch token을 평균내면 배경(전체 픽셀의 95%+)이 task object(5%)를 압도하여, 서로 다른 task phase가 cosine similarity ≥ 0.97로 구별 불가능해진다. GRPO reward가 항상 ≈ 1.0을 반환하면 group 내 advantage가 0에 수렴하여 학습이 정체된다.

Phase 2의 핵심 주장:

1. **Spatial representation 유지** — world model을 scalar space가 아닌 patch space (256×256)에서 학습하면, 배경 patch의 transition 분산이 작고 object patch의 transition 분산이 크다 → mean-pool 후에도 object 신호가 지배적
2. **Semantic goal 생성** — 고정 temporal goal (25/50/75/100%) 대신 PCA progress threshold 기반 adaptive goal → demo마다 의미론적으로 의미 있는 phase 전환점에 reward 부여
3. **Patch-selective reward** — attention map 또는 Δs magnitude로 task-relevant top-K patch를 식별하여 weighted mean-pool → 배경 noise 최소화

---

## 1. Phase 1 vs Phase 2: 핵심 구조 차이

```
Phase 1 (Scalar World Model)
────────────────────────────────────────────────────────────────
image_t ──→ encoder ──→ mean_pool(256 tokens) ──→ z_t ∈ ℝ²¹⁷⁶
                                                       │
                                             SWMTransition(z_t, a_t)
                                                       │
                                                 ẑ_{t+1} ∈ ℝ²¹⁷⁶
                                                       │
                                   reward = cos_sim(ẑ_{t+1}, z_goal)
                                   → background dominates → ≥ 0.97 always

Phase 2 (Spatial World Model)
────────────────────────────────────────────────────────────────
image_t ──→ encoder ──→ spatial_proj ──→ S_t ∈ ℝ^{256×256}
                         (Linear+LN)         (patch tokens preserved)
                                                       │
                              SpatialSWMTransition(S_t, a_t)
                              [action prepended as token; 6L Transformer]
                                                       │
                                               Ŝ_{t+1} ∈ ℝ^{256×256}
                                                       │
                             weighted_mean(Ŝ_{t+1}, w) ∈ ℝ^{256}
                                   w = patch_weights (attn or Δs)
                                                       │
                                   reward = cos_sim(z_rollout, z_goal)
                                   → object patches dominate → discriminative
```

---

## 2. 아키텍처

### 2.1 SWMEncoder 변경 (`models/encoder.py`)

```python
# Stage 1 동결, Stage 2에서 spatial_proj만 추가 학습
self.spatial_proj = nn.Sequential(
    nn.Linear(latent_dim, spatial_dim),   # 2176 → 256
    nn.LayerNorm(spatial_dim),
)

def encode_spatial_projected(self, image):
    # DINOv2+SigLIP patch tokens → spatial_proj → (B, 256, 256)
    patch_features = self._get_patch_tokens(image)   # (B, 256, 2176)
    return self.spatial_proj(patch_features)          # (B, 256, 256)
```

Stage 1 (DINOv2+SigLIP)은 완전 동결. `spatial_proj`만 Stage 2에서 학습됨.

### 2.2 SpatialSWMTransition (`models/transition.py`)

```python
class SpatialSWMTransition(nn.Module):
    """
    입력: S_t (B, N, d_s) + action (B, 7)
    출력: S_{t+1} (B, N, d_s)

    Action을 token으로 prepend → 257-token Transformer self-attention
    → action token이 어떤 patch에 attend하는지가 task-relevant patch를 결정
    """
    def forward(self, s_t, action):
        x = self.token_embed(s_t)               # (B, 256, hidden)
        a = self.action_embed(action).unsqueeze(1)  # (B, 1, hidden)
        x = torch.cat([a, x], dim=1)            # (B, 257, hidden)
        x = self.blocks(x)                      # 6-layer Transformer
        x = self.norm(x[:, 1:, :])              # drop action token
        return self.out(x)                      # (B, 256, d_s)
```

| 항목 | Phase 1 SWMTransition | Phase 2 SpatialSWMTransition |
|---|---|---|
| 입력 | (B, 2176) scalar | (B, 256, 256) spatial tokens |
| 출력 | (B, 2176) scalar | (B, 256, 256) spatial tokens |
| Action 처리 | linear concat | prepended token |
| Attention | 없음 | 257×257 full self-attention |
| Parameters | 10.4M | 19.2M |
| Loss | L2 | per-patch L1 |

**핵심 특성**: 257개 토큰이 single forward pass에서 전부 attend → action token(index 0)의 patch token(1~256)에 대한 attention weight가 곧 "이 action을 실행할 때 중요한 patch"를 나타냄.

### 2.3 Patch Weight 방법

#### 방법 A: Attention Map (`patch_weight_method: attn`)
```python
def get_patch_weights(self, s_t, action, top_k=64):
    # 마지막 Transformer layer action→patch attention 추출
    _, attn = last_layer.self_attn(x_norm, x_norm, x_norm,
        need_weights=True, average_attn_weights=True)  # (B, 257, 257)
    patch_w = attn[:, 0, 1:].mean(0)  # action token → patch tokens, (256,)
    # top-k sparse mask: top-64 patch만 활성화
    patch_w = patch_w * top_k_mask
    return patch_w / patch_w.sum()
```

- 학습된 attention이 task-relevant patch를 자동으로 식별
- top-k=64 → 전체의 25% patch만 사용

#### 방법 B: Δs Magnitude (`patch_weight_method: delta`)
```python
def get_delta_weights(self, s_t, action, top_k=64):
    s_next = self.forward(s_t, action)
    delta = (s_next - s_t).norm(dim=-1).mean(0)  # (256,) per-patch change
    delta = delta * top_k_mask
    return delta / delta.sum()
```

- Action으로 인해 실제로 변화한 patch를 중요도로 사용
- 물체 patch는 크게 변하고, 배경 patch는 거의 변하지 않음

---

## 3. Stage 2 학습

### 3.1 설정

```yaml
# configs/stage2_spatial_multitask.yaml
data:
  tasks: [square, coffee, stack_three, three_piece_assembly]
  train_pairs: 315,432   # MimicGen 4tasks + RoboMimic 3tasks
  val_pairs: 35,300

transition:
  use_spatial: true
  spatial_dim: 256       # d_s
  hidden_dim: 512
  num_layers: 6
  num_heads: 8

training:
  batch_size: 512        # GPU 2-7, 6-GPU DDP
  lr: 3.0e-4             # 안정적 fine-tune lr (1e-3은 발산 확인)
  epochs: 50
  loss: per-patch L1
```

### 3.2 학습 결과

| 시도 | lr | 결과 |
|---|---|---|
| 최초 (bs=512) | 3.2e-3 (linear scaling) | epoch 6 발산 (val spike) |
| resume (epoch3) | 1e-3 | optimizer state 미로드 → 발산 |
| **현재** | **3e-4** (optimizer state 포함) | **epoch 5: val=0.0630 ★** |

Stage 2 best checkpoint: `outputs/stage2/spatial_multitask/best.pt` (epoch 5, val=0.0630)

optimizer/scheduler/scaler state 저장 포함 (이후 resume 안정성 확보):
```python
ckpt = {
    "transition": ..., "spatial_proj": ...,
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
}
```

---

## 4. PCA-Adaptive Multi-Goal Reward

### 4.1 Phase 1 고정 temporal goal의 문제

```
Phase 1: goal frames = {T×0.25, T×0.50, T×0.75, T×1.0}
문제: 시간 기반 분할이 의미론적 phase와 불일치
→ t=25%에서 robot이 아직 물체에 접근 중이거나 이미 집었을 수 있음
→ reward가 실제 task progress와 misaligned
```

### 4.2 PCA-Adaptive Goal 알고리즘

```python
# demo trajectory를 PCA 공간에서 linear progress로 분석
pca_vecs = pca.transform(encoded_frames)       # (T', n_pca=16)
direction = pca_vecs[-1] - pca_vecs[0]         # start→end 방향벡터
proj_t = dot(pca_vecs[t] - pca_vecs[0], direction) / ||direction||²

# start→end 방향 progress가 threshold만큼 증가할 때마다 goal 추가
last_proj = 0.0
for t, proj in enumerate(projections):
    if proj - last_proj >= pca_goal_threshold:  # = 0.3
        goal_frames.append(t)
        last_proj = proj
goal_frames.append(T-1)  # 마지막 프레임 항상 포함
```

| threshold | 평균 goal 수 | 특성 |
|---|---|---|
| 0.2 | ~5–6 | 세밀, reward 분산 작음 |
| **0.3** | **~3–4** | **의미론적 phase (사용)** |
| 0.5 | ~2–3 | 거침, sparse reward |

### 4.3 Binary Reward 계산

```python
# 각 goal k에 대해:
z_pred = transition.rollout(z_init, actions)    # (B, T, d_s)
z_pred_pooled = weighted_mean(z_pred, patch_w)  # (B, T, d_s)
z_goal_pooled = weighted_mean(z_goal_k, patch_w)

cos_k = cos_sim(z_pred_pooled[:, -1], z_goal_pooled)  # per-rollout
progress_k = (cos_k - cos_init_k) / (1 - cos_init_k + ε)
binary_k = float(progress_k > phase_threshold)   # 0 or 1

reward = mean(binary_k for k in goals)
# reward ∈ {0, 1/K, 2/K, ..., 1}  where K = # adaptive goals
```

GRPO advantage: `A_i = (r_i - μ_group) / (σ_group + ε)`
Binary reward는 group 내 분산을 보장 (일부 rollout이 goal을 달성하고 일부는 못 달성).

---

## 5. 실험 설계 (2 × 3 Factorial Ablation)

### 5.1 실험 행렬

| | No Patch Weight | Attn Weight | Δs Weight |
|---|---|---|---|
| **PCA Multi-Goal** | **Exp 1** (chain1) | **Exp 3** (chain3) | **Exp 5** (chain5) |
| **PCA Binary** | **Exp 2** (chain2) | **Exp 4** (chain4) | **Exp 6** (chain6) |

- **행 (Reward Type)**: goal 생성 방식 비교
  - `pca_multi_goal`: PCA adaptive threshold (semantic phases)
  - `pca_binary`: fixed K=4 temporal goals (Phase 1 방식)
- **열 (Patch Weighting)**: reward 공간의 patch 선택 방식 비교
  - `none`: uniform mean-pool (baseline)
  - `attn`: last-layer action→patch attention
  - `delta`: ||Ŝ_{t+1} - S_t||₂ magnitude

**Phase 1 baseline** (비교 기준): 26% SR (pca_binary, scalar SWM)

### 5.2 실험별 설정

| Exp | Config | n_goals | patch_weight | temperature |
|---|---|---|---|---|
| 1 | stage3_spatial_pca_multi_goal | pca_adaptive (thresh=0.3) | none | 1.2 |
| 2 | stage3_spatial_pca_binary | 4 (fixed) | none | 1.2 |
| 3 | stage3_spatial_pca_multi_goal_attn | pca_adaptive | attn (top-64) | 1.2 |
| 4 | stage3_spatial_pca_binary_attn | 4 (fixed) | attn (top-64) | 1.2 |
| 5 | stage3_spatial_pca_multi_goal_delta | pca_adaptive | delta (top-64) | 1.2 |
| 6 | stage3_spatial_pca_binary_delta | 4 (fixed) | delta (top-64) | 1.2 |

공통 GRPO 설정:
```yaml
iterations: 200
max_demos: 300
n_states_per_iter: 4     # 4 initial states per iteration
g_rollouts: 8            # 8 rollouts per state → group size 8
temperature: 1.2
lr: 5.0e-6
clip_eps: 0.2
kl_coef: 0.05
```

### 5.3 Chain 실행 구조

```
Chain A (GPU 2,3,4)          Chain B (GPU 5,6,7)
────────────────────         ────────────────────
chain1: pca_multi_goal   ||  chain2: pca_binary
    ↓ eval done              ↓ eval done
chain3: multi_goal_attn  ||  chain4: binary_attn
    ↓ eval done              ↓ eval done
chain5: multi_goal_delta ||  chain6: binary_delta
```

- chain1 & chain2: 현재 동시 실행 중
- chain3 & chain4: chain1/chain2 eval 완료 대기 중
- chain5 & chain6: chain3/chain4 eval 완료 대기 중
- 총 소요 예상: ~18–24h (각 chain: train ~4h + eval ~3h/GPU)

---

## 6. 가설 및 예측

### 6.1 Primary Hypotheses

**H1: Spatial > Scalar** (Exp 2 vs Phase 1 baseline)
- `spatial_pca_binary` SR > 26% (Phase 1 pca_binary)
- Mechanism: spatial_proj가 배경 patch의 변화 분산을 억제하여 reward discriminability 향상

**H2: Adaptive Goal > Fixed Goal** (Exp 1 vs Exp 2)
- `spatial_pca_multi_goal` SR > `spatial_pca_binary`
- Mechanism: semantic phase goals → better credit assignment → policy reaches intermediate milestones

**H3: Patch Weighting Improves Reward** (Exp 3,4,5,6 vs Exp 1,2)
- attn/delta weighted reward > uniform mean-pool reward
- Mechanism: 배경 patch contribution 감소 → cos_sim 범위 확대 → 더 sharp한 reward signal

**H4: Attn ≈ Delta** (Exp 3 vs Exp 5, Exp 4 vs Exp 6)
- attention map과 Δs weight는 유사한 patch를 식별할 것으로 예측
- Mechanism: well-trained transition에서 action token이 크게 변하는 patch를 attend할 것

### 6.2 예상 결과 순위

```
multi_goal_attn ≈ multi_goal_delta  >  multi_goal
        >
binary_attn ≈ binary_delta  >  binary  >  Phase1 baseline(26%)
```

### 6.3 실패 시나리오 및 대응

| 시나리오 | 원인 | 대응 |
|---|---|---|
| binary == multi_goal | goal 수가 reward variance에 미치는 영향 미미 | threshold 조정 실험 |
| patch_weight 효과 없음 | val=0.063인 transition이 아직 uninformative | Stage 2 더 학습 후 재실험 |
| 전체 SR < 26% | spatial feature가 reward 공간으로 부적합 | spatial_proj fine-tuning (reward task-specific) |

---

## 7. Paper 구성 제안

### 7.1 주장 구조

```
Problem: VLA post-training with GRPO needs informative dense rewards
         → existing scalar world models suffer from background dominance

Method:
  1. Spatial World Model (SpatialSWMTransition)
     - patch-level prediction preserves task-relevant structure
     - action token attention identifies task-relevant patches
  2. PCA-Adaptive Goal Generation
     - semantic phase detection vs temporal heuristics
  3. Patch-Selective Reward Computation
     - attention/delta-based weighted mean-pool

Experiments:
  - 2×3 ablation on square manipulation task
  - Comparison with Phase 1 (scalar SWM) baseline
  - Analysis: patch weight visualization, reward variance, SR
```

### 7.2 핵심 Figure 계획

| Figure | 내용 |
|---|---|
| Fig 1 | Phase 1 vs Phase 2 pipeline 다이어그램 |
| Fig 2 | PCA adaptive goal 시각화 (demo trajectory + goal frames) |
| Fig 3 | Attention map / Δs weight 히트맵 (어떤 patch가 선택됐는지) |
| Fig 4 | 2×3 실험 결과 막대그래프 (SR ± std) |
| Fig 5 | Reward variance during training (GRPO advantage distribution) |
| Tab 1 | Phase 1 vs Phase 2 전체 비교 테이블 |

### 7.3 Ablation 스토리라인

```
[Scalar SWM baseline]: 26% SR
    ↓ + Spatial SWM (Exp 2 vs Phase 1)
[Spatial + Fixed Goal]: ? SR  → "spatial representation helps"
    ↓ + Adaptive Goal (Exp 1 vs Exp 2)
[Spatial + Adaptive Goal]: ? SR  → "semantic goals help"
    ↓ + Patch Weighting (Exp 3,5 vs Exp 1 / Exp 4,6 vs Exp 2)
[Spatial + Adaptive + Attn/Delta]: ? SR  → "patch selection helps"
```

각 단계가 독립적으로 기여하면 clean ablation story.

---

## 8. 파일 구조

```
SWM/
├── configs/
│   ├── stage2_spatial_multitask.yaml
│   ├── stage3_spatial_pca_multi_goal.yaml         # Exp 1 (chain1)
│   ├── stage3_spatial_pca_binary.yaml             # Exp 2 (chain2)
│   ├── stage3_spatial_pca_multi_goal_attn.yaml    # Exp 3 (chain3)
│   ├── stage3_spatial_pca_binary_attn.yaml        # Exp 4 (chain4)
│   ├── stage3_spatial_pca_multi_goal_delta.yaml   # Exp 5 (chain5)
│   └── stage3_spatial_pca_binary_delta.yaml       # Exp 6 (chain6)
├── models/
│   ├── encoder.py       # encode_spatial_projected()
│   └── transition.py    # SpatialSWMTransition, get_patch_weights(), get_delta_weights()
├── scripts/
│   ├── train_stage2.py  # spatial DDP, optimizer state save/load
│   ├── train_stage3.py  # spatial reward, pca_adaptive, patch_weights
│   └── phase2/
│       ├── chain1_spatial_pca_multi_goal.sh      # Chain A-1 (GPU 2,3,4)
│       ├── chain2_spatial_pca_binary.sh           # Chain B-1 (GPU 5,6,7)
│       ├── chain3_spatial_pca_multi_goal_attn.sh  # Chain A-2 (GPU 2,3,4)
│       ├── chain4_spatial_pca_binary_attn.sh      # Chain B-2 (GPU 5,6,7)
│       ├── chain5_spatial_pca_multi_goal_delta.sh # Chain A-3 (GPU 2,3,4)
│       └── chain6_spatial_pca_binary_delta.sh     # Chain B-3 (GPU 5,6,7)
└── eval/
    └── eval_swm_mimicgen.py
```

---

## 9. 현재 실행 상태 (2026-05-19)

```
Chain A (GPU 2,3,4)
├── chain1: RUNNING  — train_spatial_pca_multi_goal_20260518_235721.log
├── chain3: WAITING  — for eval_spatial_pca_multi_goal_best.log
└── chain5: WAITING  — for eval_spatial_pca_multi_goal_attn_best.log

Chain B (GPU 5,6,7)
├── chain2: RUNNING  — train_spatial_pca_binary_20260518_235723.log
├── chain4: WAITING  — for eval_spatial_pca_binary_best.log
└── chain6: WAITING  — for eval_spatial_pca_binary_attn_best.log

Stage 2 checkpoint: outputs/stage2/spatial_multitask/best.pt
  epoch=5, val_loss=0.0630 (L1, per-patch)
```
