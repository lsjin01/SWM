# SWM Phase 2 — Spatial World Model

> **상태: IN PROGRESS** | Stage 2 학습 중 (batch=512, GPU 2-7) | ETA ~01:47 AM

→ Phase 1 결과 및 한계: [README.md](README.md)

---

## Abstract

Phase 1의 핵심 한계는 **mean-pool scalar latent**의 구조적 문제다: 256 spatial token을 평균내면 배경(95%+)이 task object(5%)를 압도하여 서로 다른 task phase가 cosine sim ≥ 0.97로 구별 불가능하다. Phase 2에서는 이 문제를 **SpatialSWMTransition**으로 구조적으로 해결한다: world model이 scalar 벡터 대신 **256 패치 각각**을 독립적으로 예측하도록 재설계하여, task-relevant 패치(물체 위치)는 action에 따라 크게 변화하고 배경 패치는 고정되는 자연스러운 구조를 만든다.

---

## 1. Phase 1 vs Phase 2: 핵심 차이

```
Phase 1 (Scalar World Model)
────────────────────────────
image → encoder → mean_pool(256 tokens) → z ∈ ℝ²¹⁷⁶
                                              ↓
                                    SWMTransition(z, a) → ẑ ∈ ℝ²¹⁷⁶
                                              ↓
                              reward = f(ẑ, z_goal)  ← 배경 압도 문제

Phase 2 (Spatial World Model)
─────────────────────────────
image → encoder → spatial_proj → S ∈ ℝ^(256×256)   ← 패치별 분리 유지
                                              ↓
                           SpatialSWMTransition(S, a) → Ŝ ∈ ℝ^(256×256)
                                              ↓
                         reward = f(mean(Ŝ), mean(S_goal))
                                   ↑
                         (배경 패치 분산 ≈ 0 → 자동으로 영향 최소화)
```

---

## 2. 아키텍처 변경

### 2.1 SWMEncoder 변경 (encoder.py)

```python
# 기존: global latent만
z = encoder(image)                   # (B, 2176)

# 추가: spatial projection
S = encoder.encode_spatial_projected(image)  # (B, 256, 256)
#     DINOv2+SigLIP (2176-dim) → spatial_proj (Linear+LayerNorm) → 256-dim
```

`spatial_proj`: Linear(2176→256) + LayerNorm(256), Stage 2에서 학습됨.

### 2.2 SpatialSWMTransition (transition.py)

```python
class SpatialSWMTransition(nn.Module):
    # 입력: S_t (B, 256, 256) + action (B, 7)
    # 출력: S_{t+1} (B, 256, 256)

    def forward(self, s_t, action):
        x = self.token_embed(s_t)         # (B, 256, 512)
        a = self.action_embed(action)     # (B, 1, 512)  ← action을 token으로 prepend
        x = torch.cat([a, x], dim=1)     # (B, 257, 512)
        x = self.blocks(x)               # 6-layer Transformer
        x = self.norm(x[:, 1:, :])       # (B, 256, 512)  ← action token 제거
        return self.out(x)               # (B, 256, 256)
```

| 항목 | Phase 1 SWMTransition | Phase 2 SpatialSWMTransition |
|---|---|---|
| 입력 shape | (B, 2176) scalar | (B, 256, 256) spatial tokens |
| 출력 shape | (B, 2176) scalar | (B, 256, 256) spatial tokens |
| Action 처리 | concat to latent | prepend as token |
| Parameters | 21.2M | 19.2M |
| 손실 함수 | MSE | per-patch L1 |

### 2.3 Stage 2 학습 변경 (train_stage2.py)

```yaml
# configs/stage2_spatial_multitask.yaml
transition:
  use_spatial: true
  spatial_dim: 256
  hidden_dim: 512
  num_layers: 6
  num_heads: 8

training:
  batch_size: 512     # 6 GPU DDP (GPU 2-7), lr linear scaling
  lr: 3.2e-3          # 4e-4 × (512/64) linear scale
  epochs: 50
```

데이터: MimicGen 4 tasks + RoboMimic 3 tasks = **315,432 train pairs**, 35,300 val pairs

체크포인트 저장 키:
```python
ckpt["spatial_proj"] = encoder.spatial_proj.state_dict()
ckpt["spatial_dim"]  = 256
ckpt["transition"]   = spatial_transition.state_dict()
```

---

## 3. Phase 2 Reward 설계

### 3.1 Spatial Reward 계산

```python
# swm_rollout: spatial transition 후 reward용 mean-pool
if is_spatial:  # z.ndim == 3
    z_history.append(z.mean(dim=1).clone())  # (B, 256) for reward

# goal encoding: spatial → mean-pool
goal = encoder.encode_spatial_projected(img).mean(dim=1)  # (B, 256)
```

reward 비교는 (B, 256) mean-pool 공간에서 수행. Phase 1의 (B, 2176)보다 더 compact하고 spatial_proj가 transition-optimized feature를 담고 있어 discriminability 개선 기대.

### 3.2 Known Limitation

reward를 계산할 때도 mean-pool → 배경 패치가 여전히 영향을 미침. Phase 2에서 이것이 실제로 문제인지 확인 후, 필요시 **patch variance-weighted mean**으로 개선 예정:

```python
# 개선 후보 (Phase 2 후반)
patch_var = z.var(dim=0) + 1e-8   # (256,) 각 패치의 분산
weights = patch_var / patch_var.sum()
z_reward = (z * weights.unsqueeze(-1)).sum(dim=1)  # weighted mean
```

---

## 4. PCA-Adaptive Multi-Goal (신규)

Phase 1의 fixed temporal goal(25/50/75/100%)을 **PCA progress 기반 동적 goal 생성**으로 교체:

```
Phase 1: goal frames at t = [T×0.25, T×0.50, T×0.75, T×1.0]
         → 시간 기반, demo 내용과 무관

Phase 2: goal frames where PCA_progress(t) - PCA_progress(last_goal) ≥ threshold
         → 의미론적 phase 전환점 기반
```

**알고리즘:**
```python
# demo trajectory를 PCA 공간에 투영
pca_vecs = pca.transform(encoded_frames)  # (T', n_pca)

# start→end 방향으로 scalar projection
p0, pT = pca_vecs[0], pca_vecs[-1]
direction = pT - p0
proj_t = dot(pca_vecs[t] - p0, direction) / (||direction||² + ε)

# threshold 초과 시마다 goal 생성
when proj_t - proj_last_goal ≥ pca_goal_threshold:
    → 새 goal 추가
    → last_goal = t
```

| 설정 | 평균 goal 수 | 특징 |
|---|---|---|
| threshold=0.2 | ~6개 | 세밀한 phase |
| **threshold=0.3** | **~3–4개** | **균형잡힌 phase 분할** ← 사용 |
| threshold=0.5 | ~2–3개 | 거친 phase |

마지막 프레임은 항상 포함 보장.

---

## 5. Phase 2 실험 계획

Stage 2 학습 완료 후 (`outputs/stage2/spatial_multitask/best.pt`) 순차 실행:

| 우선순위 | 실험명 | Config | Reward 설계 | 비교 대상 |
|---|---|---|---|---|
| 1 | **spatial_pca_multi_goal** | stage3_spatial_pca_multi_goal.yaml | PCA adaptive, threshold=0.3, binary τ=0.3 | Phase 1 pca_binary(26%) |
| 2 | **spatial_pca_binary** | stage3_spatial_pca_binary.yaml | PCA binary, K=4 fixed, τ=0.3 | Phase 1 pca_binary (직접 비교) |
| 3 | **spatial_pca_binary_t2** | stage3_spatial_pca_binary_t2.yaml | PCA binary + temperature=2.0 | pca_binary + diversity 결합 |
| 4 | spatial_latent_cos | stage3_spatial_latent_cos.yaml | 정규화 progress, K=1 | Phase 1 latent_cos(24%) |
| 5 | spatial_pca_delta | stage3_spatial_pca_delta.yaml | PCA delta cosine | Phase 1 pca_delta(16%) |
| 6 | spatial_diversity | stage3_spatial_diversity.yaml | 정규화 progress, T=2.0 | Phase 1 diversity(24%) |

**실험 1이 최우선**: Phase 1 best(pca_binary)의 두 가지 개선을 동시에 적용:
- spatial transition (mean-pool 문제 해결)
- adaptive goal (temporal → PCA-progress 기반)

---

## 6. 현재 진행 상황

```
Stage 2 학습 중
  - GPU 2-7 (6 GPUs, DDP)
  - batch=512, lr=3.2e-3 (linear scaling)
  - 102 steps/epoch, 50 epochs total
  - ETA: ~01:47 AM (5월 19일)

Stage 3 debug 완료
  - spatial_debug: 2 iter 정상 완료 ✓
  - pca_multi_goal_debug: 2 iter 정상 완료, demo당 6 goals 생성 확인 ✓

Fix 완료 (train_stage3.py)
  - dtype mismatch: z.float() → z.to(dtype) (bfloat16 transition)
  - z_init mean-pool: spatial (1,256,256) → (1,256) before PCA
  - pca_adaptive: load_initial_states에 n_goals='pca_adaptive' 브랜치 추가
```

---

## 7. Phase 2 이후 후보 방향

Phase 2 실험 결과에 따라 다음 방향을 고려:

### 방향 A: Weighted Reward (patch variance-based)
- 현재 reward의 mean-pool 문제가 여전히 존재할 경우
- spatial transition 후 patch variance로 가중치 계산 → weighted mean-pool

### 방향 B: Reward-Discriminability 개선
- spatial_proj가 transition 정확도를 위해 학습되어 reward discriminability가 부족할 수 있음
- Stage 2.5: spatial_proj를 reward signal로 추가 fine-tuning

### 방향 C: Multi-task 확장
- 현재 실험: square task 단일
- Phase 2 성공 시 coffee, stack_three, three_piece_assembly로 확장
- Stage 2가 이미 4 tasks로 학습됨 → Stage 3만 task별 실행

### 방향 D: Compute 최적화
- Phase 1 대비 Phase 2 장점:
  - SWM rollout: 19.2M params (lightweight)
  - WMPO 대비: 7B SFT3 모델 rollout 불필요 (3 GPU vs 32 GPU)
- 이 이점을 g_rollouts 증가 (8→16)나 iterations 증가 (200→400)에 활용 가능

---

## 8. 파일 구조 (Phase 2 추가분)

```
SWM/
├── configs/
│   ├── stage2_spatial_multitask.yaml              # Phase 2 spatial transition 학습
│   ├── stage3_spatial_pca_multi_goal.yaml         # 실험 1 (최우선) ← 신규
│   ├── stage3_spatial_pca_binary.yaml             # 실험 2
│   ├── stage3_spatial_pca_binary_t2.yaml          # 실험 3
│   ├── stage3_spatial_latent_cos.yaml             # 실험 4
│   ├── stage3_spatial_pca_delta.yaml              # 실험 5
│   ├── stage3_spatial_diversity.yaml              # 실험 6
│   └── stage3_spatial_pca_multi_goal_debug.yaml   # 디버그 (완료 ✓)
├── models/
│   ├── encoder.py     # encode_spatial_projected() 추가
│   └── transition.py  # SpatialSWMTransition 추가
└── scripts/
    ├── train_stage2.py  # use_spatial 플래그, spatial_proj 저장
    └── train_stage3.py  # spatial 감지, z_init mean-pool fix, pca_adaptive goal
```

---

## 9. Stage 2 checkpoint 검증

```python
import torch
ckpt = torch.load("outputs/stage2/spatial_multitask/best.pt")
print(ckpt.keys())
# ['epoch', 'transition', 'val_loss', 'cfg', 'spatial_proj', 'spatial_dim']
# spatial_dim = 256 ✓
```

Stage 3에서 자동 감지:
```python
use_spatial_swm = 'spatial_proj' in s2_ckpt and 'spatial_dim' in s2_ckpt
# → SpatialSWMTransition 로드, encode_spatial_projected 사용
```
