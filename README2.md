# SWM Phase 2 — Spatial World Model for GRPO Reward Shaping

> **Status**: Chain 9–12 running (dino_attn + pca_variance) | Chain 1–8 COMPLETED
> → Phase 1 results and baseline: [README.md](README.md)

---

## Abstract

Phase 1의 핵심 실패 원인은 **mean-pool scalar latent의 구조적 정보 손실**이다. DINOv2+SigLIP 인코더가 생성한 256개 patch token을 평균내면 배경(전체 픽셀의 95%+)이 task object(5%)를 압도하여, 서로 다른 task phase가 cosine similarity ≥ 0.97로 구별 불가능해진다. GRPO reward가 항상 ≈ 1.0을 반환하면 group 내 advantage가 0에 수렴하여 학습이 정체된다.

Phase 2의 핵심 주장:

1. **Spatial representation 유지** — world model을 scalar space가 아닌 patch space (256×256)에서 학습하면 object patch의 transition이 배경 patch를 지배
2. **Semantic goal 생성** — 고정 temporal goal 대신 PCA progress threshold 기반 adaptive goal
3. **Patch-selective reward** — task-relevant top-K patch를 식별하여 weighted mean-pool

**Phase 2 실험 결과 요약 (2026-05-19 기준):**
Phase 2 spatial SWM은 Phase 1 scalar SWM (26% SR)을 넘지 못했다 (최고 22%). 이는 spatial representation이 reward signal 품질을 개선하지 못함을 시사하며, 그 원인을 reward variance 분석 및 patch weight concentration 분석을 통해 규명하였다.

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
                                   w = patch_weights (method-dependent)
                                                       │
                                   reward = cos_sim(z_rollout, z_goal)
```

---

## 2. 아키텍처

### 2.1 SWMEncoder (`models/encoder.py`)

```python
self.spatial_proj = nn.Sequential(
    nn.Linear(latent_dim, spatial_dim),   # 2176 → 256
    nn.LayerNorm(spatial_dim),
)

def encode_spatial_projected(self, image):
    patch_features = self._get_patch_tokens(image)   # (B, 256, 2176)
    return self.spatial_proj(patch_features)          # (B, 256, 256)
```

Stage 1 (DINOv2+SigLIP)은 완전 동결. `spatial_proj`만 Stage 2에서 학습됨.

### 2.2 SpatialSWMTransition (`models/transition.py`)

| 항목 | Phase 1 SWMTransition | Phase 2 SpatialSWMTransition |
|---|---|---|
| 입력 | (B, 2176) scalar | (B, 256, 256) spatial tokens |
| 출력 | (B, 2176) scalar | (B, 256, 256) spatial tokens |
| Action 처리 | linear concat | prepended token |
| Attention | 없음 | 257×257 full self-attention |
| Parameters | 10.4M | 19.2M |
| Loss | L2 | per-patch L1 |

```python
def forward(self, s_t, action):
    x = torch.cat([self.action_embed(action).unsqueeze(1),
                   self.token_embed(s_t)], dim=1)    # (B, 257, hidden)
    x = self.blocks(x)                               # 6-layer Transformer
    return self.out(self.norm(x[:, 1:, :]))          # (B, 256, d_s)
```

### 2.3 Patch Weight 방법 (총 4종)

| 방법 | 원리 | Top-1 weight | Entropy (max=4.159) |
|---|---|---|---|
| `none` | uniform mean-pool | 0.0039 | 5.545 (=log256) |
| `attn` | 마지막 layer action→patch attention | **0.097** | 3.747 |
| `delta` | \|\|Ŝ_{t+1} - S_t\|\|₂ per patch | 0.016 | **4.159** |
| `dino_attn` | DINOv2 CLS→patch attention (frozen) | TBD | TBD |
| `pca_variance` | Patch feature variance across demos | TBD | TBD |

**핵심 발견 (patch weight analysis, n=200 samples):**
- `attn`은 top-1 patch에 weight의 9.7%가 집중 (uniform 대비 6.2배). 이는 특정 patch에 reward가 과도하게 편중되어 reward saturation을 유발함
- `delta`는 거의 완벽하게 uniform (entropy = max). Task-relevant signal이 희석됨
- `attn` vs `delta` top-64 IoU = 0.229 (random baseline = 0.250): 두 방법이 서로 다른 patch를 선택하며, 어느 쪽도 실제 task-relevant region과 align되지 않을 가능성이 높음

```python
# attn: action token → patch attention (transition model 의존)
patch_w = last_layer_attn[:, 0, 1:].mean(0)  # (256,)

# delta: prediction change magnitude (transition model 의존)
delta = (s_next - s_t).norm(dim=-1).mean(0)   # (256,)

# dino_attn: DINOv2 CLS attention via forward hook (transition 불필요)
handle = dino.blocks[-1].attn.register_forward_hook(_hook)
dino.forward_features(dino_img)  # hook captures CLS→patch[-256:] attn

# pca_variance: per-patch feature norm variance across demos
var_i = E[||s_t^i||²] - E[||s_t^i||]²  # high variance = task-discriminative
```

---

## 3. Stage 2 학습

### 3.1 설정

```yaml
# configs/stage2_spatial_multitask.yaml
data:
  tasks: [square, coffee, stack_three, three_piece_assembly]
  train_pairs: 315,432
  val_pairs: 35,300
transition:
  use_spatial: true
  spatial_dim: 256
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
training:
  batch_size: 512   # 6-GPU DDP
  lr: 3.0e-4
  epochs: 50
  loss: per-patch L1
```

### 3.2 학습 곡선 및 안정성 이슈

| 시도 | lr | 결과 | 원인 |
|---|---|---|---|
| 최초 | 3.2e-3 (linear scaling) | epoch 6 발산 | lr 과다 |
| resume | 1e-3 | optimizer state 미로드 → 발산 | optimizer cold-start |
| **현재** | **3e-4** (optimizer state 포함) | **epoch 5: val=0.0630 ★** | 안정 |

optimizer/scheduler/scaler state를 체크포인트에 포함하고 resume 시 lr을 config에서 override하는 방식으로 안정화:
```python
ckpt = {"transition": ..., "spatial_proj": ...,
        "optimizer": opt.state_dict(), "scheduler": sch.state_dict(), "scaler": sc.state_dict()}
# resume:
opt.load_state_dict(ckpt["optimizer"])
for pg in opt.param_groups: pg["lr"] = cfg.training.lr  # override
```

---

## 4. PCA-Adaptive Multi-Goal Reward

### 4.1 PCA-Adaptive Goal 알고리즘

```python
pca_vecs = pca.transform(encoded_frames)          # (T, n_pca=16)
direction = pca_vecs[-1] - pca_vecs[0]
proj_t    = dot(pca_vecs[t] - pca_vecs[0], direction) / ||direction||²

goals = []
last_proj = 0.0
for t, proj in enumerate(projections):
    if proj - last_proj >= pca_goal_threshold:     # = 0.3
        goals.append(t)
        last_proj = proj
goals.append(T-1)   # 마지막 프레임 항상 포함
# → 평균 3–4 goals per demo (vs fixed K=4 temporal)
```

### 4.2 Binary Reward 계산

```python
progress_k = (cos_sim(z_pred, z_goal_k) - cos_init_k) / (1 - cos_init_k + ε)
binary_k   = float(progress_k > phase_threshold)   # 0 or 1

reward = mean(binary_k for k in goals)
# reward ∈ {0, 1/K, 2/K, ..., 1}
```

GRPO advantage: `A_i = (r_i - μ_group) / (σ_group + ε)`

---

## 5. 실험 설계

### 5.1 실험 행렬

**Round 1: 2×3 Factorial Ablation (phase_threshold=0.3, chain1–6)**

| | No Weight | Attn Weight | Δs Weight |
|---|---|---|---|
| **PCA Multi-Goal** | Exp 1 (chain1) | Exp 3 (chain3) | Exp 5 (chain5) |
| **PCA Binary** | Exp 2 (chain2) | Exp 4 (chain4) | Exp 6 (chain6) |

**Round 2: threshold ablation (chain7–8)**

| | No Weight | phase_threshold |
|---|---|---|
| **PCA Multi-Goal** | chain7 | 0.6 |
| **PCA Binary** | chain8 | 0.6 |

**Round 3: Improved patch selection (chain9–12)**

| | DINO Attn | PCA Variance |
|---|---|---|
| **PCA Multi-Goal** | chain9 | chain11 |
| **PCA Binary** | chain10 | chain12 |

공통 GRPO 설정:
```yaml
iterations: 200,  g_rollouts: 8,  lr: 5.0e-6,  clip_eps: 0.2,  kl_coef: 0.05
```

### 5.2 Chain 실행 구조

```
Chain A (GPU 2,3,4)                  Chain B (GPU 5,6,7)
─────────────────────────────────    ──────────────────────────────────
chain1: pca_multi_goal (t=0.3)   ||  chain2: pca_binary (t=0.3)
    ↓ eval done                       ↓ eval done
chain3: multi_goal_attn (t=0.3)  ||  chain4: binary_attn (t=0.3)
    ↓ eval done                       ↓ eval done
chain5: multi_goal_delta (t=0.6) ||  chain6: binary_delta (t=0.6)
    ↓ eval done                       ↓ eval done
chain7: multi_goal (t=0.6)       ||  chain8: binary (t=0.6)
    ↓ eval done                       ↓ eval done
chain9:  multi_goal_dino (t=0.6) ||  chain10: binary_dino (t=0.6)
    ↓ eval done                       ↓ eval done
chain11: multi_goal_var (t=0.6)  ||  chain12: binary_var (t=0.6)
```

---

## 6. 실험 결과 (Chain 1–8, COMPLETED)

### 6.1 Success Rate 결과

| Chain | 실험명 | Patch Weight | Phase Thresh | Best SR | Last SR |
|---|---|---|---|---|---|
| Phase 1 (scalar) | pca_binary (baseline) | — | 0.3 | — | **26%** |
| SFT (no GRPO) | OpenVLA-OFT | — | — | — | 16% |
| chain2 | pca_binary | none | 0.3 | 20% | **22%** |
| chain1 | pca_multi_goal | none | 0.3 | 14% | 18% |
| chain4 | pca_binary_attn | attn | 0.3 | 16% | 18% |
| chain3 | pca_multi_goal_attn | attn | 0.3 | 10% | 14% |
| chain6 | pca_binary_delta | delta | **0.6** | 10% | 14% |
| chain5 | pca_multi_goal_delta | delta | **0.6** | 18% | **22%** |
| chain8 | pca_binary_t06 | none | **0.6** | 18% | 18% |
| chain7 | pca_multi_goal_t06 | none | **0.6** | 12% | 18% |

> **모든 Phase 2 실험이 Phase 1 scalar baseline (26%)를 넘지 못함.**
> 최고 성능: 22% (chain2, chain5 last checkpoint)

### 6.2 Reward Variance 분석 (핵심 발견)

| 실험 | Reward mean | Reward std | Saturation (≥0.99) | 학습 상태 |
|---|---|---|---|---|
| pca_binary (chain2) | 0.158 | 0.098 | 0% | **정상** |
| pca_multi_goal (chain1) | 0.217 | 0.122 | 0% | **정상** |
| pca_binary_attn (chain4) | 0.998 | 0.012 | **96%** | ❌ saturation |
| pca_multi_goal_attn (chain3) | 1.000 | 0.000 | **100%** | ❌ saturation |
| pca_binary_delta (chain6) | 0.007 | 0.022 | 0% | ❌ collapse |
| pca_multi_goal_delta (chain5) | 0.003 | 0.013 | 0% | ❌ collapse |
| pca_binary_t06 (chain8) | 0.005 | 0.016 | 0% | ❌ collapse |
| pca_multi_goal_t06 (chain7) | 0.001 | 0.008 | 0% | ❌ collapse |

**분석:**
- `attn` (chain3,4): reward=1.0 saturation → group σ≈0 → advantage≈0 → gradient≈0. attn weight가 특정 patch에 과도하게 집중(top-1 = 9.7%, uniform의 6.2배)하여 해당 patch의 cosine similarity 변화가 모든 rollout에서 threshold를 쉽게 초과
- `delta` (chain5,6) + `t06` (chain7,8): reward≈0 → 반대 방향 붕괴. delta는 균등 분포(entropy=max)로 signal이 희석; t=0.6은 threshold가 너무 높아 모든 rollout이 credit을 받지 못함
- **patch weight 없는 t=0.3 (chain1,2)만 reward variance가 충분하여 GRPO가 실제로 작동함**

### 6.3 Phase 1 vs Phase 2 비교 분석

| 요인 | Phase 1 Scalar | Phase 2 Spatial |
|---|---|---|
| Transition space | ℝ²¹⁷⁶ scalar | ℝ^{256×256} spatial |
| Stage 2 val loss | 0.0789 (L2) | 0.0630 (L1, per-patch) |
| Reward mean | ~0.15–0.25 | ~0.16–0.22 (patch-weight 없을 때) |
| Reward variance (σ) | ~0.07–0.08 | ~0.10–0.12 |
| Best SR | **26%** | 22% |

Phase 2 spatial이 오히려 낮은 원인 가설:
1. **Spatial transition의 reward noise**: 256개 patch 각각의 cosine similarity를 aggregating하는 과정에서 noise가 증가
2. **Stage 2 transition의 품질**: val=0.063이지만, spatial transition이 배경 patch를 얼마나 잘 억제하는지는 미검증
3. **Patch weight misalignment**: attn/delta가 task-relevant patch를 제대로 선택하지 못함 (IoU=0.229, 실질적으로 random 수준)

---

## 7. Patch Weight 방법 비교 (정량 분석)

### 7.1 Patch Selection Agreement

```
분석 설정: n=200 samples, top_k=64
Random baseline IoU: 64/256 = 0.250

Results:
  attn vs delta top-64 IoU:      0.229 ± 0.075  ← random보다 낮음
  attn vs delta Spearman ρ:      0.321 ± 0.144  ← 약한 양의 상관
```

attn과 delta가 서로 다른 패치를 선택하며, IoU가 random baseline보다 낮다는 것은 두 방법이 독립적으로 다른 기준을 사용함을 의미한다.

### 7.2 Weight Concentration (n=20 random inputs)

```
                     Top-1 weight    Entropy       비고
─────────────────────────────────────────────────────
attn   (top_k=64)    0.097          3.747         고집중
delta  (top_k=64)    0.016          4.159 (max)   균등
uniform (top_k=256)  0.004          5.545 (max)   기준
─────────────────────────────────────────────────────
Uniform top-k=64:    1/64 = 0.0156  log(64)=4.159
```

**핵심 문제:**
- `attn`은 특정 patch에 reward를 편중시켜 → saturation
- `delta`는 너무 균등하게 퍼뜨려 → signal 희석 → reward collapse
- 두 방법 모두 transition model 학습의 부산물이며, task completion과 직접적으로 align되지 않음

### 7.3 Round 3 개선 방향 (chain9–12)

#### DINO CLS Attention (`dino_attn`)
DINOv2의 자기지도 학습으로 형성된 CLS→patch attention은 task object를 자연스럽게 highlight하는 것으로 알려져 있다. Transition model 학습 품질에 무관하며, 독립적인 prior로 활용 가능하다.

```python
# DINOv2 마지막 block CLS→patch attention via forward hook
def _hook(module, inp, out):
    x = inp[0]  # (B, N, C)
    qkv = module.qkv(x).reshape(B, N, 3, heads, head_dim).permute(2,0,3,1,4)
    q, k, _ = qkv.unbind(0)
    attn = (q @ k.T) * module.scale
    attn = attn.softmax(-1)
    captured['attn'] = attn[:, :, 0, -256:].mean(1)  # CLS→patch[-256:]
```

- 장점: 학습 불필요, object-centric attention이 선행연구에서 검증됨
- 예상: `attn`보다 덜 집중적이고, `delta`보다 semantically meaningful한 분포

#### PCA Variance (`pca_variance`)
Demo 전체 프레임에 걸친 patch feature variance를 기반으로 patch 중요도를 결정한다. 변동이 큰 patch = task state가 변하면서 달라지는 영역 = task-relevant.

```python
# Welford online algorithm for per-patch variance
var_i = E_t[||s_t^i||²] - (E_t[||s_t^i||])²
# top-k: 가장 변동이 큰 K개 patch 선택
```

- 장점: 실제 demo 데이터 기반, transition model에 무관
- 예상: robot arm이 움직이는 패치 + 물체가 이동하는 패치를 선택

---

## 8. 가설 검증 결과

| 가설 | 예측 | 실제 결과 | 판정 |
|---|---|---|---|
| H1: Spatial > Scalar | spatial_pca_binary > 26% | 22% < 26% | ❌ 기각 |
| H2: Multi-goal > Binary | multi_goal > binary | 14% < 20% (best) | ❌ 기각 |
| H3: Patch weight improves reward | attn/delta > none | attn: 10–16%, none: 14–22% | ❌ 기각 |
| H4: Attn ≈ Delta | IoU ≈ 0.8+ | IoU = 0.229 (random수준) | ❌ 기각 |

모든 가설이 기각됐으나, 이는 다음을 시사한다:
- Spatial feature가 reward 공간으로 부적합한 것이 아니라, **patch weight 설계의 실패**가 주 원인
- attn/delta는 서로 다른 patch를 선택하며 둘 다 reward signal 품질을 저하시킴
- `none` (uniform mean-pool)이 가장 안정적인 reward variance를 제공하는 아이러니
- **Round 3 (dino_attn + pca_variance)이 critical**: task-relevant patch selection이 가능하면 reward quality 개선 가능

---

## 9. Paper 구성 제안

### 9.1 주장 구조 (실험 결과 반영)

```
Problem: VLA post-training with GRPO needs informative dense rewards
         → scalar WM suffers from background dominance

Method:
  1. Spatial World Model (SpatialSWMTransition, 19.2M params)
  2. PCA-Adaptive Goal Generation (semantic phases vs temporal)
  3. Patch-Selective Reward (4 methods: none / attn / delta / dino_attn / pca_var)

Key Findings:
  - Spatial WM alone does not surpass scalar WM (22% vs 26%)
  - Patch weight quality is the critical bottleneck:
    * attn: reward saturation (concentration issue, top-1 = 6.2x uniform)
    * delta: reward collapse (over-uniform, signal dilution)
    * dino_attn / pca_variance: under investigation
  - Reward variance, not architecture, determines GRPO effectiveness
```

### 9.2 핵심 Figure 계획

| Figure | 내용 |
|---|---|
| Fig 1 | Phase 1 vs Phase 2 pipeline 다이어그램 |
| Fig 2 | PCA adaptive goal 시각화 (demo trajectory + goal frames) |
| Fig 3 | Patch weight 히트맵 비교: attn / delta / dino_attn / pca_variance |
| Fig 4 | Reward variance during training (8 실험, distribution) |
| Fig 5 | 전체 SR 결과 막대그래프 (chain1–12 vs baseline) |
| Tab 1 | Patch weight analysis: top-1 concentration, entropy, IoU with dino_attn |

### 9.3 Ablation 스토리라인

```
[SFT baseline]:             16% SR
[Phase 1 scalar SWM]:       26% SR  (+10%p GRPO effect)
    ↓ spatial WM, uniform pool (Exp 2 vs Phase 1)
[Spatial + Fixed Goal]:     22% SR  (-4%p: spatial does not help with uniform pool)
    ↓ adaptive vs fixed goal (Exp 1 vs Exp 2)
[Spatial + Adaptive Goal]:  18% SR  (-4%p: adaptive goal also does not help)
    ↓ patch weighting — why it fails (reward variance analysis)
[attn/delta weight]:        10–16%  (reward saturation or collapse)
    ↓ improved patch selection
[dino_attn / pca_variance]: TBD
```

Expected narrative: patch selection quality is the key variable; DINO-based selection should improve over learned transition-based selection.

---

## 10. 파일 구조

```
SWM/
├── configs/
│   ├── stage2_spatial_multitask.yaml
│   ├── stage3_spatial_pca_multi_goal.yaml              # chain1
│   ├── stage3_spatial_pca_binary.yaml                  # chain2
│   ├── stage3_spatial_pca_multi_goal_attn.yaml         # chain3
│   ├── stage3_spatial_pca_binary_attn.yaml             # chain4
│   ├── stage3_spatial_pca_multi_goal_delta.yaml        # chain5 (t=0.6)
│   ├── stage3_spatial_pca_binary_delta.yaml            # chain6 (t=0.6)
│   ├── stage3_spatial_pca_multi_goal_t06.yaml          # chain7
│   ├── stage3_spatial_pca_binary_t06.yaml              # chain8
│   ├── stage3_spatial_pca_multi_goal_dino_attn.yaml    # chain9
│   ├── stage3_spatial_pca_binary_dino_attn.yaml        # chain10
│   ├── stage3_spatial_pca_multi_goal_pca_variance.yaml # chain11
│   └── stage3_spatial_pca_binary_pca_variance.yaml     # chain12
├── models/
│   ├── encoder.py       # encode_spatial_projected(), spatial_proj
│   └── transition.py    # SpatialSWMTransition, get_patch_weights(),
│                        # get_delta_weights()
├── scripts/
│   ├── train_stage2.py  # spatial DDP, optimizer state save/load
│   ├── train_stage3.py  # spatial reward, pca_adaptive, patch_weights
│   │                    # Methods: none / attn / delta / dino_attn / pca_variance
│   └── phase2/
│       ├── chain1_spatial_pca_multi_goal.sh
│       ├── chain2_spatial_pca_binary.sh
│       ├── chain3_spatial_pca_multi_goal_attn.sh
│       ├── chain4_spatial_pca_binary_attn.sh
│       ├── chain5_spatial_pca_multi_goal_delta.sh
│       ├── chain6_spatial_pca_binary_delta.sh
│       ├── chain7_spatial_pca_multi_goal_t06.sh
│       ├── chain8_spatial_pca_binary_t06.sh
│       ├── chain9_spatial_pca_multi_goal_dino_attn.sh
│       ├── chain10_spatial_pca_binary_dino_attn.sh
│       ├── chain11_spatial_pca_multi_goal_pca_variance.sh
│       └── chain12_spatial_pca_binary_pca_variance.sh
└── eval/
    ├── eval_swm_mimicgen.py          # TF GPU block fix (CUDA_VISIBLE_DEVICES="")
    └── check_patch_weight_agreement.py  # attn/delta IoU analysis
```

---

## 12. 전체 실험 결과 정리 (2026-05-20 기준)

### Baselines

| SR | 실험 |
|---|---|
| **40%** | WMPO P_1280 (fair) |
| 30% | WMPO GRPO pixel |
| 26% | JEPA GRPO |
| 20% | SFT |

### Phase 1 (Scalar SWM, original) — best per 실험

| SR | 실험 |
|---|---|
| **24%** | diversity (best) |
| **24%** | latent_cos (nozbp) |
| 20% | action_pca_delta (best) |
| 20% | pca_goal (reeval best) |
| 16% | action_goal_v2 |
| 16% | pca_goal_single |
| 12% | pca_max |

### Phase 2 (Spatial SWM, original) — best per 실험

| SR | 실험 |
|---|---|
| **22%** | spatial_pca_binary_last |
| **22%** | spatial_pca_multi_goal_delta_last |
| 18% | spatial_pca_binary_attn_last |
| 18% | spatial_pca_binary_dino_attn |
| 18% | spatial_pca_binary_t06 |
| 18% | spatial_pca_multi_goal_last |
| 18% | spatial_pca_multi_goal_t06_last |
| 16% | spatial_pca_multi_goal_dino_attn_last |
| 14% | spatial_pca_multi_goal_attn_last |
| 10% | spatial_pca_binary_delta_best |

### Robust-B / Phase 2 — 완료된 6개 (binary 계열, 2026-05-20)

> robust_b: Stage 2 transition을 multi-step unrolled loss (k=8)로 재학습. free-run cosine@t=160: -0.684 → **+0.968**

| SR | 실험 |
|---|---|
| 18% | spatial_pca_binary_dino_attn_last |
| 18% | spatial_pca_binary_t06_last |
| 16% | spatial_pca_binary_dino_attn_best |
| 16% | spatial_pca_binary_pca_variance_best |
| 14% | spatial_pca_binary_best / delta_best |
| 12% | spatial_pca_binary_attn_best / delta_last |
| 10% | spatial_pca_binary_attn_last / pca_variance_last |

**Robust-B Phase 1 unique (7개)**: chain 진행 중, 아직 결과 없음

> 현재 `spatial_pca_multi_goal` 학습 중 (7/19번째). 이후 multi_goal 5개 → spatial_continuous_reward → Phase 1 unique 7개.

---

## 11. 현재 실행 상태 (2026-05-19 09:55)

```
COMPLETED (chain1–8):
  chain1: pca_multi_goal (t=0.3)          → best=14%, last=18%
  chain2: pca_binary (t=0.3)              → best=20%, last=22%  ← Phase2 best
  chain3: pca_multi_goal_attn (t=0.3)     → best=10%, last=14%  (reward saturated)
  chain4: pca_binary_attn (t=0.3)         → best=16%, last=18%  (reward saturated)
  chain5: pca_multi_goal_delta (t=0.6)    → best=18%, last=22%
  chain6: pca_binary_delta (t=0.6)        → best=10%, last=14%  (reward collapsed)
  chain7: pca_multi_goal_t06 (t=0.6)      → best=12%, last=18%  (reward collapsed)
  chain8: pca_binary_t06 (t=0.6)          → best=18%, last=18%  (reward collapsed)

RUNNING (chain9–12):
  chain9:  pca_multi_goal_dino_attn       GPU 2,3,4  training
  chain10: pca_binary_dino_attn           GPU 5,6,7  training
  chain11: pca_multi_goal_pca_variance    GPU 2,3,4  waiting for chain9 eval
  chain12: pca_binary_pca_variance        GPU 5,6,7  waiting for chain10 eval

Stage 2 checkpoint: outputs/stage2/spatial_multitask/best.pt
  epoch=5, val_loss=0.0630 (per-patch L1)
```
