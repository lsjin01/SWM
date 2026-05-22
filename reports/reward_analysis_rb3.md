# Reward Signal 분석 보고서 — Robust-B3 (spatial_dim=2176)

**작성일**: 2026-05-22  
**분석 대상**: rb3 chain Stage 3 학습 로그 (`logs/chain_rb3_20260521_222753.log`)  
**총 분석 iter**: 1,777개 (pair 1~9, 200 iter × 2 exp × 최대 9쌍)

---

## 1. 핵심 요약

| 지표 | 값 | 해석 |
|---|---|---|
| 평균 reward | **0.131** | 매우 낮음 (최대 이론값 1.0) |
| reward=0 비율 | **37.7%** | 전체 iter의 38%에서 학습 신호 없음 |
| 최대 관측 reward | **0.656** | reward ceiling이 낮음 |
| std ≥ 0.1 비율 (nonzero) | **91.2%** | reward 있을 때 분산은 충분 |

**결론: Variance가 아닌 Reward Sparsity가 핵심 문제**

---

## 2. Reward 분포

전체 reward 값의 분포 (1,777 iter):

```
reward=0.00: 670건 (37.7%) ██████████████████  ← dead zone
reward=0.05: 187건 (10.5%) █████
reward=0.10: 190건 (10.7%) █████
reward=0.15:  78건  (4.4%) ██
reward=0.20: 159건  (8.9%) ████
reward=0.25: 187건 (10.5%) █████
reward=0.30: 120건  (6.8%) ███
reward=0.35:  55건  (3.1%) █
reward=0.40:  72건  (4.1%) ██
reward=0.45:  28건  (1.6%)
reward=0.50:  19건  (1.1%)
reward≥0.55:  12건  (0.7%)
```

**Nonzero reward 통계** (n=1,169):

| 통계량 | 값 |
|---|---|
| 중앙값 | 0.1875 |
| 75th percentile | 0.2812 |
| 90th percentile | 0.3750 |
| 최댓값 | 0.6562 |
| reward > 0.3 | 22.4% |
| reward > 0.5 | 1.5% |

---

## 3. 학습 단계별 추이

| 단계 | Iter 범위 | 평균 reward | 평균 std | Dead rate |
|---|---|---|---|---|
| Early | 1–50 | 0.1061 | 0.1025 | **43.6%** |
| Mid | 51–150 | 0.1397 | 0.1414 | **30.0%** |
| Late | 151–200 | 0.1419 | 0.1383 | **33.3%** |

- Early → Mid: reward +31.8%, dead rate -13.6%p 로 초반 개선 있음
- Mid → Late: **정체** — reward/dead rate 모두 수렴, 추가 학습 효과 미미
- 200 iter 이후 추가 학습은 효과 없을 것으로 추정

---

## 4. 실험별 비교

| Pair | 실험 | avg reward | late avg | Dead rate | 평가 SR |
|---|---|---|---|---|---|
| 1&2 | spatial_pca_binary | 0.171 | 0.178 | **19%** | 20% |
| 7&8 | spatial_pca_multi_goal | 0.118 | 0.113 | 24% | 14% |
| 9&10 | spatial_pca_multi_goal_attn | **0.208** | **0.270** | **14%** | 진행 중 |
| 5&6 | spatial_pca_binary_dino_attn | 0.097 | 0.093 | 46% | 18% |
| 3&4 | spatial_pca_binary_attn | 0.108 | 0.148 | **57%** | 16% |

**관찰:**
- `spatial_pca_binary` (dead 19%)가 reward 수신율 가장 안정적 → 평가 SR도 최고
- attention 기반 실험 (`_attn`, `_dino_attn`)은 dead rate 46~57%로 훨씬 높음
- `multi_goal_attn` (pair 9)은 학습 중반부터 reward가 크게 오름 (0.270) → 최종 SR 기대됨

---

## 5. GRPO 관점에서의 문제 분석

GRPO advantage 계산:
```
A_i = (r_i - μ_group) / (σ_group + ε)
```

### 문제 1: Dead Batch (reward=0 in all 8 rollouts)

전체 37.7%의 iteration에서 8개 rollout 전부 reward=0:
- `μ_group = 0, σ_group = 0 → A_i = 0` → **gradient 없음**
- 이 batch는 완전히 낭비

### 문제 2: Reward Ceiling

- 관측된 최대 reward = 0.656 (이론 최대 1.0)
- 90th percentile = 0.375
- reward가 낮은 범위에 몰려 있어 **group 내 상대적 차이가 작음**
- GRPO는 절대값이 아닌 상대적 차이로 학습하므로, reward가 낮은 구간에 집중되면 학습 효율 저하

### 문제 3: Reward가 개선되지 않음 (Dead rate 수렴)

Mid(30%) → Late(33%)로 오히려 소폭 증가:
- 200 iter 이후 policy가 "local optimum"에 갇혀 있음
- reward 신호 자체가 discriminative하지 않아 더 나은 행동으로 향하는 방향을 못 잡음

### 문제 4: Variance가 아님 (중요)

nonzero reward인 경우:
- **91.2%가 std ≥ 0.1** → GRPO 학습에 충분한 분산
- std=0인 경우 (그룹 내 reward 동일) = 0.6% 로 거의 없음
- **Variance 자체는 문제가 아님**

---

## 6. 근본 원인 분석

### 왜 reward가 sparse한가?

```
image → backbone → z_t (2176) →[WM transition]→ ẑ_{t+1}
                                                    ↓
goal_latent ←──── cosine_similarity ────────────── ẑ_{t+1}
```

1. **WM transition 오차 누적**: Stage 2 val_loss=0.2327로 transition이 완벽하지 않음.  
   chunk가 늘어날수록 실제 상태와 예측 상태 괴리 증가 → reward 과소평가

2. **Cosine similarity의 한계**: 2176차원 고차원 공간에서 cosine similarity는  
   task-relevant 변화에 둔감할 수 있음. 물체가 조금 이동해도 전체 feature vector가  
   크게 바뀌지 않으면 reward≈0

3. **Goal latent 품질**: goal은 성공 demo의 마지막 프레임에서 추출.  
   중간 과정(task 50% 완료 시) reward가 거의 0으로 나오면 중간 행동 학습 불가

4. **Binary patch weight의 이산성**: `spatial_pca_binary` 계열은  
   중요 patch를 {0, 1}로만 구분 → reward가 계단식으로 변화, 미세한 개선을 반영 못 함

---

## 7. 개선 방향 제안

| 방향 | 방법 | 예상 효과 |
|---|---|---|
| **Shaped reward** | 목표까지의 거리 기반 step reward 추가 | Dead rate 감소, 중간 progress 반영 |
| **Multi-goal reward** | 여러 중간 목표 latent 설정, 단계별 누적 | reward ceiling 상승 |
| **Reward normalization** | running mean/std로 reward 정규화 | reward 분포 분산 개선 |
| **Curriculum** | 쉬운 task state → 어려운 state 순서로 샘플링 | Early dead rate 감소 |
| **WM transition 개선** | Stage 2 더 오래 학습 or val_loss 0.1 미만 | reward 예측 정확도 향상 |
| **더 많은 iter** | 200 → 400 iter | Mid plateau 이후 추가 탐색 여지 |

---

## 8. 전체 결론

rb3 (spatial_dim=2176, true circulatory) 학습의 **주 병목은 reward 희소성**이다.

- Variance 문제: **해당 없음** (reward 있을 때 분산 91%에서 충분)
- Dead batch: **37.7%** — 핵심 문제
- Reward ceiling: **0.656** (max), 대부분 0.2 이하에 분포
- 학습 수렴: **Mid 이후 정체** — 200 iter 상한이 적절하지 않을 수 있음

현재 구조에서 reward 설계를 개선하지 않으면, circulatory structure 개선만으로는  
SFT 베이스라인(20%)을 크게 넘기 어려울 것으로 판단된다.
