# EMA Reward 설계 · 구현 · 실험 계획

> SWM Stage 3 (GRPO) 에 **EMA reward LLM + task-상관 reward metric** 을 도입한 구조 전체 정리.
> 작성 기준: 2026-06-03 / square task / OpenVLA-OFT + WMPO world model.
> 코드: `scripts/train_stage3.py`

---

## 0. 한눈에 보는 구조

```
                 ┌─────────────────── 매 iteration ───────────────────┐
                 │                                                     │
  데모 last img ─┤→ reward LLM(EMA, 느림) → h_goal  ─┐                 │
  (goal, GT)     │                                   │ score_t         │
                 │  정책 rollout(상상):              ▼                 │
  z0,image0 ─────┤→ 정책 LLM(fast) ─ h_t ── metric(h_t, h_goal/scorer) │
                 │       │                  = cosine/probe/pls/tcn/    │
                 │       │ action a_t        goaldist                  │
                 │       ▼                                             │
                 │  WM transition: (z_t,a_t)→z_{t+1} (8회/chunk)       │
                 │       │  z_{t+1}→projector→다음 patch_embeds        │
                 │       └──────────── 순환(closed-loop) ──────────────┘
                 │
  reward = score_T − score_0   (rollout 동안 progress 변화량)
  GRPO: 그룹 상대 advantage → 정책 update → θ_ema ← τ·θ_ema+(1−τ)·θ
```

- **action LLM (정책)**: GRPO 로 빠르게 학습되는 OpenVLA-OFT.
- **reward LLM (EMA target)**: 정책 LM 의 느린 EMA 복사본(τ=0.99). reward 계산 기준 제공.
- **reward**: rollout 시작(chunk0, 실제 관측) 대비 끝(chunkT, 상상 관측)의 **progress 증가량**.

---

## 1. 배경

### 1.1 기존 Stage 3 (GRPO) rollout — 순환(circulatory) 구조
- 정책: OpenVLA-OFT (action LLM).
- **chunk 0 만 실제 이미지** → encoder → spatial patch features → projector → patch_embeds.
- 이후 chunk: **WM transition 이 latent z 를 상상**으로 전개 (env 재관측 없음).
  - `z_t →(wm_bridge)→ vla.projector → patch_embeds → VLA → action a_t`
  - `(z_t, a_t) →(transition)→ z_{t+1}` — chunk 당 8 sub-action = transition **8회 연쇄**
  - `z_{t+1} → projector → 다음 chunk 관측`
- 코드: `swm_rollout()`, gradient 재계산은 `recompute_logprobs_grad()`.

### 1.2 동기
기존 reward(transition latent vs goal latent 의 PCA 공간 progress)를 **두 LLM + EMA** 로 확장하고,
reward metric 을 **task 진전과 실제 상관된 지표**로 교체한다.

---

## 2. EMA Reward 핵심 메커니즘

### 2.1 두 LLM 의 역할 (target-network 패턴)
| LLM | 속도 | 학습 | 역할 |
|-----|------|------|------|
| **action LLM** (`vla_raw.language_model`) | 빠름 | GRPO 로 매 iter update | 정책 (action 생성) |
| **reward LLM** (`ema_lm`) | 느림 (τ=0.99) | EMA(action LLM) | reward 기준 (goal hidden / scorer fit) |

> 정석 배치(DQN target, BYOL/DINO teacher)와 동일: **정책은 빠르게, 기준은 느리게**.
> 느린 target 이 reward 를 안정화 → 정책이 "움직이는 과녁"을 쫓는 발산을 막는다.

### 2.2 업데이트 주기 (매 iteration)
1. (cosine·goaldist) **`goal_hidden` 재계산** — 현재 ema_lm 으로 goal 이미지 인코딩 (캐싱 안 함).
2. g_rollouts 개 closed-loop rollout → 각 rollout 의 `score_t` 누적 → `ema_cons = score_T − score_0`.
3. reward 조립(only/additive) → GRPO loss → backward → `optimizer.step()` (**action LLM update**).
4. `θ_ema ← τ·θ_ema + (1−τ)·θ` (**reward LLM update**).

→ 두 LLM 모두 매 iter 갱신, `goal_hidden` 은 갱신된 ema_lm 으로 매번 재생성.

### 2.3 reward = progress delta
- per-chunk `score_t` 는 "그 시점이 goal 에 얼마나 가까운가/진전됐는가" 스칼라.
- **reward = `score_T − score_0`** (rollout 동안의 진전 증가량).
  - `score_0`: chunk0(실제 초기 관측) — 그룹 내 모든 rollout 공통 → 분산은 `score_T`(상상)에서 발생.
  - 진전(goal 접근) > 0, 후퇴 < 0.
- **절대 스케일 무관**: GRPO 는 그룹 상대 advantage `(r−mean)/std` 만 사용.
  → metric 마다 reward 절대값이 달라도(0.3 ~ 19) 학습에는 그룹 내 **분산**만 중요.

---

## 3. reward metric 5종 (PCA 비판에 대한 답)

> PCA(최대분산축, unsupervised)는 "보기엔 유용하나 task 상관 보장 X" 라는 한계.
> 데모의 **공짜 supervision** = `frame_fraction`(0=시작 … 1=goal) 을 활용해 **진전 상관 지표**를 만든다.

| key | metric | score_t 정의 | h_goal 사용 | scorer 학습 | 의미 |
|-----|--------|-------------|:-----------:|------------|------|
| `cosine` | goal cosine | `cos(h_t, h_goal)` | ✓ | 없음 | hidden 이 goal hidden 과 정렬되는 정도 |
| `probe` | linear probe ① | `w·pool(h_t)` | ✗ | Ridge (1회) | hidden→frame_fraction 회귀 방향 w 투영 |
| `pls` | PLS ② | `w·pool(h_t)` | ✗ | PLS (1회) | 분산 대신 progress 와의 **공분산** 최대축 |
| `tcn` | time-contrastive ③ | `mlp(pool(h_t))` | ✗ | MLP, pairwise ranking (1회) | "나중 프레임=높은 score" 비선형 진전 |
| `goaldist` | contrastive goal-distance ④ | `−‖φ(h_t)−φ(h_goal)‖` | ✓ | MLP φ, ‖·‖≈남은진전 (1회) | quasimetric goal 도달성 |

### 3.1 각 metric 상세

**① probe (linear probe → progress)** — `fit_progress_probe(method='ridge')`
- 데모 hidden(pooled) → `frame_fraction` 을 Ridge 회귀, 계수 `w` 가 "진전 방향".
- score_t = `w·pool(h_t)`. PCA 비판에 가장 직접적인 답(supervised 1D 투영).
- 검증: R²≈0.94 (hidden 에 진전이 선형으로 충분히 들어있음).

**② pls (PLS, supervised PCA)** — `fit_progress_probe(method='pls')`
- PCA 가 "분산 최대축"이라면 PLS 는 "**target(progress)과 공분산 최대축**".
- score_t = `w·pool(h_t)` (PLS 1성분 방향). probe 의 변형, 다중공선성에 강함.

**③ tcn (time-contrastive)** — `fit_tcn_progress`, `ProgressMLP`
- 같은 데모 내 두 프레임 쌍에 **pairwise ranking loss**: 나중 프레임 score > 이른 프레임.
  `loss = softplus(−(s_b − s_a)·sign(later))`.
- 비선형 MLP(H→128→1) 라 선형 probe 가 못 잡는 진전 구조 포착.
- 검증: corr(pred, frac)=0.946. reward 절대값 큼(~19)이나 GRPO 는 분산만 사용 → 무해.

**④ goaldist (contrastive RL / goal-distance)** — `fit_goaldist`, `GoalDistMLP`
- 임베딩 `φ`(H→64) 학습: `‖φ(h) − φ(h_goal)‖ ≈ (1 − frac)` (goal 까지 **남은 진전**).
- score_t = `−‖φ(h_t) − φ(h_goal)‖`, reward = `d_0 − d_T` (goal 거리 감소량).
- goal-conditioned value / quasimetric 의 단순화. 보상이론적으로 가장 원리적.
- 검증: corr(score, frac)=0.999.

### 3.2 두 부류로 나뉨 (중요)
- **goal-conditioned (cosine, goaldist)**: rollout 시 `h_goal`(EMA target 이 매 iter 재계산)을
  기준으로 정책 h_t 를 비교 → **EMA target-network 패턴을 rollout 에서 실제 활용**.
- **progress detector (probe, pls, tcn)**: scorer 를 ema_lm 데모 hidden 으로 **1회 fit** 후 고정.
  rollout 은 정책 h_t 만 읽음 → EMA 는 "scorer fit 시점의 안정 기준" 역할.

---

## 4. 두 가지 mode

| mode | config | reward |
|------|--------|--------|
| **only** | `ema_reward.mode: only` | `ema_cons` 만 (WM grounding 미사용) |
| **additive** | `ema_reward.mode: additive` | `grounding_reward + β·ema_cons` (β=`ema_reward.beta`) |

- only × {cosine, probe, pls, tcn, goaldist} / additive × {…} 조합으로 변형 정의.
- config 예시:
```yaml
ema_reward:
  enable: true
  mode:   only        # only | additive
  metric: goaldist    # cosine | probe | pls | tcn | goaldist
  tau:    0.99        # EMA 계수 (클수록 느림)
  beta:   1.0         # additive 시 EMA 항 가중치
```

---

## 5. 코드 위치 (`scripts/train_stage3.py`)

| 요소 | 함수/위치 | 설명 |
|------|----------|------|
| 데모 hidden 수집 | `_collect_demo_hidden()` | (pooled hidden, frame_fraction, demo_id, goal_hidden) 반환 |
| probe/pls fit | `fit_progress_probe(method=)` | hidden→frac 회귀 방향 `w` (Ridge/PLS) |
| tcn fit | `fit_tcn_progress()` + `ProgressMLP` | pairwise ranking MLP |
| goaldist fit | `fit_goaldist()` + `GoalDistMLP` | ‖φ(h)−φ(goal)‖≈남은진전 임베딩 |
| goal hidden | `goal_pre_action_hidden()` | goal 이미지 → reward LLM → action head 직전 hidden |
| per-chunk score | `_vla_forward_patch_embeds(goal_hidden, probe_w, progress_mlp, goaldist_mlp)` | metric 분기로 score_t 계산 |
| rollout/aggregate | `swm_rollout()` | score_t 누적 → `ema_cons = score_T − score_0` 반환 |
| goal 이미지 로드 | `load_initial_states()` → `init_goal_images` | 데모 마지막 프레임 |
| EMA 생성/업데이트 | line ~1468(deepcopy/fit), ~1644(EMA step) | |
| reward 조립 | line ~1565(only) / ~1593(additive) | |

### 5.1 metric 분기 우선순위 (`_vla_forward_patch_embeds`)
```
goaldist_mlp & goal_hidden → goaldist
elif progress_mlp          → tcn
elif probe_w               → probe/pls
else (goal_hidden)         → cosine
```

---

## 6. 검증 결과

### 6.1 metric 건강성 (debug, 2 iter)
| metric | scorer fit 지표 | rollout reward / std | 판정 |
|--------|----------------|---------------------|------|
| cosine | (포화) | std≈0.001 | ✗ 신호 약함 |
| probe  | R²=0.94 | std≈0.05 (full) | ✓ |
| pls    | — | std≈0.05 (full) | ✓ |
| tcn    | corr=0.946 | reward 3.5 / std 0.28 (full) | ✓ 건강 |
| goaldist | corr=0.999 | (full 확인 중) | — |

### 6.2 풀학습 eval 결과 (square, 100 iter, q/v_proj only, `wm.enable=False`)
> verl 공식 파이프라인 / 128 고정 state / `unnorm_key=square_d0_300_demos` / greedy.

| 변형 | mode | metric | eval SR | 비고 |
|------|------|--------|--------:|------|
| **full_probe_only** | only | probe | **0.250** | 최고. SFT 동급, P_128 우위 |
| **full_probe_add** | additive | probe | **0.242** | grounding+probe |
| full_pls_only | only | pls | 0.188 | train reward는 높았으나 SR 낮음 |
| full_pls_add | additive | pls | 0.180 | |
| full_tcn_only | only | tcn | (체인 진행 중) | |
| full_goaldist_only | only | goaldist | (대기) | |
| full_tcn_add | additive | tcn | (대기) | |
| full_goaldist_add | additive | goaldist | (대기) | |

baseline(검증 완료): SFT 0.234~0.266 / P_128 0.18~0.20 / **P_1280 0.398**.
→ 현재까지 **probe 계열(only 0.250 / add 0.242)** 이 우세, pls 계열은 약함.
> 주의: train reward 절대값(예: pls 1.23 > probe 0.59)은 SR 우열과 무관 — 그룹 상대 advantage라 분산만 의미 있음. **비교는 eval SR 로만.**

### 6.3 cosine 이 실패한 이유 (교훈)
- VLA pre-action hidden cosine ≈ **0.9995 포화**(prompt/위치 구조가 지배) → within-group std≈0 → advantage 0 → 학습 신호 없음.
- `(cos_T−cos_0)/(1−cos_0)` 정규화는 분모≈0.0005 → **폭발**. → raw delta 만 사용.
- 결론: **단순 cosine 은 약함 → task 상관 metric(probe/pls/tcn/goaldist) 이 핵심.**

---

## 7. 설계상 반드시 알아둘 두 미묘점

> 두 점 모두 "**reward LLM(느린 EMA)와 scorer(w/φ)를 언제·어디에 쓰느냐**"에서 비롯된다.
> 구현이 틀린 것이 아니라, metric 별로 EMA 의 역할이 다르다는 **의도된 비대칭**이다.

### 7.1 미묘점 ① — scorer(w, φ)는 초기 ema_lm 으로 **1회만 fit 후 고정**

**무슨 일이 일어나는가**
- probe/pls/tcn/goaldist 의 scorer(`probe_w`, `progress_mlp`, `goaldist_mlp`)는
  학습 시작 시점의 ema_lm(= iter0 정책의 복사본) 으로 데모 hidden 에서 **단 한 번 fit** 한다.
- 이후 100 iter 동안 ema_lm 은 `θ_ema ← 0.99·θ_ema + 0.01·θ_policy` 로 계속 drift 하지만,
  scorer 자체는 **재fit 하지 않고 그대로 둔다**.

**왜 이렇게 했는가**
- scorer 재fit 은 매번 데모 수백 개를 인코딩(`_collect_demo_hidden`)해야 해 **비용이 크다**(수십 초~분).
- reward 함수가 매 iter 바뀌면 GRPO 가 쫓는 목표가 흔들려 **학습이 불안정**해진다.
  → 안정된 reward 기준을 위해 scorer 를 고정하는 편이 낫다(보상함수의 stationarity).
- τ=0.99 이면 ema 의 100-iter 누적 변화량이 작아, 초기 fit 한 진전 방향이 **여전히 유효**하다는 가정.

**리스크 / 모니터링**
- 정책이 크게 이동하면 hidden 분포가 초기와 달라져 scorer 가 진전과 어긋날 수 있음(분포 shift).
- 완화책(필요 시): N iter 마다 재fit, 또는 EMA 가 아닌 **고정 reference LM** 으로 fit.
- 현재는 고정 + `corr(pred,frac)` 로그(fit 시 0.94~0.99)로 초기 품질만 확인하는 단계.

### 7.2 미묘점 ② — metric 마다 **EMA target 의 사용 방식이 다르다**

EMA(느린 reward LLM)가 rollout 에서 실제로 쓰이는지에 따라 두 부류로 갈린다.

**(A) goal-conditioned 부류 — `cosine`, `goaldist` : EMA target 을 진짜 활용**
- 매 iter `h_goal = goal_pre_action_hidden(goal_img, ema_lm)` 으로 **현재 EMA 상태**의 goal hidden 을 만든다.
- rollout 의 정책 hidden `h_t` 를 이 `h_goal` 과 비교(cosine / φ-거리).
- 즉 **"느린 EMA = 안정된 goal 기준(stable reference)" / "빠른 정책 = 그 기준을 쫓는 쪽"** 이라는
  target-network 패턴(DQN target, BYOL teacher)을 **rollout 시점에 그대로 구현**.
- EMA 가 천천히 움직이므로 goal 기준이 매 step 요동치지 않고, 정책이 발산하지 않는다.

**(B) progress-detector 부류 — `probe`, `pls`, `tcn` : EMA 는 fit 에만, rollout 은 정책 hidden 의 고정 detector**
- scorer 는 ema_lm 데모 hidden 으로 한 번 fit 되지만(§7.1), **rollout 에서는 `h_goal` 을 쓰지 않는다.**
- score_t 는 오직 **정책(vla_raw)의 h_t** 를 고정 detector(w, mlp)에 통과시켜 얻는다
  (`s_t = w·pool(h_t)` 또는 `mlp(pool(h_t))`).
- 따라서 EMA 의 역할은 "**scorer 를 fit 할 때의 안정된 hidden 공급원**" 으로 한정된다.
  rollout 동안 EMA 가 능동적으로 reward 를 만드는 것은 아니다.
- 함의: 이 부류에서 두-LLM EMA 구조의 이득은 (A)보다 약하다. fit 시점 기준을 안정화하는 정도.
  iter0 엔 ema==정책 이라 detector 와 정책 hidden 이 정확히 일치, 이후 정책이 빠르게 이동하면서
  detector(초기 분포 기준)와 정책 hidden 분포가 점차 어긋날 수 있음(§7.1 리스크와 동일 뿌리).

**요약 표**

| metric | rollout 에서 h_goal(EMA) 사용 | EMA 의 실질 역할 | target-network 패턴 |
|--------|:--------------:|------------------|:-------------------:|
| cosine | ✓ (매 iter 재계산) | 안정된 goal 기준 | 강함 |
| goaldist | ✓ (매 iter 재계산) | 안정된 goal 기준 | 강함 |
| probe / pls / tcn | ✗ | scorer fit 시 안정 hidden 공급 | 약함 |

> 실험 해석 시 주의: probe/pls/tcn 의 성능은 "EMA 구조"보다 "task-상관 detector 의 질"에 더 좌우되고,
> cosine/goaldist 는 "EMA goal 기준 + metric"이 함께 작용한다. SR 비교 시 이 비대칭을 감안할 것.

### 7.3 기타
- **goaldist reward 분산이 작을 수 있음** — 상상 rollout 의 h_t 변화가 작으면 φ 거리 변화도 작음.
  full scale(g_rollouts=8, chunks=20)에서 분산 건강성 재확인 필요.
- reward 절대 스케일은 무관(그룹 상대 advantage). metric 비교는 **eval SR** 로만.

---

## 8. 실험 진행

### 8.1 자동 체인 (`run_chain.sh`)
변형마다: **train → export(HF) → verl eval → `chain_results.csv` 기록 → 정리(best.pt만 보존)**.
```bash
GPU=0 TASK=square VARIANTS="full_probe_only full_pls_only full_probe_add full_pls_add" bash run_chain.sh
```
- config: `configs/stage3_<variant>.yaml` (full_* = iter 100, g_rollouts 8, chunks 20, max_demos 300, n_states 8).
- iteration=100 → WMPO(100 epoch)와 비교 가능하게 설정.

### 8.2 평가 (공정 비교)
- WMPO 공식 verl 파이프라인(`generate_action_verl`, 128 고정 state, greedy) — `TARGET_MODEL_PATH` 만 교체.
- **커스텀 predict_action 직접 호출은 SR=0% → 금지** (반드시 verl).

### 8.3 baseline (검증 완료, square)
| 모델 | SR |
|------|-----|
| SFT | 0.234 ~ 0.266 |
| WMPO P_128 | 0.180 ~ 0.203 |
| WMPO P_1280 | **0.398** |

### 8.4 운영 주의
- 저장: `last.pt`(전체, resume용) + `best.pt`(**학습된 param만, slim**). q/v_proj only면 best.pt≈2G.
  - slim best.pt → export 는 `--base SFT_BASE` 에 `strict=False` 병합 → full export 와 **결과 동일**.
  - 전제: `vla_path == export --base` (freeze 가중치 = base 가중치). 현재 일치 ✓.
- 디스크 공유(`/` 3.5T, 단일 파티션, mipstu 2.3T 포함) → 동시 다발 학습 자제. debug 는 GPU 1.

### 8.5 운영 사고 기록 (재발 방지)
1. **디스크 full (2026-06-03)**: debug 체크포인트가 19G씩 저장(×2=74G) + 누적으로 `/` 100% →
   probe_add eval·pls_add 학습·③④ 체인이 `ENOSPC` 연쇄 사망.
   - 조치: debug/실패/옛pilot ~152G 삭제 + best.pt slim 저장으로 전환.
   - 교훈: **debug run 은 저장 끄거나 별도 dir**. 본 학습 전 `df -h /` 확인.
2. **output_dir 충돌 (2026-06-04)**: ③④ config 를 `sed` 로 만들 때 `name`/`metric` 만 바꾸고
   `experiment.output_dir` 을 안 바꿔 → tcn_only/goaldist_only 가 **probe_only dir 에 덮어씀**.
   chain 은 `outputs/stage3/<V>/<task>` 를 기대하는데 train 은 `cfg.experiment.output_dir` 에 씀 → 불일치.
   - 조치: 4개 config 의 output_dir 을 각자 이름으로 교정 후 재실행. probe_only ckpt 는 소실(SR 0.250 결과는 보존).
   - 교훈: **config 복제 시 `name`·`output_dir` 동시 교정**. chain 에 (선택) name↔output_dir 일치 assert 권장.

---

## 9. 실행 (참고)

```bash
cd /home/miplab1/sjLee/swm
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/miplab1/sjLee/swm:/home/miplab1/sjLee/WMPO:/home/mipstu/jiPark/openvla-oft/experiments/robot \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 \
MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210 \
LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH \
CPATH=/opt/conda/envs/wmpo/x86_64-conda-linux-gnu/sysroot/usr/include:/opt/conda/envs/wmpo/include:$CPATH \
conda run -n wmpo python scripts/train_stage3.py --config configs/stage3_<variant>.yaml --no-wandb
```
- 환경 셋업: `setup_wmpo_full.sh` (torch 2.8.0+cu128, transformers moojink fork, mujoco-py 2.1.2.14 등).
- stage1/2 ckpt: `outputs/stage1/verify/best.pt`, `ckpts/transition_stage2_best.pt`.
- data_root: `data/train_root` (256px WMPO 데이터).
