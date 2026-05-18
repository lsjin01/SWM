# SWM — Structured World Model for Robot Manipulation

> **Pixel-free Latent World Modeling via DINOv2+SigLIP for GRPO-based VLA Fine-tuning**

---

## Abstract

We present **SWM (Structured World Model)**, a framework that learns a structured latent world model from expert demonstrations and uses it as a dense reward signal to fine-tune a Vision-Language-Action (VLA) model via Group Relative Policy Optimization (GRPO). Unlike prior work (WMPO) that relies on pixel-level video generation, SWM operates entirely in the latent space of a frozen DINOv2+SigLIP encoder, making rollout simulation lightweight and scalable. We identify two key failure modes in naïve latent reward design—*reward variance collapse* and *mean-pool goal indistinguishability*—and propose a series of remedies. Our best reward design (PCA-projected binary phase reward) achieves **26% success rate** on the MimicGen square-assembly task, outperforming the SFT baseline (16%) and matching WMPO-P128 (24%) with substantially lower compute.

---

## 1. Introduction

Recent VLA models such as OpenVLA-OFT achieve strong performance when supervised on large expert datasets, but their sample efficiency remains limited. World model-based RL offers a promising path: by learning a predictive model of environment dynamics, a policy can be improved using imagined rollouts rather than real environment interactions. However, world models for manipulation have historically required expensive pixel-level reconstruction (e.g., video diffusion), which is both compute-intensive and prone to compounding errors.

SWM addresses this by constructing a *latent* world model: rather than predicting future pixels, we predict future latent states in the compressed representation space of a powerful pre-trained vision encoder. The central insight is that a high-quality vision encoder (DINOv2 + SigLIP) already captures the task-relevant structure of manipulation scenes; a world model operating in this space can simulate the effect of actions without generating images.

We fine-tune OpenVLA-OFT using GRPO, treating the latent world model as an online reward function. This setup requires careful reward design: the reward must provide sufficient within-group variance to drive GRPO's advantage estimation. We conduct an extensive empirical study of reward design choices and identify the binary phase reward over PCA-projected latents as the most effective approach.

---

## 2. Method

### 2.1 Architecture Overview

SWM consists of three training stages plus an optional reward model stage:

```
Stage 1   │ Encoder     : image → z_t           (DINOv2-1024 ⊕ SigLIP-1152 = 2176-dim)
Stage 2   │ Transition  : (z_t, a_t) → ẑ_{t+1}  (JEPA-style latent prediction)
Stage 2.5 │ Reward Model: z_{0:T} → P(success)  (optional, temporal transformer)
Stage 3   │ Policy      : GRPO fine-tuning with latent reward signal
```

### 2.2 Stage 1: Latent Encoder

The encoder combines two complementary vision backbones:

- **DINOv2-ViT-L** (1024-dim): self-supervised patch features with strong spatial grounding
- **SigLIP-ViT-L** (1152-dim): language-aligned visual features with semantic understanding

Both backbones are frozen. Their spatial token sequences are concatenated along the feature dimension and projected via a lightweight MLP to produce a 2176-dim latent vector per image:

```
z_t = Proj( concat(f_DINO(I_t), f_SigLIP(I_t)) )    ∈ ℝ^2176
```

Each backbone produces 256 spatial tokens (16×16 grid). The default pooling strategy is mean-pooling over all 256 tokens before projection. Crucially, the projector is **frozen during Stage 3** (`freeze_projector=True`) to prevent visual feature drift that destabilizes GRPO training.

### 2.3 Stage 2: Latent Transition Model

The transition model predicts the next latent state given the current state and action, following the JEPA (Joint Embedding Predictive Architecture) paradigm:

```
ẑ_{t+1} = SWMTransition(z_t, a_t)
```

Architecture: 6-layer Transformer (hidden=512, heads=8), total 21.2M parameters. The action embedding is concatenated to the latent before the first transformer layer. The model is trained with MSE loss in latent space, using online/target encoder EMA for stability. This eliminates the need for a dedicated rollout GPU at inference time—world model simulation is a sequence of cheap forward passes through a 21M-parameter network.

### 2.4 Stage 2.5: Temporal Reward Model (Optional)

A temporal transformer trained to classify complete latent trajectories as success or failure:

```
P(success | z_{0:T}) = SWMRewardModel(z_{0:T})
```

Trained contrastively: expert demo trajectories as positives, random Gaussian trajectories as negatives (val acc=100%). **However, this stage is deprecated in the current pipeline** due to a fundamental distribution mismatch: VLA-generated rollout trajectories are classified as "random" by the reward model (reward≈0.0003), yielding zero GRPO gradient. This finding highlights the importance of *in-distribution* reward signals for RL fine-tuning.

### 2.5 Stage 3: GRPO Policy Fine-tuning

We fine-tune OpenVLA-OFT (7B parameters) using GRPO. For each initial state, we generate a group of G=8 rollouts, simulate their latent trajectories using the world model, compute rewards, and update the policy:

```
A_i = (r_i - mean_j(r_j)) / (std_j(r_j) + ε)     # within-group advantage

L_GRPO = -E[ A_i · log π_θ(a_i | s_i) ] + β · KL(π_θ || π_ref)
```

**Key hyperparameters:**
- Group size G = 8, mini-batch size = 2
- Rollout steps = 20 chunks × 10 actions = 200 steps
- Temperature τ = 1.2 for action sampling
- KL coefficient β = 0.05
- Learning rate = 5×10⁻⁶ with cosine decay
- Training: 200 iterations, 4 initial states per iteration
- Only non-projector parameters updated; `freeze_projector=True`

---

## 3. Reward Design

### 3.1 The Core Challenge: Reward Variance Collapse

GRPO's advantage is computed as a *within-group* normalized reward:

```
A_i = (r_i - μ_group) / σ_group
```

If all 8 rollouts in a group receive nearly identical rewards (σ_group ≈ 0), the advantage collapses to zero and no gradient flows. We empirically observe this failure across multiple reward formulations, where reward standard deviation stabilizes at σ ≈ 0.07–0.08 throughout 200 training iterations. This is the central bottleneck for SWM's learning efficiency.

### 3.2 Reward Formulations

#### 3.2.1 Absolute Cosine Similarity (Failed)

```
r = cos(z_T, z_goal)
```

**Failure mode:** The high-dimensional DINOv2+SigLIP latent space places all latent vectors in a narrow cone. Even semantically different states have cosine similarity ≥ 0.97, so every rollout receives reward ≈ 0.97 regardless of behavior. σ_group ≈ 0.

#### 3.2.2 Normalized Progress (latent_cos) — Best Single-Goal Baseline

To compensate for the high baseline similarity, we normalize by the initial cosine distance:

```
r = (cos(z_T, z_goal) - cos(z_0, z_goal)) / (1 - cos(z_0, z_goal))
```

This measures *relative progress* from the initial state toward the goal state. Result: **24% SR**, the strongest single-goal reward. However, σ ≈ 0.07–0.08 remains low, and the reward curve is flat throughout training (early mean ≈ 0.12, late mean ≈ 0.11).

#### 3.2.3 Temporal Average Progress (Failed)

```
r = (1/T) Σ_t progress(z_t, z_goal)
```

**Failure mode:** Averaging over the full trajectory dilutes per-rollout variance. Result: **10% SR** — worse than single-endpoint reward.

#### 3.2.4 PCA-Projected Cosine Progress

The full 2176-dim latent space suffers from *latent indistinguishability*: goals at the 25th, 50th, 75th, and 100th percentiles of a demo trajectory have mutual cosine similarity ≥ 0.97. PCA to the top-16 principal components dramatically improves goal separability:

```
Goal inter-similarity:  2176-dim → 0.97+    (goals indistinguishable)
                        PCA-16   → 0.05–0.08 (goals well-separated)
Explained variance: 73.2%
```

The PCA-projected progress reward:

```
z_pca = PCA(z; n_components=16)       # fit on expert demo trajectories
progress_k = [cos(z_T_pca, g_k_pca) - cos(z_0_pca, g_k_pca)] / [1 - cos(z_0_pca, g_k_pca)]
r = (1/K) Σ_k progress_k              # K=4 uniformly spaced goals
```

Despite improved goal separation in latent space, PCA-goal (n_goals=4) achieves only **20% SR** — identical to the naïve multi-goal baseline. This reveals that the fundamental bottleneck is reward variance, not goal separability.

#### 3.2.5 PCA Delta Cosine

Rather than measuring progress toward a fixed goal, this reward measures whether the *direction of change* matches the goal direction:

```
δz_T    = z_T_pca - z_0_pca           # policy's movement vector
δz_goal = z_goal_pca - z_0_pca        # desired movement vector
r = cos(δz_T, δz_goal)                ∈ [-1, 1]
```

This removes the dependence on absolute goal location and focuses on trajectory direction. However, continuous cosine values in the range 0.18–0.53 still provide insufficient within-group discrimination. Result: **12% SR (best), 16% SR (last)**.

#### 3.2.6 Binary Phase Reward (Best Performing)

Inspired by the hypothesis that *discrete* rewards provide cleaner within-group variance, we threshold each goal's progress:

```
binary_k = 1(progress_k > τ),   τ = 0.3
r = (1/K) Σ_k binary_k           ∈ {0, 0.25, 0.5, 0.75, 1.0}
```

With K=4 goals, reward values are constrained to 5 discrete levels. Within a group of 8 rollouts, even a few successes on individual phase goals create strong variance. Observed σ_group increases noticeably and the reward curve shows an upward trend (early mean ≈ 0.061, late mean ≈ 0.084, Δ≈+0.023).

Result: **20% SR (best checkpoint), 26% SR (final checkpoint)** — current best overall.

**Key insight:** The `last` checkpoint outperforms `best` here because the reward improvement (binary thresholding increasing from 0 to 1) happens gradually — the best reward checkpoint may occur early when the policy is still noisy, while the final model has more consistent phase completion.

#### 3.2.7 Reward Aggregation Ablation

| Aggregation | Formula | SR (best) | SR (last) |
|---|---|---|---|
| Mean (K=4) | (1/K) Σ_k progress_k | 20% | 12% |
| **Binary Mean (K=4)** | (1/K) Σ_k 1(progress_k > 0.3) | **20%** | **26%** |
| Max (K=4) | max_k progress_k | 8% | 12% |

Max aggregation **hurts**: always picking the best goal reduces within-group variance by making it easier for any rollout to get a high reward.

---

## 4. Spatial Pooling for Goal Encoding

### 4.1 The Mean-Pool Problem

The default encoder computes z_t by mean-pooling over 256 spatial tokens:

```
z_t = Proj( (1/256) Σ_p token_p )
```

In a typical manipulation scene, the task-relevant region (e.g., a small nut or peg) occupies ≈1–5% of the image. The remaining 95–99% of tokens correspond to table surface, background, and robot body. Mean-pooling is therefore dominated by non-task tokens, producing goal latents that are nearly identical regardless of task phase (cosine sim ≥ 0.97).

### 4.2 Weighted Spatial Pooling

We replace mean-pooling with a *demo-informed weighted pooling* for goal encoding:

```
z_goal = Proj( Σ_p w_p · token_p ),   w_p ≥ 0, Σ_p w_p = 1
```

The weights **w** are computed once from expert demonstration trajectories before training. Only the top-K (K=64) tokens by weight are retained (others zeroed), ensuring sparsity. Four weight computation methods are studied:

#### Method 1: Variance Pooling

Tokens that change most across the demonstration trajectory are likely task-relevant:

```
w_p^var = || Var_t(token_p(t)) ||_2     # temporal feature variance
w_p     = softmask(top-K(w_p^var))
```

#### Method 2: Correlation Pooling

Tokens whose activation norm correlates with task progress (t/T) identify progress-coupled regions:

```
ρ_p = Pearson(||token_p(t)||_2,  t/T)
w_p = softmask(top-K(|ρ_p|))
```

#### Method 3: Activation Pooling

Tokens with consistently high activation norms are the most "active" and likely task-relevant:

```
w_p^act = E_t[ ||token_p(t)||_2 ]
w_p     = softmask(top-K(w_p^act))
```

#### Method 4: Slot Attention Pooling

Inspired by object-centric learning, we use iterative competitive attention to segment the scene into K=4 slots. The slot with the highest temporal variance in its assignments is identified as the task-relevant slot:

```python
# Initialize slots via PCA of all demo tokens
slots = PCA(all_tokens, n_components=4).components_    # (4, 2176)

# Iterative slot competition (3 iterations)
for _ in range(3):
    attn = softmax(normalize(tokens) @ normalize(slots).T)    # (N, 4)
    slots = normalize(attn.T @ tokens / attn.sum(0))

# Select most temporally varying slot
slot_var = Var_t(attn_per_frame)    # temporal variance per slot
best_slot = argmax(slot_var)

# Use best slot's attention as token weights
w_p = E_t[ attn_t[:, best_slot] ]
```

This method requires no additional training and runs at inference time using only demo observations.

### 4.3 Interaction with Reward Design

Since z_T (the final policy state) is produced by the *transition model* operating in the mean-pooled latent space, weighted pooling can only be applied to z_goal (which comes from actual demo images). The reward is therefore computed as:

```
r ∝ cos(z_T^{mean}, z_goal^{weighted})
```

This cross-space comparison is valid since both lie in ℝ^2176 and the weighted pooling shifts the goal embedding toward task-relevant subspace. The combined experiments (pca_binary + spatial pooling) test whether improved goal encoding further amplifies the binary reward signal.

---

## 5. Experiments

### 5.1 Setup

- **Task:** Square assembly (MimicGen), 50 evaluation episodes
- **VLA:** OpenVLA-OFT (7B parameters, Llama2-7B backbone)
- **Expert data:** 300 demonstrations
- **Hardware:** 3× NVIDIA B200 (183 GB VRAM each) for training, single B200 for evaluation
- **Evaluation:** Native pixel-mode VLA inference (no world model at test time)

### 5.2 Baselines

| Model | SR | Notes |
|---|---|---|
| SFT (OpenVLA-OFT) | 16% | Supervised fine-tuning only |
| WMPO-P128 | 24% | World-model PO, 128 pixel rollout steps |
| WMPO-P1280 | 36% | World-model PO, 1280 pixel rollout steps |

### 5.3 Main Results

| # | Experiment | Reward Design | SR (best ckpt) | SR (last ckpt) | Notes |
|---|---|---|---|---|---|
| 1 | Stage3-init | Graph distance | 10% | — | Projector unfrozen → feature drift |
| 2 | freeze\_proj | Graph distance | 10% | — | Projector frozen; reward design issue |
| 3 | temporal\_rm | Temporal RM classifier | — | — | reward≈0.0003; GRPO gradient≈0 |
| 4 | latent\_cos (abs) | cos(z_T, z_goal) | — | — | reward≈0.97 constant; σ≈0 |
| 5 | **latent\_cos** | Normalized progress, K=1 | **24%** | **24%** | +8%p vs SFT; WMPO-P128 parity |
| 6 | latent\_cos\_temporal | Trajectory mean progress | 10% | 10% | Variance dilution |
| 7 | multi\_goal | Normalized progress, K=4, 2176-dim | 20% | 18% | Goal sim=0.97+; reward noisy |
| 8 | pca\_goal | PCA-16 progress, K=4 | 20% | 12% | Goal sim→0.05; still variance-limited |
| 9 | pca\_goal\_single | PCA-16 progress, K=1 | 16% | 16% | PCA without multi-goal = SFT baseline |
| 10 | action\_goal | Action-velocity phase detection, K=4 | 16% | 16% | Phase detection imprecise |
| 11 | pca\_delta | PCA delta cosine, K=1 | 12% | 16% | Continuous reward; flat signal |
| 12 | **pca\_binary** | PCA binary phase, τ=0.3, K=4 | 20% | **26%** | **Current best** |
| 13 | diversity | Normalized progress, K=1, T=2.0 | 24% | 18% | Temperature increase ≈ SFT baseline |
| 14 | pca\_max | PCA max-progress, K=4 | 8% | 12% | Max aggregation reduces variance |
| 15 | action\_pca\_delta | Action phase + PCA delta | TBD | TBD | In progress |
| 16 | var\_pool\_binary | pca\_binary + variance pooling | TBD | TBD | Ongoing |
| 17 | corr\_pool\_binary | pca\_binary + correlation pooling | TBD | TBD | Ongoing |
| 18 | act\_pool\_binary | pca\_binary + activation pooling | TBD | TBD | Ongoing |
| 19 | slot\_attn\_binary | pca\_binary + slot attention | TBD | TBD | Ongoing |

### 5.4 Training Dynamics Analysis

To understand why reward design matters, we examine the training dynamics of all completed experiments:

| Experiment | Reward Mean | Reward Std | Early→Late (Δ) | SR (best) |
|---|---|---|---|---|
| action\_goal | 0.054 | 0.038 | flat | 16% |
| pca\_binary | 0.074 | **0.079** | **+0.022 ↑** | 20% |
| pca\_goal | 0.100 | 0.068 | +0.004 | 20% |
| diversity | 0.128 | 0.067 | −0.005 | 24% |
| pca\_max | 0.193 | 0.081 | +0.019 | 8% |
| pca\_delta | 0.331 | 0.107 | −0.008 | 12% |
| latent\_cos | ~0.12 | ~0.07 | flat | 24% |

**Observations:**
1. High reward mean does not imply high SR. `pca_delta` has the highest mean (0.33) but below-average SR (12%).
2. `pca_binary` is the only experiment showing consistent upward reward trend (+0.022), confirming that discrete rewards create cleaner learning signal.
3. `diversity` (T=2.0) matches the latent_cos baseline (24%) — higher temperature increases rollout diversity but does not improve goal pursuit quality.
4. `pca_max` underperforms despite high mean: max aggregation over 4 goals trivializes the reward by always finding the easiest phase goal.

---

## 6. Analysis

### 6.1 Latent Indistinguishability in High-Dimensional Space

A critical finding is that the 2176-dim DINOv2+SigLIP latent space concentrates all representations in a narrow angular cone. We measure:

```
Full 2176-dim space:
  Goal inter-cosine-similarity:  0.97+ (25/50/75/100% demo frames nearly identical)
  Δz norm across trajectory:     ~0.002 (very small)
  Trajectory curvature:          0.85–0.93 (near random orthogonality)

PCA top-16 projection:
  Explained variance:            73.2%
  Goal inter-cosine-similarity:  0.05–0.08 (well-separated)
  Train/test generalization:     consistent across held-out demos
```

This phenomenon is consistent with the known *hubness problem* in high-dimensional spaces and the concentration of measure effect: in sufficiently high dimensions, random vectors are nearly orthogonal, and even semantically different observations cluster near the uniform distribution on the sphere.

### 6.2 Why Freeze the Projector?

Early experiments (Stage3-init) allowed the projector to be updated during GRPO training. The projector rapidly adapts its output distribution to maximize the reward signal, but this *reward hacking* via representation shift degrades the alignment between z_T (from the fixed transition model) and z_goal (from the shifting encoder). Freezing the projector constrains the policy to improve actual action behavior rather than redefine the reward geometry.

### 6.3 Why Temporal Reward Fails

The temporal reward model (Stage 2.5) is trained to discriminate expert demo trajectories from random Gaussian trajectories, achieving 100% validation accuracy. However, VLA-generated rollout trajectories do not resemble either class: they are neither expert-like nor Gaussian-random, but occupy a distinct region of trajectory space that the binary classifier maps to "random" with high confidence. This *covariate shift* between training data and inference-time rollouts results in reward ≈ 0 across all rollouts, preventing any GRPO update.

### 6.4 GRPO Signal Strength

Across all completed experiments, the GRPO learning signal is weak relative to typical RL settings. Reward mean changes of < 0.02 over 200 iterations suggest that the policy is making marginal adjustments rather than fundamentally changing its behavior. Two contributing factors:

1. **7B model inertia:** Large language model backbones require strong gradients to update meaningfully; the KL penalty (β=0.05) and small learning rate (5×10⁻⁶) prevent large policy shifts.
2. **Reward scale:** All rewards are in [0, 1], but σ_group ≈ 0.07 means advantages are typically in [−1, +1], which is small relative to the cross-entropy loss magnitude.

The pca_binary reward partially addresses this by creating discrete jumps in reward that produce clearer advantage signals.

---

## 7. Key Findings and Lessons

1. **Reward variance is the primary bottleneck.** GRPO's advantage computation requires within-group reward variance; when all 8 rollouts receive similar rewards (σ ≈ 0.07), gradient flow is negligible regardless of reward magnitude.

2. **Discrete rewards outperform continuous rewards for GRPO.** Binarizing the phase progress reward (threshold τ=0.3) creates discrete {0, 0.25, 0.5, 0.75, 1.0} values that naturally produce within-group variance and a cleaner learning signal.

3. **Endpoint reward dominates temporal averaging.** Measuring reward only at the final timestep (z_T) avoids variance dilution. Temporal averaging across the trajectory reduces per-rollout discriminability (10% vs 24% SR).

4. **Mean pooling produces indistinguishable goal representations.** With 256 spatial tokens, background tokens (≥95% of tokens) dominate the mean, making latents at different demo phases nearly identical (sim=0.97+). PCA dimensionality reduction is necessary to expose task-relevant variation.

5. **Freeze the projector.** Updating the encoder projector during policy learning causes representation drift that invalidates the world model reward, leading to reward hacking rather than genuine task improvement.

6. **Initial cosine similarity normalization is essential.** Raw cosine similarity (r = cos(z_T, z_goal)) is constant ≈ 0.97 across all rollouts due to high-dimensional concentration. Normalizing by initial similarity isolates relative progress.

7. **Max aggregation over multi-goal rewards hurts.** Always selecting the easiest goal trivializes the reward, reducing variance and providing a weaker learning signal than mean aggregation.

---

## 8. Repo Structure

```
swm/
├── configs/
│   ├── stage1_multitask_dinosiglip.yaml   # Stage 1 encoder training
│   ├── stage2_multitask_dinosiglip.yaml   # Stage 2 transition model training
│   ├── stage2_5_v3.yaml                   # Stage 2.5 temporal reward model
│   ├── stage3_latent_cos.yaml             # Normalized cosine, K=1    → 24%
│   ├── stage3_latent_cos_temporal.yaml    # Temporal average progress  → 10%
│   ├── stage3_multi_goal.yaml             # Uniform K=4 goals          → 20%
│   ├── stage3_pca_goal.yaml               # PCA-16, K=4                → 20%
│   ├── stage3_pca_goal_single.yaml        # PCA-16, K=1                → 16%
│   ├── stage3_pca_delta.yaml              # PCA delta cosine            → 12%
│   ├── stage3_action_goal.yaml            # Action-velocity phases      → 16%
│   ├── stage3_pca_binary.yaml             # Binary phase, τ=0.3        → 26% ★
│   ├── stage3_diversity.yaml              # Temperature=2.0             → 24%
│   ├── stage3_pca_max.yaml                # Max aggregation             → 8%
│   ├── stage3_action_pca_delta.yaml       # Action phase + PCA delta    → TBD
│   ├── stage3_var_pool_binary.yaml        # Binary + variance pooling   → TBD
│   ├── stage3_corr_pool_binary.yaml       # Binary + correlation pooling→ TBD
│   ├── stage3_act_pool_binary.yaml        # Binary + activation pooling → TBD
│   └── stage3_slot_attn_binary.yaml       # Binary + slot attention     → TBD
├── models/
│   ├── encoder.py        # SWMEncoder: DINOv2+SigLIP, encode_weighted()
│   ├── transition.py     # SWMTransition: 6-layer Transformer, 21.2M params
│   ├── heads.py          # Auxiliary prediction heads (graph, depth)
│   └── reward_model.py   # SWMRewardModel: Temporal Transformer classifier
├── scripts/
│   ├── train_stage1.py               # Encoder training
│   ├── train_stage2.py               # Transition model training
│   ├── train_stage2_5.py             # Temporal RM training
│   ├── train_stage3.py               # GRPO fine-tuning (main)
│   ├── analyze_latent_inflection.py  # Δz / curvature analysis
│   ├── analyze_pca_latent.py         # PCA fit & goal separation
│   └── analyze_pca_generalization.py # Train/test PCA generalization
└── eval/
    └── eval_swm_mimicgen.py          # MimicGen evaluation (pixel mode)
```

---

## 9. Quick Start

```bash
# Stage 1: Train encoder projection head
python scripts/train_stage1.py --config configs/stage1_multitask_dinosiglip.yaml

# Stage 2: Train latent transition model
python scripts/train_stage2.py --config configs/stage2_multitask_dinosiglip.yaml

# Stage 3: GRPO fine-tuning (best config)
export CUDA_VISIBLE_DEVICES=2,3,7
torchrun --nproc_per_node=3 --master_port=29507 \
    scripts/train_stage3.py \
    --config configs/stage3_pca_binary.yaml

# Evaluation (pixel mode — always use native VLA pipeline)
CUDA_VISIBLE_DEVICES=2 TASK=square \
  CKPT_PATH=outputs/stage3/pca_binary/square/best.pt \
  STAGE1_CKPT=outputs/stage1/multitask_dinosiglip/best.pt \
  STAGE2_CKPT=outputs/stage2/multitask_dinosiglip/best.pt \
  VLA_BASE=<SFT_model_path> VLA_DEVICE=cuda N_EPISODES=50 \
  python eval/eval_swm_mimicgen.py
```

---

## 10. Comparison with WMPO

| Aspect | WMPO | SWM |
|---|---|---|
| World Model | V-JEPA2 + pixel video generation | DINOv2+SigLIP latent prediction |
| Reward | VideoMAE sparse binary classifier | Latent cosine progress / binary phase |
| Rollout | Dedicated rollout GPU (pixel decode) | Transition model only (lightweight) |
| Reward variance | Sparse binary → natural variance | Continuous → variance collapse problem |
| Best SR (square) | P128: 24%, P1280: 36% | 26% (pca\_binary, last ckpt) |
| Training compute | High (pixel generation) | Low (latent only) |
| Sensitivity | Pixel quality dependent | Encoder quality dependent |

SWM matches WMPO-P128 with substantially lower compute and surpasses it by 2%p on the last checkpoint. Closing the gap to WMPO-P1280 (36%) remains the primary research goal.

---

## Hardware

- **GPU:** 3× NVIDIA B200 (183 GB VRAM each) for training; single B200 for evaluation
- **Training:** `torchrun --nproc_per_node=3` (data-parallel across 3 GPUs)
- **Memory:** ~60 GB per GPU for 7B VLA + world model during GRPO training

---

## Citation

```bibtex
@misc{swm2026,
  title   = {SWM: Structured World Model for Latent-Reward GRPO Fine-tuning of Vision-Language-Action Models},
  author  = {Lee, Seungjae},
  year    = {2026},
  note    = {Technical Report}
}
```
