# SWM — Structured World Model

> **Pixel-free World Modeling via Graph & Sparse Depth for Robot Manipulation**

## Overview

SWM replaces pixel-level reconstruction with structured scene supervision
(Graph Node Positions + Sparse Depth) to learn a compact, task-relevant latent
world model.  A GRPO-based policy is then optimised entirely inside this latent
space — no image generation, no separate reward model.

```
Stage 1 │ Encoder  : O_t + G_t  →  z_t  →  Ĝ_t + D̂_t
Stage 2 │ Transition: z_t + a_t →  ẑ_t+1  ≈  z*_t+1  (JEPA)
Stage 3 │ Policy    : GRPO with dense Graph-distance reward
```

## Key Differences from WMPO

| | WMPO | SWM |
|---|---|---|
| World Model | pixel generation (OpenSora) | latent prediction (structured) |
| Reward | VideoMAE sparse (0/1) | Graph distance dense (continuous) |
| Reward Model | separately trained | no training needed |
| Rollout drift | Noisy Frame Conditioning | Teacher Forcing + Noise Injection |

## Repo Structure

```
swm/
├── configs/          # YAML experiment configs
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── stage3.yaml
├── models/           # Model definitions
│   ├── encoder.py        # SWM Encoder (V-JEPA2 backbone)
│   ├── transition.py     # Transition model (V-JEPA2-AC)
│   ├── heads.py          # Graph Node Position + Sparse Depth heads
│   └── policy.py         # Policy π_θ (OpenVLA-OFT based)
├── data/             # Dataset & dataloader
│   ├── dataset.py
│   └── graph_utils.py    # Detection → Graph builder
├── utils/
│   ├── reward.py         # Graph-distance dense reward
│   ├── grpo.py           # GRPO implementation
│   └── metrics.py
├── scripts/          # Training entry points
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── train_stage3.py
├── tests/
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/yourname/swm.git
cd swm
pip install -e ".[dev]"
```

## Quick Start

```bash
# Stage 1: Train Encoder
python scripts/train_stage1.py --config configs/stage1.yaml

# Stage 2: Train Transition
python scripts/train_stage2.py --config configs/stage2.yaml

# Stage 3: GRPO Policy
python scripts/train_stage3.py --config configs/stage3.yaml
```
