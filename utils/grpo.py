# utils/grpo.py
"""
GRPO — Group Relative Policy Optimisation
------------------------------------------
Dense Graph-distance reward replaces sparse VideoMAE reward.
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn.functional as F


def compute_graph_distance_reward(
    pred_positions: torch.Tensor,    # (B, max_nodes, 3)  predicted at final step
    goal_positions: torch.Tensor,    # (max_nodes, 3)     goal graph
    node_mask: torch.Tensor,         # (max_nodes,) bool
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Dense reward: negative mean L2 distance (per trajectory).
    r_i ∈ (−∞, 0];  r_i → 0 means perfect alignment with goal.

    Args:
        pred_positions: (B, N, 3)  final predicted node positions
        goal_positions: (N, 3)     goal node positions
        node_mask:      (N,)       which nodes are valid
        temperature:    scales reward before advantage computation
    Returns:
        rewards: (B,)
    """
    goal = goal_positions.unsqueeze(0).expand_as(pred_positions)   # (B, N, 3)
    dist = (pred_positions - goal).pow(2).sum(-1).sqrt()           # (B, N)

    # Mask out padding nodes
    mask = node_mask.unsqueeze(0).float()                          # (1, N)
    reward = -(dist * mask).sum(-1) / mask.sum(-1).clamp(min=1)   # (B,)
    return reward * temperature


def compute_grpo_loss(
    log_probs: torch.Tensor,        # (B, T, K)  log π_θ(a|s) per step per action dim
    old_log_probs: torch.Tensor,    # (B, T, K)  log π_θ_old(a|s)
    rewards: torch.Tensor,          # (B,)        one reward per trajectory
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.001,
    normalize_advantage: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    GRPO loss with group-relative advantage.

    Args:
        log_probs:      current policy log-probs
        old_log_probs:  reference policy log-probs (π_θ_old)
        rewards:        (B,) dense graph-distance rewards
        clip_epsilon:   PPO clipping
        kl_coef:        KL penalty coefficient
        entropy_coef:   entropy bonus coefficient
    Returns:
        loss:   scalar
        info:   dict with diagnostic scalars
    """
    B = rewards.shape[0]

    # ── Group-relative advantage ──────────────────────────────────────────────
    mean_r = rewards.mean()
    std_r  = rewards.std().clamp(min=1e-6)
    if normalize_advantage:
        advantage = (rewards - mean_r) / std_r   # (B,)
    else:
        advantage = rewards - mean_r             # (B,)

    # (B,) → (B, T, K) for broadcasting
    adv = advantage.view(B, 1, 1).expand_as(log_probs)

    # ── Importance ratio ─────────────────────────────────────────────────────
    ratio = torch.exp(log_probs - old_log_probs.detach())   # (B, T, K)

    # ── Clipped surrogate objective ──────────────────────────────────────────
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # ── KL penalty ───────────────────────────────────────────────────────────
    kl = (old_log_probs.detach() - log_probs).mean()
    kl_loss = kl_coef * kl

    # ── Entropy bonus ─────────────────────────────────────────────────────────
    entropy = -log_probs.mean()
    entropy_loss = -entropy_coef * entropy

    total_loss = policy_loss + kl_loss + entropy_loss

    info = {
        "loss/policy":   policy_loss.item(),
        "loss/kl":       kl_loss.item(),
        "loss/entropy":  entropy_loss.item(),
        "loss/total":    total_loss.item(),
        "reward/mean":   mean_r.item(),
        "reward/std":    std_r.item(),
        "reward/max":    rewards.max().item(),
        "reward/min":    rewards.min().item(),
        "ratio/mean":    ratio.mean().item(),
        "ratio/clip_frac": ((ratio - 1).abs() > clip_epsilon).float().mean().item(),
    }

    return total_loss, info


def sample_action_chunks(
    policy,
    z: torch.Tensor,                   # (1, latent_dim)
    instruction: str,
    G: int = 8,
    temperature: float = 1.0,
    chunk_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample G action chunks from policy.

    Returns:
        actions:    (G, chunk_size, action_dim)
        log_probs:  (G, chunk_size, action_dim)
    """
    z_expanded = z.expand(G, -1)   # (G, latent_dim)
    actions, log_probs = policy.sample(
        z_expanded,
        instruction,
        temperature=temperature,
        chunk_size=chunk_size,
    )
    return actions, log_probs
