#!/usr/bin/env python3
# scripts/train_stage3.py
"""
Stage 3: Policy Training (GRPO)
  z_t + instruction → Policy → action chunk (non-AR)
  Repeated T times  → imagined trajectory
  Reward: ‖ Ĝ_T − G*_T ‖²  (dense, no reward model needed)
  Update: GRPO
"""

import sys
import argparse
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import SWMEncoder
from models.heads import SWMHeads
from models.transition import SWMTransition
from utils.grpo import compute_graph_distance_reward, compute_grpo_loss
from data.dataset import GoalGraphDataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Policy stub — replace with your OpenVLA-OFT wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PolicyWrapper(nn.Module):
    """
    Wraps OpenVLA-OFT.

    Expected interface:
        actions, log_probs = policy.sample(z, instruction,
                                           chunk_size, temperature)
        actions:   (B, chunk_size, action_dim)
        log_probs: (B, chunk_size, action_dim)
    """

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.action_dim = cfg.policy.action_dim
        self.chunk_size = cfg.policy.action_chunk_size
        self._load_policy()

    def _load_policy(self):
        try:
            # TODO: replace with actual OpenVLA-OFT loading
            # from openvla_oft import OpenVLAOFT
            # self.model = OpenVLAOFT.from_pretrained(self.cfg.policy.openvla_ckpt)
            log.warning("PolicyWrapper: using RANDOM policy stub. "
                        "Replace with real OpenVLA-OFT.")
            self.model = None
        except Exception as e:
            log.error(f"Policy load failed: {e}")
            self.model = None

    def sample(
        self,
        z: torch.Tensor,         # (B, latent_dim)
        instruction: str,
        chunk_size: int = 4,
        temperature: float = 1.0,
    ):
        B = z.shape[0]
        if self.model is None:
            # Random stub for testing pipeline
            actions = torch.randn(B, chunk_size, self.action_dim,
                                  device=z.device) * temperature
            log_probs = torch.zeros_like(actions) - 1.0
            return actions, log_probs

        return self.model.sample(z, instruction,
                                 chunk_size=chunk_size,
                                 temperature=temperature)

    def log_prob(
        self,
        z: torch.Tensor,
        instruction: str,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Re-compute log π_θ(a|s) for given actions (for GRPO ratio)."""
        if self.model is None:
            return torch.zeros_like(actions) - 1.0
        return self.model.log_prob(z, instruction, actions)


# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_components(cfg, device):
    """Load and freeze Encoder, Heads, Transition from Stage 1 & 2."""
    encoder = SWMEncoder(backbone=cfg.encoder.backbone
                         if hasattr(cfg, "encoder") else "vjepa2").to(device)
    heads = SWMHeads(latent_dim=encoder.latent_dim).to(device)

    s1 = torch.load(cfg.stage1_ckpt, map_location=device)
    encoder.load_state_dict(s1["encoder"])
    heads.load_state_dict(s1["heads"])

    transition = SWMTransition(latent_dim=encoder.latent_dim,
                               action_dim=cfg.policy.action_dim).to(device)
    s2 = torch.load(cfg.stage2_ckpt, map_location=device)
    transition.load_state_dict(s2["transition"])

    for m in [encoder, heads, transition]:
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    log.info("Frozen: Encoder, Heads, Transition")
    return encoder, heads, transition


@torch.no_grad()
def imagined_rollout(
    z0: torch.Tensor,           # (G, latent_dim)
    policy,
    transition,
    instruction: str,
    T: int,
    chunk_size: int,
    temperature: float,
):
    """
    Run T steps of policy + transition in latent space.
    Returns final z_T and collected (actions, log_probs).
    """
    z = z0
    all_actions   = []
    all_log_probs = []

    for t in range(T):
        actions, log_probs = policy.sample(z, instruction,
                                           chunk_size=chunk_size,
                                           temperature=temperature)
        # actions: (G, chunk_size, action_dim)
        # Use mean action of chunk for transition step
        a_mean = actions.mean(dim=1)              # (G, action_dim)
        z = transition(z, a_mean, training=False) # (G, latent_dim)

        all_actions.append(actions)
        all_log_probs.append(log_probs)

    # Stack: (G, T, chunk_size, action_dim)
    all_actions   = torch.stack(all_actions,   dim=1)
    all_log_probs = torch.stack(all_log_probs, dim=1)

    return z, all_actions, all_log_probs


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage3.yaml")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    cfg     = OmegaConf.load(args.config)
    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    if not args.debug:
        wandb.init(project="swm", name=cfg.experiment.name,
                   config=OmegaConf.to_container(cfg))

    # ── Frozen components ─────────────────────────────────────────────────────
    encoder, heads, transition = load_frozen_components(cfg, device)

    # ── Policy ────────────────────────────────────────────────────────────────
    policy     = PolicyWrapper(cfg, device).to(device)
    policy_old = PolicyWrapper(cfg, device).to(device)
    policy_old.load_state_dict(policy.state_dict())
    for p in policy_old.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.training.lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    # ── Goal graphs from success trajectories ─────────────────────────────────
    goal_ds = GoalGraphDataset(
        success_traj_dir=cfg.data.success_traj_dir,
        task=cfg.data.task,
    )
    log.info(f"Goal graphs: {len(goal_ds)}")

    G           = cfg.grpo.G
    T           = cfg.grpo.T
    chunk_size  = cfg.policy.action_chunk_size
    temperature = cfg.grpo.sampling_temperature
    max_nodes   = 8

    # ── GRPO loop ──────────────────────────────────────────────────────────────
    best_reward = -float("inf")
    for iteration in range(1, cfg.training.iterations + 1):

        # Sample a random initial state from the environment
        # TODO: replace with real env.reset() → obs
        # For now: random latent as stub
        with torch.no_grad():
            z0 = torch.randn(1, encoder.latent_dim, device=device)
            z0_expanded = z0.expand(G, -1)            # (G, latent_dim)

        # Sample goal graph (random from success set)
        goal_idx = random.randint(0, len(goal_ds) - 1)
        goal_graph = goal_ds[goal_idx].to(device)     # (max_nodes, 3)
        node_mask = (goal_graph.abs().sum(-1) > 0)    # (max_nodes,)

        instruction = "pick and place"  # TODO: load from dataset

        # ── Imagined rollout ───────────────────────────────────────────────────
        with torch.no_grad():
            z_T, all_actions, old_log_probs = imagined_rollout(
                z0_expanded, policy_old, transition,
                instruction, T, chunk_size, temperature
            )
            # z_T: (G, latent_dim)

        # ── Reward: Graph distance ─────────────────────────────────────────────
        with torch.no_grad():
            graph_pred, _ = heads(z_T)   # (G, max_nodes, 3)
            rewards = compute_graph_distance_reward(
                graph_pred, goal_graph, node_mask,
                temperature=cfg.grpo.reward_temperature
            )  # (G,)

        # ── Re-compute log probs under current policy ─────────────────────────
        # all_actions: (G, T, chunk_size, action_dim)
        # Reshape for log_prob computation
        with torch.cuda.amp.autocast(enabled=cfg.training.amp):
            z_t = z0_expanded.detach()
            new_log_probs_list = []
            for t in range(T):
                a_t = all_actions[:, t]              # (G, chunk_size, action_dim)
                lp = policy.log_prob(z_t, instruction, a_t)
                new_log_probs_list.append(lp)
                a_mean = a_t.mean(dim=1)
                with torch.no_grad():
                    z_t = transition(z_t, a_mean, training=False)

            new_log_probs = torch.stack(new_log_probs_list, dim=1)
            # (G, T, chunk_size, action_dim)

            loss, info = compute_grpo_loss(
                log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                rewards=rewards,
                clip_epsilon=cfg.grpo.clip_epsilon,
                kl_coef=cfg.grpo.kl_coef,
                entropy_coef=cfg.grpo.entropy_coef,
                normalize_advantage=cfg.grpo.normalize_advantage,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Sync old policy
        policy_old.load_state_dict(policy.state_dict())

        if iteration % cfg.training.log_interval == 0:
            log.info(f"[Iter {iteration}]  "
                     f"loss={info['loss/total']:.4f}  "
                     f"reward_mean={info['reward/mean']:.4f}  "
                     f"reward_max={info['reward/max']:.4f}")
            wandb.log({**info, "iteration": iteration})

        if iteration % cfg.training.save_interval == 0:
            torch.save({
                "iteration": iteration,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "reward_mean": info["reward/mean"],
            }, out_dir / f"ckpt_iter{iteration:04d}.pt")

        mean_r = info["reward/mean"]
        if mean_r > best_reward:
            best_reward = mean_r
            torch.save({
                "iteration": iteration,
                "policy": policy.state_dict(),
                "reward_mean": mean_r,
            }, out_dir / "best.pt")
            log.info(f"  ✓ New best reward: {best_reward:.4f}")

    log.info("Stage 3 complete.")


if __name__ == "__main__":
    main()
