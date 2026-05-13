#!/usr/bin/env python3
# scripts/train_stage3.py
"""
Stage 3: Policy Training (GRPO)
================================

전체 흐름:
  1. image → V-JEPA-2 encoder → z_0  (초기 latent)
  2. z_0 + instruction → OpenVLA-OFT → action_chunk_0  (non-AR, chunk 단위)
  3. z_0 + action_mean_0 → Transition → z_1
  4. z_1 + instruction → OpenVLA-OFT → action_chunk_1
  5. ... T번 반복 → z_T
  6. z_T → frozen Graph Head → Ĝ_T
  7. Reward = -‖Ĝ_T − G*_T‖²  (dense, 별도 reward model 불필요)
  8. GRPO → Policy(OpenVLA-OFT) 업데이트

G개 trajectory 병렬 샘플링 → group-relative advantage
"""

import sys
import random
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import SWMEncoder
from models.heads import SWMHeads
from models.transition import SWMTransition
from models.policy import SWMPolicy
from data.dataset import GoalGraphDataset, InitialStateDataset
from utils.grpo import compute_graph_distance_reward, compute_grpo_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_components(cfg, device):
    """Stage 1, 2 체크포인트 로드 후 freeze."""
    # Encoder + Heads (Stage 1)
    encoder = SWMEncoder(latent_dim=cfg.get("latent_dim", 1024)).to(device)
    heads   = SWMHeads(latent_dim=encoder.latent_dim).to(device)
    s1 = torch.load(cfg.stage1_ckpt, map_location=device)
    encoder.load_state_dict(s1["encoder"])
    heads.load_state_dict(s1["heads"])

    # Transition (Stage 2)
    transition = SWMTransition(
        latent_dim=encoder.latent_dim,
        action_dim=cfg.policy.action_dim,
    ).to(device)
    s2 = torch.load(cfg.stage2_ckpt, map_location=device)
    transition.load_state_dict(s2["transition"])

    for m in [encoder, heads, transition]:
        for p in m.parameters(): p.requires_grad = False
        m.eval()

    log.info("Frozen: Encoder, Heads, Transition")
    return encoder, heads, transition


@torch.no_grad()
def imagined_rollout(
    z0: torch.Tensor,          # (G, latent_dim)
    policy: SWMPolicy,
    transition: SWMTransition,
    instruction: str,
    T: int,
    chunk_size: int,
    temperature: float,
    device,
):
    """
    T번 반복:
      z_t → Policy → action_chunk_t (non-AR)
      z_t + action_mean_t → Transition → z_t+1 (AR)

    Returns:
      z_T:        (G, latent_dim)  최종 latent
      all_actions:(G, T, chunk_size, action_dim)
      all_lp:     (G, T, chunk_size, action_dim)
    """
    z = z0
    all_actions = []
    all_lp      = []

    for t in range(T):
        # Policy: z_t → action chunk (non-AR)
        actions, log_probs = policy.sample(
            z, instruction,
            chunk_size=chunk_size,
            temperature=temperature,
        )
        # actions: (G, chunk_size, action_dim)

        # Transition: z_t + mean(action) → z_t+1
        a_mean = actions.mean(dim=1)                    # (G, action_dim)
        z = transition(z, a_mean, add_noise=False)      # (G, latent_dim)

        all_actions.append(actions)
        all_lp.append(log_probs)

    all_actions = torch.stack(all_actions, dim=1)  # (G, T, chunk_size, action_dim)
    all_lp      = torch.stack(all_lp,      dim=1)  # (G, T, chunk_size, action_dim)
    return z, all_actions, all_lp


def recompute_log_probs(
    policy: SWMPolicy,
    z0: torch.Tensor,          # (G, latent_dim)
    transition: SWMTransition,
    all_actions: torch.Tensor, # (G, T, chunk_size, action_dim)
    instruction: str,
) -> torch.Tensor:
    """
    현재 policy θ로 log π_θ(a|s) 재계산.
    GRPO ratio 계산에 필요.
    """
    G, T, chunk_size, action_dim = all_actions.shape
    z = z0
    new_lp_list = []

    for t in range(T):
        a_t = all_actions[:, t]                       # (G, chunk_size, action_dim)
        lp  = policy.log_prob(z, instruction, a_t)   # (G, chunk_size, action_dim)
        new_lp_list.append(lp)
        with torch.no_grad():
            a_mean = a_t.mean(dim=1)
            z = transition(z, a_mean, add_noise=False)

    return torch.stack(new_lp_list, dim=1)  # (G, T, chunk_size, action_dim)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3.yaml")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    cfg     = OmegaConf.load(args.config)
    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    if not args.debug:
        import wandb
        wandb.init(
            project="swm",
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Frozen components ─────────────────────────────────────────────────────
    encoder, heads, transition = load_frozen_components(cfg, device)

    # ── Policy (OpenVLA-OFT + V-JEPA-2) ──────────────────────────────────────
    policy = SWMPolicy(
        openvla_ckpt=cfg.policy.get("openvla_ckpt", None),
        vjepa2_ckpt=cfg.policy.get("vjepa2_ac_ckpt", None),
        freeze_siglip=cfg.policy.freeze_siglip,
        freeze_vjepa2=cfg.policy.freeze_vjepa,
        latent_dim=encoder.latent_dim,
        action_dim=cfg.policy.action_dim,
        action_chunk_size=cfg.policy.action_chunk_size,
        use_peft=cfg.policy.use_peft,
        peft_r=cfg.policy.peft_r,
        peft_alpha=cfg.policy.peft_alpha,
    ).to(device)

    # π_θ_old: policy 복사본 (GRPO ratio 분모)
    import copy
    policy_old = copy.deepcopy(policy)
    for p in policy_old.parameters():
        p.requires_grad = False
    policy_old.eval()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    # ── Goal Graph Dataset (성공 trajectory 마지막 프레임) ─────────────────
    goal_ds = GoalGraphDataset(
        data_root=cfg.data.data_root,
        task=cfg.data.task,
        max_nodes=8,
    )

    # ── Initial State Dataset ─────────────────────────────────────────────
    init_ds = InitialStateDataset(
        data_root=cfg.data.data_root,
        task=cfg.data.task,
        image_size=cfg.data.image_size,
    )
    init_loader = DataLoader(
        init_ds, batch_size=1,
        shuffle=True, num_workers=2,
    )
    init_iter = iter(init_loader)

    # ── GRPO 하이퍼파라미터 ───────────────────────────────────────────────
    G           = cfg.grpo.G
    T           = cfg.grpo.T
    chunk_size  = cfg.policy.action_chunk_size
    temperature = cfg.grpo.sampling_temperature
    instruction = f"pick and place square into peg"  # TODO: task별 instruction

    # ── GRPO loop ─────────────────────────────────────────────────────────
    best_reward = -float("inf")

    for iteration in range(1, cfg.training.iterations + 1):

        # ── 초기 상태 샘플링 ─────────────────────────────────────────────
        try:
            init_batch = next(init_iter)
        except StopIteration:
            init_iter = iter(init_loader)
            init_batch = next(init_iter)

        init_image = init_batch["image"].to(device)  # (1, 3, H, W)

        with torch.no_grad():
            # V-JEPA-2 encoder → z_0
            z0_single = encoder(init_image)                    # (1, latent_dim)
            z0 = z0_single.expand(G, -1).contiguous()         # (G, latent_dim)

        # ── Goal graph 샘플링 ────────────────────────────────────────────
        goal_graph = goal_ds.random_goal().to(device)          # (max_nodes, 3)
        node_mask  = (goal_graph.abs().sum(-1) > 0)            # (max_nodes,)

        # ── Imagined rollout with π_θ_old ────────────────────────────────
        with torch.no_grad():
            z_T, all_actions, old_log_probs = imagined_rollout(
                z0, policy_old, transition,
                instruction, T, chunk_size, temperature, device
            )
            # z_T: (G, latent_dim)

        # ── Dense reward: Graph distance ─────────────────────────────────
        with torch.no_grad():
            graph_pred, _ = heads(z_T)    # (G, max_nodes, 3)
            rewards = compute_graph_distance_reward(
                graph_pred, goal_graph, node_mask,
                temperature=cfg.grpo.reward_temperature,
            )  # (G,)  연속값, 가까울수록 0에 가까움

        # ── Re-compute log probs under current policy π_θ ────────────────
        with torch.cuda.amp.autocast(enabled=cfg.training.amp):
            new_log_probs = recompute_log_probs(
                policy, z0.detach(), transition,
                all_actions, instruction
            )  # (G, T, chunk_size, action_dim)

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
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, policy.parameters()),
            cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        # π_θ_old 동기화
        policy_old.load_state_dict(policy.state_dict())

        # ── Logging ──────────────────────────────────────────────────────
        if iteration % cfg.training.log_interval == 0:
            log.info(
                f"[Iter {iteration:04d}]  "
                f"loss={info['loss/total']:.4f}  "
                f"reward_mean={info['reward/mean']:.4f}  "
                f"reward_max={info['reward/max']:.4f}  "
                f"clip_frac={info['ratio/clip_frac']:.3f}"
            )
            if not args.debug:
                wandb.log({**info, "iteration": iteration})

        # ── Checkpoint ───────────────────────────────────────────────────
        if iteration % cfg.training.save_interval == 0:
            torch.save({
                "iteration":   iteration,
                "policy":      policy.state_dict(),
                "reward_mean": info["reward/mean"],
            }, out_dir / f"ckpt_iter{iteration:04d}.pt")

        mean_r = info["reward/mean"]
        if mean_r > best_reward:
            best_reward = mean_r
            torch.save({
                "iteration":   iteration,
                "policy":      policy.state_dict(),
                "reward_mean": mean_r,
            }, out_dir / "best.pt")
            log.info(f"  ★ New best reward={best_reward:.4f}")

    log.info(f"Stage 3 done.  Best reward={best_reward:.4f}")


if __name__ == "__main__":
    main()
