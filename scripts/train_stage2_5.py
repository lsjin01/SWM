#!/usr/bin/env python3
# scripts/train_stage2_5.py
"""
Stage 2.5 v3: Temporal Reward Model Training
=============================================
WMPO LatentRewardModel 방식으로 변경:
  - 입력: z_history[-T:] (temporal trajectory)  ← z_T + z_goal 방식 폐기
  - 구조: Temporal Transformer (CLS token, Pre-LN)
  - goal 비교 없음: trajectory 자체가 성공/실패를 구분

Positive (label=1): WM_transition(z0, demo_actions[:N]) 의 uniformly sampled T latents
Negative (label=0): WM_transition(z0, random_actions[:N]) 의 uniformly sampled T latents
"""

import os, sys, logging, argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/dependencies/openvla-oft')
sys.path.insert(0, '/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA')

from models.encoder      import SWMEncoder
from models.transition   import SWMTransition
from models.reward_model import SWMRewardModel, sample_traj

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RewardDataset(Dataset):
    """
    sample: (z_traj, label)
      z_traj: (n_frames, latent_dim)  — uniformly sampled WM latent trajectory
      label:  1 = demo rollout (success-like), 0 = random rollout (failure-like)
    """

    def __init__(self, z_trajs_pos, z_trajs_neg):
        trajs  = torch.cat([z_trajs_pos, z_trajs_neg], dim=0)  # (N, T, D)
        labels = torch.cat([
            torch.ones(len(z_trajs_pos)),
            torch.zeros(len(z_trajs_neg)),
        ])
        self.trajs  = trajs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.trajs[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Data builder
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_latent_dataset(hdf5_path, encoder, transition, device, dtype,
                          max_demos=300, n_rollout_steps=160,
                          n_neg_per_demo=4, n_pos_per_demo=1,
                          n_frames=8, action_dim=7):
    """
    HDF5 데모에서 temporal WM trajectory (z_traj, label) 구축.

    각 rollout의 z_history를 uniformly n_frames 샘플 → (n_frames, latent_dim)
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    z_pos_list, z_neg_list = [], []

    with h5py.File(hdf5_path, 'r') as f:
        demos = sorted(f['data'].keys())[:max_demos]
        log.info(f"Processing {len(demos)} demos  n_frames={n_frames} ...")

        for dn in demos:
            imgs         = f[f'data/{dn}/obs/agentview_image']
            demo_actions = f[f'data/{dn}/actions'][:]
            T = len(imgs)
            if T < 4:
                continue

            img0 = transform(imgs[0]).unsqueeze(0).to(device)
            z0   = encoder(img0).to(dtype)

            n_steps = min(n_rollout_steps, len(demo_actions))
            action_std = torch.from_numpy(demo_actions).float().std(0).clamp(min=0.1)

            # ── Positive: demo actions ─────────────────────────────────────
            for _ in range(n_pos_per_demo):
                z_t   = z0.clone()
                z_hist = []
                for t in range(n_steps):
                    a = torch.from_numpy(demo_actions[t]).float().to(device).unsqueeze(0)
                    z_t = transition(z_t.float(), a).to(dtype)
                    z_hist.append(z_t.cpu())
                traj = sample_traj(z_hist, n_frames)  # (1, n_frames, latent_dim)
                z_pos_list.append(traj)

            # ── Negative: random actions ───────────────────────────────────
            for _ in range(n_neg_per_demo):
                z_t   = z0.clone()
                z_hist = []
                for t in range(n_steps):
                    a = (torch.randn(1, action_dim) * action_std.unsqueeze(0)).to(device)
                    z_t = transition(z_t.float(), a).to(dtype)
                    z_hist.append(z_t.cpu())
                traj = sample_traj(z_hist, n_frames)  # (1, n_frames, latent_dim)
                z_neg_list.append(traj)

    z_pos = torch.cat(z_pos_list, dim=0)  # (N_pos, n_frames, latent_dim)
    z_neg = torch.cat(z_neg_list, dim=0)  # (N_neg, n_frames, latent_dim)

    log.info(f"Dataset: pos={len(z_pos)}, neg={len(z_neg)}")
    return z_pos, z_neg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/stage2_5_v3.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / 'train.log', mode='w'),
        ]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16
    log.info(f"Device: {device}")

    # ── Load frozen encoder ──────────────────────────────────────────────────
    s1_ckpt = torch.load(cfg.stage1_ckpt, map_location=device, weights_only=False)
    encoder = SWMEncoder(
        vla_path=cfg.vla_path,
        freeze_backbone=True,
        latent_dim=2176,
    ).to(device)
    encoder.load_state_dict(s1_ckpt['encoder'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    log.info("Encoder loaded (frozen)")

    # ── Load frozen transition ───────────────────────────────────────────────
    s2_ckpt = torch.load(cfg.stage2_ckpt, map_location=device, weights_only=False)
    transition = SWMTransition(
        latent_dim=2176,
        action_dim=7,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
    ).to(device)
    transition.load_state_dict(s2_ckpt['transition'])
    transition.eval()
    for p in transition.parameters():
        p.requires_grad = False
    log.info("Transition loaded (frozen)")

    # ── Build temporal latent dataset ────────────────────────────────────────
    task     = cfg.task
    n_frames = cfg.model.get('n_frames', 8)
    hdf5     = os.path.join(
        cfg.data_root, 'demos', 'core_datasets', task,
        f'demo_src_{task}_task_D0', 'demo.hdf5'
    )
    n_rollout_steps = cfg.training.get('n_rollout_steps', 160)
    log.info(f"Building temporal dataset  n_frames={n_frames}  n_rollout_steps={n_rollout_steps}")
    z_pos, z_neg = build_latent_dataset(
        hdf5, encoder, transition, device, dtype,
        max_demos       = cfg.training.max_demos,
        n_rollout_steps = n_rollout_steps,
        n_neg_per_demo  = cfg.training.get('n_neg_per_demo', 4),
        n_pos_per_demo  = cfg.training.get('n_pos_per_demo', 1),
        n_frames        = n_frames,
    )

    train_split  = cfg.training.get('train_split', 0.9)
    n_pos_train  = int(len(z_pos) * train_split)
    n_neg_train  = int(len(z_neg) * train_split)

    train_ds = RewardDataset(z_pos[:n_pos_train], z_neg[:n_neg_train])
    val_ds   = RewardDataset(z_pos[n_pos_train:], z_neg[n_neg_train:])
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                               shuffle=True, drop_last=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                               shuffle=False, num_workers=4)

    # ── Reward model ─────────────────────────────────────────────────────────
    rm = SWMRewardModel(
        latent_dim = cfg.model.get('latent_dim', 2176),
        hidden_dim = cfg.model.get('hidden_dim', 256),
        n_frames   = n_frames,
        n_heads    = cfg.model.get('n_heads', 4),
        n_layers   = cfg.model.get('n_layers', 2),
        dropout    = cfg.model.get('dropout', 0.0),
    ).to(device)

    n_params = sum(p.numel() for p in rm.parameters())
    log.info(f"RewardModel params: {n_params:,}")

    optimizer = torch.optim.AdamW(rm.parameters(),
                                   lr=cfg.training.lr,
                                   weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 0.1
    )

    pos_weight = torch.tensor([n_neg_train / max(n_pos_train, 1)], device=device)
    best_val_loss = float('inf')

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, cfg.training.epochs + 1):
        rm.train()
        train_loss, train_acc = 0.0, 0.0
        for z_traj, label in train_loader:
            z_traj = z_traj.to(device, torch.float32)  # (B, T, D)
            label  = label.to(device, torch.float32)

            prob = rm(z_traj)                           # (B,) already sigmoid
            loss = F.binary_cross_entropy(
                prob, label, weight=pos_weight.expand_as(label) * label + (1 - label)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_acc  += ((prob > 0.5).float() == label).float().mean().item()

        scheduler.step()

        rm.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for z_traj, label in val_loader:
                z_traj = z_traj.to(device, torch.float32)
                label  = label.to(device, torch.float32)
                prob   = rm(z_traj)
                loss   = F.binary_cross_entropy(
                    prob, label, weight=pos_weight.expand_as(label) * label + (1 - label)
                )
                val_loss += loss.item()
                val_acc  += ((prob > 0.5).float() == label).float().mean().item()

        n_tr = len(train_loader); n_vl = len(val_loader)
        log.info(
            f"[Epoch {epoch:03d}]  "
            f"train loss={train_loss/n_tr:.4f} acc={train_acc/n_tr:.3f}  "
            f"val loss={val_loss/n_vl:.4f} acc={val_acc/n_vl:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        ckpt = {
            'epoch': epoch,
            'reward_model': rm.state_dict(),
            'val_loss': val_loss / n_vl,
            'val_acc':  val_acc  / n_vl,
            'cfg': OmegaConf.to_container(cfg),
        }
        if val_loss / n_vl < best_val_loss:
            best_val_loss = val_loss / n_vl
            torch.save(ckpt, out_dir / 'best.pt')
            log.info(f"  ★ New best val_loss={best_val_loss:.4f}  acc={val_acc/n_vl:.3f}")

        if epoch % cfg.training.get('save_every', 10) == 0:
            torch.save(ckpt, out_dir / f'ckpt_epoch{epoch:03d}.pt')

    log.info(f"Stage 2.5 done.  Best val_loss={best_val_loss:.4f}")


if __name__ == '__main__':
    main()
