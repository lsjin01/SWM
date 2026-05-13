#!/usr/bin/env python3
# scripts/train_stage2.py
"""
Stage 2: Transition Training
  z_t + a_t → Transition → ẑ_t+1
  Loss: ℒ_graph + ℒ_depth + ℒ_JEPA
  Encoder + Heads: frozen (from Stage 1)
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import SWMEncoder
from models.heads import SWMHeads
from models.transition import SWMTransition
from data.dataset import Stage2Dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_stage1(ckpt_path: str, cfg, device):
    encoder = SWMEncoder(
        backbone=cfg.encoder.backbone if hasattr(cfg, "encoder") else "vjepa2",
        use_graph_branch=True,
    ).to(device)
    heads = SWMHeads(latent_dim=encoder.latent_dim).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(sd["encoder"])
    heads.load_state_dict(sd["heads"])

    for p in encoder.parameters():
        p.requires_grad = False
    for p in heads.parameters():
        p.requires_grad = False

    encoder.eval()
    heads.eval()
    log.info(f"Loaded & froze Stage 1 from {ckpt_path}")
    return encoder, heads


def get_tf_ratio(epoch: int, cfg) -> float:
    """Linear decay from initial_ratio to final_ratio over decay_epochs."""
    tf_cfg = cfg.training.teacher_forcing
    ratio = tf_cfg.initial_ratio - (tf_cfg.initial_ratio - tf_cfg.final_ratio) * \
            min(epoch / tf_cfg.decay_epochs, 1.0)
    return ratio


def train_one_epoch(transition, encoder, heads, loader, optimizer,
                    cfg, device, scaler, epoch):
    transition.train()
    lm_graph = cfg.loss.lambda_graph
    lm_depth = cfg.loss.lambda_depth
    lm_jepa  = cfg.loss.lambda_jepa
    tf_ratio = get_tf_ratio(epoch, cfg)

    graph_fn = nn.SmoothL1Loss()
    depth_fn = nn.SmoothL1Loss()
    jepa_fn  = nn.L1Loss() if cfg.loss.jepa_loss == "l1" else nn.MSELoss()

    total = total_gl = total_dl = total_jl = 0.0

    for step, batch in enumerate(loader):
        image_t    = batch["image_t"].to(device)
        graph_t    = batch["graph_feat_t"].to(device)
        mask_t     = batch["node_mask_t"].to(device)
        action     = batch["action"].to(device)

        image_t1   = batch["image_t1"].to(device)
        graph_t1   = batch["graph_feat_t1"].to(device)
        mask_t1    = batch["node_mask_t1"].to(device)
        node_pos_t1 = batch["node_pos_t1"].to(device)
        depth_t1   = batch["sparse_depth_t1"].to(device)

        with torch.no_grad():
            z_t  = encoder(image_t,  graph_t,  mask_t)     # (B, D)
            z_gt = encoder(image_t1, graph_t1, mask_t1)    # (B, D)  JEPA target

        with torch.cuda.amp.autocast(enabled=cfg.training.amp):
            z_hat = transition(z_t, action, training=True)  # ẑ_t+1 (B, D)

            # ℒ_graph + ℒ_depth via frozen heads
            graph_pred, depth_pred = heads(z_hat)
            mk = mask_t1.unsqueeze(-1).float()
            gl = graph_fn(graph_pred * mk, node_pos_t1 * mk)
            dl = depth_fn(depth_pred, depth_t1)

            # ℒ_JEPA: predicted latent vs real next-frame latent
            jl = jepa_fn(z_hat, z_gt.detach())

            loss = lm_graph * gl + lm_depth * dl + lm_jepa * jl

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            transition.parameters(), cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        total   += loss.item()
        total_gl += gl.item()
        total_dl += dl.item()
        total_jl += jl.item()

        if step % cfg.training.log_interval == 0:
            log.info(f"Epoch {epoch}  Step {step}/{len(loader)}"
                     f"  loss={loss.item():.4f}"
                     f"  graph={gl.item():.4f}"
                     f"  depth={dl.item():.4f}"
                     f"  jepa={jl.item():.4f}"
                     f"  tf={tf_ratio:.2f}")
            wandb.log({
                "train/loss": loss.item(),
                "train/graph": gl.item(),
                "train/depth": dl.item(),
                "train/jepa":  jl.item(),
                "train/tf_ratio": tf_ratio,
                "epoch": epoch,
            })

    n = len(loader)
    return total/n, total_gl/n, total_dl/n, total_jl/n


@torch.no_grad()
def evaluate(transition, encoder, heads, loader, cfg, device):
    transition.eval()
    graph_fn = nn.SmoothL1Loss()
    depth_fn = nn.SmoothL1Loss()
    jepa_fn  = nn.L1Loss()
    total = total_gl = total_dl = total_jl = 0.0

    for batch in loader:
        image_t   = batch["image_t"].to(device)
        graph_t   = batch["graph_feat_t"].to(device)
        mask_t    = batch["node_mask_t"].to(device)
        action    = batch["action"].to(device)
        image_t1  = batch["image_t1"].to(device)
        graph_t1  = batch["graph_feat_t1"].to(device)
        mask_t1   = batch["node_mask_t1"].to(device)
        node_pos_t1 = batch["node_pos_t1"].to(device)
        depth_t1  = batch["sparse_depth_t1"].to(device)

        z_t  = encoder(image_t,  graph_t,  mask_t)
        z_gt = encoder(image_t1, graph_t1, mask_t1)
        z_hat = transition(z_t, action, training=False)

        graph_pred, depth_pred = heads(z_hat)
        mk = mask_t1.unsqueeze(-1).float()
        gl = graph_fn(graph_pred * mk, node_pos_t1 * mk)
        dl = depth_fn(depth_pred, depth_t1)
        jl = jepa_fn(z_hat, z_gt)
        loss = cfg.loss.lambda_graph * gl + cfg.loss.lambda_depth * dl \
             + cfg.loss.lambda_jepa  * jl

        total   += loss.item()
        total_gl += gl.item()
        total_dl += dl.item()
        total_jl += jl.item()

    n = len(loader)
    return total/n, total_gl/n, total_dl/n, total_jl/n


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2.yaml")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    cfg     = OmegaConf.load(args.config)
    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.debug:
        wandb.init(project="swm", name=cfg.experiment.name,
                   config=OmegaConf.to_container(cfg))

    # ── Frozen Stage 1 ───────────────────────────────────────────────────────
    encoder, heads = load_frozen_stage1(cfg.stage1_ckpt, cfg, device)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = Stage2Dataset(
        data_root=cfg.data.data_root, task=cfg.data.task,
        split="train", train_ratio=cfg.data.train_split,
        image_size=cfg.data.image_size, seq_len=cfg.data.seq_len,
    )
    val_ds = Stage2Dataset(
        data_root=cfg.data.data_root, task=cfg.data.task,
        split="val",   train_ratio=cfg.data.train_split,
        image_size=cfg.data.image_size, seq_len=cfg.data.seq_len,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True,  num_workers=cfg.data.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=cfg.data.num_workers,
                              pin_memory=True)
    log.info(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")

    # ── Transition ───────────────────────────────────────────────────────────
    transition = SWMTransition(
        latent_dim=encoder.latent_dim,
        action_dim=cfg.transition.action_dim,
        hidden_dim=cfg.transition.hidden_dim,
        num_layers=cfg.transition.num_layers,
        num_heads=cfg.transition.num_heads,
        dropout=cfg.transition.dropout,
        noise_std=cfg.training.noise_injection.std
        if cfg.training.noise_injection.enabled else 0.0,
        vjepa2_ac_ckpt=cfg.transition.get("vjepa2_ac_ckpt", None),
        freeze=cfg.transition.freeze_vjepa2_ac,
    ).to(device)

    log.info(f"Transition params: "
             f"{sum(p.numel() for p in transition.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, transition.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val = float("inf")
    for epoch in range(1, cfg.training.epochs + 1):
        tr = train_one_epoch(
            transition, encoder, heads, train_loader,
            optimizer, cfg, device, scaler, epoch
        )
        vl = evaluate(transition, encoder, heads, val_loader, cfg, device)
        scheduler.step()

        log.info(f"[Epoch {epoch}]  train={tr[0]:.4f}  val={vl[0]:.4f}"
                 f"  (g={vl[1]:.4f} d={vl[2]:.4f} j={vl[3]:.4f})")
        wandb.log({"val/loss": vl[0], "val/graph": vl[1],
                   "val/depth": vl[2], "val/jepa": vl[3], "epoch": epoch})

        if epoch % cfg.training.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "transition": transition.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   vl[0],
            }, out_dir / f"ckpt_epoch{epoch:03d}.pt")

        if vl[0] < best_val:
            best_val = vl[0]
            torch.save({
                "epoch": epoch,
                "transition": transition.state_dict(),
                "val_loss":   vl[0],
            }, out_dir / "best.pt")
            log.info(f"  ✓ New best val_loss: {best_val:.4f}")

    log.info("Stage 2 complete.")


if __name__ == "__main__":
    main()
