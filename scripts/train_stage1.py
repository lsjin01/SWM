#!/usr/bin/env python3
# scripts/train_stage1.py
"""
Stage 1: Encoder Training
  Encoder(O_t, G_t) → z_t → [Graph Head, Depth Head]
  Loss: ℒ_graph + ℒ_depth
"""

import os
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
from data.dataset import Stage1Dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def build_loss(cfg):
    if cfg.loss.graph_loss == "l2":
        graph_fn = nn.MSELoss()
    elif cfg.loss.graph_loss == "l1":
        graph_fn = nn.L1Loss()
    else:
        graph_fn = nn.SmoothL1Loss()

    depth_fn = nn.SmoothL1Loss()
    return graph_fn, depth_fn


def train_one_epoch(encoder, heads, loader, optimizer, graph_fn, depth_fn,
                    cfg, device, scaler, epoch):
    encoder.train()
    heads.train()

    total_loss = total_gl = total_dl = 0.0
    for step, batch in enumerate(loader):
        image      = batch["image"].to(device)
        graph_feat = batch["graph_feat"].to(device)
        node_mask  = batch["node_mask"].to(device)
        node_pos   = batch["node_pos"].to(device)        # (B, N, 3) GT positions
        depth_gt   = batch["sparse_depth"].to(device)   # (B, G, G)

        with torch.cuda.amp.autocast(enabled=cfg.training.amp):
            z = encoder(image, graph_feat, node_mask)    # (B, latent_dim)
            graph_pred, depth_pred = heads(z)            # (B,N,3), (B,G,G)

            # Masked graph loss
            mask = node_mask.unsqueeze(-1).float()       # (B, N, 1)
            gl = graph_fn(graph_pred * mask, node_pos * mask)
            dl = depth_fn(depth_pred, depth_gt)

            loss = cfg.loss.lambda_graph * gl + cfg.loss.lambda_depth * dl

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(heads.parameters()),
            cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_gl   += gl.item()
        total_dl   += dl.item()

        if step % cfg.training.log_interval == 0:
            log.info(f"Epoch {epoch}  Step {step}/{len(loader)}"
                     f"  loss={loss.item():.4f}"
                     f"  graph={gl.item():.4f}"
                     f"  depth={dl.item():.4f}")
            wandb.log({
                "train/loss": loss.item(),
                "train/graph_loss": gl.item(),
                "train/depth_loss": dl.item(),
                "epoch": epoch, "step": step,
            })

    n = len(loader)
    return total_loss / n, total_gl / n, total_dl / n


@torch.no_grad()
def evaluate(encoder, heads, loader, graph_fn, depth_fn, cfg, device):
    encoder.eval()
    heads.eval()
    total_loss = total_gl = total_dl = 0.0

    for batch in loader:
        image      = batch["image"].to(device)
        graph_feat = batch["graph_feat"].to(device)
        node_mask  = batch["node_mask"].to(device)
        node_pos   = batch["node_pos"].to(device)
        depth_gt   = batch["sparse_depth"].to(device)

        z = encoder(image, graph_feat, node_mask)
        graph_pred, depth_pred = heads(z)

        mask = node_mask.unsqueeze(-1).float()
        gl = graph_fn(graph_pred * mask, node_pos * mask)
        dl = depth_fn(depth_pred, depth_gt)
        loss = cfg.loss.lambda_graph * gl + cfg.loss.lambda_depth * dl

        total_loss += loss.item()
        total_gl   += gl.item()
        total_dl   += dl.item()

    n = len(loader)
    return total_loss / n, total_gl / n, total_dl / n


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── WandB ────────────────────────────────────────────────────────────────
    if not args.debug:
        wandb.init(project="swm", name=cfg.experiment.name, config=OmegaConf.to_container(cfg))

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = Stage1Dataset(
        data_root=cfg.data.data_root,
        task=cfg.data.task,
        split="train",
        train_ratio=cfg.data.train_split,
        image_size=cfg.data.image_size,
        max_nodes=cfg.encoder.get("max_nodes", 8),
        grid_size=cfg.heads.depth_head.grid_size,
    )
    val_ds = Stage1Dataset(
        data_root=cfg.data.data_root,
        task=cfg.data.task,
        split="val",
        train_ratio=cfg.data.train_split,
        image_size=cfg.data.image_size,
        max_nodes=cfg.encoder.get("max_nodes", 8),
        grid_size=cfg.heads.depth_head.grid_size,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.data.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=cfg.data.num_workers,
                              pin_memory=True)
    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    encoder = SWMEncoder(
        backbone=cfg.encoder.backbone,
        backbone_ckpt=cfg.encoder.get("vjepa2_ckpt", None),
        freeze_backbone=cfg.encoder.freeze_backbone,
        use_graph_branch=cfg.encoder.use_graph_branch,
        graph_hidden_dim=cfg.encoder.graph_hidden_dim,
        graph_embed_dim=cfg.encoder.graph_embed_dim,
    ).to(device)

    heads = SWMHeads(
        latent_dim=encoder.latent_dim,
        max_nodes=cfg.encoder.get("max_nodes", 8),
        grid_size=cfg.heads.depth_head.grid_size,
        hidden_dims=list(cfg.heads.graph_head.hidden_dims),
        dropout=cfg.heads.graph_head.dropout,
    ).to(device)

    log.info(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    log.info(f"Heads params:   {sum(p.numel() for p in heads.parameters()):,}")

    # ── Optimiser (separate LR for backbone vs heads) ─────────────────────────
    backbone_params = list(encoder.visual.parameters())
    other_params    = (list(encoder.graph_enc.parameters())
                       + list(encoder.proj.parameters())
                       + list(heads.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.training.lr_backbone},
        {"params": other_params,    "lr": cfg.training.lr},
    ], weight_decay=cfg.training.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)
    graph_fn, depth_fn = build_loss(cfg)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val = float("inf")
    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_gl, tr_dl = train_one_epoch(
            encoder, heads, train_loader, optimizer,
            graph_fn, depth_fn, cfg, device, scaler, epoch
        )
        val_loss, val_gl, val_dl = evaluate(
            encoder, heads, val_loader, graph_fn, depth_fn, cfg, device
        )
        scheduler.step()

        log.info(f"[Epoch {epoch}]  train={tr_loss:.4f}  val={val_loss:.4f}")
        wandb.log({"val/loss": val_loss, "val/graph": val_gl,
                   "val/depth": val_dl, "epoch": epoch})

        # Save checkpoint
        if epoch % cfg.training.save_interval == 0:
            ckpt = {
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "heads": heads.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(ckpt, out_dir / f"ckpt_epoch{epoch:03d}.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "heads": heads.state_dict(),
                "val_loss": val_loss,
            }, out_dir / "best.pt")
            log.info(f"  ✓ New best val_loss: {best_val:.4f}")

    log.info("Stage 1 training complete.")
    log.info(f"Best val loss: {best_val:.4f}  →  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
