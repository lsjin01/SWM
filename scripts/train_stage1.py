#!/usr/bin/env python3
# scripts/train_stage1.py
"""
Stage 1: Encoder Training
=========================
V-JEPA-2 encoder → z_t → Graph Head + Depth Head
Loss: ℒ_graph + ℒ_depth

완료 후: Encoder + Heads를 freeze하여 Stage 2에 전달
"""

import sys
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
from data.dataset import Stage1Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(encoder, heads, loader, optimizer, scaler,
                    cfg, device, epoch):
    encoder.train()
    heads.train()
    graph_fn = nn.SmoothL1Loss()
    depth_fn = nn.SmoothL1Loss()
    lw_g = cfg.loss.lambda_graph
    lw_d = cfg.loss.lambda_depth

    total = g_total = d_total = 0.0

    for step, batch in enumerate(loader):
        image     = batch["image"].to(device)
        node_pos  = batch["node_pos"].to(device)     # (B, N, 3)  GT positions
        node_mask = batch["node_mask"].to(device)    # (B, N) bool
        depth_gt  = batch["sparse_depth"].to(device) # (B, G, G)

        with torch.cuda.amp.autocast(enabled=cfg.training.amp):
            z = encoder(image)                       # (B, latent_dim)
            graph_pred, depth_pred = heads(z)        # (B,N,3), (B,G,G)

            # Masked graph loss — only valid nodes
            mask = node_mask.unsqueeze(-1).float()   # (B, N, 1)
            gl   = graph_fn(graph_pred * mask, node_pos * mask)
            dl   = depth_fn(depth_pred, depth_gt)
            loss = lw_g * gl + lw_d * dl

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(heads.parameters()),
            cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        total   += loss.item()
        g_total += gl.item()
        d_total += dl.item()

        if step % cfg.training.log_interval == 0:
            log.info(f"Epoch {epoch:03d}  Step {step:04d}/{len(loader)}"
                     f"  loss={loss.item():.4f}"
                     f"  graph={gl.item():.4f}"
                     f"  depth={dl.item():.4f}")

    n = len(loader)
    return total/n, g_total/n, d_total/n


@torch.no_grad()
def evaluate(encoder, heads, loader, cfg, device):
    encoder.eval()
    heads.eval()
    graph_fn = nn.SmoothL1Loss()
    depth_fn = nn.SmoothL1Loss()
    total = g_total = d_total = 0.0

    for batch in loader:
        image     = batch["image"].to(device)
        node_pos  = batch["node_pos"].to(device)
        node_mask = batch["node_mask"].to(device)
        depth_gt  = batch["sparse_depth"].to(device)

        z = encoder(image)
        graph_pred, depth_pred = heads(z)
        mask = node_mask.unsqueeze(-1).float()
        gl   = graph_fn(graph_pred * mask, node_pos * mask)
        dl   = depth_fn(depth_pred, depth_gt)
        loss = cfg.loss.lambda_graph * gl + cfg.loss.lambda_depth * dl

        total   += loss.item()
        g_total += gl.item()
        d_total += dl.item()

    n = len(loader)
    return total/n, g_total/n, d_total/n


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--debug",  action="store_true",
                        help="skip wandb, use small dataset")
    args = parser.parse_args()

    cfg     = OmegaConf.load(args.config)
    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Output: {out_dir}")

    # ── WandB ────────────────────────────────────────────────────────────────
    if not args.debug:
        import wandb
        wandb.init(
            project="swm",
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = Stage1Dataset(
        data_root=cfg.data.data_root,
        task=cfg.data.task,
        split="train",
        train_ratio=cfg.data.train_split,
        image_size=cfg.data.image_size,
        max_nodes=cfg.encoder.max_nodes,
        grid_size=cfg.heads.depth_head.grid_size,
    )
    val_ds = Stage1Dataset(
        data_root=cfg.data.data_root,
        task=cfg.data.task,
        split="val",
        train_ratio=cfg.data.train_split,
        image_size=cfg.data.image_size,
        max_nodes=cfg.encoder.max_nodes,
        grid_size=cfg.heads.depth_head.grid_size,
    )

    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(200, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(50,  len(val_ds))))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    encoder = SWMEncoder(
        vjepa2_ckpt=cfg.encoder.get("vjepa2_ckpt", None),
        freeze_backbone=cfg.encoder.freeze_backbone,
        latent_dim=cfg.encoder.latent_dim,
    ).to(device)

    heads = SWMHeads(
        latent_dim=cfg.encoder.latent_dim,
        max_nodes=cfg.encoder.max_nodes,
        grid_size=cfg.heads.depth_head.grid_size,
        hidden_dims=list(cfg.heads.graph_head.hidden_dims),
        dropout=cfg.heads.graph_head.dropout,
    ).to(device)

    n_enc   = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_heads = sum(p.numel() for p in heads.parameters())
    log.info(f"Encoder trainable: {n_enc:,}")
    log.info(f"Heads params:      {n_heads:,}")

    # ── Optimizer: 백본은 낮은 LR, 헤드는 높은 LR ─────────────────────────
    backbone_params = list(encoder.backbone.parameters())
    proj_params     = list(encoder.proj.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.training.lr_backbone},
        {"params": proj_params,     "lr": cfg.training.lr},
    ], weight_decay=cfg.training.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_gl, tr_dl = train_one_epoch(
            encoder, heads, train_loader, optimizer, scaler, cfg, device, epoch
        )
        val_loss, val_gl, val_dl = evaluate(
            encoder, heads, val_loader, cfg, device
        )
        scheduler.step()

        lr_now = optimizer.param_groups[1]["lr"]
        log.info(
            f"[Epoch {epoch:03d}]  "
            f"train={tr_loss:.4f} (g={tr_gl:.4f} d={tr_dl:.4f})  "
            f"val={val_loss:.4f} (g={val_gl:.4f} d={val_dl:.4f})  "
            f"lr={lr_now:.2e}"
        )

        if not args.debug:
            wandb.log({
                "train/loss": tr_loss, "train/graph": tr_gl, "train/depth": tr_dl,
                "val/loss":   val_loss,"val/graph":   val_gl,"val/depth":   val_dl,
                "lr": lr_now, "epoch": epoch,
            })

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt = {
            "epoch":   epoch,
            "encoder": encoder.state_dict(),
            "heads":   heads.state_dict(),
            "val_loss": val_loss,
            "cfg":     OmegaConf.to_container(cfg),
        }

        if epoch % cfg.training.save_interval == 0:
            torch.save(ckpt, out_dir / f"ckpt_epoch{epoch:03d}.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            log.info(f"  ★ New best val_loss={best_val:.4f}  →  {out_dir}/best.pt")

    log.info(f"Stage 1 done.  Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
