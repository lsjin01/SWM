#!/usr/bin/env python3
# scripts/train_stage1.py
"""
Stage 1: Encoder Training (DDP 지원)
=====================================
torchrun --nproc_per_node=4 scripts/train_stage1.py --config configs/stage1.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import SWMEncoder
from models.heads import SWMHeads
from data.dataset import Stage1Dataset, MultiTaskStage1Dataset

log = logging.getLogger(__name__)

def setup_logging(log_file: str = None, resume: bool = False):
    handlers = [logging.StreamHandler()]
    if log_file:
        mode = "a" if resume else "w"
        handlers.append(logging.FileHandler(log_file, mode=mode))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=handlers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def reduce_mean(tensor):
    """모든 프로세스의 평균값을 main rank로 모음."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / dist.get_world_size()


# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(encoder, heads, loader, optimizer, scaler,
                    cfg, device, epoch):
    encoder.train()
    heads.train()
    graph_fn = nn.SmoothL1Loss()
    lw_g = cfg.loss.lambda_graph

    total = g_total = 0.0

    for step, batch in enumerate(loader):
        image     = batch["image"].to(device)
        node_pos  = batch["node_pos"].to(device)
        node_mask = batch["node_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=cfg.training.amp):
            z = encoder(image)
            graph_pred, _ = heads(z)

            mask = node_mask.unsqueeze(-1).float()
            gl   = graph_fn(graph_pred * mask, node_pos * mask)
            loss = lw_g * gl

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # unwrap DDP for grad clipping
        raw_enc = encoder.module if hasattr(encoder, "module") else encoder
        raw_hds = heads.module   if hasattr(heads,   "module") else heads
        nn.utils.clip_grad_norm_(
            list(raw_enc.parameters()) + list(raw_hds.parameters()),
            cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        total   += loss.item()
        g_total += gl.item()

        if is_main() and step % cfg.training.log_interval == 0:
            log.info(f"Epoch {epoch:03d}  Step {step:04d}/{len(loader)}"
                     f"  loss={loss.item():.4f}"
                     f"  graph={gl.item():.4f}")

    n = len(loader)
    return total/n, g_total/n


@torch.no_grad()
def evaluate(encoder, heads, loader, cfg, device):
    encoder.eval()
    heads.eval()
    graph_fn = nn.SmoothL1Loss()
    total = g_total = 0.0

    for batch in loader:
        image     = batch["image"].to(device)
        node_pos  = batch["node_pos"].to(device)
        node_mask = batch["node_mask"].to(device)

        z = encoder(image)
        graph_pred, _ = heads(z)
        mask = node_mask.unsqueeze(-1).float()
        gl   = graph_fn(graph_pred * mask, node_pos * mask)
        loss = cfg.loss.lambda_graph * gl

        total   += loss.item()
        g_total += gl.item()

    n = len(loader)
    return total/n, g_total/n


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from checkpoint path")
    args = parser.parse_args()

    cfg     = OmegaConf.load(args.config)
    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging 설정 ──────────────────────────────────────────────────────────
    setup_logging(str(out_dir / "train.log"), resume=bool(args.resume))

    # ── DDP 초기화 ────────────────────────────────────────────────────────────
    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    if is_main():
        log.info(f"Device: {device}  DDP: {use_ddp}  "
                 f"World: {dist.get_world_size() if use_ddp else 1}")
        log.info(f"Output: {out_dir}")

    torch.manual_seed(cfg.experiment.seed + local_rank)

    # ── WandB (main rank만) ──────────────────────────────────────────────────
    use_wandb = is_main() and not args.debug and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="swm",
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Data ─────────────────────────────────────────────────────────────────
    # multitask: cfg.data.tasks 있으면 MultiTask, 없으면 단일 task
    if hasattr(cfg.data, "tasks"):
        train_ds = MultiTaskStage1Dataset(
            data_root=cfg.data.data_root,
            tasks=list(cfg.data.tasks),
            split="train",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            max_nodes=cfg.encoder.max_nodes,
            grid_size=32,
        )
        val_ds = MultiTaskStage1Dataset(
            data_root=cfg.data.data_root,
            tasks=list(cfg.data.tasks),
            split="val",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            max_nodes=cfg.encoder.max_nodes,
            grid_size=32,
        )
    else:
        train_ds = Stage1Dataset(
            data_root=cfg.data.data_root,
            task=cfg.data.task,
            split="train",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            max_nodes=cfg.encoder.max_nodes,
            grid_size=32,
        )
        val_ds = Stage1Dataset(
            data_root=cfg.data.data_root,
            task=cfg.data.task,
            split="val",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            max_nodes=cfg.encoder.max_nodes,
            grid_size=32,
        )

    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(256, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(64,  len(val_ds))))

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    if is_main():
        log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    # DINOv2+SigLIP encoder or V-JEPA-2 encoder
    if hasattr(cfg.encoder, "vla_path"):
        encoder = SWMEncoder(
            vla_path=cfg.encoder.vla_path,
            freeze_backbone=cfg.encoder.freeze_backbone,
            latent_dim=cfg.encoder.latent_dim,
        ).to(device)
    else:
        encoder = SWMEncoder(
            vjepa2_ckpt=cfg.encoder.get("vjepa2_ckpt", None),
            freeze_backbone=cfg.encoder.freeze_backbone,
            latent_dim=cfg.encoder.latent_dim,
        ).to(device)

    heads = SWMHeads(
        latent_dim=cfg.encoder.latent_dim,
        max_nodes=cfg.encoder.max_nodes,
        grid_size=32,
        hidden_dims=list(cfg.heads.graph_head.hidden_dims),
        dropout=cfg.heads.graph_head.dropout,
    ).to(device)

    if use_ddp:
        # encoder는 완전 frozen이면 DDP 불필요 (trainable params 있을 때만)
        enc_trainable = any(p.requires_grad for p in encoder.parameters())
        if enc_trainable:
            encoder = DDP(encoder, device_ids=[local_rank])
        heads = DDP(heads, device_ids=[local_rank])

    if is_main():
        raw_enc = encoder.module if hasattr(encoder, "module") else encoder
        raw_hds = heads.module   if hasattr(heads,   "module") else heads
        n_enc   = sum(p.numel() for p in raw_enc.parameters() if p.requires_grad)
        n_heads = sum(p.numel() for p in raw_hds.parameters())
        log.info(f"Encoder trainable: {n_enc:,}")
        log.info(f"Heads params:      {n_heads:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    raw_enc = encoder.module if hasattr(encoder, "module") else encoder
    raw_hds = heads.module   if hasattr(heads,   "module") else heads
    backbone_params = [p for p in raw_enc.backbone.parameters() if p.requires_grad]
    proj_params     = list(raw_enc.proj.parameters()) + list(raw_hds.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.training.lr_backbone},
        {"params": proj_params,     "lr": cfg.training.lr},
    ], weight_decay=cfg.training.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.amp)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt_r  = torch.load(args.resume, map_location=device)
        raw_enc = encoder.module if hasattr(encoder, "module") else encoder
        raw_hds = heads.module   if hasattr(heads,   "module") else heads
        raw_enc.load_state_dict(ckpt_r["encoder"])
        raw_hds.load_state_dict(ckpt_r["heads"])
        start_epoch = ckpt_r.get("epoch", 0) + 1
        best_val    = ckpt_r.get("val_loss", float("inf"))
        if is_main():
            log.info(f"Resumed from {args.resume}  "
                     f"epoch={start_epoch-1}  val_loss={best_val:.4f}")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        tr_loss, tr_gl = train_one_epoch(
            encoder, heads, train_loader, optimizer, scaler, cfg, device, epoch
        )
        val_loss, val_gl = evaluate(
            encoder, heads, val_loader, cfg, device
        )
        scheduler.step()

        if is_main():
            lr_now = optimizer.param_groups[1]["lr"]
            log.info(
                f"[Epoch {epoch:03d}]  "
                f"train={tr_loss:.4f} (g={tr_gl:.4f})  "
                f"val={val_loss:.4f} (g={val_gl:.4f})  "
                f"lr={lr_now:.2e}"
            )

            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss": tr_loss, "train/graph": tr_gl,
                    "val/loss": val_loss,  "val/graph":   val_gl,
                    "lr": lr_now, "epoch": epoch,
                })

            # Checkpoint
            raw_enc = encoder.module if hasattr(encoder, "module") else encoder
            raw_hds = heads.module   if hasattr(heads,   "module") else heads
            ckpt = {
                "epoch":   epoch,
                "encoder": raw_enc.state_dict(),
                "heads":   raw_hds.state_dict(),
                "val_loss": val_loss,
                "cfg":     OmegaConf.to_container(cfg),
            }

            if epoch % cfg.training.save_interval == 0:
                torch.save(ckpt, out_dir / f"ckpt_epoch{epoch:03d}.pt")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, out_dir / "best.pt")
                log.info(f"  ★ New best val_loss={best_val:.4f}")

    if is_main():
        log.info(f"Stage 1 done.  Best val_loss={best_val:.4f}")

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()