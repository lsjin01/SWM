#!/usr/bin/env python3
# scripts/train_stage2.py
"""
Stage 2: Transition Training (DDP 지원)
========================================
torchrun --nproc_per_node=4 scripts/train_stage2.py --config configs/stage2.yaml --no-wandb
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
from models.transition import SWMTransition, SpatialSWMTransition
from data.dataset import Stage2Dataset, MultiTaskStage2Dataset

log = logging.getLogger(__name__)

def setup_logging(log_file: str = None, resume: bool = False):
    """로그 설정. resume=True면 파일 append 모드."""
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


# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_stage1(ckpt_path: str, cfg, device):
    """Stage 1 체크포인트 로드 후 freeze."""
    sd = torch.load(ckpt_path, map_location=device)

    # checkpoint에 저장된 cfg 사용
    s1_cfg = OmegaConf.create(sd.get("cfg", {}))
    latent_dim  = s1_cfg.get("encoder", {}).get("latent_dim", cfg.encoder.latent_dim)
    max_nodes   = s1_cfg.get("encoder", {}).get("max_nodes", 8)
    grid_size   = s1_cfg.get("heads", {}).get("depth_head", {}).get("grid_size", 32)
    hidden_dims = list(s1_cfg.get("heads", {}).get("graph_head", {}).get("hidden_dims", [1024, 512]))
    dropout     = s1_cfg.get("heads", {}).get("graph_head", {}).get("dropout", 0.1)
    vla_path    = s1_cfg.get("encoder", {}).get("vla_path", None)

    spatial_dim = cfg.encoder.get("spatial_dim", 256)
    # DINOv2+SigLIP or V-JEPA-2
    if vla_path:
        encoder = SWMEncoder(
            vla_path=vla_path,
            freeze_backbone=True,
            latent_dim=latent_dim,
            spatial_dim=spatial_dim,
        ).to(device)
    else:
        encoder = SWMEncoder(latent_dim=latent_dim, spatial_dim=spatial_dim).to(device)

    heads = SWMHeads(
        latent_dim=latent_dim,
        max_nodes=max_nodes,
        grid_size=grid_size,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    # spatial_proj는 새로 추가된 파라미터라 stage1 ckpt에 없음 → strict=False
    encoder.load_state_dict(sd["encoder"], strict=False)
    heads.load_state_dict(sd["heads"])

    for p in encoder.parameters(): p.requires_grad = False
    for p in heads.parameters():   p.requires_grad = False
    encoder.eval()
    heads.eval()

    if is_main():
        log.info(f"Loaded & froze Stage 1 from {ckpt_path}")
    return encoder, heads


def get_tf_ratio(epoch: int, cfg) -> float:
    """Teacher forcing ratio: linear decay."""
    tf = cfg.training.teacher_forcing
    progress = min(epoch / tf.decay_epochs, 1.0)
    return tf.initial_ratio - (tf.initial_ratio - tf.final_ratio) * progress


def train_one_epoch(transition, encoder, heads,
                    loader, optimizer, scaler, cfg, device, epoch):
    transition.train()

    lw_g  = cfg.loss.lambda_graph
    lw_j  = cfg.loss.lambda_jepa
    tf_ratio  = get_tf_ratio(epoch, cfg)
    noise_std = cfg.training.noise_injection.std \
                if cfg.training.noise_injection.enabled else 0.0

    graph_fn = nn.SmoothL1Loss()
    jepa_fn  = nn.L1Loss() if cfg.loss.jepa_loss == "l1" else nn.MSELoss()

    total = gl_t = jl_t = 0.0

    for step, batch in enumerate(loader):
        image_t     = batch["image_t"].to(device)
        image_t1    = batch["image_t1"].to(device)
        action      = batch["action"].to(device)
        node_pos_t1 = batch["node_pos_t1"].to(device)
        node_mask_t1= batch["node_mask_t1"].to(device)

        with torch.no_grad():
            z_t  = encoder(image_t)    # (B, D)
            z_gt = encoder(image_t1)   # (B, D)  JEPA target

        with torch.amp.autocast("cuda", enabled=cfg.training.amp):
            # Noise injection
            z_in = z_t
            if noise_std > 0:
                z_in = z_t + torch.randn_like(z_t) * noise_std

            z_hat = transition(z_in, action, add_noise=False)  # ẑ_t+1

            # ℒ_graph + ℒ_depth via frozen heads
            raw_heads = heads.module if hasattr(heads, "module") else heads
            graph_pred, _ = raw_heads(z_hat)
            mask = node_mask_t1.unsqueeze(-1).float()
            gl = graph_fn(graph_pred * mask, node_pos_t1 * mask)

            # ℒ_JEPA
            jl = jepa_fn(z_hat, z_gt.detach())

            loss = lw_g * gl + lw_j * jl

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        raw_trans = transition.module if hasattr(transition, "module") else transition
        nn.utils.clip_grad_norm_(raw_trans.parameters(), cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        gl_t  += gl.item()
        jl_t  += jl.item()

        if is_main() and step % cfg.training.log_interval == 0:
            log.info(
                f"Epoch {epoch:03d}  Step {step:04d}/{len(loader)}"
                f"  loss={loss.item():.4f}"
                f"  graph={gl.item():.4f}"
                f"  jepa={jl.item():.4f}"
                f"  tf={tf_ratio:.2f}"
            )

    n = len(loader)
    return total/n, gl_t/n, jl_t/n


@torch.no_grad()
def evaluate(transition, encoder, heads, loader, cfg, device):
    transition.eval()
    graph_fn = nn.SmoothL1Loss()
    jepa_fn  = nn.L1Loss()
    total = gl_t = jl_t = 0.0

    for batch in loader:
        image_t     = batch["image_t"].to(device)
        image_t1    = batch["image_t1"].to(device)
        action      = batch["action"].to(device)
        node_pos_t1 = batch["node_pos_t1"].to(device)
        node_mask_t1= batch["node_mask_t1"].to(device)

        z_t  = encoder(image_t)
        z_gt = encoder(image_t1)
        z_hat = transition(z_t, action, add_noise=False)

        raw_heads = heads.module if hasattr(heads, "module") else heads
        graph_pred, _ = raw_heads(z_hat)
        mask = node_mask_t1.unsqueeze(-1).float()
        gl   = graph_fn(graph_pred * mask, node_pos_t1 * mask)
        jl   = jepa_fn(z_hat, z_gt)
        loss = cfg.loss.lambda_graph * gl \
             + cfg.loss.lambda_jepa  * jl

        total += loss.item()
        gl_t  += gl.item()
        jl_t  += jl.item()

    n = len(loader)
    return total/n, gl_t/n, jl_t/n


def train_one_epoch_spatial(transition, encoder, loader, optimizer, scaler, cfg, device, epoch):
    transition.train()
    encoder.spatial_proj.train()

    lw_j     = cfg.loss.lambda_jepa
    jepa_fn  = nn.L1Loss() if cfg.loss.jepa_loss == "l1" else nn.MSELoss()
    total    = 0.0

    for step, batch in enumerate(loader):
        image_t  = batch["image_t"].to(device)
        image_t1 = batch["image_t1"].to(device)
        action   = batch["action"].to(device)

        with torch.no_grad():
            # backbone is frozen; spatial_proj is trainable but computed inside autocast
            pass

        with torch.amp.autocast("cuda", enabled=cfg.training.amp):
            s_t  = encoder.encode_spatial_projected(image_t)   # (B, 256, d_s)
            s_gt = encoder.encode_spatial_projected(image_t1)  # (B, 256, d_s)

            s_in = s_t
            if cfg.training.noise_injection.enabled:
                s_in = s_t + torch.randn_like(s_t) * cfg.training.noise_injection.std

            s_hat = transition(s_in, action)
            loss  = lw_j * jepa_fn(s_hat, s_gt.detach())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        raw_trans = transition.module if hasattr(transition, "module") else transition
        nn.utils.clip_grad_norm_(
            list(raw_trans.parameters()) + list(encoder.spatial_proj.parameters()),
            cfg.training.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()

        if is_main() and step % cfg.training.log_interval == 0:
            log.info(
                f"Epoch {epoch:03d}  Step {step:04d}/{len(loader)}"
                f"  loss={loss.item():.4f}"
            )

    return total / max(len(loader), 1)


def evaluate_spatial(transition, encoder, loader, cfg, device):
    transition.eval()
    encoder.spatial_proj.eval()
    jepa_fn = nn.L1Loss() if cfg.loss.jepa_loss == "l1" else nn.MSELoss()
    total   = 0.0
    with torch.no_grad():
        for batch in loader:
            image_t  = batch["image_t"].to(device)
            image_t1 = batch["image_t1"].to(device)
            action   = batch["action"].to(device)
            with torch.amp.autocast("cuda", enabled=cfg.training.amp):
                s_t   = encoder.encode_spatial_projected(image_t)
                s_gt  = encoder.encode_spatial_projected(image_t1)
                s_hat = transition(s_t, action)
                loss  = jepa_fn(s_hat, s_gt.detach())
            total += loss.item()
    return total / max(len(loader), 1)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/stage2.yaml")
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from checkpoint path (e.g. outputs/stage2/multitask/ckpt_epoch010.pt)")
    args = parser.parse_args()

    cfg     = OmegaConf.load(args.config)
    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging 설정 (resume이면 append) ─────────────────────────────────────
    log_file = out_dir / "train.log"
    setup_logging(str(log_file), resume=bool(args.resume))

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

    torch.manual_seed(cfg.experiment.seed + local_rank)

    use_wandb = is_main() and not args.debug and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="swm",
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Frozen Stage 1 ────────────────────────────────────────────────────────
    encoder, heads = load_frozen_stage1(cfg.stage1_ckpt, cfg, device)

    # ── Data ─────────────────────────────────────────────────────────────────
    # Combined (MimicGen + RoboMimic)
    if hasattr(cfg.data, "robomimic_root") and cfg.data.get("use_robomimic", False):
        from data.dataset import CombinedStage2Dataset
        train_ds = CombinedStage2Dataset(
            mimicgen_root=cfg.data.data_root,
            robomimic_root=cfg.data.robomimic_root,
            split="train",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            seq_len=cfg.data.seq_len,
        )
        val_ds = CombinedStage2Dataset(
            mimicgen_root=cfg.data.data_root,
            robomimic_root=cfg.data.robomimic_root,
            split="val",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            seq_len=cfg.data.seq_len,
        )
    # MultiTask (MimicGen only)
    elif hasattr(cfg.data, "tasks"):
        train_ds = MultiTaskStage2Dataset(
            data_root=cfg.data.data_root,
            tasks=list(cfg.data.tasks),
            split="train",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            seq_len=cfg.data.seq_len,
        )
        val_ds = MultiTaskStage2Dataset(
            data_root=cfg.data.data_root,
            tasks=list(cfg.data.tasks),
            split="val",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            seq_len=cfg.data.seq_len,
        )
    # Single task
    else:
        train_ds = Stage2Dataset(
            data_root=cfg.data.data_root,
            task=cfg.data.task,
            split="train",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            seq_len=cfg.data.seq_len,
        )
        val_ds = Stage2Dataset(
            data_root=cfg.data.data_root,
            task=cfg.data.task,
            split="val",
            train_ratio=cfg.data.train_split,
            image_size=cfg.data.image_size,
            seq_len=cfg.data.seq_len,
        )

    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(256, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(64,  len(val_ds))))

    train_sampler = DistributedSampler(train_ds, shuffle=True)  if use_ddp else None
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
        log.info(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")

    # ── Transition ────────────────────────────────────────────────────────────
    use_spatial = cfg.transition.get("use_spatial", False)
    if use_spatial:
        spatial_dim = cfg.transition.get("spatial_dim", 256)
        transition = SpatialSWMTransition(
            spatial_dim=spatial_dim,
            action_dim=cfg.transition.action_dim,
            hidden_dim=cfg.transition.hidden_dim,
            num_layers=cfg.transition.num_layers,
            num_heads=cfg.transition.num_heads,
            dropout=cfg.transition.dropout,
        ).to(device)
        encoder.spatial_proj.to(device)
        for p in encoder.spatial_proj.parameters():
            p.requires_grad = True
    else:
        transition = SWMTransition(
            latent_dim=cfg.encoder.latent_dim,
            action_dim=cfg.transition.action_dim,
            noise_std=cfg.training.noise_injection.std
                      if cfg.training.noise_injection.enabled else 0.0,
        ).to(device)

    if use_ddp:
        transition = DDP(transition, device_ids=[local_rank])

    if is_main():
        raw_trans = transition.module if use_ddp else transition
        n = sum(p.numel() for p in raw_trans.parameters())
        log.info(f"Transition params: {n:,}")

    if use_spatial:
        opt_params = list((transition.module if use_ddp else transition).parameters()) + \
                     list(encoder.spatial_proj.parameters())
    else:
        opt_params = list((transition.module if use_ddp else transition).parameters())
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.amp)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt_r = torch.load(args.resume, map_location=device)
        raw_trans = transition.module if use_ddp else transition
        raw_trans.load_state_dict(ckpt_r["transition"])
        start_epoch = ckpt_r.get("epoch", 0) + 1
        best_val    = ckpt_r.get("val_loss", float("inf"))
        if is_main():
            log.info(f"Resumed from {args.resume}  "
                     f"epoch={start_epoch-1}  val_loss={best_val:.4f}")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        if use_spatial:
            tr_loss = train_one_epoch_spatial(
                transition, encoder, train_loader, optimizer, scaler, cfg, device, epoch
            )
            vl_loss = evaluate_spatial(transition, encoder, val_loader, cfg, device)
            tr = (tr_loss, 0.0, tr_loss)
            vl = (vl_loss, 0.0, vl_loss)
        else:
            tr = train_one_epoch(
                transition, encoder, heads,
                train_loader, optimizer, scaler, cfg, device, epoch
            )
            vl = evaluate(transition, encoder, heads, val_loader, cfg, device)
        scheduler.step()

        if is_main():
            log.info(
                f"[Epoch {epoch:03d}]  "
                f"train={tr[0]:.4f} (g={tr[1]:.4f} j={tr[2]:.4f})  "
                f"val={vl[0]:.4f} (g={vl[1]:.4f} j={vl[2]:.4f})"
            )

            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss": tr[0], "train/graph": tr[1], "train/jepa": tr[2],
                    "val/loss": vl[0], "val/graph": vl[1], "val/jepa": vl[2],
                    "epoch": epoch,
                })

            raw_trans = transition.module if use_ddp else transition
            ckpt = {
                "epoch":      epoch,
                "transition": raw_trans.state_dict(),
                "val_loss":   vl[0],
                "cfg":        OmegaConf.to_container(cfg),
            }
            if use_spatial:
                ckpt["spatial_proj"] = encoder.spatial_proj.state_dict()
                ckpt["spatial_dim"]  = cfg.transition.get("spatial_dim", 256)

            if epoch % cfg.training.save_interval == 0:
                torch.save(ckpt, out_dir / f"ckpt_epoch{epoch:03d}.pt")

            if vl[0] < best_val:
                best_val = vl[0]
                torch.save(ckpt, out_dir / "best.pt")
                log.info(f"  ★ New best val_loss={best_val:.4f}")

    if is_main():
        log.info(f"Stage 2 done.  Best val_loss={best_val:.4f}")

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()