#!/usr/bin/env python3
# scripts/train_stage2.py
"""
Stage 2: Transition Training
=============================
z_t + a_t → SWMTransition → ẑ_t+1

Loss:
  ℒ_graph  : frozen Graph Head (ẑ_t+1 → Ĝ_t+1 vs GT)
  ℒ_depth  : frozen Depth Head (ẑ_t+1 → D̂_t+1 vs GT)
  ℒ_JEPA   : ‖ẑ_t+1 − z*_t+1‖  (z*_t+1 = frozen encoder(img_t+1))

Teacher Forcing: 1.0 → 0.5 over decay_epochs
Noise Injection: z_t에 Gaussian noise 추가

backprop: Transition만 (Encoder/Heads frozen)
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
from models.transition import SWMTransition
from data.dataset import Stage2Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_stage1(ckpt_path: str, cfg, device):
    """Stage 1 체크포인트에서 Encoder + Heads 로드 후 freeze."""
    encoder = SWMEncoder(
        latent_dim=cfg.encoder.latent_dim
        if hasattr(cfg, "encoder") else 1024,
    ).to(device)
    heads = SWMHeads(
        latent_dim=encoder.latent_dim
    ).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(sd["encoder"])
    heads.load_state_dict(sd["heads"])

    for p in encoder.parameters(): p.requires_grad = False
    for p in heads.parameters():   p.requires_grad = False
    encoder.eval()
    heads.eval()

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

    lw_g   = cfg.loss.lambda_graph
    lw_d   = cfg.loss.lambda_depth
    lw_j   = cfg.loss.lambda_jepa
    tf_ratio = get_tf_ratio(epoch, cfg)
    noise_std = cfg.training.noise_injection.std \
                if cfg.training.noise_injection.enabled else 0.0

    graph_fn = nn.SmoothL1Loss()
    depth_fn = nn.SmoothL1Loss()
    jepa_fn  = nn.L1Loss() if cfg.loss.jepa_loss == "l1" else nn.MSELoss()

    total = gl_total = dl_total = jl_total = 0.0

    for step, batch in enumerate(loader):
        image_t   = batch["image_t"].to(device)
        image_t1  = batch["image_t1"].to(device)
        action    = batch["action"].to(device)
        node_pos_t1 = batch["node_pos_t1"].to(device)
        node_mask_t1= batch["node_mask_t1"].to(device)
        depth_t1  = batch["sparse_depth_t1"].to(device)

        with torch.no_grad():
            z_t  = encoder(image_t)   # (B, D)
            z_gt = encoder(image_t1)  # (B, D)  JEPA target

        with torch.cuda.amp.autocast(enabled=cfg.training.amp):
            # Noise injection on input
            z_in = z_t
            if noise_std > 0:
                z_in = z_t + torch.randn_like(z_t) * noise_std

            # Teacher forcing: z_in vs z_gt for multi-step
            # (single-step here; multi-step TF handled in rollout)
            z_hat = transition(z_in, action, add_noise=False)  # ẑ_t+1

            # ℒ_graph + ℒ_depth via frozen heads
            graph_pred, depth_pred = heads(z_hat)
            mask = node_mask_t1.unsqueeze(-1).float()
            gl = graph_fn(graph_pred * mask, node_pos_t1 * mask)
            dl = depth_fn(depth_pred, depth_t1)

            # ℒ_JEPA: ẑ_t+1 ≈ z*_t+1 (실제 다음 프레임 latent)
            jl = jepa_fn(z_hat, z_gt.detach())

            loss = lw_g * gl + lw_d * dl + lw_j * jl

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            transition.parameters(), cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        total    += loss.item()
        gl_total += gl.item()
        dl_total += dl.item()
        jl_total += jl.item()

        if step % cfg.training.log_interval == 0:
            log.info(
                f"Epoch {epoch:03d}  Step {step:04d}/{len(loader)}"
                f"  loss={loss.item():.4f}"
                f"  graph={gl.item():.4f}"
                f"  depth={dl.item():.4f}"
                f"  jepa={jl.item():.4f}"
                f"  tf={tf_ratio:.2f}"
            )

    n = len(loader)
    return total/n, gl_total/n, dl_total/n, jl_total/n


@torch.no_grad()
def evaluate(transition, encoder, heads, loader, cfg, device):
    transition.eval()
    graph_fn = nn.SmoothL1Loss()
    depth_fn = nn.SmoothL1Loss()
    jepa_fn  = nn.L1Loss()
    total = gl_total = dl_total = jl_total = 0.0

    for batch in loader:
        image_t     = batch["image_t"].to(device)
        image_t1    = batch["image_t1"].to(device)
        action      = batch["action"].to(device)
        node_pos_t1 = batch["node_pos_t1"].to(device)
        node_mask_t1= batch["node_mask_t1"].to(device)
        depth_t1    = batch["sparse_depth_t1"].to(device)

        z_t  = encoder(image_t)
        z_gt = encoder(image_t1)
        z_hat = transition(z_t, action, add_noise=False)

        graph_pred, depth_pred = heads(z_hat)
        mask = node_mask_t1.unsqueeze(-1).float()
        gl   = graph_fn(graph_pred * mask, node_pos_t1 * mask)
        dl   = depth_fn(depth_pred, depth_t1)
        jl   = jepa_fn(z_hat, z_gt)
        loss = cfg.loss.lambda_graph * gl \
             + cfg.loss.lambda_depth * dl \
             + cfg.loss.lambda_jepa  * jl

        total    += loss.item()
        gl_total += gl.item()
        dl_total += dl.item()
        jl_total += jl.item()

    n = len(loader)
    return total/n, gl_total/n, dl_total/n, jl_total/n


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
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

    # ── Frozen Stage 1 ───────────────────────────────────────────────────────
    encoder, heads = load_frozen_stage1(cfg.stage1_ckpt, cfg, device)

    # ── Data ─────────────────────────────────────────────────────────────────
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
    ).to(device)

    n_trans = sum(p.numel() for p in transition.parameters())
    log.info(f"Transition params: {n_trans:,}")

    optimizer = torch.optim.AdamW(
        transition.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        tr = train_one_epoch(
            transition, encoder, heads,
            train_loader, optimizer, scaler, cfg, device, epoch
        )
        vl = evaluate(transition, encoder, heads, val_loader, cfg, device)
        scheduler.step()

        log.info(
            f"[Epoch {epoch:03d}]  "
            f"train={tr[0]:.4f} (g={tr[1]:.4f} d={tr[2]:.4f} j={tr[3]:.4f})  "
            f"val={vl[0]:.4f} (g={vl[1]:.4f} d={vl[2]:.4f} j={vl[3]:.4f})"
        )

        if not args.debug:
            wandb.log({
                "train/loss": tr[0], "train/graph": tr[1],
                "train/depth": tr[2], "train/jepa": tr[3],
                "val/loss": vl[0], "val/graph": vl[1],
                "val/depth": vl[2], "val/jepa": vl[3],
                "epoch": epoch,
            })

        ckpt = {
            "epoch":      epoch,
            "transition": transition.state_dict(),
            "val_loss":   vl[0],
            "cfg":        OmegaConf.to_container(cfg),
        }

        if epoch % cfg.training.save_interval == 0:
            torch.save(ckpt, out_dir / f"ckpt_epoch{epoch:03d}.pt")

        if vl[0] < best_val:
            best_val = vl[0]
            torch.save(ckpt, out_dir / "best.pt")
            log.info(f"  ★ New best val_loss={best_val:.4f}")

    log.info(f"Stage 2 done.  Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
