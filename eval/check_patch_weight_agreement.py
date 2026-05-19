"""
attn weight vs delta weight 일치도 분석.
같은 (s_t, action) 입력에서 두 방법이 top-K patch를 얼마나 공유하는지 확인.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
_tf_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = _tf_vis

import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encoder import SWMEncoder
from models.transition import SpatialSWMTransition

DEVICE = "cuda"
SWM = Path(__file__).parent.parent
STAGE1_CKPT = SWM / "outputs/stage1/multitask_dinosiglip/best.pt"
STAGE2_CKPT = SWM / "outputs/stage2/spatial_multitask/best.pt"
DATA_ROOT   = Path("/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/data")
TASK        = os.environ.get("TASK", "square")
VLA_PATH    = f"/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/{TASK}"
TOP_K       = 64
N_SAMPLES   = 200

# ── 모델 로드 ──────────────────────────────────────────────────────────────
print("Loading encoder...")
sd1 = torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=False)
latent_dim = sd1.get("cfg", {}).get("encoder", {}).get("latent_dim", 2176)
encoder = SWMEncoder(vla_path=VLA_PATH, latent_dim=latent_dim).to(DEVICE)
encoder.load_state_dict(sd1["encoder"], strict=False)

sd2 = torch.load(STAGE2_CKPT, map_location=DEVICE, weights_only=False)
spatial_dim = sd2.get("spatial_dim", 256)
encoder.spatial_proj = torch.nn.Sequential(
    torch.nn.Linear(latent_dim, spatial_dim),
    torch.nn.LayerNorm(spatial_dim),
).to(DEVICE)
encoder.spatial_proj.load_state_dict(sd2["spatial_proj"])
encoder.eval()

transition = SpatialSWMTransition(spatial_dim=spatial_dim).to(DEVICE).eval()
transition.load_state_dict(sd2["transition"])

print("Models loaded.")

# ── 데이터 로드 ────────────────────────────────────────────────────────────
import h5py
hdf5_path = (DATA_ROOT / "demos" / "core_datasets" / TASK /
             f"demo_src_{TASK}_task_D0" / "demo.hdf5")
print(f"HDF5: {hdf5_path}")

ious, rank_corrs = [], []

with h5py.File(hdf5_path, "r") as f:
    demo_keys = sorted(f["data"].keys())[:20]
    print(f"Using {len(demo_keys)} demos")

    for dk in demo_keys:
        frames  = np.array(f[f"data/{dk}/obs/agentview_image"])  # (T, H, W, 3)
        actions = np.array(f[f"data/{dk}/actions"])               # (T, 7)
        T = min(len(frames), len(actions)) - 1
        if T < 1:
            continue
        n_per_demo = max(1, N_SAMPLES // len(demo_keys))
        indices = np.random.choice(T, min(n_per_demo, T), replace=False)

        for i in indices:
            # center crop + resize to 224x224
            img_np = frames[i]  # (H, W, 3) uint8
            h, w = img_np.shape[:2]
            crop = int(min(h, w) * 0.9)
            y0, x0 = (h - crop) // 2, (w - crop) // 2
            img_np = img_np[y0:y0+crop, x0:x0+crop]
            import cv2
            img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
            action = torch.from_numpy(actions[i]).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                s_t = encoder.spatial_proj(
                    encoder.backbone.forward_spatial(frame).float()
                )   # (1, N, spatial_dim)

            # attn weight (top-64)
            w_attn  = transition.get_patch_weights(s_t, action, top_k=TOP_K)
            # delta weight (top-64)
            w_delta = transition.get_delta_weights(s_t, action, top_k=TOP_K)

            # top-K IoU
            top_attn  = set(w_attn.topk(TOP_K).indices.tolist())
            top_delta = set(w_delta.topk(TOP_K).indices.tolist())
            iou = len(top_attn & top_delta) / len(top_attn | top_delta)
            ious.append(iou)

            # Spearman rank correlation (raw, no top-k mask)
            from scipy.stats import spearmanr
            w_a_raw = transition.get_patch_weights(s_t, action, top_k=256)
            w_d_raw = transition.get_delta_weights(s_t, action, top_k=256)
            rho, _ = spearmanr(w_a_raw.numpy(), w_d_raw.numpy())
            rank_corrs.append(rho)

print(f"\n{'='*50}")
print(f"Samples analyzed : {len(ious)}")
print(f"Top-{TOP_K} IoU      : {np.mean(ious):.3f} ± {np.std(ious):.3f}  (random={TOP_K/256:.3f})")
print(f"Spearman ρ       : {np.mean(rank_corrs):.3f} ± {np.std(rank_corrs):.3f}  (random≈0)")
print(f"{'='*50}")
print("IoU > random baseline" if np.mean(ious) > TOP_K/256 + 0.05 else "IoU ≈ random (두 방법 불일치)")
