"""
Demo trajectory PCA 분석

1. demo 전체 프레임 인코딩 → Z (N_frames, 2176)
2. PCA fit → task-relevant subspace 추출
3. 원본 vs PCA 공간에서 goal 간 구분력 비교
4. PCA 누적 분산 및 trajectory 시각화
"""

import os, sys, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py
from torchvision import transforms
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encoder import SWMEncoder

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--stage1_ckpt', default='outputs/stage1/multitask_dinosiglip/best.pt')
parser.add_argument('--vla_path',    default='/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square')
parser.add_argument('--data_root',   default='/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/data')
parser.add_argument('--task',        default='square')
parser.add_argument('--n_demos',     type=int, default=20)
parser.add_argument('--n_components',type=int, default=64)
parser.add_argument('--n_goals',     type=int, default=4)
parser.add_argument('--device',      default='cuda:0')
parser.add_argument('--out',         default='outputs/analysis/pca_latent.png')
args = parser.parse_args()

device = torch.device(args.device)

# ── encoder 로드 ──────────────────────────────────────────────────────────────
print("Loading encoder ...")
s1_ckpt = torch.load(args.stage1_ckpt, map_location=device)
encoder = SWMEncoder(vla_path=args.vla_path, freeze_backbone=True, latent_dim=2176).to(device)
encoder.load_state_dict(s1_ckpt['encoder'])
encoder.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ── demo 인코딩 ───────────────────────────────────────────────────────────────
hdf5_path = os.path.join(
    args.data_root, 'demos', 'core_datasets', args.task,
    f'demo_src_{args.task}_task_D0', 'demo.hdf5'
)
print(f"Encoding {args.n_demos} demos ...")

all_z     = []   # 전체 프레임 (PCA fit용)
demo_z    = []   # demo별 trajectory [(T, 2176), ...]
demo_lens = []

@torch.no_grad()
def encode_img(img_np):
    t = transform(img_np).unsqueeze(0).to(device)
    return encoder(t).float().cpu().squeeze(0)  # (2176,)

with h5py.File(hdf5_path, 'r') as f:
    demo_names = sorted(f['data'].keys())[:args.n_demos]
    for dn in demo_names:
        imgs = np.array(f[f'data/{dn}/obs/agentview_image'])  # (T, H, W, 3)
        zs = []
        for i in range(0, len(imgs), 2):   # stride=2
            zs.append(encode_img(imgs[i]).numpy())
        zs = np.stack(zs)      # (T', 2176)
        all_z.append(zs)
        demo_z.append(zs)
        demo_lens.append(len(zs))
        print(f"  {dn}: {len(imgs)} frames → {len(zs)} encoded")

all_z = np.concatenate(all_z, axis=0)   # (N_total, 2176)
print(f"\nTotal frames: {len(all_z)}")

# ── PCA fit ───────────────────────────────────────────────────────────────────
print(f"\nFitting PCA (n_components={args.n_components}) ...")
pca = PCA(n_components=args.n_components)
pca.fit(all_z)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
print(f"  Top-1  explained: {explained[0]*100:.2f}%")
print(f"  Top-8  cumulative: {cumulative[7]*100:.2f}%")
print(f"  Top-16 cumulative: {cumulative[15]*100:.2f}%")
print(f"  Top-32 cumulative: {cumulative[31]*100:.2f}%")
print(f"  Top-64 cumulative: {cumulative[63]*100:.2f}%")

# ── goal 구분력 비교 ──────────────────────────────────────────────────────────
def uniform_goal_indices(T, n_goals):
    return [max(0, int(round((k+1)*(T-1)/n_goals))) for k in range(n_goals)]

print("\n── Goal cosine similarity: original vs PCA ──")
orig_sims, pca_sims = {k: [] for k in [4,8,16,32,64]}, {k: [] for k in [4,8,16,32,64]}

for zs in demo_z:
    T = len(zs)
    idx = uniform_goal_indices(T, args.n_goals)
    goals_orig = zs[idx]          # (K, 2176)
    goals_pca_full = pca.transform(goals_orig)  # (K, 64)

    # 원본 off-diag cosine sim
    g = torch.tensor(goals_orig)
    sim_orig = F.cosine_similarity(g.unsqueeze(1), g.unsqueeze(0), dim=-1).numpy()
    mask = ~np.eye(len(sim_orig), dtype=bool)

    for k in [4, 8, 16, 32, 64]:
        gp = torch.tensor(goals_pca_full[:, :k])
        sim_pca = F.cosine_similarity(gp.unsqueeze(1), gp.unsqueeze(0), dim=-1).numpy()
        orig_sims[k].append(sim_orig[mask].mean())
        pca_sims[k].append(sim_pca[mask].mean())

print(f"  {'dim':>6}  {'orig cos sim':>14}  {'pca cos sim':>13}  {'improvement':>12}")
print(f"  {'------':>6}  {'----------':>14}  {'-----------':>13}  {'-----------':>12}")
print(f"  {'2176':>6}  {np.mean(orig_sims[4]):>14.4f}  {'  (baseline)':>13}")
for k in [4, 8, 16, 32, 64]:
    p = np.mean(pca_sims[k])
    o = np.mean(orig_sims[k])
    print(f"  {k:>6}  {o:>14.4f}  {p:>13.4f}  {o-p:>+12.4f}  ← lower=better")

# ── 시각화 ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))

# 1) PCA 누적 분산
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(range(1, args.n_components+1), cumulative*100, color='steelblue', linewidth=1.5)
ax1.fill_between(range(1, args.n_components+1), cumulative*100, alpha=0.15, color='steelblue')
for k, c in [(4,'r'),(8,'g'),(16,'orange'),(32,'purple')]:
    ax1.axvline(k, color=c, linestyle='--', linewidth=1, label=f'top-{k}: {cumulative[k-1]*100:.1f}%')
ax1.set_xlabel('n_components'); ax1.set_ylabel('Cumulative variance (%)')
ax1.set_title('PCA Cumulative Explained Variance'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# 2) goal cosine similarity 비교
ax2 = fig.add_subplot(2, 3, 2)
dims  = [4, 8, 16, 32, 64]
means = [np.mean(pca_sims[k]) for k in dims]
ax2.plot([0]+dims, [np.mean(orig_sims[4])]+means, 'o-', color='darkorange', linewidth=1.5)
ax2.axhline(np.mean(orig_sims[4]), color='gray', linestyle='--', linewidth=1, label=f'original (2176-dim): {np.mean(orig_sims[4]):.4f}')
ax2.set_xlabel('PCA n_components'); ax2.set_ylabel('Avg off-diag cosine sim (lower=better)')
ax2.set_title('Goal Separability vs PCA dim'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_xticks(dims)

# 3-5) 대표 demo 3개의 PCA top-2 trajectory
for plot_i, demo_i in enumerate([0, 5, 10]):
    ax = fig.add_subplot(2, 3, 3+plot_i)
    zs = demo_z[demo_i]
    zp = pca.transform(zs)[:, :2]  # top-2 PC만
    T  = len(zs)
    idx = uniform_goal_indices(T, args.n_goals)

    sc = ax.scatter(zp[:, 0], zp[:, 1], c=np.arange(T), cmap='viridis', s=10, alpha=0.7)
    # goal 위치 표시
    for j, gi in enumerate(idx):
        ax.scatter(zp[gi, 0], zp[gi, 1], s=120, marker='*',
                   color='red', zorder=5, label=f'goal {j+1} ({int((j+1)*100/args.n_goals)}%)' if plot_i==0 else '')
    ax.set_title(f'demo_{demo_i} trajectory (PC1 vs PC2)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.colorbar(sc, ax=ax, label='frame index')
    if plot_i == 0:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'PCA Latent Analysis — {args.task}  (DINOv2+SigLIP, {args.n_demos} demos)', fontsize=12)
plt.tight_layout()

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
