"""
PCA Latent Generalization 검증

1. Train/Test demo 분리 → test goal 구분력 확인
2. 다른 initial state에서의 demo trajectory 정렬 확인
3. (선택) 실제 VLA rollout trajectory 투영 확인
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

parser = argparse.ArgumentParser()
parser.add_argument('--stage1_ckpt', default='outputs/stage1/multitask_dinosiglip/best.pt')
parser.add_argument('--vla_path',    default='/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square')
parser.add_argument('--data_root',   default='/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/data')
parser.add_argument('--task',        default='square')
parser.add_argument('--n_train',     type=int, default=20,  help='PCA fit용 demo 수')
parser.add_argument('--n_test',      type=int, default=20,  help='held-out test demo 수')
parser.add_argument('--n_components',type=int, default=32)
parser.add_argument('--n_goals',     type=int, default=4)
parser.add_argument('--device',      default='cuda:0')
parser.add_argument('--out',         default='outputs/analysis/pca_generalization.png')
args = parser.parse_args()

device = torch.device(args.device)

# ── encoder ───────────────────────────────────────────────────────────────────
print("Loading encoder ...")
s1_ckpt = torch.load(args.stage1_ckpt, map_location=device)
encoder = SWMEncoder(vla_path=args.vla_path, freeze_backbone=True, latent_dim=2176).to(device)
encoder.load_state_dict(s1_ckpt['encoder'])
encoder.eval()

transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
])

@torch.no_grad()
def encode_img(img_np):
    t = transform(img_np).unsqueeze(0).to(device)
    return encoder(t).float().cpu().squeeze(0).numpy()

def uniform_goal_indices(T, n_goals):
    return [max(0, int(round((k+1)*(T-1)/n_goals))) for k in range(n_goals)]

def goal_off_diag_sim(goals, pca=None, k=None):
    """goal 간 평균 cosine similarity (lower=better)"""
    if pca is not None:
        g = torch.tensor(pca.transform(goals)[:, :k])
    else:
        g = torch.tensor(goals)
    sim = F.cosine_similarity(g.unsqueeze(1), g.unsqueeze(0), dim=-1).numpy()
    mask = ~np.eye(len(sim), dtype=bool)
    return sim[mask].mean()

# ── demo 인코딩 ───────────────────────────────────────────────────────────────
hdf5_path = os.path.join(
    args.data_root, 'demos', 'core_datasets', args.task,
    f'demo_src_{args.task}_task_D0', 'demo.hdf5'
)

all_demos = []
with h5py.File(hdf5_path, 'r') as f:
    demo_names = sorted(f['data'].keys())[:args.n_train + args.n_test]
    print(f"Encoding {len(demo_names)} demos ...")
    for dn in demo_names:
        imgs = np.array(f[f'data/{dn}/obs/agentview_image'])
        zs = np.stack([encode_img(imgs[i]) for i in range(0, len(imgs), 2)])
        all_demos.append({'name': dn, 'z': zs, 'T': len(zs)})
        print(f"  {dn}: {len(imgs)} → {len(zs)} frames")

train_demos = all_demos[:args.n_train]
test_demos  = all_demos[args.n_train:]

# ── PCA fit (train only) ──────────────────────────────────────────────────────
train_z = np.concatenate([d['z'] for d in train_demos], axis=0)
print(f"\nPCA fit on {len(train_z)} train frames ...")
pca = PCA(n_components=args.n_components)
pca.fit(train_z)
cumvar = np.cumsum(pca.explained_variance_ratio_)

# ── 구분력 비교: train vs test, orig vs PCA ───────────────────────────────────
print(f"\n── Goal separation (n_goals={args.n_goals}) ──")
print(f"{'':12} {'orig(2176)':>12} {'PCA-8':>10} {'PCA-16':>10} {'PCA-32':>10}")

results = {'train': {k:[] for k in [8,16,32]},
           'test':  {k:[] for k in [8,16,32]}}
orig_results = {'train': [], 'test': []}

for split, demos in [('train', train_demos), ('test', test_demos)]:
    for d in demos:
        idx   = uniform_goal_indices(d['T'], args.n_goals)
        goals = d['z'][idx]
        orig_results[split].append(goal_off_diag_sim(goals))
        for k in [8, 16, 32]:
            results[split][k].append(goal_off_diag_sim(goals, pca, k))

for split in ['train', 'test']:
    o = np.mean(orig_results[split])
    p8  = np.mean(results[split][8])
    p16 = np.mean(results[split][16])
    p32 = np.mean(results[split][32])
    print(f"  {split:>10}   {o:>12.4f} {p8:>10.4f} {p16:>10.4f} {p32:>10.4f}")

# ── 데모 간 trajectory 정렬: PCA 공간에서 궤적이 일관된가 ─────────────────────
# 모든 demo trajectory를 PCA-2D에 투영 후 시작/끝 거리 비교
print(f"\n── Demo trajectory alignment (PCA-16) ──")
start_pts, end_pts = [], []
for d in all_demos:
    zp = pca.transform(d['z'])[:, :16]
    start_pts.append(zp[0])
    end_pts.append(zp[-1])

start_pts = np.array(start_pts)
end_pts   = np.array(end_pts)

# 시작점들의 분산 vs 끝점들의 분산
start_std = start_pts.std(axis=0).mean()
end_std   = end_pts.std(axis=0).mean()
# 시작→끝 벡터들의 방향 일관성 (cosine sim between direction vectors)
dirs = end_pts - start_pts
dirs_norm = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
dir_sims = dirs_norm @ dirs_norm.T
mask = ~np.eye(len(dirs_norm), dtype=bool)
print(f"  Start point spread (std): {start_std:.4f}")
print(f"  End point spread   (std): {end_std:.4f}")
print(f"  Direction consistency (avg cos sim between demo directions): {dir_sims[mask].mean():.4f}  (1=perfectly aligned)")

# ── 시각화 ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1) 누적 분산
ax = axes[0, 0]
ax.plot(range(1, args.n_components+1), cumvar*100, 'steelblue', linewidth=1.5)
ax.fill_between(range(1, args.n_components+1), cumvar*100, alpha=0.15, color='steelblue')
for k, c in [(8,'r'),(16,'g'),(32,'orange')]:
    ax.axvline(k, color=c, linestyle='--', linewidth=1.2, label=f'top-{k}: {cumvar[k-1]*100:.1f}%')
ax.set_xlabel('n_components'); ax.set_ylabel('Cumul. variance (%)')
ax.set_title('Cumulative Explained Variance (train demos)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2) train vs test goal separation bar chart
ax = axes[0, 1]
dims = [8, 16, 32]
x = np.arange(len(dims))
w = 0.3
ax.bar(x - w/2, [np.mean(results['train'][k]) for k in dims], w, label='train', color='steelblue', alpha=0.8)
ax.bar(x + w/2, [np.mean(results['test'][k])  for k in dims], w, label='test (held-out)', color='darkorange', alpha=0.8)
ax.axhline(np.mean(orig_results['train']), color='gray', linestyle='--', linewidth=1, label=f'orig 2176-dim: {np.mean(orig_results["train"]):.3f}')
ax.set_xticks(x); ax.set_xticklabels([f'PCA-{k}' for k in dims])
ax.set_ylabel('Avg off-diag cosine sim (lower=better)')
ax.set_title('Goal Separation: Train vs Test (Held-out)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

# 3) PCA-2 trajectory: train demos (10개)
ax = axes[0, 2]
colors = plt.cm.tab20(np.linspace(0, 1, args.n_train))
for i, d in enumerate(train_demos[:10]):
    zp = pca.transform(d['z'])[:, :2]
    ax.plot(zp[:, 0], zp[:, 1], '-', color=colors[i], alpha=0.6, linewidth=1)
    ax.scatter(zp[0, 0], zp[0, 1], marker='o', color=colors[i], s=40, zorder=5)
    ax.scatter(zp[-1, 0], zp[-1, 1], marker='*', color=colors[i], s=80, zorder=5)
ax.set_title('Train demo trajectories (PC1 vs PC2)\n○=start  ★=end'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, alpha=0.3)

# 4) PCA-2 trajectory: test demos (10개)
ax = axes[1, 0]
colors2 = plt.cm.tab20b(np.linspace(0, 1, args.n_test))
for i, d in enumerate(test_demos[:10]):
    zp = pca.transform(d['z'])[:, :2]
    ax.plot(zp[:, 0], zp[:, 1], '-', color=colors2[i], alpha=0.6, linewidth=1)
    ax.scatter(zp[0, 0], zp[0, 1], marker='o', color=colors2[i], s=40, zorder=5)
    ax.scatter(zp[-1, 0], zp[-1, 1], marker='*', color=colors2[i], s=80, zorder=5)
ax.set_title('Test demo trajectories (PC1 vs PC2)\n○=start  ★=end'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, alpha=0.3)

# 5) 시작점 vs 끝점 분포 (PCA-2)
ax = axes[1, 1]
all_start = np.array([pca.transform(d['z'][:1])[:, :2][0] for d in all_demos])
all_end   = np.array([pca.transform(d['z'][-1:])[:, :2][0] for d in all_demos])
ax.scatter(all_start[:args.n_train, 0], all_start[:args.n_train, 1],
           c='steelblue', marker='o', s=60, alpha=0.7, label='train start')
ax.scatter(all_end[:args.n_train, 0],   all_end[:args.n_train, 1],
           c='steelblue', marker='*', s=100, alpha=0.7, label='train end')
ax.scatter(all_start[args.n_train:, 0], all_start[args.n_train:, 1],
           c='darkorange', marker='o', s=60, alpha=0.7, label='test start')
ax.scatter(all_end[args.n_train:, 0],   all_end[args.n_train:, 1],
           c='darkorange', marker='*', s=100, alpha=0.7, label='test end')
ax.set_title('Start(○) vs End(★) distribution\ntrain=blue, test=orange')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 6) Per-demo goal separation 분포 (train vs test)
ax = axes[1, 2]
ax.boxplot([results['train'][16], results['test'][16]],
           labels=['train (PCA-16)', 'test (PCA-16)'], patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.5))
ax.axhline(np.mean(orig_results['train']), color='red', linestyle='--', label=f'orig: {np.mean(orig_results["train"]):.4f}')
ax.set_ylabel('off-diag cosine sim (lower=better)')
ax.set_title('Goal Separation Distribution\nPCA-16'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f'PCA Generalization Analysis — {args.task}\n'
    f'Train: {args.n_train} demos  |  Test (held-out): {args.n_test} demos  |  PCA fit on train only',
    fontsize=11
)
plt.tight_layout()
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
