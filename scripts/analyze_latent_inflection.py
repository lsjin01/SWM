"""
Latent Trajectory Inflection Analysis

demo trajectory의 연속 프레임 간 latent 변화량(Δz_t)을 plot하여
변곡점(큰 action 전환)이 어디서 발생하는지 시각화.
현재 균등 샘플링 goal(25/50/75/100%)이 실제 변곡점과 얼마나 일치하는지 확인.
"""

import os, sys, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encoder import SWMEncoder


# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--stage1_ckpt', default='outputs/stage1/multitask_dinosiglip/best.pt')
parser.add_argument('--vla_path',    default='/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square')
parser.add_argument('--data_root',   default='/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/data')
parser.add_argument('--task',        default='square')
parser.add_argument('--n_demos',     type=int, default=5,   help='분석할 demo 수')
parser.add_argument('--n_goals',     type=int, default=4,   help='uniform goal 수 (표시용)')
parser.add_argument('--top_k',       type=int, default=4,   help='adaptive: 상위 K 변곡점')
parser.add_argument('--device',      default='cuda:0')
parser.add_argument('--out',         default='outputs/analysis/latent_inflection.png')
args = parser.parse_args()

device = torch.device(args.device)
dtype  = torch.bfloat16

# ── encoder 로드 ──────────────────────────────────────────────────────────────
print("Loading encoder ...")
s1_ckpt = torch.load(args.stage1_ckpt, map_location=device)
encoder = SWMEncoder(vla_path=args.vla_path, freeze_backbone=True, latent_dim=2176).to(device)
encoder.load_state_dict(s1_ckpt['encoder'])
encoder.eval()
print("Encoder loaded.")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ── demo hdf5 로드 ────────────────────────────────────────────────────────────
hdf5_path = os.path.join(
    args.data_root, 'demos', 'core_datasets', args.task,
    f'demo_src_{args.task}_task_D0', 'demo.hdf5'
)
print(f"Loading demos from {hdf5_path}")

@torch.no_grad()
def encode_frames(imgs_np, stride=2):
    """imgs_np: (T, H, W, 3) uint8 → z: (T', 2176) float32"""
    zs = []
    for i in range(0, len(imgs_np), stride):
        t = transform(imgs_np[i]).unsqueeze(0).to(device)
        z = encoder(t).float().cpu()   # (1, 2176)
        zs.append(z.squeeze(0))
    return torch.stack(zs)  # (T', 2176)

def compute_curvature(z):
    """
    latent trajectory 곡률: 연속 velocity 벡터 간 방향 변화.
    velocity_t = z_{t+1} - z_t  (방향 벡터)
    curvature_t = 1 - cos(velocity_t, velocity_{t+1})

    반환: (T'-2,) — 클수록 direction이 급격히 꺾임
    """
    velocity = z[1:] - z[:-1]                              # (T'-1, D)
    vel_norm = F.normalize(velocity.float(), dim=-1)        # 방향만 추출
    curvature = 1.0 - F.cosine_similarity(
        vel_norm[:-1], vel_norm[1:], dim=-1
    ).numpy()                                               # (T'-2,)
    return curvature

def uniform_goal_indices(T, n_goals):
    return [max(0, int(round((k + 1) * (T - 1) / n_goals))) for k in range(n_goals)]

def adaptive_goal_indices_by_curvature(curvature, top_k):
    """곡률 피크 기준 상위 top_k 인덱스 (+ 1 offset: curvature는 idx 1부터 시작)"""
    order = np.argsort(curvature)[::-1]
    peaks = sorted((order[:top_k] + 1).tolist())   # velocity 기준 offset 보정
    return peaks

# ── 분석 루프 ─────────────────────────────────────────────────────────────────
results = []

with h5py.File(hdf5_path, 'r') as f:
    demo_names = sorted(f['data'].keys())[:args.n_demos]
    print(f"Analyzing {len(demo_names)} demos ...")

    for dn in demo_names:
        imgs = np.array(f[f'data/{dn}/obs/agentview_image'])  # (T, H, W, 3)
        T    = len(imgs)
        print(f"  {dn}: T={T} frames → encoding (stride=2) ...")

        z  = encode_frames(imgs, stride=2)   # (T', 2176)
        Tp = len(z)

        # 1차: 연속 프레임 간 cosine distance (속도 크기)
        cos_sim = F.cosine_similarity(z[:-1], z[1:], dim=-1).numpy()
        delta   = 1.0 - cos_sim                          # (T'-1,)

        # 2차: latent 방향 변화 (곡률)
        curvature = compute_curvature(z)                 # (T'-2,)

        # uniform goal 위치
        uniform_raw = uniform_goal_indices(T, args.n_goals)
        uniform_idx = [min(r // 2, Tp - 1) for r in uniform_raw]

        # adaptive goal: 곡률 기반 피크
        curv_idx = adaptive_goal_indices_by_curvature(curvature, args.top_k)

        results.append({
            'name': dn, 'T': T, 'Tp': Tp,
            'delta': delta,
            'curvature': curvature,
            'uniform_idx': uniform_idx,
            'curv_idx': curv_idx,
            'z': z,
        })

# ── 시각화 ────────────────────────────────────────────────────────────────────
n = len(results)
fig, axes = plt.subplots(n, 2, figsize=(18, 3.5 * n))
if n == 1:
    axes = axes[np.newaxis, :]

for i, res in enumerate(results):
    delta     = res['delta']
    curvature = res['curvature']
    ax_l, ax_r = axes[i]

    # ── 왼쪽: 1차 Δz_t ────────────────────────────────────────────────────
    t1 = np.arange(len(delta))
    ax_l.plot(t1, delta, color='steelblue', linewidth=1.2, label='Δz_t (speed)')
    ax_l.fill_between(t1, delta, alpha=0.15, color='steelblue')
    for idx in res['uniform_idx']:
        ax_l.axvline(idx, color='tomato', linestyle='--', linewidth=1.5,
                     label='uniform' if idx == res['uniform_idx'][0] else '')
    ax_l.axhline(delta.mean(), color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_l.set_title(f"{res['name']}  1st-order Δz_t", fontsize=9)
    ax_l.set_ylabel('1−cos(z_t, z_{t+1})')
    ax_l.legend(fontsize=7, loc='upper right')
    ax_l.grid(True, alpha=0.3)

    # ── 오른쪽: 2차 곡률 ──────────────────────────────────────────────────
    t2 = np.arange(len(curvature))
    ax_r.plot(t2, curvature, color='darkorange', linewidth=1.2, label='curvature (direction change)')
    ax_r.fill_between(t2, curvature, alpha=0.15, color='darkorange')
    # uniform goals
    for idx in res['uniform_idx']:
        ax_r.axvline(idx, color='tomato', linestyle='--', linewidth=1.5,
                     label='uniform' if idx == res['uniform_idx'][0] else '')
    # curvature-based adaptive goals (파란 점선)
    for idx in res['curv_idx']:
        ax_r.axvline(idx, color='royalblue', linestyle=':', linewidth=2.0,
                     label='curv peak' if idx == res['curv_idx'][0] else '')
    ax_r.axhline(curvature.mean(), color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_r.set_title(f"{res['name']}  2nd-order curvature", fontsize=9)
    ax_r.set_ylabel('1−cos(v_t, v_{t+1})')
    ax_r.legend(fontsize=7, loc='upper right')
    ax_r.grid(True, alpha=0.3)

    print(f"\n{res['name']}:")
    print(f"  uniform goals    : {res['uniform_idx']}")
    print(f"  curvature peaks  : {res['curv_idx']}")
    print(f"  Δz       max={delta.max():.4f}  mean={delta.mean():.4f}  std={delta.std():.4f}")
    print(f"  curvature max={curvature.max():.4f}  mean={curvature.mean():.4f}  std={curvature.std():.4f}")

plt.suptitle(
    f"Latent Trajectory Analysis — {args.task}  (DINOv2+SigLIP)\n"
    f"Left: 1st-order speed | Right: 2nd-order curvature (direction change)\n"
    f"Red dashed=uniform goals, Blue dotted=curvature-based adaptive goals",
    fontsize=10
)
plt.tight_layout()

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")

# ── 추가: goal별 pairwise cosine similarity (uniform vs curvature) ─────────────
print("\n── Goal pairwise cosine similarity ──")
for res in results:
    z = res['z']
    for label, idx_list in [('uniform', res['uniform_idx']), ('curvature', res['curv_idx'])]:
        goals = z[idx_list]
        sim   = F.cosine_similarity(goals.unsqueeze(1), goals.unsqueeze(0), dim=-1).numpy()
        # off-diagonal mean (goal 간 평균 유사도, 낮을수록 구분 잘 됨)
        mask = ~np.eye(len(sim), dtype=bool)
        off_diag_mean = sim[mask].mean()
        print(f"  {res['name']} [{label}] off-diag cosine sim mean: {off_diag_mean:.4f}  (lower=better separated)")
