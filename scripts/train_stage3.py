#!/usr/bin/env python3
# scripts/train_stage3.py
"""
Stage 3: GRPO Policy Optimization
===================================
SWM (DINOv2+SigLIP encoder + Small Transition) + OpenVLA-OFT

핵심 아이디어:
  z_t (2176) → repeat(256) → projector → LLM → action tokens
  action → Transition → z_t+1 → Graph Head → Ĝ_T
  Reward = -‖Ĝ_T - G*_T‖  (goal graph distance)

기존 WMPO train_grpo_optionA.py의 구조를 SWM에 맞게 수정:
  - decoder 제거 (z_t를 projector에 직접 주입)
  - V-JEPA-AC → SWM Transition
  - latent_reward → graph_reward
"""

import os, sys, time, random, argparse, logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions import Categorical
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/dependencies/openvla-oft')
sys.path.insert(0, '/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA')

from models.encoder   import SWMEncoder
from models.transition import SWMTransition
from models.heads     import SWMHeads

log = logging.getLogger(__name__)

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
# Constants (OpenVLA-OFT)
# ─────────────────────────────────────────────────────────────────────────────
NUM_ACTIONS_CHUNK  = 8
ACTION_DIM         = 7
NUM_ACTION_TOKENS  = ACTION_DIM * NUM_ACTIONS_CHUNK   # 56
NUM_VISION_TOKENS  = 256

_TASK_DESC = {
    "square":               "pick up the square nut and place it on the round peg",
    "coffee":               "place the coffee pod into the coffee machine",
    "stack_three":          "stack the three cubes on top of each other",
    "three_piece_assembly": "assemble the three pieces together",
}


# ─────────────────────────────────────────────────────────────────────────────
# z_t → VLA forward (vision bypass)
# ─────────────────────────────────────────────────────────────────────────────

def image_to_patch_embeddings(
    image: torch.Tensor,
    encoder: 'SWMEncoder',
    vla,
) -> torch.Tensor:
    """
    image (B, 3, H, W) → projected spatial patch embeddings (B, 256, 4096)

    encoder.encode_spatial() → (B, 256, 2176) 공간 patch features
    → projector → (B, 256, 4096)

    ▸ 각 256개 token이 서로 다른 공간 정보를 보존 (mean pool 제거)
    ▸ train/eval 파이프라인 완전 일치
    """
    dtype = next(vla.projector.parameters()).dtype
    spatial = encoder.encode_spatial(image).to(dtype)   # (B, 256, 2176)
    return vla.projector(spatial)                        # (B, 256, 4096)


@torch.no_grad()
def goal_pre_action_hidden(goal_image, encoder, vla, reward_lm, prompt_ids, attn_mask,
                           device, dtype):
    """goal 이미지(데모 마지막 프레임) → patch_embeds → reward LLM → action head 직전 hidden.
    Returns (1, 56, H). EMA reward LLM이 goal을 인코딩한 target 표현(h_goal)."""
    patch_embeds = image_to_patch_embeddings(goal_image.to(device), encoder, vla)  # (1,256,4096)
    input_ids_ext  = prompt_ids.clone()
    placeholder    = torch.ones((1, NUM_ACTION_TOKENS), device=device, dtype=input_ids_ext.dtype)
    stop           = torch.ones((1, 1), device=device, dtype=input_ids_ext.dtype) * 2
    input_ids_full = torch.cat([input_ids_ext, placeholder, stop], dim=-1)
    input_embeds   = vla.get_input_embeddings()(input_ids_full)
    full_embeds    = torch.cat([patch_embeds, input_embeds], dim=1)
    vis_mask  = torch.ones((1, NUM_VISION_TOKENS),       device=device, dtype=attn_mask.dtype)
    ext_mask  = torch.ones((1, input_ids_full.shape[1]), device=device, dtype=attn_mask.dtype)
    full_mask = torch.cat([vis_mask, ext_mask], dim=1)
    out = reward_lm(inputs_embeds=full_embeds, attention_mask=full_mask,
                    use_cache=False, output_hidden_states=True)
    return out.hidden_states[-1][:, -NUM_ACTION_TOKENS-2:-2].detach()   # (1,56,H)


@torch.no_grad()
def fit_progress_probe(demo_dir, task, encoder, vla, reward_lm, prompt_ids, attn_mask,
                       max_demos, device, dtype, n_frames_per_demo=8, method='ridge'):
    """데모 프레임의 pre-action hidden → frame_fraction(0→1) 으로 'task 진전 상관축' w 학습.
    PCA(최대 분산, unsupervised)와 달리 진전과 상관된 방향을 supervised로 추출.
      method='ridge' : Ridge 선형회귀 계수 (hidden→progress 예측)
      method='pls'   : PLS 1st component (분산 대신 progress와의 공분산 최대화 방향)
    Returns w: (H,) tensor.  reward는 rollout에서 w·pool(h_T) − w·pool(h_0)."""
    import h5py
    from torchvision import transforms

    hdf5 = os.path.join(demo_dir, 'demos', 'core_datasets', task,
                        f'demo_src_{task}_task_D0', 'demo.hdf5')
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
    ])
    X, y = [], []
    with h5py.File(hdf5, 'r') as f:
        demos = sorted(f['data'].keys())[:max_demos]
        for dn in demos:
            imgs = f[f'data/{dn}/obs/agentview_image']
            T = len(imgs)
            idxs = np.linspace(0, T - 1, n_frames_per_demo).astype(int)
            for i in idxs:
                img = transform(np.array(imgs[i])).unsqueeze(0)
                h = goal_pre_action_hidden(img, encoder, vla, reward_lm,
                                           prompt_ids, attn_mask, device, dtype)  # (1,56,H)
                X.append(h.mean(dim=1).float().cpu().numpy().flatten())           # pool→(H,)
                y.append(float(i) / max(T - 1, 1))
    X = np.stack(X); y = np.array(y)
    if method == 'pls':
        from sklearn.cross_decomposition import PLSRegression
        reg = PLSRegression(n_components=1).fit(X, y)
        w = reg.coef_.reshape(-1)               # (H,) progress 공분산 방향
        r2 = reg.score(X, y)
    else:  # ridge
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0).fit(X, y)
        w = reg.coef_.reshape(-1)
        r2 = reg.score(X, y)
    log.info(f"[progress_probe] method={method}  fit on {len(X)} frames ({len(demos)} demos)  R2={r2:.3f}")
    return torch.tensor(w, dtype=torch.float32)   # (H,)


class ProgressMLP(nn.Module):
    """pooled pre-action hidden (H,) → progress scalar. time-contrastive(③)용 비선형 scorer."""
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):  # x: (B, H) → (B,)
        return self.net(x).squeeze(-1)


@torch.no_grad()
def _collect_demo_hidden(demo_dir, task, encoder, vla, reward_lm, prompt_ids, attn_mask,
                         max_demos, device, dtype, n_frames_per_demo=8):
    """데모 프레임의 pooled pre-action hidden + (demo_id, frame_fraction) 수집."""
    import h5py
    from torchvision import transforms
    hdf5 = os.path.join(demo_dir, 'demos', 'core_datasets', task,
                        f'demo_src_{task}_task_D0', 'demo.hdf5')
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
    X, frac, dids, G = [], [], [], []
    with h5py.File(hdf5, 'r') as f:
        demos = sorted(f['data'].keys())[:max_demos]
        for di, dn in enumerate(demos):
            imgs = f[f'data/{dn}/obs/agentview_image']; T = len(imgs)
            # 데모 goal(마지막 프레임) pooled hidden 1회 계산
            gimg = tf(np.array(imgs[T - 1])).unsqueeze(0)
            gh = goal_pre_action_hidden(gimg, encoder, vla, reward_lm, prompt_ids, attn_mask, device, dtype)
            gh_pooled = gh.mean(dim=1).float().cpu().numpy().flatten()
            for i in np.linspace(0, T - 1, n_frames_per_demo).astype(int):
                img = tf(np.array(imgs[i])).unsqueeze(0)
                h = goal_pre_action_hidden(img, encoder, vla, reward_lm, prompt_ids, attn_mask, device, dtype)
                X.append(h.mean(dim=1).float().cpu().numpy().flatten())
                frac.append(float(i) / max(T - 1, 1)); dids.append(di); G.append(gh_pooled)
    return np.stack(X), np.array(frac), np.array(dids), np.stack(G)


def fit_tcn_progress(demo_dir, task, encoder, vla, reward_lm, prompt_ids, attn_mask,
                     max_demos, device, dtype, n_frames_per_demo=8, steps=800):
    """time-contrastive(③): 같은 데모 내 '나중 프레임이 더 높은 progress'가 되도록
    MLP φ를 pairwise ranking loss로 학습. PCA/선형과 달리 비선형 시간구조 인코딩.
    Returns: frozen ProgressMLP (GPU)."""
    X, frac, dids, _G = _collect_demo_hidden(demo_dir, task, encoder, vla, reward_lm,
                                         prompt_ids, attn_mask, max_demos, device, dtype, n_frames_per_demo)
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    fg = torch.tensor(frac, dtype=torch.float32, device=device)
    dg = torch.tensor(dids, device=device)
    mlp = ProgressMLP(Xg.shape[1]).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    N = Xg.shape[0]
    for step in range(steps):
        ia = torch.randint(0, N, (256,), device=device); ib = torch.randint(0, N, (256,), device=device)
        same = (dg[ia] == dg[ib]) & (fg[ia] != fg[ib])    # 같은 데모, 다른 시점
        if same.sum() < 4:
            continue
        ia, ib = ia[same], ib[same]
        sa, sb = mlp(Xg[ia]), mlp(Xg[ib])
        later = (fg[ib] > fg[ia]).float()                 # ib가 더 나중이면 1
        # 나중 프레임 score가 더 크도록: margin ranking
        loss = F.softplus(-(sb - sa) * (2 * later - 1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    # 단조성 점검: progress와 frac의 상관
    with torch.no_grad():
        pred = mlp(Xg).cpu().numpy()
    corr = float(np.corrcoef(pred, frac)[0, 1])
    log.info(f"[tcn_progress] fit on {N} frames  final_loss={loss.item():.3f}  corr(pred,frac)={corr:.3f}")
    return mlp


class GoalDistMLP(nn.Module):
    """pooled pre-action hidden (H,) → d차원 임베딩 φ. contrastive RL / goal-distance(④)용.
    ||φ(h) − φ(goal)|| 가 goal까지 '남은 진전'을 근사하도록 학습."""
    def __init__(self, in_dim, emb=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, emb),
        )
    def forward(self, x):           # x: (B, H) → (B, emb)
        return self.net(x)


def fit_goaldist(demo_dir, task, encoder, vla, reward_lm, prompt_ids, attn_mask,
                 max_demos, device, dtype, n_frames_per_demo=8, steps=1200):
    """contrastive goal-distance(④): 임베딩 φ를 학습하여 ||φ(h)−φ(goal)|| ≈ (1−frac)
    (goal까지 남은 진전)이 되도록. quasimetric/goal-conditioned value의 단순화.
    rollout score_t = −||φ(h_t)−φ(goal)|| → goal에 가까울수록 ↑.
    Returns: frozen GoalDistMLP (GPU)."""
    X, frac, dids, G = _collect_demo_hidden(demo_dir, task, encoder, vla, reward_lm,
                                            prompt_ids, attn_mask, max_demos, device, dtype, n_frames_per_demo)
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    Gg = torch.tensor(G, dtype=torch.float32, device=device)
    fg = torch.tensor(frac, dtype=torch.float32, device=device)
    remain = (1.0 - fg)                                   # goal까지 남은 진전 (0=goal)
    mlp = GoalDistMLP(Xg.shape[1]).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    N = Xg.shape[0]
    for step in range(steps):
        idx = torch.randint(0, N, (256,), device=device)
        d = (mlp(Xg[idx]) - mlp(Gg[idx])).norm(dim=-1)    # ||φ(h)−φ(goal)||
        loss = F.mse_loss(d, remain[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    # 단조성 점검: −거리(=score)와 frac의 상관 (1에 가까울수록 진전과 상관)
    with torch.no_grad():
        score = -(mlp(Xg) - mlp(Gg)).norm(dim=-1).cpu().numpy()
    corr = float(np.corrcoef(score, frac)[0, 1])
    log.info(f"[goaldist] fit on {N} frames  final_loss={loss.item():.3f}  corr(score,frac)={corr:.3f}")
    return mlp


def _vla_forward_patch_embeds(
    patch_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    vla,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
    goal_hidden=None,   # (1,56,H) EMA reward LLM의 goal-이미지 pre-action hidden (cosine metric)
    probe_w=None,       # (H,) progress probe 방향 (probe/pls metric): s_t = w·pool(h_t)
    progress_mlp=None,  # ProgressMLP (tcn metric): s_t = mlp(pool(h_t))
    goaldist_mlp=None,  # GoalDistMLP φ (goaldist metric): s_t = −||φ(pool(h_t))−φ(pool(goal))||
):
    """patch_embeds (1, 256, 4096) + prompt → (tids (1,56), logprob float, score or None)

    metric에 따라 per-chunk scalar 반환 (swm_rollout이 cos_T−cos_0 progress로 변환):
      - goal_hidden: cosine(h_t, h_goal)               (goal-conditioned)
      - probe_w:     w·pool(h_t)                        (진전 상관축 투영)
    """
    input_ids_ext = prompt_ids.clone()
    placeholder   = torch.ones((1, NUM_ACTION_TOKENS), device=device, dtype=input_ids_ext.dtype)
    stop          = torch.ones((1, 1),                  device=device, dtype=input_ids_ext.dtype) * 2
    input_ids_full = torch.cat([input_ids_ext, placeholder, stop], dim=-1)

    input_embeds = vla.get_input_embeddings()(input_ids_full)
    full_embeds  = torch.cat([patch_embeds, input_embeds], dim=1)

    vis_mask  = torch.ones((1, NUM_VISION_TOKENS),        device=device, dtype=attn_mask.dtype)
    ext_mask  = torch.ones((1, input_ids_full.shape[1]),  device=device, dtype=attn_mask.dtype)
    full_mask = torch.cat([vis_mask, ext_mask], dim=1)

    need_hidden = (goal_hidden is not None) or (probe_w is not None) or (progress_mlp is not None) or (goaldist_mlp is not None)
    out    = vla.language_model(inputs_embeds=full_embeds, attention_mask=full_mask,
                                use_cache=False, output_hidden_states=need_hidden)
    logits = out.logits[:, -NUM_ACTION_TOKENS-2:-2]  # (1, 56, V)  off-by-one fix
    if temperature != 1.0:
        logits = logits / temperature
    dist = Categorical(logits=logits.reshape(-1, logits.size(-1)).float())
    tids = dist.sample().reshape(1, -1)
    lp   = dist.log_prob(tids.reshape(-1)).sum().item()

    score = None
    if need_hidden:
        # 정책 LLM의 action head 직전 hidden (action 토큰 위치, logits와 동일 slice)
        h_pol = out.hidden_states[-1][:, -NUM_ACTION_TOKENS-2:-2]              # (1, 56, H)
        pooled = h_pol.float().mean(dim=1).squeeze(0)                          # (H,)
        if goaldist_mlp is not None and goal_hidden is not None:
            # contrastive goal-distance: score = −||φ(pool(h_t)) − φ(pool(goal))||
            g_pooled = goal_hidden.to(pooled.device).float().mean(dim=1).squeeze(0)   # (H,)
            d = (goaldist_mlp(pooled.unsqueeze(0)) - goaldist_mlp(g_pooled.unsqueeze(0))).norm(dim=-1)
            score = float(-d.squeeze())
        elif progress_mlp is not None:
            # time-contrastive 비선형 progress
            score = float(progress_mlp(pooled.unsqueeze(0)).squeeze())
        elif probe_w is not None:
            # progress 상관축 투영: w · pool(h_t)
            score = float(torch.dot(pooled, probe_w.to(pooled.device).float()))
        else:
            score = F.cosine_similarity(
                h_pol.float(), goal_hidden.to(h_pol.device).float(), dim=-1
            ).mean().item()
    return tids, lp, score


def swm_rollout(
    z0: torch.Tensor,           # (1, 2176)  global latent (transition / reward 전용)
    image0: torch.Tensor,       # (1, 3, H, W)  초기 관측 이미지 (VLA spatial 인코딩용)
    encoder: 'SWMEncoder',
    transition: 'SWMTransition',
    heads: 'SWMHeads',
    vla,
    processor,
    prompt_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    tokens_to_actions,
    n_chunks: int,
    temperature: float,
    device: torch.device,
    dtype: torch.dtype,
    patch_weights: torch.Tensor = None,  # (256,) CPU — spatial weighted mean pool
    wm_bridge=None,   # NEW: nn.Linear(spatial_dim, 2176), spatial WM 전용
    goal_hidden=None, # (1,56,H) EMA reward LLM의 goal-이미지 pre-action hidden (cosine)
    probe_w=None,     # (H,) progress probe 방향 (probe/pls metric)
    progress_mlp=None,# ProgressMLP (tcn metric)
    goaldist_mlp=None,# GoalDistMLP φ (goaldist metric)
):
    """
    수정된 SWM rollout (circulatory 구조):
      - VLA: 매 chunk마다 WM transition 출력 z_t → wm_bridge → vla.projector → patch_embeds
             (chunk 0: 초기 이미지 spatial patch features 사용)
      - WM:  global z_t → transition → z_t+1 (reward 계산 전용)

    Returns:
        token_ids_list   : list[Tensor(56,)]   각 chunk의 action tokens
        logprobs_list    : list[float]          log-probabilities
        rollout_data     : (z_at_chunk_start list, init_patch_embeds_cpu) tuple
        z_history        : list[Tensor(1, 2176)]  WM latent history
    """
    z = z0.clone().to(device, dtype)
    is_spatial = (z.ndim == 3)

    # ── 초기 이미지 → spatial patch embeddings (chunk 0용) ────────────────
    with torch.no_grad():
        patch_embeds = image_to_patch_embeddings(image0, encoder, vla)  # (1, 256, 4096)

    init_patch_embeds_cpu = patch_embeds.cpu()   # gradient recompute용 CPU 저장

    token_ids_list   = []
    logprobs_list    = []
    z_history        = []
    z_at_chunk_start = []  # 각 chunk 시작 z (gradient recompute용 CPU 저장)
    ema_cos_list     = []  # chunk별 pre-action hidden consistency (EMA reward용)

    for chunk_i in range(n_chunks):
        z_at_chunk_start.append(z.clone().cpu())

        # ── VLA: 현재 patch_embeds로 action 생성 ──────────────────────────
        with torch.no_grad():
            tids, lp, ema_cos = _vla_forward_patch_embeds(
                patch_embeds, prompt_ids, attn_mask, vla, device, dtype, temperature,
                goal_hidden=goal_hidden, probe_w=probe_w, progress_mlp=progress_mlp,
                goaldist_mlp=goaldist_mlp,
            )
        if ema_cos is not None:
            ema_cos_list.append(ema_cos)

        token_ids_list.append(tids[0].cpu())
        logprobs_list.append(lp)

        # ── WM: global z_t + action → z_t+1 (reward 전용) ────────────────
        actions = tokens_to_actions(tids)  # (1, 8, 7)
        with torch.no_grad():
            for sub in range(NUM_ACTIONS_CHUNK):
                a = actions[0, sub].to(device, dtype)
                z = transition(z.to(dtype), a.unsqueeze(0))
                if is_spatial:
                    if patch_weights is not None:
                        w = patch_weights.to(z.device, z.dtype)   # (256,)
                        z_r = (z * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                    else:
                        z_r = z.mean(dim=1)
                    z_history.append(z_r.clone())              # (B, d_s)
                else:
                    z_history.append(z.clone())

        # ── Circulatory: patch_embeds 업데이트 (다음 chunk VLA 입력용) ────
        if is_spatial and wm_bridge is not None:
            # spatial_dim<2176: bridge로 업샘플 후 projector
            with torch.no_grad():
                vla_dtype = next(vla.projector.parameters()).dtype
                z_up = wm_bridge(z.to(vla_dtype))          # (1, 256, 2176)
                patch_embeds = vla.projector(z_up)         # (1, 256, 4096)
        elif is_spatial and z.shape[-1] == 2176:
            # spatial_dim=2176: transition 출력을 projector에 직접 입력
            with torch.no_grad():
                vla_dtype = next(vla.projector.parameters()).dtype
                patch_embeds = vla.projector(z.to(vla_dtype))  # (1, 256, 4096)
        elif not is_spatial:
            # Scalar WM: z (1, 2176) → repeat → vla.projector
            with torch.no_grad():
                vla_dtype = next(vla.projector.parameters()).dtype
                z_rep = z.to(vla_dtype).unsqueeze(1).expand(-1, 256, -1)  # (1, 256, 2176)
                patch_embeds = vla.projector(z_rep)        # (1, 256, 4096)
        # is_spatial, dim≠2176, no bridge: keep same patch_embeds (backward compat)

    rollout_data = (z_at_chunk_start, init_patch_embeds_cpu)
    # goal-conditioned reward: VLA hidden cosine은 ≈1로 포화 →
    #   (1−cos_0) 분모 정규화는 분모≈0으로 폭발 → raw delta(cos_T − cos_0)만 사용.
    #   cos_0 = chunk0(초기 관측) vs h_goal,  cos_T = 마지막 chunk(상상 관측) vs h_goal
    #   진전(goal에 가까워짐) > 0, 멀어짐 < 0. 그룹 내 상대 advantage라 스케일은 무관.
    ema_consistency = None
    if ema_cos_list:
        cos_0 = ema_cos_list[0]
        cos_T = ema_cos_list[-1]
        ema_consistency = float(cos_T - cos_0)
    return token_ids_list, logprobs_list, rollout_data, z_history, ema_consistency


def _patch_embeds_from_z(z_cpu, vla, wm_bridge, device, dtype):
    """z (CPU tensor) → patch_embeds (GPU tensor with grad).
    Spatial WM (bridge): z (1, N, d_s) → wm_bridge → (1, N, 2176) → projector → (1, N, 4096)
    Spatial WM (2176):   z (1, N, 2176) → projector directly → (1, N, 4096)
    Scalar WM:           z (1, 2176)   → repeat(N) → projector → (1, N, 4096)
    """
    vla_dtype = next(vla.projector.parameters()).dtype
    z = z_cpu.to(device, vla_dtype)
    if z.ndim == 3 and wm_bridge is not None:
        z_up = wm_bridge(z)                             # (1, N, 2176)
    elif z.ndim == 3:                                   # spatial_dim=2176, no bridge
        z_up = z                                        # (1, N, 2176)
    else:                                               # scalar
        z_up = z.unsqueeze(1).expand(-1, 256, -1)       # (1, 256, 2176)
    return vla.projector(z_up).to(dtype)                # (1, 256, 4096)


def recompute_logprobs_grad(
    rollout_data,             # (z_at_chunk_cpu list, init_patch_embeds_cpu) OR legacy Tensor
    token_ids_list: list,     # list of (56,) Tensor
    vla,
    prompt_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    wm_bridge=None,           # nn.Linear(spatial_dim, 2176) — circulatory 구조용
):
    """
    gradient 있는 log-prob 재계산 (GRPO backward용).
    circulatory: chunk마다 z_t → wm_bridge → vla.projector → 개별 LLM forward.
    legacy fallback: 단일 patch_embeds로 한 번만 LLM forward.
    """
    # ── circulatory mode ─────────────────────────────────────────────────────
    if isinstance(rollout_data, tuple):
        z_at_chunk_cpu, init_patch_embeds_cpu = rollout_data
        input_ids_ext  = prompt_ids.clone()
        placeholder    = torch.ones((1, NUM_ACTION_TOKENS), device=device, dtype=input_ids_ext.dtype)
        stop           = torch.ones((1, 1),                  device=device, dtype=input_ids_ext.dtype) * 2
        input_ids_full = torch.cat([input_ids_ext, placeholder, stop], dim=-1)
        input_embeds   = vla.get_input_embeddings()(input_ids_full)
        vis_mask  = torch.ones((1, NUM_VISION_TOKENS),       device=device, dtype=attn_mask.dtype)
        ext_mask  = torch.ones((1, input_ids_full.shape[1]), device=device, dtype=attn_mask.dtype)
        full_mask = torch.cat([vis_mask, ext_mask], dim=1)

        # spatial_dim=2176: transition output → projector directly (circulatory)
        is_spatial_2176 = (
            len(z_at_chunk_cpu) > 0 and
            z_at_chunk_cpu[0].ndim == 3 and
            z_at_chunk_cpu[0].shape[-1] == 2176
        )
        lp_list = []
        for i, tids in enumerate(token_ids_list):
            if i == 0:
                patch_embeds = init_patch_embeds_cpu.to(device, dtype)
            elif wm_bridge is not None or is_spatial_2176:
                patch_embeds = _patch_embeds_from_z(
                    z_at_chunk_cpu[i], vla, wm_bridge, device, dtype)
            else:
                patch_embeds = init_patch_embeds_cpu.to(device, dtype)
            full_embeds = torch.cat([patch_embeds, input_embeds], dim=1)
            out    = vla.language_model(inputs_embeds=full_embeds,
                                        attention_mask=full_mask, use_cache=False)
            logits = out.logits[:, -NUM_ACTION_TOKENS-2:-2]
            lp = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                tids.to(device).reshape(-1), reduction='sum',
            )
            lp_list.append(lp)
        return lp_list

    # ── legacy mode (단일 patch_embeds) ──────────────────────────────────────
    patch_embeds = rollout_data.to(device, dtype)
    input_ids_ext  = prompt_ids.clone()
    placeholder    = torch.ones((1, NUM_ACTION_TOKENS), device=device, dtype=input_ids_ext.dtype)
    stop           = torch.ones((1, 1),                  device=device, dtype=input_ids_ext.dtype) * 2
    input_ids_full = torch.cat([input_ids_ext, placeholder, stop], dim=-1)
    input_embeds   = vla.get_input_embeddings()(input_ids_full)
    full_embeds    = torch.cat([patch_embeds, input_embeds], dim=1)
    vis_mask  = torch.ones((1, NUM_VISION_TOKENS),        device=device, dtype=attn_mask.dtype)
    ext_mask  = torch.ones((1, input_ids_full.shape[1]),  device=device, dtype=attn_mask.dtype)
    full_mask = torch.cat([vis_mask, ext_mask], dim=1)
    out    = vla.language_model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    logits = out.logits[:, -NUM_ACTION_TOKENS-2:-2]
    lp_list = []
    for tids in token_ids_list:
        lp = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            tids.to(device).reshape(-1), reduction='sum',
        )
        lp_list.append(lp)
    return lp_list


def grpo_loss_fn(lp_new_list, lp_old_list, advantages, clip_eps, kl_coef):
    losses = []
    for lp_new, lp_old, adv in zip(lp_new_list, lp_old_list, advantages):
        ratio = torch.exp(lp_new - lp_old)
        adv_t = torch.tensor(adv, device=lp_new.device, dtype=lp_new.dtype)
        surr  = -torch.min(
            ratio * adv_t,
            torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_t
        )
        if kl_coef > 0:
            surr = surr + kl_coef * (lp_new - lp_old)
        losses.append(surr)
    return torch.stack(losses).mean()


@torch.no_grad()
def graph_reward(z_history, heads, goal_pos, device):
    """
    Reward = -‖Ĝ_T - G*_T‖   (goal graph distance)
    z_history: list of (1, 2176) latents
    goal_pos:  (N, 3) goal node positions
    """
    if len(z_history) == 0:
        return 0.0

    z_final = z_history[-1].float().to(device)
    raw_heads = heads.module if hasattr(heads, 'module') else heads
    graph_pred, _ = raw_heads(z_final)  # (1, max_nodes, 3)

    N = goal_pos.shape[0]
    pred = graph_pred[0, :N]
    goal = goal_pos.to(device, torch.float32)
    dist = (pred - goal).norm(dim=-1).mean().item()
    return -dist   # reward는 distance의 음수


@torch.no_grad()
def transition_l2_reward(z_history, z_goal_transition, device):
    """
    Reward = -||z_T_policy - z_goal_demo||_2   (둘 다 transition 공간)
    z_history:         list of (1, 2176) — policy rollout via transition
    z_goal_transition: (1, 2176) — demo actions 64스텝을 transition으로 돌린 끝 latent
    → encoder vs transition 분포 불일치 없음
    """
    if len(z_history) == 0:
        return 0.0
    z_final = z_history[-1].float().to(device)
    z_g     = z_goal_transition.float().to(device)
    return -(z_final - z_g).norm(dim=-1).mean().item()


def _pca_project(z, pca):
    """numpy array (1, D) or (D,) → PCA projected tensor (1, K)"""
    z_np = z.float().cpu().numpy().reshape(1, -1)
    return torch.tensor(pca.transform(z_np), dtype=torch.float32)


@torch.no_grad()
def latent_reward(z_history, z_goal, device, metric='cosine', z_init=None, pca=None,
                  phase_threshold=None, aggregation='mean'):
    """
    Reward = normalized cosine progress at ENDPOINT (z_T only) toward goal(s).

    phase_threshold: float or None. If set, binarize each goal's progress:
                     1.0 if progress > threshold, else 0.0 (direction 1)
    aggregation:     'mean' (default) or 'max' — how to combine multi-goal rewards (direction 2)
    """
    if len(z_history) == 0:
        return 0.0

    # Multi-goal: recurse per goal, optionally binarize, then aggregate
    if isinstance(z_goal, (list, tuple)):
        rewards = [latent_reward(z_history, g, device, metric, z_init, pca) for g in z_goal]
        if phase_threshold is not None:
            rewards = [1.0 if r > phase_threshold else 0.0 for r in rewards]
        if aggregation == 'max':
            return float(max(rewards))
        return float(sum(rewards) / len(rewards))

    # PCA projection (CPU)
    if pca is not None:
        z_final_raw = z_history[-1].float()
        z_final = _pca_project(z_final_raw, pca).to(device)
        z_g     = _pca_project(z_goal.float(), pca).to(device)
        z_i     = _pca_project(z_init.float(), pca).to(device) if z_init is not None else None
    else:
        z_final = z_history[-1].float().to(device)
        z_g     = z_goal.float().to(device)
        z_i     = z_init.float().to(device) if z_init is not None else None

    if metric == 'cosine':
        if z_i is not None:
            cos_0 = F.cosine_similarity(z_i, z_g).mean().item()
            denom = max(1.0 - cos_0, 1e-4)
            cos_T = F.cosine_similarity(z_final, z_g).mean().item()
            return (cos_T - cos_0) / denom
        return F.cosine_similarity(z_final, z_g).mean().item()
    elif metric == 'delta_cosine':
        # cos(z_T - z_0, z_goal - z_0): 변화 방향 비교
        # pca=None이면 전체 latent, pca!=None이면 PCA 공간에서 delta
        if z_i is None:
            return 0.0
        dz_T    = z_final - z_i
        dz_goal = z_g - z_i
        return F.cosine_similarity(dz_T, dz_goal).mean().item()
    else:
        return -(z_final - z_g).norm(dim=-1).mean().item()


@torch.no_grad()
def _get_dino_cls_attn(dino, dino_img):
    """
    DINOv2 마지막 block의 CLS→patch attention 추출.
    Returns: (B, 256) float CPU tensor
    """
    captured = {}

    def _hook(module, inp, out):
        x = inp[0]                                        # (B, N, C)
        B, N, C = x.shape
        head_dim = C // module.num_heads
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)                          # (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * module.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        # patch tokens are the last 256 (after CLS + any register tokens)
        captured['attn'] = attn[:, :, 0, -256:].mean(1).float().cpu()  # (B, 256)

    handle = dino.blocks[-1].attn.register_forward_hook(_hook)
    with torch.no_grad():
        dino.forward_features(dino_img)
    handle.remove()
    return captured['attn']


def compute_spatial_patch_weights(
    demo_dir, task, encoder, transition, max_demos, device, dtype,
    method='attn', top_k=64, stride=4,
):
    """
    Demo transitions에서 spatial patch weights 계산.
    method='attn'       : SpatialSWMTransition 마지막 layer action→patch attention
    method='delta'      : Δs = ||s_{t+1} - s_t||₂ per patch
    method='dino_attn'  : DINOv2 CLS→patch attention (마지막 block, 학습 불필요)
    method='pca_variance': 패치별 feature variance across demos (고변동 = task-relevant)
    Returns: (256,) normalized weight tensor (CPU)
    """
    import h5py
    from torchvision import transforms

    hdf5 = os.path.join(
        demo_dir, 'demos', 'core_datasets', task,
        f'demo_src_{task}_task_D0', 'demo.hdf5'
    )
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
    ])

    # pca_variance: Welford online variance per patch position
    if method == 'pca_variance':
        n_patches = 256
        sum_s  = torch.zeros(n_patches)          # sum of ||s_t^i||^2 (scalar per patch)
        sum_sq = torch.zeros(n_patches)          # sum of ||s_t^i||^2
        count  = 0
        with h5py.File(hdf5, 'r') as f:
            demos = sorted(f['data'].keys())[:max_demos]
            for dn in demos:
                imgs = f[f'data/{dn}/obs/agentview_image']
                T = len(imgs)
                for t in range(0, T, stride):
                    img_t = transform(np.array(imgs[t])).unsqueeze(0).to(device)
                    with torch.no_grad():
                        s_t = encoder.encode_spatial_projected(img_t).float().cpu()  # (1, 256, d_s)
                    s = s_t.squeeze(0)                          # (256, d_s)
                    norms_sq = (s * s).sum(-1)                  # (256,) ||s^i||^2
                    norms    = norms_sq.sqrt()                  # (256,)
                    sum_s  += norms
                    sum_sq += norms_sq
                    count  += 1
        # Var[||s^i||] ≈ E[||s^i||^2] - E[||s^i||]^2
        mean_sq = sum_sq / max(count, 1)
        mean    = sum_s  / max(count, 1)
        var     = (mean_sq - mean ** 2).clamp(min=0)          # (256,)
        # top-k hard mask
        if top_k < 256:
            mask = torch.zeros(256)
            mask[var.topk(top_k).indices] = 1.0
            var = var * mask
        weights = var / var.sum().clamp(min=1e-8)
        if is_main():
            nonzero = int((weights > 0).sum())
            log.info(f"[patch_weights/pca_variance] demos={len(demos)}  steps={count}  "
                     f"nonzero={nonzero}/256  top_k={top_k}")
        return weights

    # dino_attn: DINOv2 CLS attention (no transition needed)
    if method == 'dino_attn':
        dino = encoder.backbone.dino
        weights_acc = torch.zeros(256)
        count = 0
        with h5py.File(hdf5, 'r') as f:
            demos = sorted(f['data'].keys())[:max_demos]
            for dn in demos:
                imgs = f[f'data/{dn}/obs/agentview_image']
                T = len(imgs)
                for t in range(0, T, stride):
                    img_t = transform(np.array(imgs[t])).unsqueeze(0).to(device)
                    # DINO normalization
                    m = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1,3,1,1)
                    s = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1,3,1,1)
                    dino_img = (img_t.to(dtype) - m) / s
                    w = _get_dino_cls_attn(dino, dino_img).squeeze(0)  # (256,)
                    weights_acc += w
                    count += 1
        weights = weights_acc / max(count, 1)
        if top_k < 256:
            mask = torch.zeros(256)
            mask[weights.topk(top_k).indices] = 1.0
            weights = weights * mask
        weights = weights / weights.sum().clamp(min=1e-8)
        if is_main():
            nonzero = int((weights > 0).sum())
            log.info(f"[patch_weights/dino_attn] demos={len(demos)}  steps={count}  "
                     f"nonzero={nonzero}/256  top_k={top_k}")
        return weights

    # attn / delta (original methods)
    weights_acc = torch.zeros(256)
    count = 0
    with h5py.File(hdf5, 'r') as f:
        demos = sorted(f['data'].keys())[:max_demos]
        for dn in demos:
            imgs    = f[f'data/{dn}/obs/agentview_image']
            actions = np.array(f[f'data/{dn}/actions'])
            T = len(imgs)
            for t in range(0, T - 1, stride):
                img_t = transform(np.array(imgs[t])).unsqueeze(0).to(device)
                s_t   = encoder.encode_spatial_projected(img_t).to(dtype)
                a_t   = torch.from_numpy(actions[t]).to(device, dtype).unsqueeze(0)
                if method == 'attn':
                    w = transition.get_patch_weights(s_t, a_t, top_k=top_k)
                else:
                    w = transition.get_delta_weights(s_t, a_t, top_k=top_k)
                weights_acc += w.float()
                count += 1

    weights = weights_acc / max(count, 1)
    weights = weights / weights.sum().clamp(min=1e-8)
    if is_main():
        nonzero = int((weights > 0).sum())
        log.info(f"[patch_weights/{method}] demos={len(demos)}  steps={count}  "
                 f"nonzero={nonzero}/256  top_k={top_k}")
    return weights   # (256,) CPU


def fit_demo_pca(demo_dir, task, encoder, max_demos, device, dtype, n_components=16, stride=2, token_weights=None):
    """
    demo 전체 프레임을 인코딩해 PCA fit → task-relevant subspace 추출.
    token_weights 제공 시 encode_weighted 사용 (spatial pooling과 동일 분포).
    반환: sklearn PCA object (CPU-side, numpy 기반)
    """
    import h5py
    from torchvision import transforms
    from sklearn.decomposition import PCA as SklearnPCA

    hdf5 = os.path.join(
        demo_dir, 'demos', 'core_datasets', task,
        f'demo_src_{task}_task_D0', 'demo.hdf5'
    )
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
    ])
    all_z = []
    with h5py.File(hdf5, 'r') as f:
        demos = sorted(f['data'].keys())[:max_demos]
        for dn in demos:
            imgs = f[f'data/{dn}/obs/agentview_image']
            for i in range(0, len(imgs), stride):
                t = transform(np.array(imgs[i])).unsqueeze(0).to(device)
                if hasattr(encoder, 'spatial_proj') and getattr(encoder, 'spatial_dim', None) is not None and token_weights is None:
                    z = encoder.encode_spatial_projected(t).mean(dim=1).float().cpu().squeeze(0).numpy()
                elif token_weights is not None:
                    z = encoder.encode_weighted(t, token_weights).float().cpu().squeeze(0).numpy()
                else:
                    z = encoder(t).float().cpu().squeeze(0).numpy()
                all_z.append(z)
    all_z = np.stack(all_z)   # (N, 2176)
    pca = SklearnPCA(n_components=n_components)
    pca.fit(all_z)
    cumvar = float(np.sum(pca.explained_variance_ratio_) * 100)
    log.info(f"Demo PCA fit: {len(all_z)} frames → top-{n_components} dims  "
             f"cumulative variance={cumvar:.1f}%")
    return pca


@torch.no_grad()
def compute_spatial_weights(demo_dir, task, encoder, max_demos, device, dtype,
                             method='variance', top_k=64, stride=4):
    """
    Demo 궤적에서 task-relevant spatial token weight 계산.
      variance   : temporal variance가 큰 토큰 (가장 많이 변하는 위치)
      correlation: task progress(t/T)와 norm이 가장 상관된 토큰
      activation : 평균 activation norm이 가장 큰 토큰
    Returns: (256,) normalized weight tensor (CPU)
    """
    import h5py
    from torchvision import transforms

    hdf5 = os.path.join(
        demo_dir, 'demos', 'core_datasets', task,
        f'demo_src_{task}_task_D0', 'demo.hdf5'
    )
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    all_spatial  = []
    all_progress = []

    with h5py.File(hdf5, 'r') as f:
        demos = sorted(f['data'].keys())[:max_demos]
        for dn in demos:
            imgs = f[f'data/{dn}/obs/agentview_image']
            T = len(imgs)
            indices = list(range(0, T, stride))
            seq = []
            for idx in indices:
                img_t = transform(imgs[idx]).unsqueeze(0).to(device, dtype)
                sp = encoder.backbone.forward_spatial(img_t)   # (1, 256, 2176)
                seq.append(sp.squeeze(0).float().cpu())
            all_spatial.append(torch.stack(seq))               # (T_s, 256, 2176)
            all_progress.append(torch.linspace(0, 1, len(indices)))

    if method == 'variance':
        parts = []
        for sp in all_spatial:
            parts.append(sp.var(dim=0).norm(dim=-1))           # (256,)
        weights = torch.stack(parts).mean(dim=0)

    elif method == 'correlation':
        parts = []
        for sp, prog in zip(all_spatial, all_progress):
            norms  = sp.norm(dim=-1)                           # (T_s, 256)
            p      = prog.unsqueeze(-1)                        # (T_s, 1)
            n_c    = norms - norms.mean(dim=0, keepdim=True)
            p_c    = p - p.mean()
            cov    = (n_c * p_c).mean(dim=0)                   # (256,)
            std_n  = norms.std(dim=0).clamp(min=1e-8)
            std_p  = prog.std().clamp(min=1e-8)
            parts.append((cov / (std_n * std_p)).abs())
        weights = torch.stack(parts).mean(dim=0)

    elif method == 'activation':
        parts = []
        for sp in all_spatial:
            parts.append(sp.norm(dim=-1).mean(dim=0))          # (256,)
        weights = torch.stack(parts).mean(dim=0)

    elif method == 'slot_attention':
        import torch.nn.functional as F_sa
        n_slots = 4
        D = all_spatial[0].shape[-1]

        # 모든 demo 토큰을 합쳐서 (N_total, D)
        all_tokens = torch.cat([sp.reshape(-1, D) for sp in all_spatial], dim=0)

        # PCA로 slot 초기화 (task-relevant 방향)
        _, _, V = torch.pca_lowrank(all_tokens, q=n_slots, niter=4)
        slots = F_sa.normalize(V.T, dim=-1)  # (n_slots, D)

        # Iterative slot competition (3회)
        for _ in range(3):
            attn = F_sa.softmax(
                torch.einsum('nd,kd->nk', F_sa.normalize(all_tokens, dim=-1), slots),
                dim=-1)                                         # (N, n_slots)
            slot_sum = torch.einsum('nk,nd->kd', attn, all_tokens)
            slots = F_sa.normalize(slot_sum / attn.sum(0).unsqueeze(-1).clamp(min=1e-8), dim=-1)

        # 시간적으로 가장 많이 변하는 slot 선택 (task-relevant)
        slot_var = torch.zeros(n_slots)
        for sp in all_spatial:
            T_s, P, _ = sp.shape
            a = F_sa.softmax(
                torch.einsum('nd,kd->nk',
                             F_sa.normalize(sp.reshape(-1, D), dim=-1), slots),
                dim=-1).reshape(T_s, P, n_slots)
            slot_var += a.var(dim=0).mean(dim=0)               # (n_slots,)
        best_slot = int(slot_var.argmax())

        # 선택된 slot의 attention을 토큰 가중치로
        w_sum = torch.zeros(256)
        for sp in all_spatial:
            T_s, P, _ = sp.shape
            a = F_sa.softmax(
                torch.einsum('nd,kd->nk',
                             F_sa.normalize(sp.reshape(-1, D), dim=-1), slots),
                dim=-1).reshape(T_s, P, n_slots)
            w_sum += a[:, :, best_slot].mean(dim=0)            # (256,)
        weights = w_sum / len(all_spatial)
        if is_main():
            log.info(f"[slot_attention] best_slot={best_slot}  slot_var={slot_var.tolist()}")

    else:
        raise ValueError(f"Unknown spatial pooling method: {method}")

    if top_k < 256:
        mask = torch.zeros(256)
        mask[weights.topk(top_k).indices] = 1.0
        weights = weights * mask

    weights = weights / weights.sum().clamp(min=1e-8)
    if is_main():
        log.info(f"[spatial_weights] method={method}  top_k={top_k}  "
                 f"non-zero={int((weights > 0).sum())}")
    return weights


@torch.no_grad()
def reward_model_reward(z_history, z_goal, reward_model, device):
    """
    Temporal Transformer reward: sample_traj(z_history) → SWMRewardModel → P(success)
    z_history: list of (1, 2176) latents from WM transition
    z_goal:    unused (kept for API compatibility)
    """
    from models.reward_model import sample_traj
    if len(z_history) == 0:
        return 0.0
    n_frames = reward_model.n_frames
    z_traj = sample_traj(z_history, n_frames).float().to(device)  # (1, T, 2176)
    return reward_model.reward(z_traj)


@torch.no_grad()
def load_initial_states(demo_dir, task, encoder, max_demos, device, dtype,
                        reward_type='graph', transition=None, rollout_steps=64,
                        n_goals=1, token_weights=None, pca=None,
                        pca_goal_threshold=0.2, patch_weights=None):
    """
    데모 첫 프레임을 인코딩.

    Returns:
        init_images  : list of (1, 3, H, W) CPU float tensor  (VLA spatial 인코딩용)
        init_latents : list of (1, 2176) CPU tensor            (WM transition 시작점)
        init_goals   : list — content depends on reward_type:
                       'graph'          → list of (N, 3) object position tensors
                       'latent'         → list of (1, 2176) encoder(last_frame) latents
                       'reward_model'   → list of (1, 2176) encoder(last_frame) latents
                       'transition_l2'  → list of (1, 2176) transition(z0, demo_actions[:rollout_steps])
    """
    import h5py
    from torchvision import transforms

    hdf5 = os.path.join(
        demo_dir, 'demos', 'core_datasets', task,
        f'demo_src_{task}_task_D0', 'demo.hdf5'
    )
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize는 encoder 내부에서 backbone별로 적용
    ])

    init_images  = []
    init_latents = []
    init_goals   = []
    init_goal_images = []   # 데모 마지막 프레임(goal) 이미지 — EMA reward LLM 입력용

    with h5py.File(hdf5, 'r') as f:
        demos = sorted(f['data'].keys())[:max_demos]
        for dn in demos:
            img_np    = f[f'data/{dn}/obs/agentview_image'][0]
            image_t   = transform(img_np).unsqueeze(0)
            image_gpu = image_t.to(device)
            # goal 이미지 = 데모 마지막 프레임
            goal_img_np = f[f'data/{dn}/obs/agentview_image'][-1]
            init_goal_images.append(transform(goal_img_np).unsqueeze(0).cpu())
            if hasattr(encoder, 'spatial_proj') and getattr(encoder, 'spatial_dim', None) is not None and token_weights is None:
                z = encoder.encode_spatial_projected(image_gpu).to(dtype)  # (1, 256, d_s)
            elif token_weights is not None:
                z = encoder.encode_weighted(image_gpu, token_weights).to(dtype)
            else:
                z = encoder(image_gpu).to(dtype)

            init_images.append(image_t.cpu())
            init_latents.append(z.cpu())

            if reward_type == 'transition_l2':
                # demo actions를 transition으로 rollout → z_goal도 transition 공간
                demo_actions = f[f'data/{dn}/actions'][:]      # (T, 7) raw actions
                n_steps = min(rollout_steps, len(demo_actions))
                z_t = z.clone()
                for t in range(n_steps):
                    a = torch.from_numpy(demo_actions[t]).to(device, dtype).unsqueeze(0)
                    z_t = transition(z_t.to(dtype), a)
                init_goals.append(z_t.cpu())

            elif reward_type in ('latent', 'reward_model'):
                imgs = f[f'data/{dn}/obs/agentview_image']
                T_demo = len(imgs)
                if n_goals == 'pca_adaptive':
                    assert pca is not None, "n_goals='pca_adaptive' requires fitted PCA"
                    # ── PCA 공간에서 데모 trajectory 투영 ─────────────────────────
                    local_stride = max(1, T_demo // 200)
                    frame_indices = list(range(0, T_demo, local_stride))
                    if frame_indices[-1] != T_demo - 1:
                        frame_indices.append(T_demo - 1)

                    def _encode_np(idx):
                        img_t = transform(np.array(imgs[idx])).unsqueeze(0).to(device)
                        if hasattr(encoder, 'spatial_proj') and getattr(encoder, 'spatial_dim', None) is not None and token_weights is None:
                            return encoder.encode_spatial_projected(img_t).mean(dim=1).float().cpu().numpy().flatten()
                        elif token_weights is not None:
                            return encoder.encode_weighted(img_t, token_weights).float().cpu().numpy().flatten()
                        else:
                            return encoder(img_t).float().cpu().numpy().flatten()

                    pca_vecs = np.stack([pca.transform(_encode_np(i).reshape(1, -1))[0]
                                         for i in frame_indices])  # (T', n_pca)

                    # ── start→end 방향으로 normalized scalar projection 계산 ──────
                    p0, pT = pca_vecs[0], pca_vecs[-1]
                    direction = pT - p0
                    total_sq  = float(np.dot(direction, direction)) + 1e-8
                    projections = [float(np.dot(pv - p0, direction) / total_sq)
                                   for pv in pca_vecs]

                    # ── threshold 초과 시마다 goal frame 선택 ────────────────────
                    goal_frame_indices = []
                    last_proj = 0.0
                    for fi, proj in zip(frame_indices, projections):
                        if proj - last_proj >= pca_goal_threshold:
                            goal_frame_indices.append(fi)
                            last_proj = proj
                    if not goal_frame_indices or goal_frame_indices[-1] != T_demo - 1:
                        goal_frame_indices.append(T_demo - 1)

                    indices = goal_frame_indices

                elif n_goals == 1:
                    indices = [T_demo - 1]
                elif n_goals == 'action':
                    # action velocity 기반 phase 감지:
                    # ||a_t - a_{t-1}|| 피크 = task 전환 순간 → goal frame으로 사용
                    acts = np.array(f[f'data/{dn}/actions'])    # (T, 7)
                    vel  = np.linalg.norm(np.diff(acts, axis=0), axis=1)  # (T-1,)
                    # gaussian smoothing으로 noise 제거
                    from scipy.ndimage import gaussian_filter1d
                    vel_smooth = gaussian_filter1d(vel, sigma=3.0)
                    # 균등 분할된 K=4 구간에서 각 구간 내 최대 피크 선택
                    K = 4
                    seg_len = max(1, (T_demo - 1) // K)
                    indices = []
                    for k in range(K):
                        lo = k * seg_len
                        hi = (k + 1) * seg_len if k < K - 1 else T_demo - 1
                        seg = vel_smooth[lo:hi]
                        if len(seg) == 0:
                            indices.append(min(lo + seg_len // 2, T_demo - 1))
                        else:
                            peak_in_seg = int(np.argmax(seg)) + lo + 1  # +1: diff offset
                            indices.append(min(peak_in_seg, T_demo - 1))
                else:
                    # K goals uniformly sampled: 1/K, 2/K, ..., K/K of demo
                    indices = [max(0, int(round((k + 1) * (T_demo - 1) / n_goals)))
                               for k in range(n_goals)]
                goal_latents = []
                for idx in indices:
                    goal_t = transform(imgs[idx]).unsqueeze(0).to(device)
                    if hasattr(encoder, 'spatial_proj') and getattr(encoder, 'spatial_dim', None) is not None and token_weights is None:
                        sp = encoder.encode_spatial_projected(goal_t).to(dtype)  # (1, 256, d_s)
                        if patch_weights is not None:
                            w = patch_weights.to(sp.device, sp.dtype)
                            goal_latents.append((sp * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1).cpu())
                        else:
                            goal_latents.append(sp.mean(dim=1).cpu())
                    elif token_weights is not None:
                        goal_latents.append(encoder.encode_weighted(goal_t, token_weights).to(dtype).cpu())
                    else:
                        goal_latents.append(encoder(goal_t).to(dtype).cpu())
                # single goal → Tensor, multi-goal or pca_adaptive → list[Tensor]
                if n_goals == 1:
                    init_goals.append(goal_latents[0])
                else:
                    if n_goals == 'pca_adaptive' and is_main():
                        log.info(f"  [{dn}] pca_adaptive: {len(goal_latents)} goals "
                                 f"(threshold={pca_goal_threshold:.2f})")
                    init_goals.append(goal_latents)

            else:  # 'graph'
                obj = f[f'data/{dn}/obs/object'][-1]
                from data.dataset import parse_object_positions
                pos = parse_object_positions(obj, task)
                init_goals.append(torch.from_numpy(pos))

    log.info(f"[init_states] {task}: {len(init_latents)} demos encoded  reward_type={reward_type}")
    return init_images, init_latents, init_goals, init_goal_images


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',           default='configs/stage3.yaml')
    p.add_argument('--no-wandb',         action='store_true')
    p.add_argument('--resume',           type=str, default=None)
    p.add_argument('--debug',            action='store_true')
    p.add_argument('--probe',            type=int, default=None,
                   help='Run only N iters then print reward diagnostics and exit')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    out_dir / 'train.log',
                    mode='a' if args.resume else 'w'
                ),
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
    dtype  = torch.bfloat16
    if is_main():
        log.info(f"Device: {device}  DDP: {use_ddp}  dtype: {dtype}")
        latest = out_dir.parent / 'latest'
        try:
            latest.unlink()
        except FileNotFoundError:
            pass
        try:
            latest.symlink_to(out_dir.name)
        except FileExistsError:
            pass

    # ── Load SWM ─────────────────────────────────────────────────────────────
    log.info("Loading SWM checkpoints ...")
    # Stage 2: transition
    s2_ckpt = torch.load(cfg.swm_ckpt, map_location=device)
    s2_cfg  = OmegaConf.create(s2_ckpt.get('cfg', {}))
    # Stage 1: encoder + heads
    s1_ckpt = torch.load(cfg.stage1_ckpt, map_location=device)
    s1_cfg  = OmegaConf.create(s1_ckpt.get('cfg', {}))

    use_spatial_swm = 'spatial_proj' in s2_ckpt and 'spatial_dim' in s2_ckpt
    spatial_dim_swm = s2_ckpt.get('spatial_dim', 256) if use_spatial_swm else None

    spatial_dim_for_enc = spatial_dim_swm if use_spatial_swm else 256
    encoder = SWMEncoder(
        vla_path=cfg.vla_path,
        freeze_backbone=True,
        latent_dim=2176,
        spatial_dim=spatial_dim_for_enc,
    ).to(device)
    # spatial_proj는 stage1 ckpt에 없음 → strict=False
    encoder.load_state_dict(s1_ckpt['encoder'], strict=False)

    if use_spatial_swm:
        from models.transition import SpatialSWMTransition
        transition = SpatialSWMTransition(
            spatial_dim=spatial_dim_swm,
            action_dim=7,
        ).to(device).to(dtype)
        transition.load_state_dict(s2_ckpt['transition'])
        if spatial_dim_swm == 2176:
            encoder.spatial_proj = nn.Identity().to(device)
        else:
            encoder.spatial_proj = nn.Sequential(
                nn.Linear(2176, spatial_dim_swm),
                nn.LayerNorm(spatial_dim_swm),
            ).to(device)
            encoder.spatial_proj.load_state_dict(s2_ckpt['spatial_proj'])
        encoder.spatial_dim = spatial_dim_swm
        for p in encoder.spatial_proj.parameters():
            p.requires_grad = False
        encoder.spatial_proj.eval()
    else:
        transition = SWMTransition(
            latent_dim=2176,
            action_dim=7,
            hidden_dim=s2_cfg.get('transition', {}).get('hidden_dim', 512),
            num_layers=s2_cfg.get('transition', {}).get('num_layers', 6),
            num_heads =s2_cfg.get('transition', {}).get('num_heads',  8),
        ).to(device)
        transition.load_state_dict(s2_ckpt['transition'])

    heads = SWMHeads(
        latent_dim=2176,
        max_nodes=s1_cfg.get('encoder', {}).get('max_nodes', 8),
    ).to(device)
    heads.load_state_dict(s1_ckpt['heads'])

    # freeze encoder, transition, heads
    for m in [encoder, transition, heads]:
        for p in m.parameters():
            p.requires_grad = False

    encoder.eval(); transition.eval(); heads.eval()
    log.info("SWM loaded (frozen)")

    # ── Load VLA ─────────────────────────────────────────────────────────────
    log.info(f"Loading VLA from {cfg.vla_path} ...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from verl.utils.openvla_utils import update_auto_map

    # rank 0만 config.json 수정 → 나머지 rank는 barrier 후 읽기
    if not use_ddp or dist.get_rank() == 0:
        update_auto_map(cfg.vla_path)
    if use_ddp:
        dist.barrier()
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    # full fine-tune (p128-style) or LoRA-only (q/v_proj, projector optionally frozen)
    full_finetune    = cfg.training.get('full_finetune', False)
    freeze_projector = cfg.training.get('freeze_projector', True)
    if full_finetune:
        for p in vla.parameters():
            p.requires_grad = True
        if cfg.training.get('gradient_checkpointing', False):
            try:
                vla.gradient_checkpointing_enable()
                if is_main():
                    log.info("gradient_checkpointing enabled")
            except Exception as e:
                if is_main():
                    log.warning(f"gradient_checkpointing failed: {e}")
    else:
        for p in vla.parameters():
            p.requires_grad = False
        if not freeze_projector:
            for p in vla.projector.parameters():
                p.requires_grad = True
        for name, p in vla.language_model.named_parameters():
            if 'q_proj' in name or 'v_proj' in name:
                p.requires_grad = True

    n_total = sum(p.numel() for p in vla.parameters())
    n_trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    if is_main():
        log.info(f"VLA trainable: {n_trainable/1e9:.2f}B / total {n_total/1e9:.2f}B  full_finetune={full_finetune}")

    vla_raw = vla

    # action codec
    from data.dataset import TASK_OBJECT_CONFIG
    task = cfg.task
    unnorm_key = cfg.get('unnorm_key', task)

    if unnorm_key not in vla_raw.norm_stats:
        import json
        ds_stats = json.load(open(Path(cfg.vla_path) / 'dataset_statistics.json'))
        vla_raw.norm_stats.update(ds_stats)

    stats = vla_raw.norm_stats[unnorm_key]['action']
    q01 = torch.tensor(stats['q01'], device=device, dtype=torch.float32)
    q99 = torch.tensor(stats['q99'], device=device, dtype=torch.float32)
    n_bins = vla_raw.bin_centers.shape[0] + 1
    vocab_size = vla_raw.vocab_size

    def tokens_to_actions(tids):
        """(1, 56) vocab token ids → (1, 8, 7) continuous actions"""
        tids = tids.to(device)
        # action tokens occupy the last n_bins-1 vocab IDs (highest IDs)
        bin_idx = (vocab_size - tids - 1).clamp(0, n_bins - 2)
        norm = (bin_idx.float() / (n_bins - 1)) * 2.0 - 1.0   # [-1, 1]
        norm = norm.reshape(1, NUM_ACTIONS_CHUNK, ACTION_DIM)
        acts = (norm + 1.0) / 2.0 * (q99 - q01) + q01
        return acts

    # ── Prompt (text-only tokenization — image tokens은 patch_embeds로 직접 주입) ──
    task_desc = _TASK_DESC.get(task, task)
    prompt    = f"In: What action should the robot take to {task_desc}?\nOut:"
    feat      = processor.tokenizer(prompt, return_tensors='pt')
    p_ids     = feat['input_ids'].to(device)
    a_mask    = feat['attention_mask'].to(device)

    # ── Initial states (images + global latents + goals) ─────────────────────
    reward_type = cfg.get('reward_type', 'graph')
    reward_metric = cfg.get('reward_metric', 'cosine')
    phase_threshold = cfg.get('phase_threshold', None)
    reward_aggregation = cfg.get('reward_aggregation', 'mean')
    reward_rank_normalize = cfg.get('reward_rank_normalize', False)
    spatial_pooling = cfg.get('spatial_pooling', 'mean')
    spatial_top_k   = cfg.get('spatial_top_k', 64)

    # ── Spatial token weights (variance / correlation / activation) ───────────
    # PCA보다 먼저 계산: PCA fitting 시 동일 인코딩 방식 사용 위함
    token_weights = None
    if spatial_pooling in ('variance', 'correlation', 'activation', 'slot_attention'):
        if is_main():
            log.info(f"Computing spatial weights (method={spatial_pooling}, top_k={spatial_top_k}) ...")
        token_weights = compute_spatial_weights(
            cfg.data_root, task, encoder, cfg.training.max_demos,
            device, dtype, method=spatial_pooling, top_k=spatial_top_k,
        )

    # ── Spatial patch weights (attn / delta) — Phase 2 전용 ──────────────────
    patch_weights = None
    patch_weight_method = cfg.get('patch_weight_method', None)
    if patch_weight_method in ('attn', 'delta', 'dino_attn', 'pca_variance') and use_spatial_swm:
        spatial_top_k_pw = cfg.get('patch_weight_top_k', 64)
        if is_main():
            log.info(f"Computing spatial patch weights (method={patch_weight_method}, "
                     f"top_k={spatial_top_k_pw}) ...")
        patch_weights = compute_spatial_patch_weights(
            cfg.data_root, task, encoder, transition, cfg.training.max_demos,
            device, dtype, method=patch_weight_method, top_k=spatial_top_k_pw,
        )

    # ── PCA fit (pca_cosine / pca_delta_cosine) ──────────────────────────────
    # token_weights 전달 → spatial pooling 시 encode_weighted로 PCA 학습 (분포 일치)
    demo_pca = None
    n_goals = cfg.get('n_goals', 1)
    if reward_metric in ('pca_cosine', 'pca_delta_cosine') or n_goals == 'pca_adaptive':
        n_pca = cfg.get('pca_components', 16)
        if is_main():
            log.info(f"Fitting demo PCA (n_components={n_pca}) ...")
        demo_pca = fit_demo_pca(
            cfg.data_root, task, encoder, cfg.training.max_demos,
            device, dtype, n_components=n_pca, token_weights=token_weights,
        )

    if is_main():
        log.info(f"Pre-encoding initial states ...  reward_type={reward_type}")
    # transition_l2: rollout_steps = n_rollout_chunks × NUM_ACTIONS_CHUNK
    rollout_steps = cfg.training.n_rollout_chunks * NUM_ACTIONS_CHUNK
    init_images, init_latents, init_goals, init_goal_images = load_initial_states(
        cfg.data_root, task, encoder, cfg.training.max_demos, device, dtype,
        reward_type=reward_type,
        transition=transition,
        rollout_steps=rollout_steps,
        n_goals=n_goals,
        token_weights=token_weights,
        pca=demo_pca,
        pca_goal_threshold=cfg.get('pca_goal_threshold', 0.2),
        patch_weights=patch_weights,
    )

    # ── Reward model 로딩 (reward_type='reward_model'일 때만) ─────────────────
    rm = None
    if reward_type == 'reward_model':
        from models.reward_model import SWMRewardModel
        rm_ckpt_path = cfg.get('reward_model_ckpt',
                                f'outputs/stage2_5/{task}/best.pt')
        rm_ckpt = torch.load(rm_ckpt_path, map_location=device, weights_only=False)
        rm = SWMRewardModel(
            latent_dim=cfg.get('reward_model', {}).get('latent_dim', 2176),
            hidden_dim=cfg.get('reward_model', {}).get('hidden_dim', 512),
            n_layers  =cfg.get('reward_model', {}).get('n_layers', 3),
        ).to(device)
        rm.load_state_dict(rm_ckpt['reward_model'])
        rm.eval()
        for p in rm.parameters():
            p.requires_grad = False
        if is_main():
            log.info(f"RewardModel loaded from {rm_ckpt_path}  (frozen)")

    # ── WM Bridge (spatial_dim<2176 전용: transition 출력을 projector 입력으로 업샘플)
    # spatial_dim=2176이면 transition 출력이 이미 projector 입력과 동일 → 불필요
    wm_bridge = None
    if use_spatial_swm and spatial_dim_swm != 2176:
        wm_bridge = nn.Linear(spatial_dim_swm, 2176).to(device).to(dtype)
        if is_main():
            log.info(f"wm_bridge: Linear({spatial_dim_swm}, 2176)  [trainable]")

    # ── DDP 감싸기 ────────────────────────────────────────────────────────────
    if use_ddp:
        vla = DDP(vla, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable = [p for p in vla_raw.parameters() if p.requires_grad]
    if wm_bridge is not None:
        trainable = trainable + list(wm_bridge.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.training.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.iterations, eta_min=cfg.training.lr * 0.1
    )

    # ── EMA reward LLM (느린 target = EMA(action LLM)) ─────────────────────────
    #   변형1 (mode=only):     reward = pre-action hidden cosine(policy, EMA)  [grounding 없음]
    #   변형2 (mode=additive): reward = WM grounding + beta * 위 cosine
    import copy as _copy
    ema_cfg        = cfg.get('ema_reward', {}) or {}
    use_ema_reward = bool(ema_cfg.get('enable', False))
    ema_mode       = ema_cfg.get('mode', 'additive')      # 'only' | 'additive'
    ema_metric     = ema_cfg.get('metric', 'cosine')      # 'cosine'(goal hidden) | 'probe'(progress 상관축)
    ema_tau        = float(ema_cfg.get('tau', 0.99))
    ema_beta       = float(ema_cfg.get('beta', 1.0))
    ema_lm         = None
    probe_w        = None
    progress_mlp   = None
    goaldist_mlp   = None
    if use_ema_reward:
        ema_lm = _copy.deepcopy(vla_raw.language_model).eval()
        for p in ema_lm.parameters():
            p.requires_grad_(False)
        if is_main():
            log.info(f"[EMA reward] enabled  mode={ema_mode}  metric={ema_metric}  tau={ema_tau}  beta={ema_beta}")
        if ema_metric in ('probe', 'pls'):
            # 진전 상관축 w 를 데모 hidden→frame_fraction 으로 1회 fit (EMA LLM 사용)
            #   metric=probe → Ridge,  metric=pls → PLS(공분산 최대화)
            probe_w = fit_progress_probe(
                cfg.data_root, task, encoder, vla_raw, ema_lm, p_ids, a_mask,
                cfg.training.max_demos, device, dtype,
                method=('pls' if ema_metric == 'pls' else 'ridge'),
            )
        elif ema_metric == 'tcn':
            # time-contrastive: 비선형 progress MLP (시간 ranking)
            progress_mlp = fit_tcn_progress(
                cfg.data_root, task, encoder, vla_raw, ema_lm, p_ids, a_mask,
                cfg.training.max_demos, device, dtype,
            )
        elif ema_metric == 'goaldist':
            # contrastive goal-distance: 임베딩 φ (goal까지 거리=남은 진전)
            goaldist_mlp = fit_goaldist(
                cfg.data_root, task, encoder, vla_raw, ema_lm, p_ids, a_mask,
                cfg.training.max_demos, device, dtype,
            )
    else:
        ema_mode = None   # 비활성: 기존 grounding reward 그대로

    # ── Resume ────────────────────────────────────────────────────────────────
    start_iter  = 1
    best_reward = -float('inf')

    if args.resume and Path(args.resume).exists():
        ckpt_r = torch.load(args.resume, map_location=device)
        vla_raw.load_state_dict(ckpt_r['vla'])
        if wm_bridge is not None and 'wm_bridge' in ckpt_r:
            wm_bridge.load_state_dict(ckpt_r['wm_bridge'])
        if 'optimizer' in ckpt_r:
            optimizer.load_state_dict(ckpt_r['optimizer'])
        if 'scheduler' in ckpt_r:
            scheduler.load_state_dict(ckpt_r['scheduler'])
        start_iter  = ckpt_r.get('iteration', 0) + 1
        best_reward = ckpt_r.get('reward_mean', -float('inf'))
        log.info(f"Resumed from {args.resume}  iter={start_iter-1}  reward={best_reward:.4f}")

    use_wandb = not args.no_wandb and not args.debug
    if use_wandb:
        import wandb
        wandb.init(project='swm', name=cfg.experiment.name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    rng = random.Random(cfg.experiment.seed)

    # ── GRPO loop ─────────────────────────────────────────────────────────────
    total_iters = args.probe if args.probe else cfg.training.iterations
    if args.probe:
        log.info(f"\n[PROBE MODE] {args.probe} iters — reward diagnostics only")
    log.info(f"\n[GRPO] {total_iters} iterations  "
             f"task={task}  g_rollouts={cfg.training.g_rollouts}")

    probe_rewards_all = []   # for probe diagnostic summary

    for iteration in range(start_iter, total_iters + 1):
        z0_batch = rng.sample(
            list(range(len(init_latents))),
            min(cfg.training.n_states_per_iter, len(init_latents))
        )
        iter_rewards = []
        iter_loss_sum = 0.0
        optimizer.zero_grad()

        for z0_idx in z0_batch:
            z0     = init_latents[z0_idx].to(device, dtype)   # global latent (WM용)
            image0 = init_images[z0_idx].to(device)            # 관측 이미지 (VLA spatial용)
            goal   = init_goals[z0_idx]

            # ── EMA reward: metric=cosine이면 goal 이미지 → reward LLM → h_goal ──
            #               metric=probe이면 probe_w(고정) 사용, goal_hidden 불필요
            goal_hidden = None
            if use_ema_reward and ema_metric in ('cosine', 'goaldist'):
                # cosine: h_goal과 직접 cosine / goaldist: φ(h_t)와 φ(h_goal) 거리
                goal_hidden = goal_pre_action_hidden(
                    init_goal_images[z0_idx], encoder, vla_raw, ema_lm,
                    p_ids, a_mask, device, dtype,
                )

            group_rewards = []
            group_data    = []   # (tids_ep, lp_ep, rollout_data) 저장

            for _ in range(cfg.training.g_rollouts):
                tids_ep, lp_ep, rollout_data, z_hist, ema_cons = swm_rollout(
                    z0, image0, encoder, transition, heads,
                    vla_raw, processor, p_ids, a_mask,
                    tokens_to_actions,
                    cfg.training.n_rollout_chunks,
                    cfg.training.temperature,
                    device, dtype,
                    patch_weights=patch_weights,
                    wm_bridge=wm_bridge,
                    goal_hidden=goal_hidden,
                    probe_w=probe_w,
                    progress_mlp=progress_mlp,
                    goaldist_mlp=goaldist_mlp,
                )
                if ema_mode == 'only':
                    # 변형1: grounding 없이 EMA consistency가 메인 reward
                    rew = ema_cons if ema_cons is not None else 0.0
                    group_rewards.append(float(rew))
                    group_data.append((tids_ep, lp_ep, rollout_data))
                    continue
                if reward_type == 'transition_l2':
                    rew = transition_l2_reward(z_hist, goal, device)
                elif reward_type == 'latent':
                    use_pca = demo_pca if reward_metric in ('pca_cosine', 'pca_delta_cosine') else None
                    if reward_metric == 'pca_cosine':
                        actual_metric = 'cosine'
                    elif reward_metric == 'pca_delta_cosine':
                        actual_metric = 'delta_cosine'
                    else:
                        actual_metric = reward_metric
                    z_init_1d = z0.mean(dim=1).cpu() if z0.ndim == 3 else z0.cpu()
                    rew = latent_reward(z_hist, goal, device,
                                        metric=actual_metric,
                                        z_init=z_init_1d,
                                        pca=use_pca,
                                        phase_threshold=phase_threshold,
                                        aggregation=reward_aggregation)
                elif reward_type == 'reward_model':
                    rew = reward_model_reward(z_hist, goal, rm, device)
                else:
                    rew = graph_reward(z_hist, heads, goal, device)
                # 변형2: 기존 grounding reward + β·EMA consistency
                if ema_mode == 'additive' and ema_cons is not None:
                    rew = float(rew) + ema_beta * float(ema_cons)
                group_rewards.append(rew)
                group_data.append((tids_ep, lp_ep, rollout_data))

            iter_rewards.extend(group_rewards)
            gr = np.array(group_rewards)
            if reward_rank_normalize:
                # force variance: replace raw rewards with rank scores in [-1, 1]
                ranks = np.argsort(np.argsort(gr)).astype(float)
                gr = (ranks / max(len(ranks) - 1, 1)) * 2.0 - 1.0
            adv_z = (gr - gr.mean()) / max(gr.std(), 1e-8)

            mini_g = cfg.training.get('mini_g', 0) or cfg.training.g_rollouts
            n_mb = (cfg.training.g_rollouts + mini_g - 1) // mini_g
            for mb_start in range(0, cfg.training.g_rollouts, mini_g):
                mb_end = min(mb_start + mini_g, cfg.training.g_rollouts)
                for g_idx in range(mb_start, mb_end):
                    tids_ep, lp_ep, rollout_data = group_data[g_idx]
                    adv_scalar = float(adv_z[g_idx])

                    lp_new_list = recompute_logprobs_grad(
                        rollout_data, tids_ep, vla_raw, p_ids, a_mask, device, dtype,
                        wm_bridge=wm_bridge,
                    )
                    lp_old_list = lp_ep
                    adv_list    = [adv_scalar] * len(lp_new_list)

                    loss = grpo_loss_fn(
                        lp_new_list, lp_old_list, adv_list,
                        cfg.training.clip_eps,
                        cfg.training.kl_coef,
                    )
                    (loss / (n_mb * len(z0_batch))).backward()
                    iter_loss_sum += loss.item()
                    torch.cuda.empty_cache()

        nn.utils.clip_grad_norm_(trainable, cfg.training.grad_clip)
        optimizer.step()
        scheduler.step()

        # ── EMA(reward) 업데이트: θ_ema ← τ·θ_ema + (1-τ)·θ_policy ──────────────
        if use_ema_reward:
            with torch.no_grad():
                for pe, p in zip(ema_lm.parameters(), vla_raw.language_model.parameters()):
                    pe.mul_(ema_tau).add_(p.detach().to(pe.dtype), alpha=1.0 - ema_tau)

        mean_r = np.mean(iter_rewards) if iter_rewards else 0.0
        std_r  = np.std(iter_rewards)  if iter_rewards else 0.0
        min_r  = np.min(iter_rewards)  if iter_rewards else 0.0
        max_r  = np.max(iter_rewards)  if iter_rewards else 0.0
        nonzero_frac = np.mean(np.array(iter_rewards) != 0.0) if iter_rewards else 0.0
        if use_ddp:
            dist.barrier()

        if args.probe:
            probe_rewards_all.extend(iter_rewards)

        n_grads = len(z0_batch) * cfg.training.g_rollouts
        mean_loss = iter_loss_sum / max(n_grads, 1)
        if is_main() and iteration % cfg.training.log_every == 0:
            log.info(
                f"[Iter {iteration:04d}]  reward={mean_r:.4f}  std={std_r:.4f}  "
                f"[{min_r:.3f},{max_r:.3f}]  nonzero={nonzero_frac:.2f}  "
                f"loss={mean_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if use_wandb:
                import wandb
                wandb.log({'reward': mean_r, 'reward_std': std_r,
                           'reward_min': min_r, 'reward_max': max_r,
                           'nonzero_frac': nonzero_frac,
                           'loss': mean_loss, 'iteration': iteration})

        # 저장: save_every 미사용 — 매 iteration last.pt 갱신, 개선 시 best.pt만 (디스크 절약)
        #   last.pt : 전체 state(+optimizer) — resume 용
        #   best.pt : 학습된(requires_grad) 파라미터만 — export 는 strict=False 로 base 에 병합
        #             (q/v_proj only 학습 시 19G → ~2G. full_finetune 시는 전체 저장됨)
        if not args.probe and is_main():
            ckpt = {
                'iteration': iteration,
                'vla': vla_raw.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'reward_mean': mean_r,
            }
            if wm_bridge is not None:
                ckpt['wm_bridge'] = wm_bridge.state_dict()
            torch.save(ckpt, out_dir / 'last.pt')
            if mean_r > best_reward:
                best_reward = mean_r
                tnames = {n for n, p in vla_raw.named_parameters() if p.requires_grad}
                slim_vla = {k: v for k, v in vla_raw.state_dict().items() if k in tnames}
                best_ckpt = {'iteration': iteration, 'vla': slim_vla, 'reward_mean': mean_r}
                if wm_bridge is not None:
                    best_ckpt['wm_bridge'] = wm_bridge.state_dict()
                torch.save(best_ckpt, out_dir / 'best.pt')
                log.info(f"  ★ New best reward={best_reward:.4f}  (slim ckpt: {len(slim_vla)} tensors)")

    if is_main() and args.probe:
        arr = np.array(probe_rewards_all)
        log.info(f"\n{'='*60}")
        log.info(f"[PROBE RESULT]  n={len(arr)}  iters={args.probe}")
        log.info(f"  mean={arr.mean():.4f}  std={arr.std():.4f}")
        log.info(f"  min={arr.min():.4f}   max={arr.max():.4f}")
        log.info(f"  nonzero={np.mean(arr != 0):.2f}")
        verdict = "SIGNAL OK" if arr.std() > 0.05 else "DEAD SIGNAL (std<0.05)"
        log.info(f"  → {verdict}")
        log.info(f"{'='*60}")
    elif is_main():
        log.info(f"Stage 3 done.  Best reward={best_reward:.4f}")
    if use_wandb:
        import wandb; wandb.finish()
    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
    