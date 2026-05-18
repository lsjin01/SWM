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


def _vla_forward_patch_embeds(
    patch_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    vla,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
):
    """patch_embeds (1, 256, 4096) + prompt → (tids (1,56), logprob float)"""
    input_ids_ext = prompt_ids.clone()
    placeholder   = torch.ones((1, NUM_ACTION_TOKENS), device=device, dtype=input_ids_ext.dtype)
    stop          = torch.ones((1, 1),                  device=device, dtype=input_ids_ext.dtype) * 2
    input_ids_full = torch.cat([input_ids_ext, placeholder, stop], dim=-1)

    input_embeds = vla.get_input_embeddings()(input_ids_full)
    full_embeds  = torch.cat([patch_embeds, input_embeds], dim=1)

    vis_mask  = torch.ones((1, NUM_VISION_TOKENS),        device=device, dtype=attn_mask.dtype)
    ext_mask  = torch.ones((1, input_ids_full.shape[1]),  device=device, dtype=attn_mask.dtype)
    full_mask = torch.cat([vis_mask, ext_mask], dim=1)

    out    = vla.language_model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    logits = out.logits[:, -NUM_ACTION_TOKENS-2:-2]  # (1, 56, V)  off-by-one fix
    if temperature != 1.0:
        logits = logits / temperature
    dist = Categorical(logits=logits.reshape(-1, logits.size(-1)).float())
    tids = dist.sample().reshape(1, -1)
    lp   = dist.log_prob(tids.reshape(-1)).sum().item()
    return tids, lp


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
):
    """
    수정된 SWM rollout:
      - VLA: 초기 관측 image의 spatial patch features → projector → LLM
             (256개 token 각자 다른 공간 정보 → train/eval 완전 일치)
      - WM:  global z_t → transition → z_t+1 (reward 계산 전용)

    Returns:
        token_ids_list : list[Tensor(56,)]   각 chunk의 action tokens
        logprobs_list  : list[float]          log-probabilities
        patch_embeds_cpu: Tensor(1, 256, 4096) CPU (gradient recompute용)
        z_history      : list[Tensor(1, 2176)]  WM latent history
    """
    z = z0.clone().to(device, dtype)
    is_spatial = (z.ndim == 3)  # (B, N, d_s) vs (B, D)

    # ── 초기 이미지 → spatial patch embeddings (한 번만 계산) ────────────
    with torch.no_grad():
        patch_embeds = image_to_patch_embeddings(image0, encoder, vla)  # (1, 256, 4096)

    patch_embeds_cpu = patch_embeds.cpu()   # gradient recompute용 CPU 저장

    token_ids_list = []
    logprobs_list  = []
    z_history      = []

    for _ in range(n_chunks):
        # ── VLA: 동일한 spatial patch_embeds 사용 ─────────────────────────
        with torch.no_grad():
            tids, lp = _vla_forward_patch_embeds(
                patch_embeds, prompt_ids, attn_mask, vla, device, dtype, temperature
            )

        token_ids_list.append(tids[0].cpu())
        logprobs_list.append(lp)

        # ── WM: global z_t + action → z_t+1 (reward 전용) ────────────────
        actions = tokens_to_actions(tids)  # (1, 8, 7)
        with torch.no_grad():
            for sub in range(NUM_ACTIONS_CHUNK):
                a = actions[0, sub].to(device, dtype)
                z = transition(z.to(dtype), a.unsqueeze(0))
                if is_spatial:
                    z_history.append(z.mean(dim=1).clone())  # (B, d_s) for reward
                else:
                    z_history.append(z.clone())

    return token_ids_list, logprobs_list, patch_embeds_cpu, z_history


def recompute_logprobs_grad(
    patch_embeds_cpu: torch.Tensor,   # (1, 256, 4096)  공유 spatial embeddings
    token_ids_list: list,             # list of (56,) Tensor
    vla,
    prompt_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    gradient 있는 log-prob 재계산 (GRPO backward용).
    모든 chunk가 같은 patch_embeds를 공유 → 한 번만 GPU로 이동.
    """
    patch_embeds = patch_embeds_cpu.to(device, dtype)

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
    logits = out.logits[:, -NUM_ACTION_TOKENS-2:-2]  # (1, 56, V)  off-by-one fix

    lp_list = []
    for tids in token_ids_list:
        tgt = tids.to(device)
        lp  = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            tgt.reshape(-1),
            reduction='sum',
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
            surr = surr + kl_coef * (lp_old - lp_new)
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
                        pca_goal_threshold=0.2):
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

    with h5py.File(hdf5, 'r') as f:
        demos = sorted(f['data'].keys())[:max_demos]
        for dn in demos:
            img_np    = f[f'data/{dn}/obs/agentview_image'][0]
            image_t   = transform(img_np).unsqueeze(0)
            image_gpu = image_t.to(device)
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
                        goal_latents.append(encoder.encode_spatial_projected(goal_t).to(dtype).mean(dim=1).cpu())
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
    return init_images, init_latents, init_goals


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',           default='configs/stage3.yaml')
    p.add_argument('--no-wandb',         action='store_true')
    p.add_argument('--resume',           type=str, default=None)
    p.add_argument('--debug',            action='store_true')
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

    # ── PCA fit (pca_cosine / pca_delta_cosine) ──────────────────────────────
    # token_weights 전달 → spatial pooling 시 encode_weighted로 PCA 학습 (분포 일치)
    demo_pca = None
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
    n_goals = cfg.get('n_goals', 1)
    init_images, init_latents, init_goals = load_initial_states(
        cfg.data_root, task, encoder, cfg.training.max_demos, device, dtype,
        reward_type=reward_type,
        transition=transition,
        rollout_steps=rollout_steps,
        n_goals=n_goals,
        token_weights=token_weights,
        pca=demo_pca,
        pca_goal_threshold=cfg.get('pca_goal_threshold', 0.2),
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

    # ── DDP 감싸기 ────────────────────────────────────────────────────────────
    if use_ddp:
        vla = DDP(vla, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable = [p for p in vla_raw.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.training.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.iterations, eta_min=cfg.training.lr * 0.1
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_iter  = 1
    best_reward = -float('inf')

    if args.resume and Path(args.resume).exists():
        ckpt_r = torch.load(args.resume, map_location=device)
        vla_raw.load_state_dict(ckpt_r['vla'])
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
    log.info(f"\n[GRPO] {cfg.training.iterations} iterations  "
             f"task={task}  g_rollouts={cfg.training.g_rollouts}")

    for iteration in range(start_iter, cfg.training.iterations + 1):
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

            group_rewards = []
            group_data    = []   # (tids_ep, lp_ep, patch_embeds_cpu) 저장

            for _ in range(cfg.training.g_rollouts):
                tids_ep, lp_ep, patch_embeds_cpu, z_hist = swm_rollout(
                    z0, image0, encoder, transition, heads,
                    vla_raw, processor, p_ids, a_mask,
                    tokens_to_actions,
                    cfg.training.n_rollout_chunks,
                    cfg.training.temperature,
                    device, dtype,
                )
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
                group_rewards.append(rew)
                group_data.append((tids_ep, lp_ep, patch_embeds_cpu))

            iter_rewards.extend(group_rewards)
            gr    = np.array(group_rewards)
            adv_z = (gr - gr.mean()) / max(gr.std(), 1e-8)

            mini_g = cfg.training.get('mini_g', 0) or cfg.training.g_rollouts
            n_mb = (cfg.training.g_rollouts + mini_g - 1) // mini_g
            for mb_start in range(0, cfg.training.g_rollouts, mini_g):
                mb_end = min(mb_start + mini_g, cfg.training.g_rollouts)
                for g_idx in range(mb_start, mb_end):
                    tids_ep, lp_ep, patch_embeds_cpu = group_data[g_idx]
                    adv_scalar = float(adv_z[g_idx])

                    # 같은 patch_embeds로 n_chunks log-probs 재계산
                    lp_new_list = recompute_logprobs_grad(
                        patch_embeds_cpu, tids_ep, vla_raw, p_ids, a_mask, device, dtype
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

        mean_r = np.mean(iter_rewards) if iter_rewards else 0.0
        if use_ddp:
            dist.barrier()

        n_grads = len(z0_batch) * cfg.training.g_rollouts
        mean_loss = iter_loss_sum / max(n_grads, 1)
        if is_main() and iteration % cfg.training.log_every == 0:
            log.info(
                f"[Iter {iteration:04d}]  reward={mean_r:.4f}  "
                f"loss={mean_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if use_wandb:
                import wandb
                wandb.log({'reward': mean_r, 'loss': mean_loss, 'iteration': iteration})

        if is_main() and iteration % cfg.training.save_every == 0:
            ckpt = {
                'iteration': iteration,
                'vla': vla_raw.state_dict(),
                'reward_mean': mean_r,
            }
            torch.save(ckpt, out_dir / f'ckpt_iter{iteration:04d}.pt')
            if mean_r > best_reward:
                best_reward = mean_r
                torch.save(ckpt, out_dir / 'best.pt')
                log.info(f"  ★ New best reward={best_reward:.4f}")

    if is_main():
        log.info(f"Stage 3 done.  Best reward={best_reward:.4f}")
    if use_wandb:
        import wandb; wandb.finish()
    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
    