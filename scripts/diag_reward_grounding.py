#!/usr/bin/env python3
"""
진단: 상상(WM closed-loop) reward 가 실제 환경 성공과 상관되는가?
================================================================
동기(비판적 평가 #1): Stage3 의 reward 는 전부 WM 이 상상한 rollout 위에서 계산된다.
이 reward 가 '실제 task 성공' 과 상관이 없다면, 우리는 환각을 최적화하는 것이다.

방법 (action replay, predict_action 함정 회피):
  각 초기 상태에서
    ① 상상 rollout(swm_rollout)  → imagined_reward(=score_T−score_0) + action 토큰 시퀀스
    ② 같은 상태로 real env reset_to → ①의 action 을 그대로 replay → 실제 success / return
  → imagined_reward 와 (success, return) 의 상관(Pearson/Spearman/point-biserial/AUC) 보고.

한계: open-loop replay 라 실제 closed-loop 보정이 없어 성공률 자체는 낮게 나올 수 있음.
      그래도 '상상 reward 가 높을수록 실제로 더 진전/성공' 하는 단조 상관이 있으면 grounding 의 증거,
      상관≈0 이면 reward 가 ungrounded 라는 강한 증거.

사용:
  python scripts/diag_reward_grounding.py --config configs/stage3_full_probe_only.yaml \
      --ckpt outputs/stage3/full_probe_only/square/best.pt --metric probe \
      --n_states 20 --g_rollouts 2 --out eval_results/diag_probe.json
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/miplab1/sjLee/swm')
sys.path.insert(0, '/home/miplab1/sjLee/WMPO')
sys.path.insert(0, '/home/mipstu/jiPark/openvla-oft/experiments/robot')

from omegaconf import OmegaConf
import scripts.train_stage3 as T   # 모듈 함수/상수 재사용


def build_models(cfg, device, dtype):
    """train_stage3.main 의 모델 빌드 부분을 복제 (spatial_dim=2176 경로)."""
    s2_ckpt = torch.load(cfg.swm_ckpt, map_location=device)
    s2_cfg  = OmegaConf.create(s2_ckpt.get('cfg', {}))
    s1_ckpt = torch.load(cfg.stage1_ckpt, map_location=device)

    use_spatial = 'spatial_proj' in s2_ckpt and 'spatial_dim' in s2_ckpt
    spatial_dim = s2_ckpt.get('spatial_dim', 256) if use_spatial else None
    spatial_dim_for_enc = spatial_dim if use_spatial else 256

    encoder = T.SWMEncoder(vla_path=cfg.vla_path, freeze_backbone=True,
                           latent_dim=2176, spatial_dim=spatial_dim_for_enc).to(device)
    encoder.load_state_dict(s1_ckpt['encoder'], strict=False)

    if use_spatial:
        from models.transition import SpatialSWMTransition
        transition = SpatialSWMTransition(spatial_dim=spatial_dim, action_dim=7).to(device).to(dtype)
        transition.load_state_dict(s2_ckpt['transition'])
        if spatial_dim == 2176:
            encoder.spatial_proj = nn.Identity().to(device)
        else:
            encoder.spatial_proj = nn.Sequential(
                nn.Linear(2176, spatial_dim), nn.LayerNorm(spatial_dim)).to(device)
            encoder.spatial_proj.load_state_dict(s2_ckpt['spatial_proj'])
        encoder.spatial_dim = spatial_dim
        for p in encoder.spatial_proj.parameters(): p.requires_grad = False
        encoder.spatial_proj.eval()
    else:
        transition = T.SWMTransition(latent_dim=2176, action_dim=7).to(device)
        transition.load_state_dict(s2_ckpt['transition'])

    for m in (encoder, transition):
        for p in m.parameters(): p.requires_grad = False
    encoder.eval(); transition.eval()

    wm_bridge = None
    if use_spatial and spatial_dim != 2176:
        wm_bridge = nn.Linear(spatial_dim, 2176).to(device).to(dtype)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from verl.utils.openvla_utils import update_auto_map
    update_auto_map(cfg.vla_path)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=dtype, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    for p in vla.parameters(): p.requires_grad = False
    vla.eval()
    return encoder, transition, vla, processor, wm_bridge


def load_stage3_ckpt(vla, ckpt_path, device):
    if not ckpt_path:
        print('[diag] --ckpt 없음 → base SFT 정책 그대로 진단')
        return
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('vla', ck)
    missing, unexpected = vla.load_state_dict(state, strict=False)
    print(f'[diag] stage3 ckpt 로드: {Path(ckpt_path).name}  '
          f'적용 tensor={len(state)}  missing={len(missing)}  unexpected={len(unexpected)}')


def make_action_codec(cfg, vla, task, device):
    unnorm_key = cfg.get('unnorm_key', task)
    if unnorm_key not in vla.norm_stats:
        ds = json.load(open(Path(cfg.vla_path) / 'dataset_statistics.json'))
        vla.norm_stats.update(ds)
    stats = vla.norm_stats[unnorm_key]['action']
    q01 = torch.tensor(stats['q01'], device=device, dtype=torch.float32)
    q99 = torch.tensor(stats['q99'], device=device, dtype=torch.float32)
    n_bins = vla.bin_centers.shape[0] + 1
    vocab = vla.vocab_size

    def tokens_to_actions(tids):
        tids = tids.to(device)
        bin_idx = (vocab - tids - 1).clamp(0, n_bins - 2)
        norm = (bin_idx.float() / (n_bins - 1)) * 2.0 - 1.0
        norm = norm.reshape(1, T.NUM_ACTIONS_CHUNK, T.ACTION_DIM)
        return (norm + 1.0) / 2.0 * (q99 - q01) + q01
    return tokens_to_actions


def load_states(cfg, task, encoder, device, dtype, n_states):
    """demo.hdf5 에서 초기 이미지·z0·goal 이미지·raw env state 로드."""
    import h5py
    from torchvision import transforms
    hdf5 = os.path.join(cfg.data_root, 'demos', 'core_datasets', task,
                        f'demo_src_{task}_task_D0', 'demo.hdf5')
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
    out = []
    with h5py.File(hdf5, 'r') as f:
        demos = sorted(f['data'].keys())[:n_states]
        for dn in demos:
            img0 = tf(np.array(f[f'data/{dn}/obs/agentview_image'][0])).unsqueeze(0)
            goal = tf(np.array(f[f'data/{dn}/obs/agentview_image'][-1])).unsqueeze(0)
            state0 = np.array(f[f'data/{dn}/states'][0])
            with torch.no_grad():
                z = (encoder.encode_spatial_projected(img0.to(device)).to(dtype)
                     if hasattr(encoder, 'spatial_proj') else encoder(img0.to(device)).to(dtype))
            out.append(dict(image0=img0.cpu(), z0=z.cpu(), goal_img=goal.cpu(), state0=state0))
    return out, hdf5


def make_env(hdf5):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
    return env


def real_replay(env, state0, token_ids_list, tokens_to_actions, max_chunks):
    """초기 state 로 reset 후 imagined action 토큰을 open-loop replay → (success, return)."""
    env.reset()
    env.reset_to({"states": state0})
    ret, succ = 0.0, False
    for tids in token_ids_list[:max_chunks]:
        acts = tokens_to_actions(tids.unsqueeze(0))[0].cpu().numpy()   # (8,7)
        for k in range(acts.shape[0]):
            _, r, done, _ = env.step(acts[k])
            ret += float(r)
            try:
                if env.is_success()["task"]:
                    succ = True
            except Exception:
                succ = succ or (r > 0)
            if succ or done:
                break
        if succ or done:
            break
    return succ, ret


def correlations(imag, succ, ret):
    imag, succ, ret = np.array(imag), np.array(succ, float), np.array(ret, float)
    res = {}
    def safe_corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9: return None
        return float(np.corrcoef(a, b)[0, 1])
    res['pearson_reward_return'] = safe_corr(imag, ret)
    res['pointbiserial_reward_success'] = safe_corr(imag, succ)
    # Spearman (rank)
    def rank(x): return np.argsort(np.argsort(x)).astype(float)
    res['spearman_reward_return'] = safe_corr(rank(imag), rank(ret))
    # AUC: imagined reward 가 success 를 구분하는가
    pos, neg = imag[succ == 1], imag[succ == 0]
    if len(pos) and len(neg):
        auc = np.mean([1.0*(p > n) + 0.5*(p == n) for p in pos for n in neg])
        res['auc_reward_predicts_success'] = float(auc)
    else:
        res['auc_reward_predicts_success'] = None
    # top-half vs bottom-half imagined reward 의 실제 성공률
    med = np.median(imag)
    res['success_top_half'] = float(succ[imag >= med].mean()) if (imag >= med).any() else None
    res['success_bottom_half'] = float(succ[imag < med].mean()) if (imag < med).any() else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--metric', default='probe', choices=['probe', 'pls', 'tcn', 'goaldist', 'cosine'])
    ap.add_argument('--n_states', type=int, default=20)
    ap.add_argument('--g_rollouts', type=int, default=2)
    ap.add_argument('--max_chunks', type=int, default=20)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max_demos_fit', type=int, default=100)
    ap.add_argument('--out', default='eval_results/diag_reward_grounding.json')
    args = ap.parse_args()

    device = torch.device('cuda')
    dtype = torch.bfloat16
    cfg = OmegaConf.load(args.config)
    task = cfg.task

    print('[diag] 모델 빌드 ...')
    encoder, transition, vla, processor, wm_bridge = build_models(cfg, device, dtype)
    load_stage3_ckpt(vla, args.ckpt, device)
    tokens_to_actions = make_action_codec(cfg, vla, task, device)

    # prompt
    desc = T._TASK_DESC.get(task, task)
    feat = processor.tokenizer(f"In: What action should the robot take to {desc}?\nOut:", return_tensors='pt')
    p_ids = feat['input_ids'].to(device); a_mask = feat['attention_mask'].to(device)

    # reward LLM (정책 복사본) + scorer fit
    import copy
    ema_lm = copy.deepcopy(vla.language_model).eval()
    for p in ema_lm.parameters(): p.requires_grad_(False)
    probe_w = progress_mlp = goaldist_mlp = None
    if args.metric in ('probe', 'pls'):
        probe_w = T.fit_progress_probe(cfg.data_root, task, encoder, vla, ema_lm, p_ids, a_mask,
                                       args.max_demos_fit, device, dtype,
                                       method=('pls' if args.metric == 'pls' else 'ridge'))
    elif args.metric == 'tcn':
        progress_mlp = T.fit_tcn_progress(cfg.data_root, task, encoder, vla, ema_lm, p_ids, a_mask,
                                          args.max_demos_fit, device, dtype)
    elif args.metric == 'goaldist':
        goaldist_mlp = T.fit_goaldist(cfg.data_root, task, encoder, vla, ema_lm, p_ids, a_mask,
                                      args.max_demos_fit, device, dtype)

    print(f'[diag] 초기 상태 {args.n_states}개 로드 ...')
    states, hdf5 = load_states(cfg, task, encoder, device, dtype, args.n_states)
    print('[diag] real env 생성 ...')
    env = make_env(hdf5)

    patch_weights = None
    imag_rewards, successes, returns, meta = [], [], [], []
    for si, st in enumerate(states):
        z0 = st['z0'].to(device, dtype); image0 = st['image0'].to(device)
        goal_hidden = None
        if args.metric in ('cosine', 'goaldist'):
            goal_hidden = T.goal_pre_action_hidden(st['goal_img'], encoder, vla, ema_lm,
                                                   p_ids, a_mask, device, dtype)
        for g in range(args.g_rollouts):
            with torch.no_grad():
                tids_list, _, _, _, ema_cons = T.swm_rollout(
                    z0, image0, encoder, transition, None, vla, processor, p_ids, a_mask,
                    tokens_to_actions, args.max_chunks, args.temperature, device, dtype,
                    patch_weights=patch_weights, wm_bridge=wm_bridge,
                    goal_hidden=goal_hidden, probe_w=probe_w,
                    progress_mlp=progress_mlp, goaldist_mlp=goaldist_mlp)
            succ, ret = real_replay(env, st['state0'], tids_list, tokens_to_actions, args.max_chunks)
            imag_rewards.append(float(ema_cons) if ema_cons is not None else 0.0)
            successes.append(int(succ)); returns.append(ret)
            meta.append(dict(state=si, rollout=g))
        print(f'  state {si+1}/{len(states)}  imag_reward(last)={imag_rewards[-1]:.4f}  '
              f'succ={successes[-1]}  ret={returns[-1]:.2f}')

    res = correlations(imag_rewards, successes, returns)
    summary = dict(
        config=args.config, ckpt=args.ckpt, metric=args.metric,
        n_pairs=len(imag_rewards), n_states=args.n_states, g_rollouts=args.g_rollouts,
        real_success_rate=float(np.mean(successes)),
        imag_reward_mean=float(np.mean(imag_rewards)), imag_reward_std=float(np.std(imag_rewards)),
        correlations=res,
        raw=dict(imag_reward=imag_rewards, success=successes, ret=returns, meta=meta),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, 'w'), indent=2)

    print('\n' + '=' * 60)
    print(f'[diag] metric={args.metric}  pairs={len(imag_rewards)}  '
          f'real_SR(open-loop)={summary["real_success_rate"]:.3f}')
    print(f'  imagined reward: mean={summary["imag_reward_mean"]:.4f} std={summary["imag_reward_std"]:.4f}')
    print('  --- 상상 reward ↔ 실제 결과 상관 ---')
    for k, v in res.items():
        print(f'    {k}: {v}')
    print(f'\n  해석: pointbiserial/auc 가 0 근처면 reward 가 실제 성공과 무관(=ungrounded).')
    print(f'        success_top_half >> success_bottom_half 면 reward 가 실제 진전을 반영(grounded).')
    print(f'  저장: {args.out}')
    print('=' * 60)


if __name__ == '__main__':
    main()
