"""
measure_transition_drift.py
============================
Transition model의 multi-step rollout drift를 측정합니다.

핵심 아이디어:
  Stage 2는 1-step teacher-forced loss(val=0.063)로 학습되지만
  Stage 3 GRPO rollout은 160-step free-run autoregressive rollout을 사용합니다.
  이 gap이 실제로 얼마나 큰지 측정합니다.

측정 모드 (2가지):
  [A] Teacher-forced:  매 스텝마다 real z_t 사용 → 1-step 품질 측정
  [B] Free-run:        z_0만 실제, 이후 ẑ_t 자동회귀 → 누적 오차 측정

측정 지표:
  - cosine similarity: cos_sim(ẑ_t, z_encoder(img_t))
  - L1 distance
  - Phase 1 reward 재현: cos_sim을 PCA 공간에서 계산 (pca_binary와 동일)
  - Phase 2 reward 재현: patch-level + mean-pooled cosine

비교 대상:
  - Phase 1: SWMTransition (scalar, 2176-dim)
  - Phase 2: SpatialSWMTransition (spatial, 256×256)

사용법:
  # Phase 2 spatial만 측정 (기본)
  python eval/measure_transition_drift.py \\
    --spatial_ckpt outputs/stage2/spatial_multitask/best.pt \\
    --data_root data/ --task square --max_demos 50 --max_steps 160

  # Phase 1 + Phase 2 비교
  python eval/measure_transition_drift.py \\
    --scalar_ckpt  outputs/stage2/scalar/best.pt \\
    --spatial_ckpt outputs/stage2/spatial_multitask/best.pt \\
    --data_root data/ --task square --max_demos 50 --max_steps 160

  # 출력 디렉터리 지정
  python eval/measure_transition_drift.py \\
    --spatial_ckpt outputs/stage2/spatial_multitask/best.pt \\
    --output_dir outputs/drift_analysis/
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# matplotlib: 헤드리스 환경 대응
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── repo root를 sys.path에 추가 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.encoder    import SWMEncoder
from models.transition import SWMTransition, SpatialSWMTransition


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_demo_trajectories(data_root: str, task: str, max_demos: int, split: str = "val"):
    """
    MimicGen HDF5 데모 파일에서 (images, actions) 시퀀스를 로드합니다.

    반환값:
        List of (images, actions)
          images  : torch.Tensor (T, 3, H, W)  float32, [0, 1]
          actions : torch.Tensor (T-1, action_dim)  float32
    """
    import h5py

    data_dir = Path(data_root) / task
    hdf5_files = sorted(data_dir.glob(f"{split}*.hdf5")) or \
                 sorted(data_dir.glob("*.hdf5")) or \
                 sorted(data_dir.glob(f"**/{split}*.hdf5")) or \
                 sorted(data_dir.glob("**/*.hdf5"))

    if not hdf5_files:
        raise FileNotFoundError(
            f"HDF5 파일을 찾을 수 없습니다: {data_dir}\n"
            "data_root와 task 인자를 확인하세요."
        )

    trajectories = []
    for hdf5_path in hdf5_files[:max_demos]:
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys())
            for dk in demo_keys:
                if len(trajectories) >= max_demos:
                    break
                demo = f["data"][dk]

                # 이미지 (T, H, W, 3) uint8 → (T, 3, H, W) float32
                imgs_np = demo["obs"]["agentview_image"][:]   # (T, H, W, 3)
                imgs = torch.from_numpy(imgs_np).float() / 255.0
                imgs = imgs.permute(0, 3, 1, 2)              # (T, 3, H, W)
                # DINOv2/SigLIP expects 224×224
                if imgs.shape[-1] != 224 or imgs.shape[-2] != 224:
                    import torch.nn.functional as F_img
                    imgs = F_img.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)

                # 액션 (T-1, action_dim)
                acts_np = demo["actions"][:]
                acts = torch.from_numpy(acts_np).float()

                # 길이 맞추기
                T = min(len(imgs), len(acts) + 1)
                imgs = imgs[:T]
                acts = acts[:T - 1]

                trajectories.append((imgs, acts))

        if len(trajectories) >= max_demos:
            break

    print(f"[데이터] {len(trajectories)}개 demo 로드 완료 (task={task})")
    return trajectories


# ─────────────────────────────────────────────────────────────────────────────
# 2. 모델 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_scalar_transition(ckpt_path: str, device: torch.device) -> SWMTransition:
    ckpt = torch.load(ckpt_path, map_location=device)
    transition = SWMTransition()
    state = ckpt.get("transition", ckpt)
    transition.load_state_dict(state)
    transition.to(device).eval()
    print(f"[모델] Scalar Transition 로드: {ckpt_path}")
    return transition


def load_spatial_transition(ckpt_path: str, device: torch.device, vla_path: str = "/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square"):
    """SpatialSWMTransition + spatial_proj 동시 로드"""
    ckpt = torch.load(ckpt_path, map_location=device)

    transition = SpatialSWMTransition(
        spatial_dim=256, hidden_dim=512, num_layers=6, num_heads=8
    )
    transition.load_state_dict(ckpt["transition"])
    transition.to(device).eval()

    # spatial_proj는 SWMEncoder 안에 있으므로 encoder에 주입
    encoder = SWMEncoder(vla_path=vla_path)
    encoder.spatial_proj.load_state_dict(ckpt["spatial_proj"])
    encoder.to(device).eval()
    # DINOv2 + SigLIP frozen 확인
    for p in encoder.parameters():
        p.requires_grad_(False)

    print(f"[모델] Spatial Transition + Encoder 로드: {ckpt_path}")
    return encoder, transition


# ─────────────────────────────────────────────────────────────────────────────
# 3. Drift 측정 함수
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_scalar_drift(
    encoder:    SWMEncoder,
    transition: SWMTransition,
    trajectories: list,
    max_steps: int,
    device: torch.device,
) -> dict:
    """
    Phase 1 Scalar Transition drift 측정

    Returns:
        {
          "teacher_cosine": (N, T) - teacher-forced cosine similarity
          "freerun_cosine": (N, T) - free-run autoregressive cosine similarity
          "teacher_l1":     (N, T)
          "freerun_l1":     (N, T)
        }
    """
    teacher_cosines, freerun_cosines = [], []
    teacher_l1s,     freerun_l1s     = [], []

    for imgs, acts in tqdm(trajectories, desc="Scalar drift"):
        imgs = imgs.to(device)
        acts = acts.to(device)
        T    = min(len(acts), max_steps)

        # ── 실제 latent 시퀀스 미리 계산 ──────────────────────────
        # (T+1,) 길이의 z_real 캐시
        z_reals = []
        for t in range(T + 1):
            z_reals.append(encoder.encode(imgs[t:t+1]))    # (1, 2176)

        # ── A. Teacher-forced (1-step 품질) ───────────────────────
        t_cos, t_l1 = [], []
        for t in range(T):
            z_pred = transition(z_reals[t], acts[t:t+1])   # (1, 2176)
            cos = F.cosine_similarity(z_pred, z_reals[t+1], dim=-1).item()
            l1  = F.l1_loss(z_pred, z_reals[t+1]).item()
            t_cos.append(cos)
            t_l1.append(l1)
        teacher_cosines.append(t_cos)
        teacher_l1s.append(t_l1)

        # ── B. Free-run (누적 오차) ────────────────────────────────
        z_hat = z_reals[0]   # z_0만 real
        f_cos, f_l1 = [], []
        for t in range(T):
            z_hat = transition(z_hat, acts[t:t+1])         # autoregressive
            cos = F.cosine_similarity(z_hat, z_reals[t+1], dim=-1).item()
            l1  = F.l1_loss(z_hat, z_reals[t+1]).item()
            f_cos.append(cos)
            f_l1.append(l1)
        freerun_cosines.append(f_cos)
        freerun_l1s.append(f_l1)

    # 길이 맞추기 (demo마다 T가 다를 수 있음)
    def pad(arr_list):
        max_len = max(len(a) for a in arr_list)
        out = np.full((len(arr_list), max_len), np.nan)
        for i, a in enumerate(arr_list):
            out[i, :len(a)] = a
        return out

    return {
        "teacher_cosine": pad(teacher_cosines),
        "freerun_cosine": pad(freerun_cosines),
        "teacher_l1":     pad(teacher_l1s),
        "freerun_l1":     pad(freerun_l1s),
    }


@torch.no_grad()
def measure_spatial_drift(
    encoder:    SWMEncoder,
    transition: SpatialSWMTransition,
    trajectories: list,
    max_steps: int,
    device: torch.device,
) -> dict:
    """
    Phase 2 Spatial Transition drift 측정

    patch-level cosine + mean-pooled cosine (GRPO reward와 동일) 모두 측정
    """
    teacher_patch_cos,  freerun_patch_cos  = [], []
    teacher_pooled_cos, freerun_pooled_cos = [], []
    teacher_l1,         freerun_l1         = [], []

    for imgs, acts in tqdm(trajectories, desc="Spatial drift"):
        imgs = imgs.to(device)
        acts = acts.to(device)
        T    = min(len(acts), max_steps)

        # ── 실제 spatial latent 시퀀스 캐시 ──────────────────────
        s_reals = []
        for t in range(T + 1):
            s_reals.append(encoder.encode_spatial_projected(imgs[t:t+1]))  # (1, 256, 256)

        # ── A. Teacher-forced ─────────────────────────────────────
        tp_cos, tpool_cos, tl1 = [], [], []
        for t in range(T):
            s_pred = transition(s_reals[t], acts[t:t+1])  # (1, 256, 256)
            s_real = s_reals[t+1]

            # per-patch cosine (patch 차원 기준)
            patch_cos = F.cosine_similarity(s_pred, s_real, dim=-1).mean().item()

            # mean-pooled cosine (GRPO reward 계산과 동일)
            pooled_cos = F.cosine_similarity(
                s_pred.mean(dim=1), s_real.mean(dim=1), dim=-1
            ).item()

            tp_cos.append(patch_cos)
            tpool_cos.append(pooled_cos)
            tl1.append(F.l1_loss(s_pred, s_real).item())

        teacher_patch_cos.append(tp_cos)
        teacher_pooled_cos.append(tpool_cos)
        teacher_l1.append(tl1)

        # ── B. Free-run ───────────────────────────────────────────
        s_hat = s_reals[0]
        fp_cos, fpool_cos, fl1 = [], [], []
        for t in range(T):
            s_hat   = transition(s_hat, acts[t:t+1])
            s_real  = s_reals[t+1]

            patch_cos = F.cosine_similarity(s_hat, s_real, dim=-1).mean().item()
            pooled_cos = F.cosine_similarity(
                s_hat.mean(dim=1), s_real.mean(dim=1), dim=-1
            ).item()

            fp_cos.append(patch_cos)
            fpool_cos.append(pooled_cos)
            fl1.append(F.l1_loss(s_hat, s_real).item())

        freerun_patch_cos.append(fp_cos)
        freerun_pooled_cos.append(fpool_cos)
        freerun_l1.append(fl1)

    def pad(arr_list):
        max_len = max(len(a) for a in arr_list)
        out = np.full((len(arr_list), max_len), np.nan)
        for i, a in enumerate(arr_list):
            out[i, :len(a)] = a
        return out

    return {
        "teacher_patch_cosine":  pad(teacher_patch_cos),
        "teacher_pooled_cosine": pad(teacher_pooled_cos),
        "teacher_l1":            pad(teacher_l1),
        "freerun_patch_cosine":  pad(freerun_patch_cos),
        "freerun_pooled_cosine": pad(freerun_pooled_cos),
        "freerun_l1":            pad(freerun_l1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. 통계 요약
# ─────────────────────────────────────────────────────────────────────────────

def summarize(metrics: np.ndarray, name: str) -> dict:
    """
    (N, T) 배열에서 per-step 통계를 계산합니다.
    NaN은 해당 demo의 길이 초과 스텝이므로 nanmean/nanstd 사용.
    """
    mean  = np.nanmean(metrics, axis=0)
    std   = np.nanstd(metrics,  axis=0)
    p10   = np.nanpercentile(metrics, 10, axis=0)
    p90   = np.nanpercentile(metrics, 90, axis=0)

    # 주요 체크포인트 스텝 출력
    checkpoints = [1, 10, 40, 80, 120, 159]
    T = metrics.shape[1]
    valid_ckpts = [s for s in checkpoints if s < T]

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  {'Step':>6}  {'Mean':>8}  {'Std':>8}  {'P10':>8}  {'P90':>8}")
    for s in valid_ckpts:
        print(f"  {s:>6}  {mean[s]:>8.4f}  {std[s]:>8.4f}"
              f"  {p10[s]:>8.4f}  {p90[s]:>8.4f}")

    # t=0 vs t=159 drop
    if T > 1:
        drop = mean[0] - mean[min(159, T-1)]
        print(f"\n  t=0 → t={min(159,T-1)} drop: {drop:+.4f}")

    return {"mean": mean, "std": std, "p10": p10, "p90": p90}


# ─────────────────────────────────────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_drift(
    scalar_results: dict | None,
    spatial_results: dict | None,
    output_dir: Path,
):
    """
    multi-panel drift 시각화

    패널 구성:
      [0] Teacher-forced cosine:  scalar vs spatial (1-step 품질 비교)
      [1] Free-run cosine:        scalar vs spatial (누적 오차 비교)
      [2] Teacher vs Free-run gap: scalar, spatial 각각의 gap
      [3] L1 distance (free-run): 절대적 오차 크기
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Transition Model Drift Analysis\n"
        "(Teacher-forced vs Free-run, Phase 1 Scalar vs Phase 2 Spatial)",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    COLORS = {
        "scalar_teacher":  "#2196F3",   # blue
        "scalar_freerun":  "#F44336",   # red
        "spatial_teacher": "#4CAF50",   # green
        "spatial_freerun": "#FF9800",   # orange
    }

    def _plot_band(ax, stats, color, label, alpha_band=0.2):
        T = len(stats["mean"])
        x = np.arange(T)
        ax.plot(x, stats["mean"], color=color, label=label, linewidth=1.8)
        ax.fill_between(
            x,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color, alpha=alpha_band
        )

    # ── 패널 0: Teacher-forced cosine ────────────────────────────────────────
    ax = axes[0]
    ax.set_title("① Teacher-forced Cosine Similarity\n(1-step 품질: 높을수록 좋음)")
    ax.set_xlabel("Step t")
    ax.set_ylabel("cos_sim(ẑ_{t+1}, z_real_{t+1})")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.97, color="gray", linestyle="--", linewidth=1,
               label="Phase1 background dom. (≥0.97)")

    if scalar_results:
        s = summarize(scalar_results["teacher_cosine"], "Scalar Teacher-forced Cosine")
        _plot_band(ax, s, COLORS["scalar_teacher"], "Phase1 Scalar")

    if spatial_results:
        s = summarize(spatial_results["teacher_pooled_cosine"], "Spatial Teacher-forced Cosine (pooled)")
        _plot_band(ax, s, COLORS["spatial_teacher"], "Phase2 Spatial (pooled)")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 패널 1: Free-run cosine ───────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("② Free-run Autoregressive Cosine\n(누적 오차: GRPO rollout과 동일 조건)")
    ax.set_xlabel("Step t")
    ax.set_ylabel("cos_sim(ẑ_t_freerun, z_real_t)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.97, color="gray", linestyle="--", linewidth=1,
               label="background dominance baseline (≥0.97)")

    # t=160 위치 표시
    ax.axvline(159, color="black", linestyle=":", linewidth=1.5,
               label="GRPO rollout end (t=160)")

    if scalar_results:
        s = summarize(scalar_results["freerun_cosine"], "Scalar Free-run Cosine")
        _plot_band(ax, s, COLORS["scalar_freerun"], "Phase1 Scalar")

    if spatial_results:
        s = summarize(spatial_results["freerun_pooled_cosine"], "Spatial Free-run Cosine (pooled)")
        _plot_band(ax, s, COLORS["spatial_freerun"], "Phase2 Spatial (pooled)")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 패널 2: Teacher vs Free-run gap ──────────────────────────────────────
    ax = axes[2]
    ax.set_title("③ Teacher vs Free-run Gap\n(gap이 클수록 compounding error가 심각)")
    ax.set_xlabel("Step t")
    ax.set_ylabel("Cosine Gap (Teacher - Free-run)")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)

    if scalar_results:
        t_mean = np.nanmean(scalar_results["teacher_cosine"], axis=0)
        f_mean = np.nanmean(scalar_results["freerun_cosine"], axis=0)
        ax.plot(t_mean - f_mean, color=COLORS["scalar_freerun"],
                label="Phase1 Scalar gap", linewidth=1.8)

    if spatial_results:
        t_mean = np.nanmean(spatial_results["teacher_pooled_cosine"], axis=0)
        f_mean = np.nanmean(spatial_results["freerun_pooled_cosine"], axis=0)
        ax.plot(t_mean - f_mean, color=COLORS["spatial_freerun"],
                label="Phase2 Spatial gap", linewidth=1.8)

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 패널 3: Free-run L1 distance ─────────────────────────────────────────
    ax = axes[3]
    ax.set_title("④ Free-run L1 Distance\n(절대적 예측 오차 크기)")
    ax.set_xlabel("Step t")
    ax.set_ylabel("L1 distance")

    if scalar_results:
        s = summarize(scalar_results["freerun_l1"], "Scalar Free-run L1")
        _plot_band(ax, s, COLORS["scalar_freerun"], "Phase1 Scalar")

    if spatial_results:
        s = summarize(spatial_results["freerun_l1"], "Spatial Free-run L1")
        _plot_band(ax, s, COLORS["spatial_freerun"], "Phase2 Spatial")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "transition_drift.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[시각화] 저장 완료: {save_path}")
    plt.close()


def plot_spatial_patch_vs_pooled(spatial_results: dict, output_dir: Path):
    """
    Spatial 모델 전용: patch-level vs mean-pooled cosine 비교
    mean-pool이 왜 배경에 지배당하는지 시각화
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Spatial: Per-patch Cosine vs Mean-pooled Cosine (Free-run)\n"
        "mean-pool이 배경 패치에 얼마나 지배당하는지 확인",
        fontsize=11, fontweight="bold"
    )

    for ax, key, title in [
        (axes[0], "freerun_patch_cosine",  "Per-patch Cosine (mean over 256 patches)"),
        (axes[1], "freerun_pooled_cosine", "Mean-pooled Cosine (GRPO reward와 동일)"),
    ]:
        data = spatial_results[key]
        mean = np.nanmean(data, axis=0)
        std  = np.nanstd(data,  axis=0)
        x    = np.arange(len(mean))

        ax.plot(x, mean, linewidth=1.8, color="#4CAF50")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="#4CAF50")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Step t")
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.97, color="gray", linestyle="--", linewidth=1,
                   label="≥0.97 (구분 불가 threshold)")
        ax.axvline(159, color="black", linestyle=":", linewidth=1.2,
                   label="t=160 (GRPO rollout end)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    save_path = output_dir / "spatial_patch_vs_pooled.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[시각화] 저장 완료: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. 결과 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_stats(results: dict, model_name: str, output_dir: Path):
    """
    per-step mean/std를 JSON으로 저장합니다.
    이후 LLM EMA 실험에서 prior로 활용 가능.
    """
    stats = {}
    for key, val in results.items():
        mean = np.nanmean(val, axis=0).tolist()
        std  = np.nanstd(val,  axis=0).tolist()
        stats[key] = {"mean": mean, "std": std}

    # 주요 지점 요약
    summary = {}
    primary_key = (
        "freerun_cosine"         if "freerun_cosine"         in results else
        "freerun_pooled_cosine"  if "freerun_pooled_cosine"  in results else
        list(results.keys())[0]
    )
    freerun_mean = np.nanmean(results[primary_key], axis=0)
    T = len(freerun_mean)
    for step in [0, 10, 40, 80, 120, min(159, T-1)]:
        if step < T:
            summary[f"t={step}"] = float(freerun_mean[step])

    output = {
        "model":   model_name,
        "summary": summary,
        "per_step_stats": stats,
    }

    save_path = output_dir / f"drift_stats_{model_name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[통계] 저장 완료: {save_path}")

    # 핵심 해석 출력
    t0  = freerun_mean[0]
    t_end = freerun_mean[min(159, T-1)]
    print(f"\n{'='*60}")
    print(f"  [{model_name}] Free-run Cosine 요약")
    print(f"{'='*60}")
    print(f"  t=0   : {t0:.4f}")
    print(f"  t={min(159,T-1):<3}: {t_end:.4f}  (drop: {t0 - t_end:+.4f})")
    if t_end >= 0.97:
        print("  ⚠️  t=160에서도 cosine ≥ 0.97 → 배경 지배 여전함")
        print("     reward signal이 task progress를 반영하지 못할 가능성 높음")
    elif t_end >= 0.90:
        print("  ⚠️  cosine 0.90~0.97 → 약한 signal, GRPO σ 매우 작을 것")
    else:
        print("  ✅  cosine < 0.90 → 어느 정도 구분 가능한 signal 존재")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Transition Drift Measurement")
    p.add_argument("--scalar_ckpt",  type=str, default=None,
                   help="Phase 1 scalar transition checkpoint (.pt)")
    p.add_argument("--spatial_ckpt", type=str, default=None,
                   help="Phase 2 spatial transition checkpoint (.pt)")
    p.add_argument("--data_root",    type=str, default="data/",
                   help="데모 데이터 루트 디렉터리")
    p.add_argument("--task",         type=str, default="square",
                   choices=["square", "coffee", "stack_three", "three_piece_assembly"],
                   help="측정할 task (default: square)")
    p.add_argument("--split",        type=str, default="val",
                   help="데이터 split (default: val)")
    p.add_argument("--max_demos",    type=int, default=50,
                   help="최대 demo 수 (default: 50)")
    p.add_argument("--max_steps",    type=int, default=160,
                   help="최대 rollout steps (default: 160, GRPO와 동일)")
    p.add_argument("--output_dir",   type=str, default="outputs/drift_analysis/",
                   help="결과 저장 디렉터리")
    p.add_argument("--device",       type=str, default="cuda",
                   help="device (default: cuda)")
    p.add_argument("--vla_path",     type=str,
                   default="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square",
                   help="VLA backbone path for SWMEncoder")
    return p.parse_args()


def main():
    args = parse_args()

    if args.scalar_ckpt is None and args.spatial_ckpt is None:
        print("오류: --scalar_ckpt 또는 --spatial_ckpt 중 하나 이상 지정 필요")
        sys.exit(1)

    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Transition Drift Measurement")
    print(f"  task={args.task}, max_demos={args.max_demos}, max_steps={args.max_steps}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    trajectories = load_demo_trajectories(
        args.data_root, args.task, args.max_demos, args.split
    )

    scalar_results  = None
    spatial_results = None

    # ── Phase 1: Scalar ──────────────────────────────────────────────────────
    if args.scalar_ckpt:
        print("\n[Phase 1] Scalar Transition 측정 시작...")
        # scalar encoder는 mean-pool만 사용 (spatial_proj 없음)
        scalar_encoder    = SWMEncoder(vla_path=args.vla_path)
        scalar_encoder.to(device).eval()
        scalar_transition = load_scalar_transition(args.scalar_ckpt, device)

        scalar_results = measure_scalar_drift(
            scalar_encoder, scalar_transition, trajectories, args.max_steps, device
        )
        save_stats(scalar_results, "scalar", output_dir)

    # ── Phase 2: Spatial ─────────────────────────────────────────────────────
    if args.spatial_ckpt:
        print("\n[Phase 2] Spatial Transition 측정 시작...")
        spatial_encoder, spatial_transition = load_spatial_transition(
            args.spatial_ckpt, device, args.vla_path
        )
        spatial_results = measure_spatial_drift(
            spatial_encoder, spatial_transition, trajectories, args.max_steps, device
        )
        save_stats(spatial_results, "spatial", output_dir)

        # Spatial 전용 patch vs pooled 시각화
        plot_spatial_patch_vs_pooled(spatial_results, output_dir)

    # ── 통합 시각화 ──────────────────────────────────────────────────────────
    plot_drift(scalar_results, spatial_results, output_dir)

    # ── LLM EMA 설계를 위한 핵심 수치 출력 ──────────────────────────────────
    print("\n" + "="*60)
    print("  LLM EMA 설계를 위한 핵심 수치")
    print("="*60)

    if spatial_results is not None:
        freerun = np.nanmean(spatial_results["freerun_pooled_cosine"], axis=0)
        t160    = freerun[min(159, len(freerun)-1)]
        print(f"  t=160 free-run cosine (reward 계산 기준): {t160:.4f}")
        print()
        if t160 >= 0.97:
            print("  진단: ẑ_160이 real z_160과 구분 불가능 수준")
            print("  → LLM EMA reward도 동일한 signal collapse 예상")
            print("  → Transition 품질 개선이 선행되어야 함")
            print()
            print("  권장 next step:")
            print("    1) Multi-step loss로 Stage 2 재학습")
            print("       loss = Σ L(ẑ_{t+k}, z_{t+k}) for k=1..N")
            print("    2) Teacher forcing schedule 도입")
            print("       초반: real z_t 사용 → 후반: ẑ_t 사용")
        else:
            delta = 1.0 - t160
            print(f"  진단: t=160에서 cosine drop = {1.0 - freerun[0]:.4f}")
            print(f"  → reward signal 존재 (delta={delta:.4f})")
            print("  → LLM EMA 실험 진행 가능")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()