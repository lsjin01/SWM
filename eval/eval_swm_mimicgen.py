#!/usr/bin/env python3
"""
SWM Evaluation Script
======================
WMPO eval_wmpo_mimicgen.py와 동일한 인터페이스로 SWM 모델 평가.

환경 변수:
  TASK          : square | coffee | stack_three | three_piece_assembly
  CKPT_PATH     : SWM Stage 3 checkpoint (.pt)
  STAGE1_CKPT   : Stage 1 encoder+heads checkpoint
  STAGE2_CKPT   : Stage 2 transition checkpoint
  VLA_BASE      : OpenVLA-OFT base model 경로 (SFT checkpoint)
  N_EPISODES    : 평가 episode 수 (default: 50)
  VLA_DEVICE    : cuda | cpu (default: cuda)
  SAVE_JSON     : 결과 저장 경로 (optional)

실행 예시:
  CUDA_VISIBLE_DEVICES=0 \\
  TASK=square \\
  CKPT_PATH=outputs/stage3/square/best.pt \\
  STAGE1_CKPT=outputs/stage1/square/best.pt \\
  STAGE2_CKPT=outputs/stage2/square/best.pt \\
  VLA_BASE=/path/to/SFT_models/square \\
  VLA_DEVICE=cuda N_EPISODES=50 \\
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \\
  LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH" \\
  OMP_NUM_THREADS=4 \\
  python eval/eval_swm_mimicgen.py
"""

# TF가 GPU 잡지 못하게 먼저 차단
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
_tf_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# TF import 전 GPU 차단
os.environ["CUDA_VISIBLE_DEVICES_TF_BLOCK"] = "-1"
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# 환경변수 복원
os.environ["CUDA_VISIBLE_DEVICES"] = _tf_vis

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TASK_CONFIG = {
    "square": {
        "env_name":   "Square_D0",
        "unnorm_key": "square_d0_300_demos",
        "max_steps":  184,
        "instruction": "pick up the square nut and insert it onto the peg",
    },
    "coffee": {
        "env_name":   "Coffee_D0",
        "unnorm_key": "coffee_d0_300_demos",
        "max_steps":  256,
        "instruction": "pick up the coffee pod and place it into the coffee machine",
    },
    "stack_three": {
        "env_name":   "StackThree_D0",
        "unnorm_key": "stack_three_d0_300_demos",
        "max_steps":  320,
        "instruction": "stack the three cubes on top of each other",
    },
    "three_piece_assembly": {
        "env_name":   "ThreePieceAssembly_D0",
        "unnorm_key": "three_piece_assembly_d0_300_demos",
        "max_steps":  384,
        "instruction": "assemble the three pieces together",
    },
}

STATES_ROOT = "/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/data/states"


# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────

TASK_TASK_ID = {
    "square":               "square_D0",
    "coffee":               "coffee_D0",
    "stack_three":          "stack_three_D0",
    "three_piece_assembly": "three_piece_assembly_D0",
}

DATA_DIR = "/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/data"


def make_env(task: str):
    """robomimic 방식으로 MimicGen 환경 생성 (WMPO eval과 동일한 방법)."""
    import json as _json
    import mimicgen.envs.robosuite  # noqa: F401
    from robomimic.config import config_factory
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    task_id   = TASK_TASK_ID[task]
    cfg_path  = os.path.join(DATA_DIR, "core_train_configs",
                             f"bc_rnn_image_ds_{task_id}_seed_101.json")
    hdf5_path = os.path.join(DATA_DIR, "demos", "core_datasets",
                             task, f"demo_src_{task}_task_D0", "demo.hdf5")

    with open(cfg_path) as f:
        cfg_dict = _json.load(f)
    cfg_dict["train"]["data"] = hdf5_path
    cfg = config_factory(cfg_dict["algo_name"])
    with cfg.values_unlocked():
        def _safe_update(dst, src):
            for k, v in src.items():
                try:
                    _safe_update(dst[k], v) if isinstance(v, dict) else setattr(dst, k, None) or dst.__setitem__(k, v)
                except Exception:
                    pass
        _safe_update(cfg, cfg_dict)
    cfg.lock()
    ObsUtils.initialize_obs_utils_with_config(cfg)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, env_name=env_meta["env_name"],
        render=False, render_offscreen=True, use_image_obs=True,
    )
    return EnvUtils.wrap_env_from_config(env, config=cfg)


def load_initial_states(task: str) -> list:
    pkl_path = Path(STATES_ROOT) / f"{task}_d0_states.pkl"
    with open(pkl_path, "rb") as f:
        states = pickle.load(f)
    return states


def reset_to_state(env, state: dict, warmup_steps: int = 10):
    """robomimic-wrapped env에서 특정 state로 리셋 후 warmup."""
    env.reset_to(state)
    obs = None
    for _ in range(warmup_steps):
        obs, _, _, _ = env.step(np.zeros(7))
    return obs


def get_image(obs: dict) -> np.ndarray:
    """obs → (H, W, 3) uint8."""
    img = obs["agentview_image"]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:  # (C,H,W) → (H,W,C)
        img = img.transpose(1, 2, 0)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# SWM Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_swm_model(
    stage3_ckpt: str,
    stage1_ckpt: str,
    stage2_ckpt: str,
    vla_base: str,
    device: torch.device,
):
    """SWM 전체 모델 로드."""
    from models.encoder import SWMEncoder
    from models.heads import SWMHeads
    from models.transition import SWMTransition
    from omegaconf import OmegaConf

    # ── Stage 1: Encoder + Heads ──────────────────────────────────────────
    sd1 = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    s1_cfg = OmegaConf.create(sd1.get("cfg", {}))
    latent_dim  = s1_cfg.get("encoder", {}).get("latent_dim", 2176)
    max_nodes   = s1_cfg.get("encoder", {}).get("max_nodes", 8)
    grid_size   = s1_cfg.get("heads", {}).get("depth_head", {}).get("grid_size", 32)
    hidden_dims = list(s1_cfg.get("heads", {}).get("graph_head", {}).get("hidden_dims", [1024, 512]))

    # vla_path required by SWMEncoder to load DINOv2+SigLIP backbone
    encoder = SWMEncoder(
        vla_path=vla_base,
        freeze_backbone=True,
        latent_dim=latent_dim,
    ).to(device)
    heads   = SWMHeads(
        latent_dim=latent_dim,
        max_nodes=max_nodes,
        grid_size=grid_size,
        hidden_dims=hidden_dims,
    ).to(device)
    encoder.load_state_dict(sd1["encoder"], strict=False)
    heads.load_state_dict(sd1["heads"], strict=False)
    encoder.eval()
    heads.eval()
    print(f"[SWM] Encoder+Heads loaded from {stage1_ckpt}")

    # ── Stage 2: Transition ───────────────────────────────────────────────
    sd2 = torch.load(stage2_ckpt, map_location=device, weights_only=False)
    transition = SWMTransition(latent_dim=latent_dim).to(device)
    transition.load_state_dict(sd2["transition"])
    transition.eval()
    print(f"[SWM] Transition loaded from {stage2_ckpt}")

    # ── Stage 3: Policy (OpenVLA-OFT) ────────────────────────────────────
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"[SWM] Loading OpenVLA-OFT base from {vla_base} ...")
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(vla_base, trust_remote_code=True)

    # Stage 3 checkpoint 적용
    if stage3_ckpt and Path(stage3_ckpt).exists():
        sd3 = torch.load(stage3_ckpt, map_location=device, weights_only=False)
        policy_sd = sd3.get("vla", sd3.get("policy", sd3))
        vla.load_state_dict(policy_sd, strict=False)
        print(f"[SWM] Policy loaded from {stage3_ckpt}")
    else:
        print(f"[SWM] No stage3 ckpt — using base VLA only")

    vla.eval()
    return encoder, heads, transition, vla, processor


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def center_crop_and_resize(image_np: np.ndarray) -> 'Image.Image':
    """(H,W,3) uint8 → center-cropped 224×224 PIL (crop_scale=0.9, WMPO와 동일)"""
    import tensorflow as tf
    from PIL import Image
    crop_scale = 0.9
    img = tf.convert_to_tensor(image_np)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)
    h = w = tf.cast(tf.sqrt(crop_scale), tf.float32)
    off = (1 - h) / 2
    boxes = tf.reshape(tf.stack([off, off, off + h, off + w]), (1, 4))
    img = tf.image.crop_and_resize(img, boxes, [0], (224, 224))
    img = tf.clip_by_value(img, 0, 1)
    img = tf.image.convert_image_dtype(img[0], tf.uint8, saturate=True)
    return Image.fromarray(img.numpy()).convert("RGB")


def preprocess_image(image_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H,W,3) uint8 → (1,3,224,224) float tensor."""
    from torchvision import transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    pil = center_crop_and_resize(image_np)
    return transform(pil).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Policy inference
# ─────────────────────────────────────────────────────────────────────────────

NUM_VISION_TOKENS = 256
NUM_ACTIONS_CHUNK = 8
ACTION_DIM        = 7
NUM_ACTION_TOKENS = NUM_ACTIONS_CHUNK * ACTION_DIM  # 56


@torch.no_grad()
def get_action_chunk(
    vla,
    processor,
    image_np: np.ndarray,
    instruction: str,
    unnorm_key: str,
    device: torch.device,
    chunk_size: int = 8,
) -> np.ndarray:
    """OpenVLA-OFT 픽셀 파이프라인 (pixel→VLA→action). 기준선 비교용."""
    pil_image = center_crop_and_resize(image_np)   # WMPO와 동일한 center crop

    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    inputs = processor(
        prompt,
        pil_image,
        return_tensors="pt",
    ).to(device, dtype=torch.bfloat16)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = vla.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
        )

    # predict_action returns (actions, ...) tuple or just actions
    actions = result[0] if isinstance(result, tuple) else result

    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()

    if actions.ndim == 1:
        actions = np.tile(actions[np.newaxis], (chunk_size, 1))
    return actions[:chunk_size]


_IMAGE_TRANSFORM = None


def _get_transform():
    """center crop 후 ToTensor만. Normalize는 encoder 내부에서 backbone별로 적용."""
    from torchvision import transforms as T
    return T.ToTensor()  # center_crop_and_resize가 이미 224×224 PIL 반환


@torch.no_grad()
def get_action_chunk_spatial(
    encoder,
    vla,
    processor,
    image_np: np.ndarray,
    instruction: str,
    unnorm_key: str,
    device: torch.device,
    chunk_size: int = 8,
) -> np.ndarray:
    """
    SWM spatial 파이프라인 — vision_backbone monkey-patch 방식.

    1. processor로 input_ids / attention_mask 준비 (pixel_values는 더미)
    2. SWM encode_spatial → (1, 256, 2176) 산출
    3. vision_backbone.forward를 임시 교체 → projector가 SWM features 받음
    4. predict_action 정상 호출 (special token, action decoding 모두 보존)
    """
    pil   = center_crop_and_resize(image_np)
    img_t = _get_transform()(pil).unsqueeze(0).to(device)

    # SWM spatial features — frozen DINOv2+SigLIP (projector input dim = 2176)
    vb_dtype = next(vla.vision_backbone.parameters()).dtype
    swm_feats = encoder.encode_spatial(img_t).to(device, vb_dtype)  # (1, 256, 2176)

    # processor로 tokenize (pixel_values는 monkey-patch이 무시하므로 더미)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, pil, return_tensors="pt")
    inputs = {k: v.to(device, dtype=torch.bfloat16)
              if torch.is_floating_point(v) else v.to(device)
              for k, v in inputs.items()}

    # vision_backbone.forward를 SWM features 반환으로 임시 교체
    _orig_vb_forward = vla.vision_backbone.forward

    def _swm_vb_forward(pixel_values, *args, **kwargs):
        return swm_feats

    vla.vision_backbone.forward = _swm_vb_forward
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = vla.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
        actions = result[0] if isinstance(result, tuple) else result
    finally:
        vla.vision_backbone.forward = _orig_vb_forward

    if isinstance(actions, torch.Tensor):
        actions = actions.float().cpu().numpy()
    if actions.ndim == 1:
        actions = np.tile(actions[np.newaxis], (chunk_size, 1))
    return actions[:chunk_size]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    task: str,
    stage3_ckpt: str,
    stage1_ckpt: str,
    stage2_ckpt: str,
    vla_base: str,
    n_episodes: int = 50,
    device_str: str = "cuda",
    save_json: str = None,
    chunk_size: int = 8,
    use_z_bypass: bool = True,
):
    task_cfg    = TASK_CONFIG[task]
    device      = torch.device(device_str)
    instruction = task_cfg["instruction"]
    unnorm_key  = task_cfg["unnorm_key"]
    max_steps   = task_cfg["max_steps"]

    # ── Load models ──────────────────────────────────────────────────────
    encoder, heads, transition, vla, processor = load_swm_model(
        stage3_ckpt, stage1_ckpt, stage2_ckpt, vla_base, device
    )

    # ── dataset_statistics 항상 로드 (pixel mode도 predict_action에서 필요) ──
    import json as _json
    if unnorm_key not in vla.norm_stats:
        ds_stats_path = Path(vla_base) / "dataset_statistics.json"
        if ds_stats_path.exists():
            vla.norm_stats.update(_json.load(open(ds_stats_path)))
            print(f"[Eval] Loaded dataset_statistics.json  key={unnorm_key}")
    if unnorm_key not in vla.norm_stats:
        fallback = next(iter(vla.norm_stats))
        print(f"[Eval] unnorm_key '{unnorm_key}' not found, using '{fallback}'")
        unnorm_key = fallback

    if use_z_bypass:
        print(f"[Eval] spatial mode: SWM encode_spatial → vision_backbone monkey-patch → predict_action")
    else:
        print(f"[Eval] pixel mode: image→VLA.predict_action (SFT baseline)")

    # ── Load initial states ───────────────────────────────────────────────
    all_states = load_initial_states(task)
    n_episodes = min(n_episodes, len(all_states))
    print(f"\n{'='*60}")
    print(f"Task: {task}  Episodes: {n_episodes}  Device: {device}")
    print(f"z_bypass={use_z_bypass}")
    print(f"{'='*60}")

    # ── Environment ───────────────────────────────────────────────────────
    env = make_env(task)

    # ── Eval loop ─────────────────────────────────────────────────────────
    n_success  = 0
    step_list  = []
    t_start    = time.time()

    for ep_idx in range(n_episodes):
        obs = reset_to_state(env, all_states[ep_idx])
        ep_success = False
        step       = 0

        while step < max_steps:
            img = get_image(obs)

            # action chunk 생성
            if use_z_bypass:
                actions = get_action_chunk_spatial(
                    encoder, vla, processor, img,
                    instruction, unnorm_key, device,
                    chunk_size=chunk_size,
                )
            else:
                actions = get_action_chunk(
                vla, processor, img,
                instruction, unnorm_key, device,
                chunk_size=chunk_size,
            )

            # chunk 실행
            for a in actions:
                obs, reward, done, info = env.step(a.tolist())
                step += 1
                if reward > 0:
                    ep_success = True
                    break
                if done or step >= max_steps:
                    break
            if ep_success or step >= max_steps:
                break

        n_success += int(ep_success)
        step_list.append(step)
        sr_so_far = n_success / (ep_idx + 1)
        elapsed   = time.time() - t_start
        symbol    = "✓" if ep_success else "✗"

        print(f"ep{ep_idx+1:03d}/{n_episodes}: {symbol}"
              f"  steps={step:4d}"
              f"  sr={sr_so_far:.3f}"
              f"  elapsed={elapsed:.0f}s")

    try:
        env.env.close() if hasattr(env, 'env') else None
    except Exception:
        pass

    # ── Results ───────────────────────────────────────────────────────────
    sr         = n_success / n_episodes
    mean_steps = np.mean(step_list)
    total_time = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"  Success Rate : {sr:.3f}  ({n_success}/{n_episodes})")
    print(f"  Mean Steps   : {mean_steps:.2f}")
    print(f"  Total Time   : {total_time:.0f}s")
    print(f"{'='*60}\n")

    result = {
        "task":          task,
        "stage3_ckpt":   stage3_ckpt,
        "stage1_ckpt":   stage1_ckpt,
        "stage2_ckpt":   stage2_ckpt,
        "vla_base":      vla_base,
        "unnorm_key":    unnorm_key,
        "n_episodes":    n_episodes,
        "success_rate":  sr,
        "n_success":     n_success,
        "mean_steps":    float(mean_steps),
        "total_time_s":  float(total_time),
        "use_z_bypass":  use_z_bypass,
        "timestamp":     datetime.now().isoformat(),
    }

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {save_json}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task        = os.environ.get("TASK", "square")
    stage3_ckpt = os.environ.get("CKPT_PATH", "")
    stage1_ckpt = os.environ.get("STAGE1_CKPT",
                    "outputs/stage1/multitask_dinosiglip/best.pt")
    stage2_ckpt = os.environ.get("STAGE2_CKPT",
                    "outputs/stage2/multitask_dinosiglip/best.pt")
    vla_base    = os.environ.get("VLA_BASE", "")
    n_episodes  = int(os.environ.get("N_EPISODES", 50))
    device_str  = os.environ.get("VLA_DEVICE", "cuda")
    save_json   = os.environ.get("SAVE_JSON", "")
    chunk_size  = int(os.environ.get("CHUNK_SIZE", 8))
    use_z_bypass = os.environ.get("USE_Z_BYPASS", "1") != "0"

    if not vla_base:
        print("ERROR: VLA_BASE 환경변수를 설정하세요.")
        sys.exit(1)

    evaluate(
        task         = task,
        stage3_ckpt  = stage3_ckpt,
        stage1_ckpt  = stage1_ckpt,
        stage2_ckpt  = stage2_ckpt,
        vla_base     = vla_base,
        n_episodes   = n_episodes,
        device_str   = device_str,
        save_json    = save_json or None,
        chunk_size   = chunk_size,
        use_z_bypass = use_z_bypass,
    )
