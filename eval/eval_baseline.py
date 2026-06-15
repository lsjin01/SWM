#!/usr/bin/env python3
"""
Baseline Evaluation: SFT / WMPO-P128 / WMPO-P1280
===================================================
SWM 인코더 없이 VLA 픽셀 모드만 사용하는 순수 베이스라인 평가.

환경 변수:
  TASK        : square | coffee | stack_three | three_piece_assembly
  VLA_BASE    : 모델 경로 (SFT 또는 WMPO checkpoint 폴더)
  MODEL_NAME  : 결과 레이블용 이름 (e.g. "SFT", "WMPO_P128", "WMPO_P1280")
  N_EPISODES  : 평가 에피소드 수 (default: 50)
  GPU         : CUDA_VISIBLE_DEVICES 값 (default: 0)
  SAVE_JSON   : 결과 저장 경로 (optional)

실행 예시:
  TASK=square VLA_BASE=ckpts/SFT_models/square/checkpoint_files/SFT_models/square \\
  MODEL_NAME=SFT GPU=0 N_EPISODES=50 python eval/eval_baseline.py
"""

import os, sys, json, time, pickle
from pathlib import Path
from datetime import datetime

# wmpo env site-packages 명시적 추가 (conda run 환경에서 누락될 수 있음)
_sp = '/opt/conda/envs/wmpo/lib/python3.11/site-packages'
if _sp not in sys.path:
    sys.path.insert(0, _sp)

import numpy as np
import torch

# GPU 설정
_gpu = os.environ.get("GPU", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

# EGL 헤드리스 렌더링 (osmesa 대신)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_DEVICE_ID", _gpu.split(",")[0])
_cuda_vis = os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_vis

SWM_ROOT    = Path(__file__).parent.parent
DATA_ROOT   = SWM_ROOT / "data" / "wmpo_data"   # fangqi/WMPO 256×256 데이터
STATES_ROOT = SWM_ROOT / "data" / "states"       # states pkl (별도 경로)
CKPTS_ROOT  = SWM_ROOT / "ckpts"

# openvla-oft: 로컬 dependencies 또는 환경변수로 경로 결정
_openvla_default = SWM_ROOT / "dependencies" / "openvla-oft"
OPENVLA_OFT = Path(os.environ.get("OPENVLA_OFT_PATH", str(_openvla_default)))
OPENVLA_ROBOT = OPENVLA_OFT / "experiments" / "robot"
for p in [str(SWM_ROOT), str(OPENVLA_ROBOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ─────────────────────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "square": {
        "env_name":    "Square_D0",
        "task_id":     "square_D0",
        "unnorm_key":  "square_d0_300_demos",
        "max_steps":   184,
        "instruction": "square",  # official WMPO instruction (eval.py line 170: args.task_description = args.task)
    },
    "coffee": {
        "env_name":    "Coffee_D0",
        "task_id":     "coffee_D0",
        "unnorm_key":  "coffee_d0_300_demos",
        "max_steps":   256,
        "instruction": "coffee",  # official WMPO instruction (rob_rollout.py line 196)
    },
    "stack_three": {
        "env_name":    "StackThree_D0",
        "task_id":     "stack_three_D0",
        "unnorm_key":  "stack_three_d0_300_demos",
        "max_steps":   320,
        "instruction": "stack_three",  # official WMPO instruction (eval.py line 170)
    },
    "three_piece_assembly": {
        "env_name":    "ThreePieceAssembly_D0",
        "task_id":     "three_piece_assembly_D0",
        "unnorm_key":  "three_piece_assembly_d0_300_demos",
        "max_steps":   384,
        "instruction": "three_piece_assembly",  # official WMPO instruction (eval.py line 170)
    },
}


def make_env(task: str):
    import mimicgen.envs.robosuite  # noqa
    from robomimic.config import config_factory
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    task_id  = TASK_CONFIG[task]["task_id"]
    # config는 기존 경로, HDF5는 WMPO 256×256 데이터
    cfg_path = DATA_ROOT / "data_files" / "core_train_configs" / f"bc_rnn_image_ds_{task_id}_seed_101.json"
    hdf5     = DATA_ROOT / "data_files" / "core_datasets" / task / f"demo_src_{task}_task_D0" / "demo.hdf5"

    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    assert hdf5.exists(),     f"HDF5 not found: {hdf5}"

    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg_dict["train"]["data"] = str(hdf5)

    cfg = config_factory(cfg_dict["algo_name"])
    with cfg.values_unlocked():
        def _safe_update(dst, src):
            for k, v in src.items():
                try:
                    _safe_update(dst[k], v) if isinstance(v, dict) else dst.__setitem__(k, v)
                except Exception:
                    pass
        _safe_update(cfg, cfg_dict)
    cfg.lock()

    ObsUtils.initialize_obs_utils_with_config(cfg)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(hdf5))
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, env_name=env_meta["env_name"],
        render=False, render_offscreen=True, use_image_obs=True,
    )
    return EnvUtils.wrap_env_from_config(env, config=cfg)


def load_initial_states(task: str) -> list:
    pkl = STATES_ROOT / f"{task}_d0_states.pkl"
    assert pkl.exists(), f"States not found: {pkl}"
    with open(pkl, "rb") as f:
        return pickle.load(f)


def center_crop_resize(image_np: np.ndarray):
    """(H,W,3) uint8 → 224×224 PIL, crop_scale=0.9 (WMPO와 동일)"""
    import tensorflow as tf
    from PIL import Image
    crop_scale = 0.9
    img = tf.image.convert_image_dtype(tf.convert_to_tensor(image_np), tf.float32)
    img = tf.expand_dims(img, 0)
    h = w = float(crop_scale ** 0.5)
    off = (1 - h) / 2
    boxes = tf.reshape(tf.stack([off, off, off + h, off + w]), (1, 4))
    img = tf.image.crop_and_resize(img, boxes, [0], (224, 224))
    img = tf.image.convert_image_dtype(tf.clip_by_value(img[0], 0, 1), tf.uint8, saturate=True)
    return Image.fromarray(img.numpy()).convert("RGB")


def load_vla(vla_base: str, device: torch.device):
    """OpenVLA-OFT 모델 로드. openvla-oft auto_map 패치 포함. LoRA 자동 적용."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # openvla-oft auto_map 패치 (custom modeling_prismatic.py 경로 교정)
    try:
        sys.path.insert(0, str(OPENVLA_OFT / "experiments" / "robot"))
        from openvla_utils import update_auto_map
        update_auto_map(vla_base)
        print(f"[VLA] auto_map updated for {vla_base}")
    except Exception as e:
        print(f"[VLA] auto_map update skipped: {e}")

    print(f"[VLA] Loading from {vla_base} ...")
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # LoRA adapter가 있으면 적용 후 merge (SFT 모델: base + lora_adapter/)
    lora_dir = Path(vla_base) / "lora_adapter"
    if lora_dir.exists():
        from peft import PeftModel
        print(f"[VLA] Applying LoRA from {lora_dir} ...")
        vla = PeftModel.from_pretrained(vla, str(lora_dir))
        vla = vla.merge_and_unload()
        print("[VLA] LoRA merged OK")

    processor = AutoProcessor.from_pretrained(vla_base, trust_remote_code=True)

    # dataset_statistics 로드
    ds_path = Path(vla_base) / "dataset_statistics.json"
    if ds_path.exists():
        vla.norm_stats.update(json.load(open(ds_path)))
        print(f"[VLA] dataset_statistics loaded")

    vla.eval()
    return vla, processor


_PAD_TOKEN_ID = 32000  # tokenizer added_tokens.json: <PAD> → 32000

@torch.no_grad()
def get_action_chunk(vla, processor, image_np, instruction, unnorm_key, device,
                     chunk_size=8, wrist_image_np=None):
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    pil = center_crop_resize(image_np)
    batch_feature = processor(prompt, pil)

    if wrist_image_np is not None:
        wrist_pil = center_crop_resize(wrist_image_np)
        wrist_feature = processor(prompt, wrist_pil)
        batch_feature["pixel_values"] = torch.cat(
            [batch_feature["pixel_values"], wrist_feature["pixel_values"]], dim=1
        )

    inputs = {k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
              for k, v in batch_feature.items()}

    # 공식 WMPO 표준: rob_rollout.py L272-276 — input이 빈 토큰(29871)으로 끝나지 않으면 추가
    _EMPTY_TOKEN_ID = 29871
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == _EMPTY_TOKEN_ID):
        extra = torch.full((input_ids.shape[0], 1), _EMPTY_TOKEN_ID,
                           dtype=input_ids.dtype, device=input_ids.device)
        inputs["input_ids"] = torch.cat([input_ids, extra], dim=-1)
        attn = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=attn.device)], dim=-1
        )

    # 공식 WMPO 표준: generate_action_verl (rob_rollout.py line 506)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actions, response, normalized_actions = vla.generate_action_verl(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            temperature=1.0,
            padding_idx=_PAD_TOKEN_ID,
        )

    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    if actions.ndim == 3:
        actions = actions[0]
    if actions.ndim == 1:
        actions = np.tile(actions[np.newaxis], (chunk_size, 1))
    return actions[:chunk_size]


def evaluate(task, vla_base, model_name, n_episodes=50, device_str="cuda", save_json=None):
    cfg    = TASK_CONFIG[task]
    device = torch.device(device_str)

    vla, processor = load_vla(vla_base, device)

    unnorm_key = cfg["unnorm_key"]
    if unnorm_key not in vla.norm_stats:
        fallback = next(iter(vla.norm_stats))
        print(f"[Eval] unnorm_key '{unnorm_key}' not found → using '{fallback}'")
        unnorm_key = fallback

    states     = load_initial_states(task)
    n_episodes = min(n_episodes, len(states))
    env        = make_env(task)

    print(f"\n{'='*60}")
    print(f"  Model : {model_name}")
    print(f"  Task  : {task}  |  Episodes: {n_episodes}")
    print(f"{'='*60}")

    n_success, step_list, t0 = 0, [], time.time()

    for ep in range(n_episodes):
        env.reset_to(states[ep])
        obs = None
        for _ in range(10):                             # warmup
            obs, _, _, _ = env.step(np.zeros(7))

        def _to_hwc_uint8(im):
            if im.dtype != np.uint8:
                im = (im * 255).astype(np.uint8)
            if im.ndim == 3 and im.shape[0] == 3:
                im = im.transpose(1, 2, 0)
            return im

        success, step = False, 0
        while step < cfg["max_steps"]:
            img = _to_hwc_uint8(obs["agentview_image"])

            # 공식 evaluate.sh: num_images_in_input=1 (agentview만)
            actions = get_action_chunk(vla, processor, img,
                                       cfg["instruction"], unnorm_key, device,
                                       wrist_image_np=None)
            for a in actions:
                obs, reward, done, _ = env.step(a.tolist())
                step += 1
                if reward > 0:
                    success = True
                    break
                if done or step >= cfg["max_steps"]:
                    break
            if success or step >= cfg["max_steps"]:
                break

        n_success += int(success)
        step_list.append(step)
        sr = n_success / (ep + 1)
        print(f"  ep{ep+1:03d}/{n_episodes}: {'✓' if success else '✗'}"
              f"  steps={step:4d}  sr={sr:.3f}  t={time.time()-t0:.0f}s")

    try:
        env.env.close() if hasattr(env, "env") else None
    except Exception:
        pass

    sr         = n_success / n_episodes
    mean_steps = float(np.mean(step_list))
    total_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  {model_name} / {task}")
    print(f"  Success Rate : {sr:.3f}  ({n_success}/{n_episodes})")
    print(f"  Mean Steps   : {mean_steps:.1f}")
    print(f"  Total Time   : {total_time:.0f}s")
    print(f"{'='*60}\n")

    result = {
        "model_name":   model_name,
        "task":         task,
        "vla_base":     vla_base,
        "n_episodes":   n_episodes,
        "success_rate": sr,
        "n_success":    n_success,
        "mean_steps":   mean_steps,
        "total_time_s": total_time,
        "timestamp":    datetime.now().isoformat(),
    }
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(save_json, "w"), indent=2)
        print(f"  Saved → {save_json}")

    return result


if __name__ == "__main__":
    task       = os.environ.get("TASK", "square")
    vla_base   = os.environ.get("VLA_BASE", "")
    model_name = os.environ.get("MODEL_NAME", "baseline")
    n_episodes = int(os.environ.get("N_EPISODES", "50"))
    device_str = os.environ.get("VLA_DEVICE", "cuda")
    save_json  = os.environ.get("SAVE_JSON", "") or None

    if not vla_base:
        print("ERROR: VLA_BASE 환경변수를 설정하세요.")
        sys.exit(1)

    evaluate(task, vla_base, model_name, n_episodes, device_str, save_json)
