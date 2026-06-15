#!/usr/bin/env python3
"""
rob_rollout.py (WMPO 공식 eval 추론 엔진) 의 핵심 로직을 standalone으로 포팅.
- generate_action_verl 사용 (공식과 동일)
- center_crop_image (TensorFlow, 공식과 동일)
- process_input: empty token 추가 + left-padding (공식과 동일)
- num_steps_wait=10 warmup (공식과 동일)
- do_sample=False (validation 기준, ray_trainer.py line 393)
- instruction: rob_rollout.py 기준
    square  → "Insert the square into the stick"
    others  → task name ("coffee", "stack_three", ...)
- GPU: 단일 프로세스, CUDA_VISIBLE_DEVICES 로 지정
"""
import argparse
import json
import os
import sys
import time
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import AutoProcessor

SWM = os.environ.get("SWM_ROOT", "/scratch/mip25/sjLee/SWM")
sys.path.insert(0, f"{SWM}/dependencies/openvla-oft")
sys.path.insert(0, f"{SWM}/dependencies/openvla-oft/experiments/robot")

from transformers import AutoModelForVision2Seq, AutoProcessor

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import mimicgen.envs.robosuite  # noqa

# ── task 설정 (rob_rollout.py 기준) ──────────────────────────────────────────
TASK_CONFIG = {
    "square": {
        "env_config": f"{SWM}/data/wmpo_data/data_files/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json",
        "states_pkl": f"{SWM}/data/states/square_d0_states.pkl",
        "unnorm_key":  "square_d0_300_demos",
        "max_steps":   184,
        "instruction": "Insert the square into the stick",  # rob_rollout.py line 194
    },
    "coffee": {
        "env_config": f"{SWM}/data/wmpo_data/data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json",
        "states_pkl": f"{SWM}/data/states/coffee_d0_states.pkl",
        "unnorm_key":  "coffee_d0_300_demos",
        "max_steps":   256,
        "instruction": "coffee",  # rob_rollout.py line 196: self.task_description = self.task
    },
}

# ── center_crop (TensorFlow, rob_rollout.py 완전 동일) ────────────────────────
def center_crop_image(image):
    import tensorflow as tf
    batch_size = 1
    crop_scale = 0.9
    image_tf = tf.convert_to_tensor(np.array(image))
    orig_dtype = image_tf.dtype
    image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
    image_tf = tf.expand_dims(image_tf, axis=0)
    new_h = tf.clip_by_value(tf.sqrt(crop_scale), 0, 1)
    new_w = new_h
    h_off = (1 - new_h) / 2
    w_off = (1 - new_w) / 2
    boxes = tf.stack([h_off, w_off, h_off + new_h, w_off + new_w])
    boxes = tf.reshape(boxes, [1, 4])
    image_tf = tf.image.crop_and_resize(image_tf, boxes, [0], (224, 224))
    image_tf = tf.clip_by_value(image_tf, 0, 1)
    image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
    image_tf = image_tf[0]
    return Image.fromarray(image_tf.numpy()).convert("RGB")


# ── 환경 생성 ─────────────────────────────────────────────────────────────────
def create_env(task):
    cfg_data = TASK_CONFIG[task]
    env_config_path = cfg_data["env_config"]
    # HDF5 경로를 로컬로 패치 (eval_baseline.py 방식)
    hdf5 = (f"{SWM}/data/wmpo_data/data_files/core_datasets/"
            f"{task}/demo_src_{task}_task_D0/demo.hdf5")
    ext_cfg = json.load(open(env_config_path))
    ext_cfg["train"]["data"] = hdf5

    cfg = config_factory(ext_cfg["algo_name"])
    with cfg.values_unlocked():
        def _safe_update(dst, src):
            for k, v in src.items():
                try:
                    _safe_update(dst[k], v) if isinstance(v, dict) else dst.__setitem__(k, v)
                except Exception:
                    pass
        _safe_update(cfg, ext_cfg)
    cfg.lock()
    ObsUtils.initialize_obs_utils_with_config(cfg)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=hdf5, all_obs_keys=cfg.all_obs_keys, verbose=False)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, env_name=env_meta["env_name"],
        render=False, render_offscreen=True,
        use_image_obs=shape_meta["use_images"], use_depth_obs=shape_meta["use_depths"])
    return EnvUtils.wrap_env_from_config(env, config=cfg)


# ── 모델 로드 (eval_baseline.py의 load_vla 와 동일) ──────────────────────────
def load_model(ckpt_path, device):
    from pathlib import Path
    try:
        from openvla_utils import update_auto_map
        update_auto_map(ckpt_path)
        print(f"[VLA] auto_map updated for {ckpt_path}")
    except Exception as e:
        print(f"[VLA] auto_map update skipped: {e}")

    print(f"[VLA] Loading from {ckpt_path} (float32) ...")
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt_path, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)

    lora_path = Path(ckpt_path) / "lora_adapter"
    if lora_path.exists():
        from peft import PeftModel
        print(f"[VLA] Applying LoRA from {lora_path} ...")
        vla = PeftModel.from_pretrained(vla, str(lora_path), torch_dtype=torch.float32)
        vla = vla.merge_and_unload()
        print("[VLA] LoRA merged OK (float32)")

    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)

    ds_path = Path(ckpt_path) / "dataset_statistics.json"
    if ds_path.exists():
        vla.norm_stats.update(json.load(open(ds_path)))
        print("[VLA] dataset_statistics loaded")

    vla.eval()
    return vla, processor


# ── process_input (rob_rollout.py process_input 완전 동일) ────────────────────
def process_input(processor, image_np, instruction, device):
    """단일 이미지 처리. rob_rollout.py process_input (batch_size=1) 과 동일."""
    image = Image.fromarray(image_np).convert("RGB")
    image = center_crop_image(image)                        # TF center_crop
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    feat = processor(prompt, image)

    input_ids     = feat["input_ids"]       # (1, L)
    attention_mask = feat["attention_mask"]
    pixel_values   = feat["pixel_values"]

    # empty token (rob_rollout.py line 272-279)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            [input_ids, torch.tensor([[29871]], dtype=input_ids.dtype)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=1)

    # left-pad sort (rob_rollout.py line 288-300, batch_size=1 이므로 trivial)
    pad_id = processor.tokenizer.pad_token_id
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values   = pixel_values.to(device)

    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "pixel_values": pixel_values}


# ── 메인 평가 루프 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(task, ckpt_path, model_name, n_episodes, device, save_json):
    cfg = TASK_CONFIG[task]

    print(f"\n{'='*60}")
    print(f"  [rob_rollout.py port / generate_action_verl]")
    print(f"  Model      : {model_name}")
    print(f"  Task       : {task}  |  Episodes: {n_episodes}")
    print(f"  Instruction: '{cfg['instruction']}'")
    print(f"{'='*60}")

    vla, processor = load_model(ckpt_path, device)
    pad_id = processor.tokenizer.pad_token_id

    with open(cfg["states_pkl"], "rb") as f:
        reset_states = pickle.load(f)

    env = create_env(task)
    instruction = cfg["instruction"]
    unnorm_key   = cfg["unnorm_key"]
    max_steps    = cfg["max_steps"]

    successes, lengths = 0, []
    t0 = time.time()

    for ep in range(n_episodes):
        env.reset_to(reset_states[ep])
        obs = None

        # num_steps_wait = 10 (rob_rollout.py line 353-356)
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(7))
            obs["agentview_image"] = (obs["agentview_image"] * 255).astype(np.uint8).transpose(1, 2, 0)

        success, step = False, 0
        while step < max_steps:
            img = obs["agentview_image"]  # already HWC uint8 from warmup/prev step

            inputs = process_input(processor, img, instruction, device)

            # do_sample=False (ray_trainer.py line 393, validation)
            with torch.autocast("cuda", dtype=torch.float32):
                actions, response, normalized_actions = vla.generate_action_verl(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["attention_mask"],
                    padding_idx=pad_id,
                    do_sample=False,
                    unnorm_key=unnorm_key,
                    temperature=1.0,
                )

            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            if actions.ndim == 3:
                actions = actions[0]     # (8, 7)

            for a in actions:
                obs, reward, done, _ = env.step(a.tolist())
                obs["agentview_image"] = (obs["agentview_image"] * 255).astype(np.uint8).transpose(1, 2, 0)
                step += 1
                if reward > 0:
                    success = True
                if step >= max_steps:
                    break

        successes += int(success)
        lengths.append(step)
        sr = successes / (ep + 1)
        elapsed = int(time.time() - t0)
        mark = "✓" if success else "✗"
        print(f"  ep{ep+1:03d}/{n_episodes}: {mark}  steps={step:4d}  sr={sr:.3f}  t={elapsed}s")

    sr_final = successes / n_episodes
    print(f"\n{'='*60}")
    print(f"  {model_name} / {task}")
    print(f"  Success Rate : {sr_final:.3f}  ({successes}/{n_episodes})")
    print(f"  Mean Steps   : {np.mean(lengths):.1f}")
    print(f"  Total Time   : {int(time.time()-t0)}s")
    print(f"{'='*60}\n")

    if save_json:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w") as f:
            json.dump({
                "model_name": model_name,
                "task": task,
                "n_episodes": n_episodes,
                "n_success": successes,
                "success_rate": sr_final,
                "avg_steps": float(np.mean(lengths)),
                "method": "rob_rollout port / generate_action_verl",
                "instruction": cfg["instruction"],
                "ckpt": ckpt_path,
            }, f, indent=2)
        print(f"  Saved → {save_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       type=str, required=True)
    parser.add_argument("--ckpt",       type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--runs",       type=int, default=128)
    parser.add_argument("--gpu",        type=str, default="0")
    parser.add_argument("--save_json",  type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda")

    evaluate(args.task, args.ckpt, args.model_name, args.runs, device, args.save_json)


if __name__ == "__main__":
    main()
