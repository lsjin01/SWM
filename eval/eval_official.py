#!/usr/bin/env python3
"""
eval.py (WMPO/dependencies/openvla-oft/eval.py) 를 우리 환경에 맞게 수정한 버전.
- predict_action 사용 (공식 eval.py 와 동일)
- LoRA 로딩 처리 추가 (SFT 모델)
- 경로를 로컬로 수정
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import pickle
import torch

SWM = os.environ.get("SWM_ROOT", "/scratch/mip25/sjLee/SWM")
sys.path.insert(0, f"{SWM}/dependencies/openvla-oft")
sys.path.insert(0, f"{SWM}/dependencies/openvla-oft/experiments/robot")

from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import mimicgen.envs.robosuite  # noqa
from PIL import Image


TASK_CONFIG = {
    "square": {
        "env_config": f"{SWM}/data/wmpo_data/data_files/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json",
        "states_pkl": f"{SWM}/data/states/square_d0_states.pkl",
        "unnorm_key": "square_d0_300_demos",
        "max_steps":  184,
        "instruction": "square",  # eval.py line 170: args.task_description = args.task
    },
    "coffee": {
        "env_config": f"{SWM}/data/wmpo_data/data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json",
        "states_pkl": f"{SWM}/data/states/coffee_d0_states.pkl",
        "unnorm_key": "coffee_d0_300_demos",
        "max_steps":  256,
        "instruction": "coffee",
    },
}


def _create_env(cfg):
    ext_cfg = json.load(open(cfg["env_config"]))
    rb_cfg = config_factory(ext_cfg["algo_name"])
    with rb_cfg.values_unlocked():
        rb_cfg.update(ext_cfg)
    rb_cfg.lock()
    ObsUtils.initialize_obs_utils_with_config(rb_cfg)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=rb_cfg.train.data,
        all_obs_keys=rb_cfg.all_obs_keys,
        verbose=False,
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=rb_cfg.train.data)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
        use_depth_obs=shape_meta["use_depths"],
    )
    return EnvUtils.wrap_env_from_config(env, config=rb_cfg)


def _pick_agent_image(obs):
    img = obs["agentview_image"]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def load_vla_official(ckpt_path, device):
    """get_vla 사용 + LoRA 자동 감지 로딩."""
    from types import SimpleNamespace
    cfg_ns = SimpleNamespace(
        pretrained_checkpoint=ckpt_path,
        center_crop=True,
        num_images_in_input=1,
        use_proprio=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        load_in_8bit=False,
        load_in_4bit=False,
        num_open_loop_steps=8,
        unnorm_key=None,  # get_vla_action에서 별도 지정
    )
    vla = get_vla(cfg_ns).to(device).eval()

    lora_path = os.path.join(ckpt_path, "lora_adapter")
    if os.path.isdir(lora_path):
        print(f"[VLA] Applying LoRA from {lora_path} ...")
        from peft import PeftModel
        vla = PeftModel.from_pretrained(vla, lora_path)
        vla = vla.merge_and_unload()
        print("[VLA] LoRA merged OK")

    processor = get_processor(cfg_ns)
    return vla, processor, cfg_ns


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

    cfg = TASK_CONFIG[args.task]
    instruction = cfg["instruction"]
    unnorm_key   = cfg["unnorm_key"]
    max_steps    = cfg["max_steps"]

    print(f"\n{'='*60}")
    print(f"  [official eval.py method / predict_action]")
    print(f"  Model      : {args.model_name}")
    print(f"  Task       : {args.task}  |  Episodes: {args.runs}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"{'='*60}")

    vla, processor, cfg_ns = load_vla_official(args.ckpt, device)
    cfg_ns.unnorm_key = unnorm_key

    with open(cfg["states_pkl"], "rb") as f:
        reset_states = pickle.load(f)

    env = _create_env(cfg)

    successes, lengths = 0, []
    t0 = time.time()

    for ep in range(args.runs):
        env.reset_to(reset_states[ep])
        obs = None
        for _ in range(10):  # num_steps_wait=10
            obs, _, _, _ = env.step(np.zeros(7))

        success, ep_len = False, 0
        while ep_len < max_steps:
            img = _pick_agent_image(obs)
            policy_obs = {"full_image": img, "task_description": instruction}
            actions = get_vla_action(cfg_ns, vla, processor, policy_obs, instruction, None, None)
            actions = np.asarray(actions)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            for action in actions[:8]:
                obs, reward, done, _ = env.step(action)
                ep_len += 1
                if reward > 0:
                    success = True
                if ep_len >= max_steps:
                    break

        successes += int(success)
        lengths.append(ep_len)
        sr = successes / (ep + 1)
        elapsed = int(time.time() - t0)
        mark = "✓" if success else "✗"
        print(f"  ep{ep+1:03d}/{args.runs}: {mark}  steps={ep_len:4d}  sr={sr:.3f}  t={elapsed}s")

    success_rate = successes / args.runs
    print(f"\n{'='*60}")
    print(f"  {args.model_name} / {args.task}")
    print(f"  Success Rate : {success_rate:.3f}  ({successes}/{args.runs})")
    print(f"  Mean Steps   : {np.mean(lengths):.1f}")
    print(f"  Total Time   : {int(time.time()-t0)}s")
    print(f"{'='*60}\n")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({
                "model_name": args.model_name,
                "task": args.task,
                "n_episodes": args.runs,
                "n_success": successes,
                "success_rate": success_rate,
                "avg_steps": float(np.mean(lengths)),
                "ckpt": args.ckpt,
                "method": "predict_action (official eval.py)",
            }, f, indent=2)
        print(f"  Saved → {args.save_json}")


if __name__ == "__main__":
    main()
