#!/bin/bash
#SBATCH --job-name=swm_dbg3
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/debug3_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210

module purge
module load Singularity/4.3.4

echo "=== Debug3: $(date) / Node: $(hostname) ==="

singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
    "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:/opt/conda/lib:\$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot:$SWM/dependencies/robosuite:$SWM/dependencies/robomimic:$SWM/dependencies/mimicgen:\$PYTHONPATH

python3 - << 'PYEOF'
import os, sys, json, pickle
import numpy as np
import torch
import h5py
from pathlib import Path

SWM = Path('/scratch/mip25/sjLee/SWM')
CKPTS = SWM / 'ckpts/checkpoint_files'
DATA_ROOT = SWM / 'data/wmpo_data'
STATES_PKL = SWM / 'data/states/square_d0_states.pkl'

# ── [A] PKL 상태로 데모 replay 테스트 ─────────────────────────────
print('=== [A] PKL States Demo Replay Test ===')

import mimicgen.envs.robosuite
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

hdf5 = DATA_ROOT / 'data_files/core_datasets/square/demo_src_square_task_D0/demo.hdf5'
cfg_path = DATA_ROOT / 'data_files/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json'

with open(cfg_path) as f:
    cfg_dict = json.load(f)
cfg_dict['train']['data'] = str(hdf5)
cfg = config_factory(cfg_dict['algo_name'])
with cfg.values_unlocked():
    def _su(d, s):
        for k, v in s.items():
            try: _su(d[k], v) if isinstance(v, dict) else d.__setitem__(k, v)
            except: pass
    _su(cfg, cfg_dict)
cfg.lock()
ObsUtils.initialize_obs_utils_with_config(cfg)
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(hdf5))
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, env_name=env_meta['env_name'],
    render=False, render_offscreen=True, use_image_obs=True)
env = EnvUtils.wrap_env_from_config(env, config=cfg)
print('[Env] Created Square_D0')

with open(STATES_PKL, 'rb') as f:
    states = pickle.load(f)
print(f'[States] count={len(states)}, keys={list(states[0].keys())}')
print(f'[States] model key present: {\"model\" in states[0]}, model len={len(states[0][\"model\"]) if \"model\" in states[0] else 0}')

with h5py.File(hdf5, 'r') as f:
    demo_keys = sorted([k for k in f['data'].keys()], key=lambda x: int(x.split('_')[1]))
    for di in range(3):
        dk = demo_keys[di]
        grp = f[f'data/{dk}']
        actions = grp['actions'][()]
        # PKL state reset (now with 'model' key)
        env.reset_to(states[di])
        obs = None
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(7))
        total_reward = 0
        success_step = -1
        for t, a in enumerate(actions):
            obs, reward, done, info = env.step(a.tolist())
            total_reward += reward
            if reward > 0 and success_step < 0:
                success_step = t
            if done: break
        print(f'  PKL demo_{di}: total_reward={total_reward:.4f}, success={success_step>=0} (step={success_step})')

print()

# ── [B] SFT 모델 로드 + 액션 디버깅 ─────────────────────────────
print('=== [B] SFT Action Debug ===')

os.environ['CUDA_VISIBLE_DEVICES'] = ''
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except: pass
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import AutoModelForVision2Seq, AutoProcessor
from openvla_utils import update_auto_map
from peft import PeftModel

model_path = CKPTS / 'SFT_models/square'
update_auto_map(str(model_path))
device = torch.device('cuda')
vla = AutoModelForVision2Seq.from_pretrained(
    str(model_path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device)
lora_dir = model_path / 'lora_adapter'
if lora_dir.exists():
    vla = PeftModel.from_pretrained(vla, str(lora_dir))
    vla = vla.merge_and_unload()
    print('[VLA] LoRA merged')
processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
ds_path = model_path / 'dataset_statistics.json'
if ds_path.exists():
    vla.norm_stats.update(json.load(open(ds_path)))
vla.eval()
print('[VLA] Model loaded')

# ep0 reset
env.reset_to(states[0])
obs = None
for _ in range(10):
    obs, _, _, _ = env.step(np.zeros(7))

img = obs['agentview_image']
print(f'[Obs] agentview_image: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.3f},{img.max():.3f}]')

# center crop (같은 방식)
import tensorflow as tf2
from PIL import Image
img_np = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
if img_np.ndim == 3 and img_np.shape[0] == 3:
    img_np = img_np.transpose(1, 2, 0)
crop_scale = 0.9
t = tf2.convert_to_tensor(img_np)
t = tf2.image.convert_image_dtype(t, tf2.float32)
t = tf2.expand_dims(t, 0)
h = w = float(crop_scale ** 0.5)
off = (1 - h) / 2
boxes = tf2.reshape(tf2.stack([off, off, off + h, off + w]), (1, 4))
t = tf2.image.crop_and_resize(t, boxes, [0], (224, 224))
t = tf2.clip_by_value(t, 0, 1)
t = tf2.image.convert_image_dtype(t[0], tf2.uint8, saturate=True)
pil = Image.fromarray(t.numpy()).convert('RGB')
print(f'[Img] PIL size: {pil.size}')

instruction = 'pick up the square nut and insert it onto the peg'
unnorm_key = 'square_d0_300_demos'
prompt = f'In: What action should the robot take to {instruction}?\nOut:'
inputs = processor(prompt, pil)
inputs = {k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
          for k, v in inputs.items()}

# predict_action (원본 방식)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    r1 = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
a1 = r1[0] if isinstance(r1, tuple) else r1
if isinstance(a1, torch.Tensor): a1 = a1.cpu().numpy()
print(f'[predict_action] shape={a1.shape}')
print(f'[predict_action] chunk[0]: {np.round(a1[0] if a1.ndim>1 else a1, 4).tolist()}')
print(f'[predict_action] range: [{a1.min():.4f}, {a1.max():.4f}]')

# generate_action_verl 방식
_PAD_TOKEN_ID = 32000
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    r2 = vla.generate_action_verl(**inputs, unnorm_key=unnorm_key,
                                   do_sample=False, temperature=1.0, padding_idx=_PAD_TOKEN_ID)
a2 = r2[0]
if isinstance(a2, torch.Tensor): a2 = a2.cpu().numpy()
if a2.ndim == 3: a2 = a2[0]
print(f'[generate_action_verl] shape={a2.shape}')
print(f'[generate_action_verl] chunk[0]: {np.round(a2[0] if a2.ndim>1 else a2, 4).tolist()}')
print(f'[generate_action_verl] range: [{a2.min():.4f}, {a2.max():.4f}]')

# HDF5 demo actions (ground truth)
with h5py.File(hdf5, 'r') as f:
    demo_keys2 = sorted([k for k in f['data'].keys()], key=lambda x: int(x.split('_')[1]))
    gt_actions = f[f'data/{demo_keys2[0]}/actions'][()]
print(f'[GT actions] demo_0, step 0: {np.round(gt_actions[0], 4).tolist()}')
print(f'[GT actions] range: [{gt_actions.min():.4f}, {gt_actions.max():.4f}]')

# 실제 10 스텝 실행 (VLA 액션으로)
print()
print('[eval] 10 steps with predict_action:')
env.reset_to(states[0])
obs = None
for _ in range(10):
    obs, _, _, _ = env.step(np.zeros(7))
for step in range(10):
    img_np2 = obs['agentview_image']
    if img_np2.dtype != np.uint8:
        img_np2 = (img_np2 * 255).astype(np.uint8)
    if img_np2.ndim == 3 and img_np2.shape[0] == 3:
        img_np2 = img_np2.transpose(1, 2, 0)
    t2 = tf2.convert_to_tensor(img_np2)
    t2 = tf2.image.convert_image_dtype(t2, tf2.float32)
    t2 = tf2.expand_dims(t2, 0)
    t2 = tf2.image.crop_and_resize(t2, boxes, [0], (224, 224))
    t2 = tf2.clip_by_value(t2, 0, 1)
    t2 = tf2.image.convert_image_dtype(t2[0], tf2.uint8, saturate=True)
    pil2 = Image.fromarray(t2.numpy()).convert('RGB')
    inp2 = processor(prompt, pil2)
    inp2 = {k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
            for k, v in inp2.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        r3 = vla.predict_action(**inp2, unnorm_key=unnorm_key, do_sample=False)
    a3 = r3[0] if isinstance(r3, tuple) else r3
    if isinstance(a3, torch.Tensor): a3 = a3.cpu().numpy()
    act = a3[0] if a3.ndim > 1 else a3
    obs, reward, done, info = env.step(act.tolist())
    print(f'  step{step}: act={np.round(act[:4],3).tolist()}... reward={reward:.4f}')

PYEOF
" 2>&1

echo "=== Done: $(date) ==="
