#!/bin/bash
#SBATCH --job-name=swm_debug
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/debug_eval_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
CKPTS=$SWM/ckpts/checkpoint_files
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210

module purge
module load Singularity/4.3.4

echo "=== Debug Eval: $(date) / Node: $(hostname) ==="

singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
    "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_PATH
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:/opt/conda/lib:\$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot:\$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python3 - << 'PYEOF'
import os, sys, pickle, json
import numpy as np
import torch
from pathlib import Path

SWM = Path('/scratch/mip25/sjLee/SWM')
VLA_BASE = str(SWM / 'ckpts/checkpoint_files/SFT_models/square')

# TF GPU 차단
os.environ['CUDA_VISIBLE_DEVICES'] = ''
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except: pass
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 모델 로드
sys.path.insert(0, str(SWM / 'dependencies/openvla-oft'))
sys.path.insert(0, str(SWM / 'dependencies/openvla-oft/experiments/robot'))

from transformers import AutoModelForVision2Seq, AutoProcessor
from openvla_utils import update_auto_map

update_auto_map(VLA_BASE)
print('[VLA] Loading...')
device = torch.device('cuda')
vla = AutoModelForVision2Seq.from_pretrained(
    VLA_BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(VLA_BASE, trust_remote_code=True)
ds_path = Path(VLA_BASE) / 'dataset_statistics.json'
vla.norm_stats.update(json.load(open(ds_path)))
vla.eval()
print('[VLA] Loaded OK')

# env 로드
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import json as _json
import mimicgen.envs.robosuite
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

DATA_ROOT = SWM / 'data/wmpo_data'
cfg_path = DATA_ROOT / 'data_files/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json'
hdf5 = DATA_ROOT / 'data_files/core_datasets/square/demo_src_square_task_D0/demo.hdf5'

with open(cfg_path) as f:
    cfg_dict = _json.load(f)
cfg_dict['train']['data'] = str(hdf5)
cfg = config_factory(cfg_dict['algo_name'])
with cfg.values_unlocked():
    def _su(d, s):
        for k,v in s.items():
            try: _su(d[k], v) if isinstance(v, dict) else d.__setitem__(k, v)
            except: pass
    _su(cfg, cfg_dict)
cfg.lock()
ObsUtils.initialize_obs_utils_with_config(cfg)
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(hdf5))
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, env_name=env_meta['env_name'],
    render=False, render_offscreen=True, use_image_obs=True)
env = EnvUtils.wrap_env_from_config(env, config=cfg)
print('[Env] Created Square_D0 OK')

# states 로드
with open(SWM / 'data/states/square_d0_states.pkl', 'rb') as f:
    states = pickle.load(f)
print(f'[States] {len(states)} states loaded')
print(f'[States] keys: {list(states[0].keys())}')
print(f'[States] states shape: {states[0][\"states\"].shape}')

# 에피소드 0 실행
print()
print('=== Episode 0 debug ===')
env.reset_to(states[0])
obs = None
for _ in range(10):
    obs, _, _, _ = env.step(np.zeros(7))
print(f'[Obs] keys: {list(obs.keys())}')
img = obs['agentview_image']
print(f'[Obs] agentview_image: shape={img.shape}, dtype={img.dtype}, min={img.min():.3f}, max={img.max():.3f}')

# 이미지 전처리
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf2
from PIL import Image

img_np = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
if img_np.ndim == 3 and img_np.shape[0] == 3:
    img_np = img_np.transpose(1, 2, 0)
print(f'[Img] after format: shape={img_np.shape}, dtype={img_np.dtype}')

# center crop (eval_swm_mimicgen 방식)
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
print(f'[Crop] PIL size: {pil.size}')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# predict_action
prompt = 'In: What action should the robot take to pick up the square nut and insert it onto the peg?\nOut:'
inputs = processor(prompt, pil, return_tensors='pt').to(device, dtype=torch.bfloat16)
print(f'[Input] input_ids shape: {inputs[\"input_ids\"].shape}')
print(f'[Input] pixel_values shape: {inputs[\"pixel_values\"].shape}')
print(f'[Input] last 5 tokens: {inputs[\"input_ids\"][0, -5:].tolist()}')

with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    result = vla.predict_action(**inputs, unnorm_key='square_d0_300_demos', do_sample=False)

actions = result[0] if isinstance(result, tuple) else result
if isinstance(actions, torch.Tensor):
    actions = actions.cpu().numpy()
print(f'[Actions] shape: {actions.shape}')
print(f'[Actions] chunk 0: {actions[0]}')
print(f'[Actions] chunk all:')
for i, a in enumerate(actions):
    print(f'  [{i}] {np.round(a, 4).tolist()}')

# 5 step 실행 후 reward 확인
print()
print('=== 5 steps ===')
for step in range(5):
    a = actions[step % len(actions)]
    obs2, reward, done, info = env.step(a.tolist())
    img2 = obs2['agentview_image']
    print(f'  step {step}: action={np.round(a[:4], 3).tolist()}... reward={reward:.4f} done={done}')
PYEOF
" 2>&1

echo "=== Done: $(date) ==="
