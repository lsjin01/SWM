import os, sys, json, pickle
import numpy as np
import torch
from pathlib import Path

SWM = Path('/scratch/mip25/sjLee/SWM')
CKPTS = SWM / 'ckpts/checkpoint_files'
DATA_ROOT = SWM / 'data/wmpo_data'
STATES_PKL = SWM / 'data/states/square_d0_states.pkl'

os.environ['CUDA_VISIBLE_DEVICES'] = ''
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except: pass
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

sys.path.insert(0, str(SWM / 'dependencies/openvla-oft'))
sys.path.insert(0, str(SWM / 'dependencies/openvla-oft/experiments/robot'))
sys.path.insert(0, str(SWM / 'dependencies/robosuite'))
sys.path.insert(0, str(SWM / 'dependencies/robomimic'))
sys.path.insert(0, str(SWM / 'dependencies/mimicgen'))

from transformers import AutoModelForVision2Seq, AutoProcessor
from openvla_utils import update_auto_map

# P128 로드
model_path = CKPTS / 'WMPO_models/square/P_128'
update_auto_map(str(model_path))
device = torch.device('cuda')
vla = AutoModelForVision2Seq.from_pretrained(
    str(model_path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
vla.norm_stats.update(json.load(open(model_path / 'dataset_statistics.json')))
vla.eval()
print('[VLA] P128 loaded')

# 환경 생성
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
print('[Env] created')

with open(STATES_PKL, 'rb') as f:
    states = pickle.load(f)

env.reset_to(states[0])
obs = None
for _ in range(10):
    obs, _, _, _ = env.step(np.zeros(7))

img = obs['agentview_image']
if img.dtype != np.uint8:
    img = (img * 255).astype(np.uint8)
if img.ndim == 3 and img.shape[0] == 3:
    img = img.transpose(1, 2, 0)

import tensorflow as tf2
from PIL import Image
crop_scale = 0.9
t = tf2.convert_to_tensor(img)
t = tf2.image.convert_image_dtype(t, tf2.float32)
t = tf2.expand_dims(t, 0)
h = w = float(crop_scale ** 0.5)
off = (1-h)/2
boxes = tf2.reshape(tf2.stack([off, off, off+h, off+w]), (1,4))
t = tf2.image.crop_and_resize(t, boxes, [0], (224,224))
t = tf2.clip_by_value(t, 0, 1)
t = tf2.image.convert_image_dtype(t[0], tf2.uint8, saturate=True)
pil = Image.fromarray(t.numpy()).convert('RGB')

instruction = 'pick up the square nut and insert it onto the peg'
unnorm_key = 'square_d0_300_demos'
prompt = f'In: What action should the robot take to {instruction}?\nOut:'
inputs = processor(prompt, pil)
inputs = {k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
          for k, v in inputs.items()}

print(f'[Input] input_ids shape: {inputs["input_ids"].shape}')
print(f'[Input] last 5 tokens: {inputs["input_ids"][0,-5:].tolist()}')

# predict_action
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    r1 = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
a1 = r1[0] if isinstance(r1, tuple) else r1
if isinstance(a1, torch.Tensor): a1 = a1.cpu().numpy()
print(f'[predict_action] chunk[0]: {np.round(a1[0] if a1.ndim>1 else a1, 4).tolist()}')

# generate_action_verl
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    r2, resp, norm = vla.generate_action_verl(**inputs, unnorm_key=unnorm_key,
                                              do_sample=False, temperature=1.0, padding_idx=32000)
a2 = r2
if isinstance(a2, torch.Tensor): a2 = a2.cpu().numpy()
if a2.ndim == 3: a2 = a2[0]
print(f'[generate_action_verl] shape={a2.shape}, chunk[0]: {np.round(a2[0] if a2.ndim>1 else a2, 4).tolist()}')

# norm stats 확인
print(f'[norm_stats] keys: {list(vla.norm_stats.keys())}')
ns = vla.norm_stats[unnorm_key]['action']
print(f'  q01: {ns["q01"]}')
print(f'  q99: {ns["q99"]}')
