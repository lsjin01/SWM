#!/bin/bash
#SBATCH --job-name=swm_replay
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/replay_demo_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210

module purge
module load Singularity/4.3.4

echo "=== Demo Replay: $(date) / $(hostname) ==="

singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
    "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:/opt/conda/lib:\$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/robosuite:$SWM/dependencies/robomimic:$SWM/dependencies/mimicgen:\$PYTHONPATH

python3 - << 'PYEOF'
import os, sys, json, pickle
import numpy as np
import h5py
from pathlib import Path

SWM = Path('/scratch/mip25/sjLee/SWM')
DATA_ROOT = SWM / 'data/wmpo_data'
STATES_PKL = SWM / 'data/states/square_d0_states.pkl'

# robomimic env 생성
import mimicgen.envs.robosuite
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

task_id = 'square_D0'
cfg_path = DATA_ROOT / 'data_files/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json'
hdf5 = DATA_ROOT / 'data_files/core_datasets/square/demo_src_square_task_D0/demo.hdf5'

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
print('[Env] action dim:', env.action_dimension)

# pkl 상태 확인
with open(STATES_PKL, 'rb') as f:
    states = pickle.load(f)
print(f'[States] count={len(states)}, shape={states[0][\"states\"].shape}')

# HDF5에서 데모 actions 읽기
print()
print('=== Demo Replay Test ===')
with h5py.File(hdf5, 'r') as f:
    demo_keys = sorted([k for k in f['data'].keys()], key=lambda x: int(x.split('_')[1]))
    print(f'[HDF5] demo count: {len(demo_keys)}')

    # 처음 3개 데모 replay
    for di, dk in enumerate(demo_keys[:3]):
        grp = f[f'data/{dk}']
        actions = grp['actions'][()]  # (T, 7)
        demo_states = grp['states']   # (T, D)
        print(f'\\n  Demo {dk}: actions={actions.shape}, states_shape={demo_states[0].shape}')
        print(f'  action sample[0]: {np.round(actions[0], 4).tolist()}')
        print(f'  action range: [{actions.min():.3f}, {actions.max():.3f}]')

        # states[0] (initial state) reset
        init_state = {'states': demo_states[0]}
        if 'model_file' in grp.attrs:
            init_state['model_file'] = grp.attrs['model_file']

        env.reset_to(init_state)

        # warmup 10 zero-action steps
        obs = None
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(7))

        # demo actions replay
        total_reward = 0
        success_step = -1
        for t, a in enumerate(actions):
            obs, reward, done, info = env.step(a.tolist())
            total_reward += reward
            if reward > 0 and success_step < 0:
                success_step = t
            if done:
                break

        print(f'  total_reward={total_reward:.4f}, success_step={success_step}, done={done}')
        if success_step >= 0:
            print(f'  ✓ SUCCESS at step {success_step}')
        else:
            print(f'  ✗ FAILED (all {len(actions)} steps exhausted)')

# pkl states replay (우리가 저장한 것)
print()
print('=== PKL States + Demo Actions Test ===')
with h5py.File(hdf5, 'r') as f:
    for di in range(3):
        dk = demo_keys[di]
        grp = f[f'data/{dk}']
        actions = grp['actions'][()]

        # pkl에서 state 로드
        pkl_state = states[di]
        env.reset_to(pkl_state)
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
            if done:
                break

        print(f'  PKL demo_{di}: total_reward={total_reward:.4f}, success={success_step>=0}')

PYEOF
" 2>&1

echo "=== Done: $(date) ==="
