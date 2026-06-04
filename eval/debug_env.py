import os, sys, json, numpy as np, h5py
sys.path.insert(0, '/opt/conda/envs/wmpo/lib/python3.11/site-packages')
import mimicgen.envs.robosuite
import robomimic.utils.file_utils as FileUtils, robomimic.utils.env_utils as EnvUtils, robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory

hdf5_path = '/home/miplab1/sjLee/swm/data/demos/core_datasets/square/demo_src_square_task_D0/demo.hdf5'
cfg_path  = '/home/miplab1/sjLee/swm/data/data_files/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json'

with open(cfg_path) as f: cd = json.load(f)
cd['train']['data'] = hdf5_path
cfg = config_factory(cd['algo_name'])
with cfg.values_unlocked():
    def su(d,s):
        for k,v in s.items():
            try: su(d[k],v) if isinstance(v,dict) else d.__setitem__(k,v)
            except: pass
    su(cfg,cd)
cfg.lock()
ObsUtils.initialize_obs_utils_with_config(cfg)
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5_path)
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, env_name=env_meta['env_name'], render=False, render_offscreen=True, use_image_obs=True)
env = EnvUtils.wrap_env_from_config(env, config=cfg)
print('Env created')

# Demo 0 재생
with h5py.File(hdf5_path, 'r') as f:
    states = f['data/demo_0/states'][:]
    actions = f['data/demo_0/actions'][:]
print(f'Demo actions: {actions.shape}, states: {states.shape}')

# 초기 상태로 리셋
env.reset_to({'states': states[0]})
obs, r, done, _ = env.step(np.zeros(7))
print(f'Image shape: {obs["agentview_image"].shape}, dtype: {obs["agentview_image"].dtype}')
print(f'Image min/max: {obs["agentview_image"].min()}, {obs["agentview_image"].max()}')

# expert actions 재생
total_r = 0
for i, a in enumerate(actions[:50]):
    obs, r, done, info = env.step(a.tolist())
    if r > 0:
        print(f'  step {i}: REWARD={r:.3f} !!!')
        total_r += r
        break
print(f'Total reward after 50 expert steps: {total_r}')
print(f'Done: {done}')
