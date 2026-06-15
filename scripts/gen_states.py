"""
demo.hdf5에서 각 데모의 초기 sim state를 추출해서
data/states/{task}_d0_states.pkl 로 저장.

eval_baseline.py 의 env.reset_to(states[ep]) 에서 사용됨.
"""
import argparse, os, pickle
import h5py
import numpy as np

def extract_states(hdf5_path: str, out_pkl: str, max_demos: int = 300):
    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = sorted(
            [k for k in f['data'].keys()],
            key=lambda x: int(x.split('_')[1])
        )
        demo_keys = demo_keys[:max_demos]
        states = []
        for dk in demo_keys:
            grp = f[f'data/{dk}']
            # robosuite/MimicGen: 'states' shape = [T, D]
            sim_state = grp['states'][0]   # t=0 → initial state
            entry = {'states': sim_state}
            # robomimic env_robosuite.reset_to()는 'model' 키를 봄 ('model_file' 아님)
            if 'model_file' in grp.attrs:
                entry['model'] = grp.attrs['model_file']
            states.append(entry)

    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(states, f)
    print(f'[gen_states] Saved {len(states)} states → {out_pkl}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task',     default='square')
    p.add_argument('--swm_root', default='/scratch/mip25/sjLee/SWM')
    p.add_argument('--max_demos', type=int, default=300)
    args = p.parse_args()

    hdf5 = (f'{args.swm_root}/data/wmpo_data/data_files/core_datasets'
            f'/{args.task}/demo_src_{args.task}_task_D0/demo.hdf5')
    out  = f'{args.swm_root}/data/states/{args.task}_d0_states.pkl'

    assert os.path.exists(hdf5), f'HDF5 not found: {hdf5}'
    extract_states(hdf5, out, args.max_demos)
