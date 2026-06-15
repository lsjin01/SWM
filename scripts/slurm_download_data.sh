#!/bin/bash
#SBATCH --job-name=swm_data
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/download_data_${SLURM_JOB_ID}.log"
exec 2>&1

SANDBOX=/scratch/mip25/sjLee/SWM/containers/swm_sandbox
SWM=/scratch/mip25/sjLee/SWM

module purge
module load Singularity/4.3.4

echo "=== Data Download: $(date) / Node: $(hostname) ==="

singularity exec \
    --bind /scratch/mip25:/scratch/mip25 \
    "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export HF_HUB_CACHE=/scratch/mip25/sjLee/.cache/huggingface
export HF_HOME=/scratch/mip25/sjLee/.cache/huggingface
mkdir -p \$HF_HUB_CACHE

python3 - << 'PYEOF'
from huggingface_hub import snapshot_download
import os

HF_TOKEN = open('/scratch/mip25/sjLee/.hf_token').read().strip()
SWM = '/scratch/mip25/sjLee/SWM'
DATA_DIR = f'{SWM}/data/wmpo_data'
REPO_ID = 'fangqi/WMPO'

# [1] square HDF5 + train configs
print('[1] Downloading square dataset (demo.hdf5 ~4GB) ...')
snapshot_download(
    repo_id=REPO_ID,
    repo_type='model',
    local_dir=DATA_DIR,
    allow_patterns=[
        'data_files/core_datasets/square/**',
        'data_files/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json',
    ],
    token=HF_TOKEN,
)
print('[1] DONE')

# 결과 확인
import os.path as osp
hdf5 = f'{DATA_DIR}/data_files/core_datasets/square/demo_src_square_task_D0/demo.hdf5'
cfg  = f'{DATA_DIR}/data_files/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json'
print(f'  HDF5 : {hdf5}  exists={osp.exists(hdf5)}  size={osp.getsize(hdf5)//1024**2 if osp.exists(hdf5) else \"N/A\"}MB')
print(f'  cfg  : {cfg}  exists={osp.exists(cfg)}')
PYEOF
"

echo "=== DONE: $(date) ==="
