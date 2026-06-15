#!/bin/bash
#SBATCH --job-name=swm_coffee
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/coffee_setup_eval_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
CKPTS=$SWM/ckpts/checkpoint_files
DATA=$SWM/data/wmpo_data
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210
TASK=coffee
N_EPISODES=128

module purge
module load Singularity/4.3.4

echo "=== Coffee Setup + Eval: $(date) / Node: $(hostname) ==="

# ── [1] HuggingFace 다운로드 ──────────────────────────────────────────────
echo ""
echo "[1] Downloading coffee models and data from HuggingFace..."

# 이미 다운로드된 경우 스킵
SFT_COFFEE=$CKPTS/SFT_models/coffee
P128_COFFEE=$CKPTS/WMPO_models/coffee/P_128
P1280_COFFEE=$CKPTS/WMPO_models/coffee/P_1280
if [ -d "$SFT_COFFEE" ] && [ -d "$P128_COFFEE" ] && [ -d "$P1280_COFFEE" ]; then
    echo "[1] Coffee models already present, skipping download."
else

singularity exec \
    --bind /scratch/mip25:/scratch/mip25 \
    "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python3 - << 'PYEOF'
from huggingface_hub import snapshot_download
import os

CKPTS_DIR = '/scratch/mip25/sjLee/SWM/ckpts'  # parent of checkpoint_files/
DATA_DIR  = '/scratch/mip25/sjLee/SWM/data/wmpo_data'

print('Downloading SFT_models/coffee ...')
snapshot_download(repo_id='fangqi/WMPO', repo_type='model',
    local_dir=CKPTS_DIR, local_dir_use_symlinks=False,
    allow_patterns=['checkpoint_files/SFT_models/coffee/**'])

print('Downloading WMPO_models/coffee ...')
snapshot_download(repo_id='fangqi/WMPO', repo_type='model',
    local_dir=CKPTS_DIR, local_dir_use_symlinks=False,
    allow_patterns=['checkpoint_files/WMPO_models/coffee/**'])

print('Downloading coffee data files ...')
snapshot_download(repo_id='fangqi/WMPO', repo_type='model',
    local_dir=DATA_DIR, local_dir_use_symlinks=False,
    allow_patterns=[
        'data_files/core_datasets/coffee/**',
        'data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json',
        'data_files/statistics/**',
    ])

print('Download complete!')
PYEOF
"

echo "[1] Download done: $(date)"
fi  # end download skip check

# ── [2] states pkl 생성 ───────────────────────────────────────────────────
STATES_PKL=$SWM/data/states/coffee_d0_states.pkl
echo ""
echo "[2] Generating coffee states pkl..."

if [ -f "$STATES_PKL" ]; then
    echo "[2] States pkl already exists: $STATES_PKL"
else
    singularity exec \
        --bind /scratch/mip25:/scratch/mip25 \
        --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
        "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_PATH
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:/opt/conda/lib:\$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
python3 $SWM/scripts/gen_states.py --task coffee --swm_root $SWM
"
    echo "[2] States pkl done: $(date)"
fi

# ── [3] 평가 실행 ──────────────────────────────────────────────────────────
run_eval() {
    local MODEL_NAME=$1
    local VLA_BASE=$2
    local GPU=$3
    local SAVE_JSON=$SWM/eval_results/coffee_${MODEL_NAME}.json
    mkdir -p $SWM/eval_results

    echo ""
    echo "[eval] $MODEL_NAME on GPU $GPU"

    singularity exec --nv \
        --bind /scratch/mip25:/scratch/mip25 \
        --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
        "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_PATH
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:/opt/conda/lib:\$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot:\$PYTHONPATH

TASK=coffee \
VLA_BASE=$VLA_BASE \
MODEL_NAME=$MODEL_NAME \
N_EPISODES=$N_EPISODES \
GPU=$GPU \
SAVE_JSON=$SAVE_JSON \
python3 $SWM/eval/eval_baseline.py \
    2>&1 | tee $LOGDIR/eval_coffee_${MODEL_NAME}_${SLURM_JOB_ID}.log
"
}

echo ""
echo "[3] Starting coffee evaluations (128 episodes each)..."

run_eval "SFT"       "$CKPTS/SFT_models/coffee"         "0" &
PID_SFT=$!
run_eval "WMPO_P128" "$CKPTS/WMPO_models/coffee/P_128"  "1" &
PID_P128=$!

wait $PID_SFT $PID_P128

run_eval "WMPO_P1280" "$CKPTS/WMPO_models/coffee/P_1280" "0"

# ── 결과 요약 ─────────────────────────────────────────────────────────────
echo ""
echo "====== COFFEE RESULTS (128 episodes) ======"
for MODEL in SFT WMPO_P128 WMPO_P1280; do
    JSON=$SWM/eval_results/coffee_${MODEL}.json
    if [ -f "$JSON" ]; then
        python3 -c "
import json
d = json.load(open('$JSON'))
print(f\"  {d['model_name']:<15} SR = {d['success_rate']:.3f}  ({d['n_success']}/{d['n_episodes']})\")
"
    else
        echo "  $MODEL: result not found"
    fi
done
echo "====== DONE: $(date) ======"
