#!/bin/bash
# SWM Evaluation Script — runs vs_p128 and 500iter sequentially on GPU 2
set -e
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

SWM_DIR=/home/dgist_shyo/sjLee/SWM
VLA_BASE=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square
STAGE1=outputs/stage1/multitask_dinosiglip/best.pt
STAGE2=outputs/stage2/multitask_dinosiglip/best.pt

cd $SWM_DIR

echo "====== [1/2] SWM vs_p128 (100 iter, full_finetune) ======"
TASK=square \
CKPT_PATH=outputs/stage3/vs_p128/square/best.pt \
STAGE1_CKPT=$STAGE1 \
STAGE2_CKPT=$STAGE2 \
VLA_BASE=$VLA_BASE \
VLA_DEVICE=cuda \
N_EPISODES=50 \
USE_Z_BYPASS=1 \
SAVE_JSON=eval_results/swm_vs_p128_square_zbp.json \
python eval/eval_swm_mimicgen.py 2>&1 | tee logs/eval_vs_p128_zbp.log

echo ""
echo "====== [2/2] SWM 500iter (projector+q/v only) ======"
TASK=square \
CKPT_PATH=outputs/stage3/square/best.pt \
STAGE1_CKPT=$STAGE1 \
STAGE2_CKPT=$STAGE2 \
VLA_BASE=$VLA_BASE \
VLA_DEVICE=cuda \
N_EPISODES=50 \
USE_Z_BYPASS=1 \
SAVE_JSON=eval_results/swm_500iter_square_zbp.json \
python eval/eval_swm_mimicgen.py 2>&1 | tee logs/eval_500iter_zbp.log

echo ""
echo "====== ALL DONE ======"
