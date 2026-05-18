#!/bin/bash
# Stage 3: Learned Reward Model로 GRPO 학습
# GPU 2,3,7 DDP

source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2,3,7
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM

echo "[Stage3-RM] Starting DDP training on GPU 2,3,7 ..."
torchrun --nproc_per_node=3 --master_port=29506 \
    scripts/train_stage3.py \
    --config configs/stage3_vs_p128_rm.yaml \
    --no-wandb \
    2>&1 | tee logs/stage3_rm_vs_p128.log

echo "[Stage3-RM] Done."
