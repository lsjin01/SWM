#!/bin/bash
# Stage 3 재학습 — spatial patch embeddings (mean pool 제거)
# GPU 0,1 DDP, vs_p128 config

source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2,3,7
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM

echo "[Stage3-Spatial] Starting DDP training on GPU 0,1 ..."
torchrun --nproc_per_node=3 --master_port=29504 \
    scripts/train_stage3.py \
    --config configs/stage3_vs_p128.yaml \
    --no-wandb \
    2>&1 | tee logs/stage3_spatial_vs_p128.log

echo "[Stage3-Spatial] Done."
