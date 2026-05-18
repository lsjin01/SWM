#!/bin/bash
# Stage 3 재학습 — Latent Reward (graph reward → cosine_sim(z_T, z_goal))
# GPU 2,3,7 DDP, vs_p128_latent config

source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2,3,7
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM

echo "[Stage3-Latent] Starting DDP training on GPU 2,3,7 ..."
torchrun --nproc_per_node=3 --master_port=29505 \
    scripts/train_stage3.py \
    --config configs/stage3_vs_p128_latent.yaml \
    --no-wandb \
    2>&1 | tee logs/stage3_latent_vs_p128.log

echo "[Stage3-Latent] Done."
