#!/bin/bash
# Stage 2 재학습 (normalization fix 반영)
# GPU 2,3 DDP

source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2,3
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM

echo "[Stage2] Retraining with fixed normalization ..."
torchrun --nproc_per_node=2 --master_port=29511 \
    scripts/train_stage2.py \
    --config configs/stage2_multitask.yaml \
    2>&1 | tee logs/stage2_multitask_retrain.log

echo "[Stage2] Done."
