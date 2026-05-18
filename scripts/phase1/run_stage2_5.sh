#!/bin/bash
# Stage 2.5: Goal-conditioned Reward Model 학습
# GPU 2 단일, 빠름 (~5분)

source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM

echo "[Stage2.5] Training goal-conditioned reward model ..."
python scripts/train_stage2_5.py \
    --config configs/stage2_5.yaml \
    2>&1 | tee logs/stage2_5_square.log

echo "[Stage2.5] Done."
