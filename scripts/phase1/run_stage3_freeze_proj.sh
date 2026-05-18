#!/bin/bash
# Stage 3 — projector frozen, q/v LoRA only
set -e
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM

mkdir -p logs
torchrun --nproc_per_node=3 --master_port=29504 \
    scripts/train_stage3.py \
    --config configs/stage3_freeze_proj.yaml \
    2>&1 | tee logs/stage3_freeze_proj.log

echo "[$(date '+%H:%M:%S')] Stage 3 (freeze_proj) done."
