#!/bin/bash
# Stage 3: Transition-L2 + LoRA + KL penalty
# GPU 2,3,7 DDP

source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2,3,7
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM

echo "[Stage3-TL2-LoRA] Starting DDP training on GPU 2,3,7 ..."
torchrun --nproc_per_node=3 --master_port=29508 \
    scripts/train_stage3.py \
    --config configs/stage3_vs_p128_tl2_lora.yaml \
    --no-wandb \
    2>&1 | tee logs/stage3_tl2_lora.log

echo "[Stage3-TL2-LoRA] Done."
