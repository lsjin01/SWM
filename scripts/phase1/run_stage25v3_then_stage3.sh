#!/bin/bash
set -e
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM
mkdir -p logs

echo "[$(date '+%H:%M:%S')] Starting Stage 2.5 v3 (temporal reward model) ..."
python scripts/train_stage2_5.py --config configs/stage2_5_v3.yaml \
  2>&1 | tee logs/stage2_5_v3_square.log

if [ ! -f outputs/stage2_5_v3/square/best.pt ]; then
    echo "ERROR: Stage 2.5 v3 checkpoint not found!"
    exit 1
fi
echo "[$(date '+%H:%M:%S')] Stage 2.5 v3 done. Starting Stage 3 ..."

export CUDA_VISIBLE_DEVICES=5,6,7
torchrun --nproc_per_node=3 --master_port=29505 \
    scripts/train_stage3.py \
    --config configs/stage3_temporal_rm.yaml \
    2>&1 | tee logs/stage3_temporal_rm.log

echo "[$(date '+%H:%M:%S')] Stage 3 (temporal_rm) done."
