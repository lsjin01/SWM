#!/bin/bash
# Stage 2.5 v2 완료 후 자동으로 Stage 3 실행
set -e
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2,3,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM

# ── Stage 2.5 v2 완료 대기 ───────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Waiting for Stage 2.5 v2 to finish ..."
until grep -q "Stage 2.5 done" logs/stage2_5_v2_square.log 2>/dev/null; do
    sleep 10
done

# 체크포인트 확인
if [ ! -f outputs/stage2_5_v2/square/best.pt ]; then
    echo "ERROR: Stage 2.5 v2 checkpoint not found!"
    exit 1
fi
echo "[$(date '+%H:%M:%S')] Stage 2.5 v2 done. Starting Stage 3 ..."

# ── Stage 3 (reward_model, n_rollout_chunks=20) ──────────────────────────────
mkdir -p logs
torchrun --nproc_per_node=3 --master_port=29503 \
    scripts/train_stage3.py \
    --config configs/stage3_vs_p128_rm_lora.yaml \
    2>&1 | tee logs/stage3_rm_lora_v2.log

echo "[$(date '+%H:%M:%S')] Stage 3 done."
