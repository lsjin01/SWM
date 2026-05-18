#!/bin/bash
# Phase 2 Chain 3: spatial_pca_binary_t2
# pca_binary + temperature=2.0 (Phase 1 best combo 재현)
# GPU 0,1 — chain2 eval 완료 후 실행

WMPO_PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
WMPO_TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

log() { echo "[CHAIN3_GPU01 $(date '+%H:%M:%S')] $*"; }

log "Waiting for chain2 eval (spatial_pca_binary_best.log) ..."
until grep -l "Success Rate" "$SWM/logs/eval_spatial_pca_binary_best.log" 2>/dev/null | grep -q .; do
    sleep 60
done
log "chain2 done. Starting spatial_pca_binary_t2 on GPU 0,1 ..."

TRAIN_LOG="$SWM/logs/train_spatial_pca_binary_t2_$(date +%Y%m%d_%H%M%S).log"
( cd $SWM && CUDA_VISIBLE_DEVICES=0,1 $WMPO_TORCHRUN \
    --nproc_per_node=2 --master_port=29552 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_spatial_pca_binary_t2.yaml" \
    > "$TRAIN_LOG" 2>&1 )
log "Training DONE. Log: $TRAIN_LOG"

OUT="$SWM/outputs/stage3/spatial_pca_binary_t2/square"
log "Evaluating best(GPU0) + last(GPU1) ..."

CUDA_VISIBLE_DEVICES=0 TASK=square CKPT_PATH="$OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_binary_t2_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_binary_t2_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 TASK=square CKPT_PATH="$OUT/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_binary_t2_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_binary_t2_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "Eval DONE."
grep -h "Success Rate" \
    "$SWM/logs/eval_spatial_pca_binary_t2_best.log" \
    "$SWM/logs/eval_spatial_pca_binary_t2_last.log" 2>/dev/null | sed 's/^/  /'
log "chain3 ALL DONE."
