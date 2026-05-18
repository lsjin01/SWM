#!/bin/bash
# Phase 2 Chain B-2: spatial_pca_binary + attn patch weights
# Chain B (GPU 5,6,7): 2 → 4 → 6
# waits for chain2 eval (eval_spatial_pca_binary_best.log)

WMPO_PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
WMPO_TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

log() { echo "[CHAIN4_GPU567 $(date '+%H:%M:%S')] $*"; }

log "Waiting for chain2 eval (eval_spatial_pca_binary_best.log) ..."
until grep -l "Success Rate" "$SWM/logs/eval_spatial_pca_binary_best.log" 2>/dev/null | grep -q .; do
    sleep 60
done
log "chain2 done. Starting spatial_pca_binary_attn on GPU 5,6,7 ..."

TRAIN_LOG="$SWM/logs/train_spatial_pca_binary_attn_$(date +%Y%m%d_%H%M%S).log"
( cd $SWM && CUDA_VISIBLE_DEVICES=5,6,7 $WMPO_TORCHRUN \
    --nproc_per_node=3 --master_port=29553 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_spatial_pca_binary_attn.yaml" \
    > "$TRAIN_LOG" 2>&1 )
log "Training DONE. Log: $TRAIN_LOG"

OUT="$SWM/outputs/stage3/spatial_pca_binary_attn/square"
log "Evaluating best(GPU5) + last(GPU6) ..."

CUDA_VISIBLE_DEVICES=5 TASK=square CKPT_PATH="$OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_binary_attn_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_binary_attn_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=6 TASK=square CKPT_PATH="$OUT/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_binary_attn_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_binary_attn_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "Eval DONE."
grep -h "Success Rate" \
    "$SWM/logs/eval_spatial_pca_binary_attn_best.log" \
    "$SWM/logs/eval_spatial_pca_binary_attn_last.log" 2>/dev/null | sed 's/^/  /'
log "chain4 ALL DONE."
