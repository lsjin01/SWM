#!/bin/bash
WMPO_PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
WMPO_TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN11_GPU234 $(date '+%H:%M:%S')] $*"; }

log "Waiting for chain9 eval (eval_spatial_pca_multi_goal_dino_attn_best.log) ..."
until grep -q "Success Rate" "$SWM/logs/eval_spatial_pca_multi_goal_dino_attn_best.log" 2>/dev/null; do
    sleep 60
done
log "chain9 done. Starting pca_multi_goal_pca_variance on GPU 2,3,4 ..."

TRAIN_LOG="$SWM/logs/train_spatial_pca_multi_goal_pca_variance_$(date +%Y%m%d_%H%M%S).log"
( cd $SWM && CUDA_VISIBLE_DEVICES=2,3,4 $WMPO_TORCHRUN \
    --nproc_per_node=3 --master_port=29560 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_spatial_pca_multi_goal_pca_variance.yaml" \
    > "$TRAIN_LOG" 2>&1 )
log "Training DONE. Log: $TRAIN_LOG"

OUT="$SWM/outputs/stage3/spatial_pca_multi_goal_pca_variance/square"
CUDA_VISIBLE_DEVICES=2 TASK=square CKPT_PATH="$OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_multi_goal_pca_variance_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON -u "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_multi_goal_pca_variance_best.log" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=3 TASK=square CKPT_PATH="$OUT/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_multi_goal_pca_variance_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON -u "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_multi_goal_pca_variance_last.log" 2>&1 &
P2=$!

wait $P1 $P2
log "Eval DONE."
grep -h "Success Rate" \
    "$SWM/logs/eval_spatial_pca_multi_goal_pca_variance_best.log" \
    "$SWM/logs/eval_spatial_pca_multi_goal_pca_variance_last.log" 2>/dev/null | sed 's/^/  /'
log "chain11 ALL DONE."
