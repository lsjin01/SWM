#!/bin/bash
# Phase 2 Chain 5: delta magnitude patch weights 실험
# chain3 eval 완료 후 실행 (GPU 0,1)
# pca_binary + delta weights / pca_multi_goal + delta weights

WMPO_PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
WMPO_TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

log() { echo "[CHAIN5_GPU01 $(date '+%H:%M:%S')] $*"; }

log "Waiting for chain3 eval (spatial_pca_binary_t2_best.log) ..."
until grep -l "Success Rate" "$SWM/logs/eval_spatial_pca_binary_t2_best.log" 2>/dev/null | grep -q .; do
    sleep 60
done
log "chain3 done. Starting delta weights experiments ..."

# ── 5a: pca_binary + delta weights ───────────────────────────────────────────
TRAIN_LOG_A="$SWM/logs/train_spatial_pca_binary_delta_$(date +%Y%m%d_%H%M%S).log"
( cd $SWM && CUDA_VISIBLE_DEVICES=0,1 $WMPO_TORCHRUN \
    --nproc_per_node=2 --master_port=29556 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_spatial_pca_binary_delta.yaml" \
    > "$TRAIN_LOG_A" 2>&1 )
log "pca_binary_delta training DONE."

OUT_A="$SWM/outputs/stage3/spatial_pca_binary_delta/square"
CUDA_VISIBLE_DEVICES=0 TASK=square CKPT_PATH="$OUT_A/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_binary_delta_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_binary_delta_best.log" 2>&1 &
PID_A1=$!

CUDA_VISIBLE_DEVICES=1 TASK=square CKPT_PATH="$OUT_A/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_binary_delta_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_binary_delta_last.log" 2>&1 &
PID_A2=$!

wait $PID_A1 $PID_A2
log "pca_binary_delta eval DONE."
grep -h "Success Rate" \
    "$SWM/logs/eval_spatial_pca_binary_delta_best.log" \
    "$SWM/logs/eval_spatial_pca_binary_delta_last.log" 2>/dev/null | sed 's/^/  /'

# ── 5b: pca_multi_goal + delta weights ───────────────────────────────────────
TRAIN_LOG_B="$SWM/logs/train_spatial_pca_multi_goal_delta_$(date +%Y%m%d_%H%M%S).log"
( cd $SWM && CUDA_VISIBLE_DEVICES=0,1 $WMPO_TORCHRUN \
    --nproc_per_node=2 --master_port=29557 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_spatial_pca_multi_goal_delta.yaml" \
    > "$TRAIN_LOG_B" 2>&1 )
log "pca_multi_goal_delta training DONE."

OUT_B="$SWM/outputs/stage3/spatial_pca_multi_goal_delta/square"
CUDA_VISIBLE_DEVICES=0 TASK=square CKPT_PATH="$OUT_B/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_multi_goal_delta_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_multi_goal_delta_best.log" 2>&1 &
PID_B1=$!

CUDA_VISIBLE_DEVICES=1 TASK=square CKPT_PATH="$OUT_B/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/spatial_pca_multi_goal_delta_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" \
> "$SWM/logs/eval_spatial_pca_multi_goal_delta_last.log" 2>&1 &
PID_B2=$!

wait $PID_B1 $PID_B2
log "pca_multi_goal_delta eval DONE."
grep -h "Success Rate" \
    "$SWM/logs/eval_spatial_pca_multi_goal_delta_best.log" \
    "$SWM/logs/eval_spatial_pca_multi_goal_delta_last.log" 2>/dev/null | sed 's/^/  /'

log "chain5 ALL DONE."
