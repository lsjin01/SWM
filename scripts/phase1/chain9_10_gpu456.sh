#!/bin/bash
WMPO_PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
WMPO_TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN9_10_GPU456 $(date '+%H:%M:%S')] $*"; }

# ── chain9: pca_max (즉시 시작) ───────────────────────────────────────────────
log "Starting pca_max training on GPU 4,5,6 ..."

TRAIN_LOG="$SWM/logs/train_pca_max_$(date +%Y%m%d_%H%M%S).log"
cd $SWM && CUDA_VISIBLE_DEVICES=4,5,6 $WMPO_TORCHRUN --nproc_per_node=3 --master_port=29524 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_pca_max.yaml" \
    > "$TRAIN_LOG" 2>&1
log "pca_max training DONE."

OUT="$SWM/outputs/stage3/pca_max/square"
log "Evaluating pca_max best(GPU4) + last(GPU5) ..."
CUDA_VISIBLE_DEVICES=4 TASK=square CKPT_PATH="$OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/pca_max_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_pca_max_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=5 TASK=square CKPT_PATH="$OUT/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/pca_max_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_pca_max_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "pca_max eval DONE."
grep -h "Success Rate" "$SWM/logs/eval_pca_max_best.log" "$SWM/logs/eval_pca_max_last.log" 2>/dev/null | sed 's/^/  /'

# ── chain10: action_pca_delta ─────────────────────────────────────────────────
log "Starting action_pca_delta training on GPU 4,5,6 ..."

TRAIN_LOG="$SWM/logs/train_action_pca_delta_$(date +%Y%m%d_%H%M%S).log"
cd $SWM && CUDA_VISIBLE_DEVICES=4,5,6 $WMPO_TORCHRUN --nproc_per_node=3 --master_port=29525 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_action_pca_delta.yaml" \
    > "$TRAIN_LOG" 2>&1
log "action_pca_delta training DONE."

OUT="$SWM/outputs/stage3/action_pca_delta/square"
log "Evaluating action_pca_delta best(GPU4) + last(GPU5) ..."
CUDA_VISIBLE_DEVICES=4 TASK=square CKPT_PATH="$OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/action_pca_delta_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_action_pca_delta_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=5 TASK=square CKPT_PATH="$OUT/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/action_pca_delta_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_action_pca_delta_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "action_pca_delta eval DONE."
grep -h "Success Rate" "$SWM/logs/eval_action_pca_delta_best.log" "$SWM/logs/eval_action_pca_delta_last.log" 2>/dev/null | sed 's/^/  /'
log "chain9_10 GPU456 ALL DONE."
