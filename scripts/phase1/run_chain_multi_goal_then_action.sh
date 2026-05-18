#!/bin/bash
# 체인 실행 스크립트:
#   1) multi_goal 학습 완료 대기
#   2) multi_goal best/last eval (GPU 2, GPU 3)
#   3) action_goal 학습 (GPU 2,3,7)
#   4) action_goal best/last eval (GPU 2, GPU 3)

set -e
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

MULTI_GOAL_LOG="$SWM/logs/train_multi_goal_20260517_225946.log"
MULTI_GOAL_OUT="$SWM/outputs/stage3/multi_goal/square"
ACTION_LOG="$SWM/logs/train_action_goal_$(date +%Y%m%d_%H%M%S).log"
ACTION_OUT="$SWM/outputs/stage3/action_goal/square"

log() { echo "[CHAIN $(date '+%H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: multi_goal 학습 완료 대기
# ─────────────────────────────────────────────────────────────────────────────
log "Waiting for multi_goal training to finish ..."
until grep -q "\[Iter 0300\]\|Training complete\|Done" "$MULTI_GOAL_LOG" 2>/dev/null; do
    sleep 60
done
log "multi_goal training DONE."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: multi_goal eval — best(GPU2), last(GPU3) 병렬
# ─────────────────────────────────────────────────────────────────────────────
log "Evaluating multi_goal best on GPU 2 ..."
CUDA_VISIBLE_DEVICES=2 \
TASK=square \
CKPT_PATH="$MULTI_GOAL_OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/multi_goal_best_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" \
    > "$SWM/logs/eval_multi_goal_best.log" 2>&1 &
PID_BEST=$!

log "Evaluating multi_goal last on GPU 3 ..."
CUDA_VISIBLE_DEVICES=3 \
TASK=square \
CKPT_PATH="$MULTI_GOAL_OUT/last.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/multi_goal_last_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" \
    > "$SWM/logs/eval_multi_goal_last.log" 2>&1 &
PID_LAST=$!

wait $PID_BEST $PID_LAST
log "multi_goal eval DONE."
log "=== multi_goal results ==="
grep -h "success_rate\|SR\|n_success" \
    "$SWM/logs/eval_multi_goal_best.log" \
    "$SWM/logs/eval_multi_goal_last.log" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: action_goal 학습 (GPU 2,3,7)
# ─────────────────────────────────────────────────────────────────────────────
log "Starting action_goal training on GPU 2,3,7 ..."
CUDA_VISIBLE_DEVICES=2,3,7 \
torchrun --nproc_per_node=3 --master_port=29508 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_action_goal.yaml" \
    > "$ACTION_LOG" 2>&1

log "action_goal training DONE."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: action_goal eval — best(GPU2), last(GPU3) 병렬
# ─────────────────────────────────────────────────────────────────────────────
log "Evaluating action_goal best on GPU 2 ..."
CUDA_VISIBLE_DEVICES=2 \
TASK=square \
CKPT_PATH="$ACTION_OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/action_goal_best_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" \
    > "$SWM/logs/eval_action_goal_best.log" 2>&1 &
PID_BEST=$!

log "Evaluating action_goal last on GPU 3 ..."
CUDA_VISIBLE_DEVICES=3 \
TASK=square \
CKPT_PATH="$ACTION_OUT/last.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/action_goal_last_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" \
    > "$SWM/logs/eval_action_goal_last.log" 2>&1 &
PID_LAST=$!

wait $PID_BEST $PID_LAST
log "action_goal eval DONE."
log "=== action_goal results ==="
grep -h "success_rate\|SR\|n_success" \
    "$SWM/logs/eval_action_goal_best.log" \
    "$SWM/logs/eval_action_goal_last.log" 2>/dev/null || true

log "ALL DONE. Check eval_results/ for json files."
