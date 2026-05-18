#!/bin/bash
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN $(date '+%H:%M:%S')] $*"; }

# Step 1: multi_goal eval 완료 대기
log "Waiting for multi_goal eval PIDs 1920078 1920079 ..."
wait 1920078 2>/dev/null; wait 1920079 2>/dev/null
log "multi_goal eval DONE."

# 결과 출력
for f in best last; do
    SR=$(grep -o "success_rate.*\|SR.*\|n_success.*" "$SWM/logs/eval_multi_goal_${f}.log" 2>/dev/null | tail -3 || true)
    log "multi_goal $f: $SR"
done

# Step 2: action_goal 학습
log "Starting action_goal training (GPU 2,3,7) ..."
CUDA_VISIBLE_DEVICES=2,3,7 \
torchrun --nproc_per_node=3 --master_port=29508 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_action_goal.yaml" \
    > "$SWM/logs/train_action_goal_$(date +%Y%m%d_%H%M%S).log" 2>&1
log "action_goal training DONE."

ACTION_OUT="$SWM/outputs/stage3/action_goal/square"

# Step 3: action_goal eval 병렬
log "Evaluating action_goal best (GPU2) and last (GPU3) ..."
CUDA_VISIBLE_DEVICES=2 TASK=square \
CKPT_PATH="$ACTION_OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/action_goal_best_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_action_goal_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=3 TASK=square \
CKPT_PATH="$ACTION_OUT/last.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/action_goal_last_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_action_goal_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "action_goal eval DONE. ALL DONE."
for f in best last; do
    SR=$(grep -o "success_rate.*\|SR.*\|n_success.*" "$SWM/logs/eval_action_goal_${f}.log" 2>/dev/null | tail -3 || true)
    log "action_goal $f: $SR"
done
