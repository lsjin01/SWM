#!/bin/bash
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN5 $(date '+%H:%M:%S')] $*"; }

# Step 1: pca_goal_single eval 완료 대기 (chain4)
log "Waiting for pca_goal_single eval (chain4) ..."
until ls "$SWM/logs/eval_pca_goal_single_best.log" 2>/dev/null | xargs grep -l "Success Rate" 2>/dev/null | grep -q .; do
    sleep 60
done
log "chain4 DONE."

# Step 2: action_goal 학습 (GPU 2,3,7) — 50분 버전
ACTION_LOG="$SWM/logs/train_action_goal_v2_$(date +%Y%m%d_%H%M%S).log"
log "Starting action_goal (n_states=4, iter=200) training (GPU 2,3,7) ..."
CUDA_VISIBLE_DEVICES=2,3,7 \
torchrun --nproc_per_node=3 --master_port=29508 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_action_goal.yaml" \
    > "$ACTION_LOG" 2>&1
log "action_goal training DONE."

# Step 3: action_goal eval 병렬
ACTION_OUT="$SWM/outputs/stage3/action_goal/square"
log "Evaluating action_goal best(GPU2) + last(GPU3) ..."
CUDA_VISIBLE_DEVICES=2 TASK=square CKPT_PATH="$ACTION_OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 USE_Z_BYPASS=0 \
SAVE_JSON="$SWM/eval_results/action_goal_v2_best_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_action_goal_v2_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=3 TASK=square CKPT_PATH="$ACTION_OUT/last.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 USE_Z_BYPASS=0 \
SAVE_JSON="$SWM/eval_results/action_goal_v2_last_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_action_goal_v2_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "action_goal eval DONE."
grep -h "Success Rate" "$SWM/logs/eval_action_goal_v2_best.log" "$SWM/logs/eval_action_goal_v2_last.log" 2>/dev/null | sed 's/^/  /'
log "ALL DONE."
