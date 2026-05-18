#!/bin/bash
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN6 $(date '+%H:%M:%S')] $*"; }

# Step 1: action_goal eval 완료 대기 (chain5)
log "Waiting for action_goal eval (chain5) ..."
until ls "$SWM/logs/eval_action_goal_v2_best.log" 2>/dev/null | xargs grep -l "Success Rate" 2>/dev/null | grep -q .; do
    sleep 60
done
log "chain5 DONE."

# Step 2: pca_delta 학습 (GPU 2,3,7)
DELTA_LOG="$SWM/logs/train_pca_delta_$(date +%Y%m%d_%H%M%S).log"
log "Starting pca_delta training (GPU 2,3,7) ..."
CUDA_VISIBLE_DEVICES=2,3,7 \
torchrun --nproc_per_node=3 --master_port=29511 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_pca_delta.yaml" \
    > "$DELTA_LOG" 2>&1
log "pca_delta training DONE."

# Step 3: pca_delta eval (best GPU2, last GPU3) 병렬
DELTA_OUT="$SWM/outputs/stage3/pca_delta/square"
log "Evaluating pca_delta best(GPU2) + last(GPU3) ..."
CUDA_VISIBLE_DEVICES=2 TASK=square CKPT_PATH="$DELTA_OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 USE_Z_BYPASS=0 \
SAVE_JSON="$SWM/eval_results/pca_delta_best_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_pca_delta_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=3 TASK=square CKPT_PATH="$DELTA_OUT/last.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 USE_Z_BYPASS=0 \
SAVE_JSON="$SWM/eval_results/pca_delta_last_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_pca_delta_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "pca_delta eval DONE."
grep -h "Success Rate" "$SWM/logs/eval_pca_delta_best.log" "$SWM/logs/eval_pca_delta_last.log" 2>/dev/null | sed 's/^/  /'
log "ALL DONE."
