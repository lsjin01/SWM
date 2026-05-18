#!/bin/bash
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN3 $(date '+%H:%M:%S')] $*"; }

# Step 1: pca_goal (n_goals=4) 학습
PCA_LOG="$SWM/logs/train_pca_goal_$(date +%Y%m%d_%H%M%S).log"
log "Starting pca_goal (n_goals=4) training (GPU 2,3,7) ..."
CUDA_VISIBLE_DEVICES=2,3,7 \
torchrun --nproc_per_node=3 --master_port=29509 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_pca_goal.yaml" \
    > "$PCA_LOG" 2>&1
log "pca_goal training DONE."

# Step 2: pca_goal eval (best GPU2, last GPU3) 병렬
PCA_OUT="$SWM/outputs/stage3/pca_goal/square"
log "Evaluating pca_goal best(GPU2) + last(GPU3) ..."
CUDA_VISIBLE_DEVICES=2 TASK=square CKPT_PATH="$PCA_OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/pca_goal_best_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_pca_goal_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=3 TASK=square CKPT_PATH="$PCA_OUT/last.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/pca_goal_last_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_pca_goal_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "pca_goal eval DONE."
grep -h "Success Rate" "$SWM/logs/eval_pca_goal_best.log" "$SWM/logs/eval_pca_goal_last.log" 2>/dev/null | sed 's/^/  /'
log "DONE. chain4 will handle pca_goal_single."
