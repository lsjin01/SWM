#!/bin/bash
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN4 $(date '+%H:%M:%S')] $*"; }

PCA_MULTI_LOG_PATTERN="$SWM/logs/train_pca_goal_*.log"

# Step 1: pca_goal(multi, n_goals=4) 학습 완료 대기
log "Waiting for pca_goal (n_goals=4) training ..."
until ls $PCA_MULTI_LOG_PATTERN 2>/dev/null | xargs grep -l "\[Iter 0200\]" 2>/dev/null | grep -q .; do
    sleep 60
done
log "pca_goal (n_goals=4) training DONE."

# Step 2: pca_goal_single 학습 (GPU 2,3,7)
SINGLE_LOG="$SWM/logs/train_pca_goal_single_$(date +%Y%m%d_%H%M%S).log"
log "Starting pca_goal_single (n_goals=1) training (GPU 2,3,7) ..."
CUDA_VISIBLE_DEVICES=2,3,7 \
torchrun --nproc_per_node=3 --master_port=29510 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_pca_goal_single.yaml" \
    > "$SINGLE_LOG" 2>&1
log "pca_goal_single training DONE."

# Step 3: pca_goal_single eval (best GPU2, last GPU3) 병렬
SINGLE_OUT="$SWM/outputs/stage3/pca_goal_single/square"
log "Evaluating pca_goal_single best(GPU2) + last(GPU3) ..."
CUDA_VISIBLE_DEVICES=2 TASK=square CKPT_PATH="$SINGLE_OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/pca_goal_single_best_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_pca_goal_single_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=3 TASK=square CKPT_PATH="$SINGLE_OUT/last.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/pca_goal_single_last_$(date +%Y%m%d_%H%M%S).json" \
python "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_pca_goal_single_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "pca_goal_single eval DONE."
grep -h "Success Rate" \
    "$SWM/logs/eval_pca_goal_single_best.log" \
    "$SWM/logs/eval_pca_goal_single_last.log" 2>/dev/null | sed 's/^/  /'
log "ALL DONE."
