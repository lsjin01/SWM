#!/bin/bash
WMPO_PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
WMPO_TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN8 $(date '+%H:%M:%S')] $*"; }

log "Waiting for eval_pca_binary_best.log ..."
until grep -l "Success Rate" "$SWM/logs/eval_pca_binary_best.log" 2>/dev/null | grep -q .; do
    sleep 60
done
log "Dependency DONE. Starting diversity training ..."

TRAIN_LOG="$SWM/logs/train_diversity_$(date +%Y%m%d_%H%M%S).log"
cd $SWM && CUDA_VISIBLE_DEVICES=2,3,7 $WMPO_TORCHRUN --nproc_per_node=3 --master_port=29513 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_diversity.yaml" \
    > "$TRAIN_LOG" 2>&1
log "diversity training DONE."

OUT="$SWM/outputs/stage3/diversity/square"
log "Evaluating diversity best(GPU2) + last(GPU3) ..."
CUDA_VISIBLE_DEVICES=2 TASK=square CKPT_PATH="$OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/diversity_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_diversity_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=3 TASK=square CKPT_PATH="$OUT/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/diversity_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_diversity_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "diversity eval DONE."
grep -h "Success Rate" "$SWM/logs/eval_diversity_best.log" "$SWM/logs/eval_diversity_last.log" 2>/dev/null | sed 's/^/  /'
log "chain8 ALL DONE."
