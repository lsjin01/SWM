#!/bin/bash
WMPO_PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
WMPO_TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
SWM=/home/dgist_shyo/sjLee/SWM
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
log() { echo "[CHAIN14_GPU456 $(date '+%H:%M:%S')] $*"; }

log "Waiting for eval_act_pool_binary_best.log (GPU 4,5,6 chain) ..."
until grep -l "Success Rate" "$SWM/logs/eval_act_pool_binary_best.log" 2>/dev/null | grep -q .; do
    sleep 60
done
log "Dependency DONE. Starting slot_attn_binary training on GPU 4,5,6 ..."

TRAIN_LOG="$SWM/logs/train_slot_attn_binary_$(date +%Y%m%d_%H%M%S).log"
cd $SWM && CUDA_VISIBLE_DEVICES=4,5,6 $WMPO_TORCHRUN --nproc_per_node=3 --master_port=29533 \
    "$SWM/scripts/train_stage3.py" \
    --config "$SWM/configs/stage3_slot_attn_binary.yaml" \
    > "$TRAIN_LOG" 2>&1
log "slot_attn_binary training DONE."

OUT="$SWM/outputs/stage3/slot_attn_binary/square"
log "Evaluating slot_attn_binary best(GPU4) + last(GPU5) ..."
CUDA_VISIBLE_DEVICES=4 TASK=square CKPT_PATH="$OUT/best.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/slot_attn_binary_best_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_slot_attn_binary_best.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=5 TASK=square CKPT_PATH="$OUT/ckpt_iter0200.pt" \
STAGE1_CKPT="$SWM/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="$SWM/outputs/stage2/multitask_dinosiglip/best.pt" \
VLA_BASE="/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square" \
VLA_DEVICE=cuda N_EPISODES=50 \
SAVE_JSON="$SWM/eval_results/slot_attn_binary_last_$(date +%Y%m%d_%H%M%S).json" \
$WMPO_PYTHON "$SWM/eval/eval_swm_mimicgen.py" > "$SWM/logs/eval_slot_attn_binary_last.log" 2>&1 &
PID2=$!

wait $PID1 $PID2
log "slot_attn_binary eval DONE."
grep -h "Success Rate" "$SWM/logs/eval_slot_attn_binary_best.log" "$SWM/logs/eval_slot_attn_binary_last.log" 2>/dev/null | sed 's/^/  /'
log "chain14 ALL DONE."
