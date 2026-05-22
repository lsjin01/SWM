#!/bin/bash
# Re-eval pairs 1-4 rb3 on GPU6 and GPU7
set -e

SWM=/home/dgist_shyo/sjLee/SWM
PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
EVAL_SCRIPT=${SWM}/eval/eval_swm_mimicgen.py
STAGE1=${SWM}/outputs/stage1/multitask_dinosiglip/best.pt
STAGE2=${SWM}/outputs/stage2/robust_b3/best.pt
VLA_BASE=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square
N=50
TS=$(date +%Y%m%d_%H%M%S)

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

run_eval() {
    local GPU=$1 CKPT=$2 JSON=$3
    CUDA_VISIBLE_DEVICES=${GPU} TASK=square N_EPISODES=${N} \
    CKPT_PATH=${CKPT} STAGE1_CKPT=${STAGE1} STAGE2_CKPT=${STAGE2} \
    VLA_BASE=${VLA_BASE} SAVE_JSON=${JSON} ${PYTHON} ${EVAL_SCRIPT}
}

GPU=${1:-6}
LOG=${SWM}/logs/rb3_reeval_gpu${GPU}_${TS}.log

if [ "${GPU}" = "6" ]; then
    echo "GPU6: spatial_pca_binary_t06/best → spatial_pca_binary_delta/best"
    run_eval 6 ${SWM}/outputs/stage3/robust_b3/spatial_pca_binary_t06/square/best.pt \
               ${SWM}/eval_results/rb3_spatial_pca_binary_t06_best_${TS}.json
    run_eval 6 ${SWM}/outputs/stage3/robust_b3/spatial_pca_binary_delta/square/best.pt \
               ${SWM}/eval_results/rb3_spatial_pca_binary_delta_best_${TS}.json
elif [ "${GPU}" = "7" ]; then
    echo "GPU7: spatial_pca_binary_t06/last → spatial_pca_binary_delta/last"
    run_eval 7 ${SWM}/outputs/stage3/robust_b3/spatial_pca_binary_t06/square/last.pt \
               ${SWM}/eval_results/rb3_spatial_pca_binary_t06_last_${TS}.json
    run_eval 7 ${SWM}/outputs/stage3/robust_b3/spatial_pca_binary_delta/square/last.pt \
               ${SWM}/eval_results/rb3_spatial_pca_binary_delta_last_${TS}.json
fi
