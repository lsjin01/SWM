#!/bin/bash
# SWM 평가 실행 스크립트
# 사용: bash eval/run_eval.sh [stage3_ckpt] [task] [n_episodes]
# 예시: bash eval/run_eval.sh outputs/stage3/square/best.pt square 50

set -e

JEPA=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA
WMPO=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO
SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

STAGE3_CKPT="${1:-outputs/stage3/square/best.pt}"
TASK="${2:-square}"
N_EPISODES="${3:-50}"

STAGE1_CKPT="${SWM}/outputs/stage1/${TASK}/best.pt"
STAGE2_CKPT="${SWM}/outputs/stage2/${TASK}/best.pt"
VLA_BASE="${WMPO}/checkpoint_files/SFT_models/${TASK}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_JSON="${SWM}/eval_results/swm_${TASK}_${TIMESTAMP}.json"
mkdir -p "${SWM}/eval_results"

echo "================================================================"
echo "  SWM Evaluation"
echo "  Task:       ${TASK}"
echo "  Stage3:     ${STAGE3_CKPT}"
echo "  Stage1:     ${STAGE1_CKPT}"
echo "  Stage2:     ${STAGE2_CKPT}"
echo "  VLA Base:   ${VLA_BASE}"
echo "  N episodes: ${N_EPISODES}"
echo "  Save:       ${SAVE_JSON}"
echo "================================================================"

CUDA_VISIBLE_DEVICES=0 \
TASK="${TASK}" \
CKPT_PATH="${STAGE3_CKPT}" \
STAGE1_CKPT="${STAGE1_CKPT}" \
STAGE2_CKPT="${STAGE2_CKPT}" \
VLA_BASE="${VLA_BASE}" \
VLA_DEVICE=cuda \
N_EPISODES="${N_EPISODES}" \
SAVE_JSON="${SAVE_JSON}" \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
python "${SWM}/eval/eval_swm_mimicgen.py"

echo ""
echo "--- Comparison with baselines ---"
python "${SWM}/eval/compare_results.py" \
    --task "${TASK}" \
    --files "${SAVE_JSON}"
