#!/bin/bash
# 공정 비교 평가 스크립트
# SWM / WMPO-JEPA / WMPO 모델을 동일 조건(N_EPISODES, states, crop)으로 평가
#
# 사용법:
#   bash eval/run_comparison_eval.sh [GPU] [TASK] [N_EPISODES]
#
# 예시 (SWM best 평가):
#   bash eval/run_comparison_eval.sh 2 square 50
#
# 예시 (특정 ckpt 비교):
#   SWM_CKPT=outputs/stage3/spatial_continuous/square/best.pt \
#   JEPA_CKPT=/path/to/grpo_optionA/best.pt \
#   bash eval/run_comparison_eval.sh 2 square 50

set -e

GPU="${1:-2}"
TASK="${2:-square}"
N_EPISODES="${3:-50}"

PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WMPO=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO
JEPA=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA
TS=$(date +%Y%m%d_%H%M%S)

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

# ── 공통 eval 조건 ──────────────────────────────────────────────────
STATES_PKL="${JEPA}/data/states/${TASK}_d0_states.pkl"
UNNORM_KEY="${TASK}_d0_300_demos"
CENTER_CROP=1
ACTION_CHUNK_LEN=8
NUM_STEPS_WAIT=10

echo "================================================================"
echo "  Fair Comparison Eval"
echo "  Task:        ${TASK}"
echo "  N_EPISODES:  ${N_EPISODES}  (states[0:${N_EPISODES}])"
echo "  States:      ${STATES_PKL}"
echo "  GPU:         ${GPU}"
echo "================================================================"

mkdir -p "${SWM}/eval_results"

# ── 1. SFT baseline ─────────────────────────────────────────────────
SFT_CKPT="${WMPO}/checkpoint_files/SFT_models/${TASK}"
SFT_JSON="${SWM}/eval_results/fair_sft_${TASK}_${TS}.json"
echo ""
echo "[1/N] SFT baseline (${SFT_CKPT})"
CUDA_VISIBLE_DEVICES=${GPU} \
TASK=${TASK} CKPT_PATH=${SFT_CKPT} \
STAGE1_CKPT="${SWM}/outputs/stage1/multitask_dinosiglip/best.pt" \
STAGE2_CKPT="${SWM}/outputs/stage2/spatial_multitask/best.pt" \
VLA_BASE=${SFT_CKPT} \
N_EPISODES=${N_EPISODES} SAVE_JSON=${SFT_JSON} \
  ${PYTHON} "${SWM}/eval/eval_swm_mimicgen.py"
echo "  → ${SFT_JSON}"

# ── 2. SWM ckpt (환경변수로 지정, 없으면 Phase1 best) ───────────────
SWM_CKPT="${SWM_CKPT:-${SWM}/outputs/stage3/scalar_pca_binary/square/best.pt}"
SWM_S2_CKPT="${SWM_S2_CKPT:-${SWM}/outputs/stage2/spatial_multitask/best.pt}"
SWM_JSON="${SWM}/eval_results/fair_swm_${TASK}_${TS}.json"
if [ -f "${SWM_CKPT}" ]; then
    echo ""
    echo "[2/N] SWM (${SWM_CKPT})"
    CUDA_VISIBLE_DEVICES=${GPU} \
    TASK=${TASK} CKPT_PATH=${SWM_CKPT} \
    STAGE1_CKPT="${SWM}/outputs/stage1/multitask_dinosiglip/best.pt" \
    STAGE2_CKPT=${SWM_S2_CKPT} \
    VLA_BASE="${WMPO}/checkpoint_files/SFT_models/${TASK}" \
    N_EPISODES=${N_EPISODES} SAVE_JSON=${SWM_JSON} \
      ${PYTHON} "${SWM}/eval/eval_swm_mimicgen.py"
    echo "  → ${SWM_JSON}"
else
    echo "[2/N] SWM ckpt not found, skip: ${SWM_CKPT}"
fi

# ── 3. WMPO-JEPA GRPO ckpt (환경변수로 지정) ────────────────────────
JEPA_CKPT="${JEPA_CKPT:-${JEPA}/outputs/grpo_optionA/square/latest/best.pt}"
JEPA_JSON="${SWM}/eval_results/fair_jepa_${TASK}_${TS}.json"
if [ -f "${JEPA_CKPT}" ]; then
    echo ""
    echo "[3/N] WMPO-JEPA GRPO (${JEPA_CKPT})"
    CUDA_VISIBLE_DEVICES=${GPU} \
    TASK=${TASK} \
    CKPT_PATH=${JEPA_CKPT} \
    VLA_BASE="${WMPO}/checkpoint_files/SFT_models/${TASK}" \
    STATES_PKL=${STATES_PKL} \
    UNNORM_KEY=${UNNORM_KEY} \
    N_EPISODES=${N_EPISODES} \
    CENTER_CROP=${CENTER_CROP} \
    ACTION_CHUNK_LEN=${ACTION_CHUNK_LEN} \
    NUM_STEPS_WAIT=${NUM_STEPS_WAIT} \
    SAVE_JSON=${JEPA_JSON} \
      ${PYTHON} "${JEPA}/eval/eval_wmpo_mimicgen.py"
    echo "  → ${JEPA_JSON}"
else
    echo "[3/N] WMPO-JEPA ckpt not found, skip: ${JEPA_CKPT}"
fi

# ── 4. WMPO P_1280 baseline ─────────────────────────────────────────
WMPO_CKPT="${WMPO}/checkpoint_files/WMPO_models/${TASK}/P_1280"
WMPO_JSON="${SWM}/eval_results/fair_wmpo_p1280_${TASK}_${TS}.json"
if [ -d "${WMPO_CKPT}" ]; then
    echo ""
    echo "[4/N] WMPO P_1280 (${WMPO_CKPT})"
    CUDA_VISIBLE_DEVICES=${GPU} \
    TASK=${TASK} \
    CKPT_PATH=${WMPO_CKPT} \
    VLA_BASE=${WMPO_CKPT} \
    STATES_PKL=${STATES_PKL} \
    UNNORM_KEY=${UNNORM_KEY} \
    N_EPISODES=${N_EPISODES} \
    CENTER_CROP=${CENTER_CROP} \
    ACTION_CHUNK_LEN=${ACTION_CHUNK_LEN} \
    NUM_STEPS_WAIT=${NUM_STEPS_WAIT} \
    SAVE_JSON=${WMPO_JSON} \
      ${PYTHON} "${JEPA}/eval/eval_wmpo_mimicgen.py"
    echo "  → ${WMPO_JSON}"
else
    echo "[4/N] WMPO P_1280 not found, skip: ${WMPO_CKPT}"
fi

# ── 결과 집계 ────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Results Summary (N_EPISODES=${N_EPISODES}, same states[0:${N_EPISODES}])"
echo "================================================================"
for f in ${SFT_JSON} ${SWM_JSON} ${JEPA_JSON} ${WMPO_JSON}; do
    [ -f "$f" ] || continue
    name=$(basename $f .json | sed "s/_${TS}//")
    ${PYTHON} -c "
import json
d = json.load(open('$f'))
sr = d.get('success_rate', '?')
n  = d.get('n_success', '?')
ep = d.get('n_episodes', '?')
print(f'  {sr:.2%}  ({n}/{ep})  $name')
"
done
echo "================================================================"
