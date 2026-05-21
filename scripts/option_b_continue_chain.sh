#!/bin/bash
# Option B: 기존 chain 이어서 실행 (fix 효과 없음 → 원래 하던대로)
# - experiment 7 (multi_goal): 학습 완료됐으나 eval 스킵됨 → eval만
# - experiments 8-19: 학습+eval 전체
# 사용법: bash scripts/option_b_continue_chain.sh [GPU] [N_GPUS]

set -e

GPUS="${1:-4,5,6,7}"
N_GPUS="${2:-4}"
SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
EVAL_SCRIPT="${SWM}/eval/eval_swm_mimicgen.py"
VLA_BASE=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square
STAGE1_CKPT="${SWM}/outputs/stage1/multitask_dinosiglip/best.pt"
STAGE2_CKPT="${SWM}/outputs/stage2/robust_b/best.pt"
N_EPISODES=50

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

mkdir -p "${SWM}/logs" "${SWM}/eval_results"

TS=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo "  Option B: 기존 chain 이어서 실행 (experiments 7 eval + 8-19)"
echo "  GPUs: ${GPUS} (N=${N_GPUS})   N_EPISODES=${N_EPISODES}"
echo "================================================================"

RESULTS=()

# ── Experiment 7: multi_goal (학습 완료, eval만 실행) ─────────────────────────
EXP="stage3_spatial_pca_multi_goal"
OUT_DIR="outputs/stage3/robust_b/spatial_pca_multi_goal/square"
echo ""
echo "[7/19] EVAL ONLY: ${EXP}  (학습 완료됨)"
for CKPT_LABEL in best last; do
    CKPT_PATH="${SWM}/${OUT_DIR}/${CKPT_LABEL}.pt"
    JSON_OUT="${SWM}/eval_results/rb_${EXP}_${CKPT_LABEL}_${TS}.json"
    if [ -f "${CKPT_PATH}" ]; then
        echo "  [eval/${CKPT_LABEL}] ${CKPT_PATH}"
        CUDA_VISIBLE_DEVICES=${GPUS%,*} \
        TASK=square CKPT_PATH=${CKPT_PATH} STAGE1_CKPT=${STAGE1_CKPT} \
        STAGE2_CKPT=${STAGE2_CKPT} VLA_BASE=${VLA_BASE} \
        N_EPISODES=${N_EPISODES} SAVE_JSON=${JSON_OUT} \
          ${PYTHON} "${EVAL_SCRIPT}" 2>&1 | tail -5
        SR=$( ${PYTHON} -c "import json; d=json.load(open('${JSON_OUT}')); print(f\"{d.get('success_rate',0):.2%}\")" 2>/dev/null || echo "?")
        echo "  → SR=${SR}"
        RESULTS+=("${EXP}/${CKPT_LABEL}: ${SR}")
    fi
done

# ── Experiments 8-19: 학습+eval ───────────────────────────────────────────────
REMAINING=(
    "stage3_spatial_pca_multi_goal_t06"
    "stage3_spatial_pca_multi_goal_attn"
    "stage3_spatial_pca_multi_goal_delta"
    "stage3_spatial_pca_multi_goal_dino_attn"
    "stage3_spatial_continuous_reward"
    "stage3_latent_cos"
    "stage3_pca_goal"
    "stage3_pca_goal_single"
    "stage3_pca_max"
    "stage3_diversity"
    "stage3_action_goal"
    "stage3_action_pca_delta"
)

TOTAL_R=${#REMAINING[@]}
for i in "${!REMAINING[@]}"; do
    EXP="${REMAINING[$i]}"
    CFG="${SWM}/configs/robust_b/${EXP}.yaml"
    NUM=$((i + 8))
    LOG="${SWM}/logs/rb_${EXP}_${TS}.log"

    echo ""
    echo "[${NUM}/19] TRAIN: ${EXP}"

    CUDA_VISIBLE_DEVICES=${GPUS} \
      ${TORCHRUN} --nproc_per_node=${N_GPUS} \
        "${SWM}/scripts/train_stage3.py" \
        --config "${CFG}" \
        2>&1 | tee "${LOG}"

    OUT_DIR=$( ${PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CFG}')
print(cfg.experiment.output_dir)
" )

    for CKPT_LABEL in best last; do
        CKPT_PATH="${SWM}/${OUT_DIR}/${CKPT_LABEL}.pt"
        JSON_OUT="${SWM}/eval_results/rb_${EXP}_${CKPT_LABEL}_${TS}.json"
        if [ -f "${CKPT_PATH}" ]; then
            echo "  [eval/${CKPT_LABEL}] ${CKPT_PATH}"
            CUDA_VISIBLE_DEVICES=${GPUS%,*} \
            TASK=square CKPT_PATH=${CKPT_PATH} STAGE1_CKPT=${STAGE1_CKPT} \
            STAGE2_CKPT=${STAGE2_CKPT} VLA_BASE=${VLA_BASE} \
            N_EPISODES=${N_EPISODES} SAVE_JSON=${JSON_OUT} \
              ${PYTHON} "${EVAL_SCRIPT}" 2>&1 | tail -5
            SR=$( ${PYTHON} -c "import json; d=json.load(open('${JSON_OUT}')); print(f\"{d.get('success_rate',0):.2%}\")" 2>/dev/null || echo "?")
            echo "  → SR=${SR}"
            RESULTS+=("${EXP}/${CKPT_LABEL}: ${SR}")
        fi
    done
done

echo ""
echo "================================================================"
echo "  Option B 완료 — Results"
echo "================================================================"
for r in "${RESULTS[@]}"; do echo "  ${r}"; done
echo "================================================================"
