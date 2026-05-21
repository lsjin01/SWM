#!/bin/bash
# Phase 2 실험 전체를 robust_b 체크포인트로 재학습 + 평가
# 사용법: bash scripts/train_phase2_robust_b_chain.sh [GPU] [N_GPUS]
# 예시:   bash scripts/train_phase2_robust_b_chain.sh 4,5,6,7 4

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

EXPERIMENTS=(
    # ── Phase 2 (spatial, 12개) ────────────────────────────────────────
    "stage3_spatial_pca_binary"
    "stage3_spatial_pca_binary_t06"
    "stage3_spatial_pca_binary_attn"
    "stage3_spatial_pca_binary_delta"
    "stage3_spatial_pca_binary_dino_attn"
    "stage3_spatial_pca_binary_pca_variance"
    "stage3_spatial_pca_multi_goal"
    "stage3_spatial_pca_multi_goal_t06"
    "stage3_spatial_pca_multi_goal_attn"
    "stage3_spatial_pca_multi_goal_delta"
    "stage3_spatial_pca_multi_goal_dino_attn"
    "stage3_spatial_continuous_reward"
    # ── Phase 1 unique (spatial 버전으로 부활, 7개) ────────────────────
    "stage3_latent_cos"
    "stage3_pca_goal"
    "stage3_pca_goal_single"
    "stage3_pca_max"
    "stage3_diversity"
    "stage3_action_goal"
    "stage3_action_pca_delta"
)

TOTAL=${#EXPERIMENTS[@]}
echo "================================================================"
echo "  Phase 2 Robust-B Chain  (${TOTAL} experiments)"
echo "  GPUs: ${GPUS} (N=${N_GPUS})   N_EPISODES=${N_EPISODES}"
echo "================================================================"

RESULTS=()

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    CFG="${SWM}/configs/robust_b/${EXP}.yaml"
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="${SWM}/logs/rb_${EXP}_${TS}.log"
    NUM=$((i + 1))

    echo ""
    echo "[${NUM}/${TOTAL}] TRAIN: ${EXP}"
    echo "  config: ${CFG}"
    echo "  log:    ${LOG}"

    CUDA_VISIBLE_DEVICES=${GPUS} \
      ${TORCHRUN} --nproc_per_node=${N_GPUS} \
        "${SWM}/scripts/train_stage3.py" \
        --config "${CFG}" \
        2>&1 | tee "${LOG}"

    # output_dir 파싱
    OUT_DIR=$( ${PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CFG}')
print(cfg.experiment.output_dir)
" )

    # ── 평가: best ──────────────────────────────────────────────────────
    BEST_CKPT="${SWM}/${OUT_DIR}/best.pt"
    LAST_CKPT="${SWM}/${OUT_DIR}/last.pt"
    JSON_BEST="${SWM}/eval_results/rb_${EXP}_best_${TS}.json"
    JSON_LAST="${SWM}/eval_results/rb_${EXP}_last_${TS}.json"

    for CKPT_LABEL in best last; do
        CKPT_PATH="${SWM}/${OUT_DIR}/${CKPT_LABEL}.pt"
        JSON_OUT="${SWM}/eval_results/rb_${EXP}_${CKPT_LABEL}_${TS}.json"

        if [ -f "${CKPT_PATH}" ]; then
            echo "  [eval/${CKPT_LABEL}] ${CKPT_PATH}"
            CUDA_VISIBLE_DEVICES=${GPUS%,*}  \
            TASK=square \
            CKPT_PATH=${CKPT_PATH} \
            STAGE1_CKPT=${STAGE1_CKPT} \
            STAGE2_CKPT=${STAGE2_CKPT} \
            VLA_BASE=${VLA_BASE} \
            N_EPISODES=${N_EPISODES} \
            SAVE_JSON=${JSON_OUT} \
              ${PYTHON} "${EVAL_SCRIPT}" 2>&1 | tail -5
            SR=$( ${PYTHON} -c "import json; d=json.load(open('${JSON_OUT}')); print(f\"{d.get('success_rate',0):.2%}\")" 2>/dev/null || echo "?")
            echo "  → SR=${SR}  (${JSON_OUT})"
            RESULTS+=("${EXP}/${CKPT_LABEL}: ${SR}")
        fi
    done
done

echo ""
echo "================================================================"
echo "  Phase 2 Robust-B — Final Results"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "  ${r}"
done
echo "================================================================"
