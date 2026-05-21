#!/bin/bash
# 수정 코드 효과 검증: spatial_pca_binary 재실행 (fix 적용본)
# 원본 결과: best=14%, last=14%
# 사용법: bash scripts/run_fixtest.sh [GPU] [N_GPUS]
# 예시:   bash scripts/run_fixtest.sh 0,1 2

set -e

GPUS="${1:-0,1}"
N_GPUS="${2:-2}"
SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
EVAL_SCRIPT="${SWM}/eval/eval_swm_mimicgen.py"
VLA_BASE=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square
STAGE1_CKPT="${SWM}/outputs/stage1/multitask_dinosiglip/best.pt"
STAGE2_CKPT="${SWM}/outputs/stage2/robust_b/best.pt"
N_EPISODES=50
CFG="${SWM}/configs/robust_b/stage3_spatial_pca_binary_fixtest.yaml"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${SWM}/logs/fixtest_pca_binary_${TS}.log"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

mkdir -p "${SWM}/logs" "${SWM}/eval_results"

echo "================================================================"
echo "  Fix Test: spatial_pca_binary (수정 코드 검증)"
echo "  원본 결과: best=14%, last=14%"
echo "  GPUs: ${GPUS} (N=${N_GPUS})"
echo "  log: ${LOG}"
echo "================================================================"

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
    JSON_OUT="${SWM}/eval_results/fixtest_pca_binary_${CKPT_LABEL}_${TS}.json"

    if [ -f "${CKPT_PATH}" ]; then
        echo ""
        echo "[eval/${CKPT_LABEL}] ${CKPT_PATH}"
        CUDA_VISIBLE_DEVICES=${GPUS%,*} \
        TASK=square \
        CKPT_PATH=${CKPT_PATH} \
        STAGE1_CKPT=${STAGE1_CKPT} \
        STAGE2_CKPT=${STAGE2_CKPT} \
        VLA_BASE=${VLA_BASE} \
        N_EPISODES=${N_EPISODES} \
        SAVE_JSON=${JSON_OUT} \
          ${PYTHON} "${EVAL_SCRIPT}" 2>&1 | tail -5
        SR=$( ${PYTHON} -c "import json; d=json.load(open('${JSON_OUT}')); print(f\"{d.get('success_rate',0):.2%}\")" 2>/dev/null || echo "?")
        echo "  → SR=${SR}  (원본 대비: best=14%, last=14%)"
    fi
done

echo ""
echo "================================================================"
echo "  Fix Test 완료"
echo "================================================================"
