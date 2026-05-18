#!/bin/bash
# ============================================================
# SWM 전체 재학습 파이프라인 (Stage 1 → 2 → 3)
# GPU 2,3,7  /  debug 검증 후 본학습 자동 실행
#
# 사용법:
#   bash run_retrain_all.sh              # Stage 1→2→3 전부
#   bash run_retrain_all.sh --from 2     # Stage 2→3부터 (Stage 1 이미 완료된 경우)
#   bash run_retrain_all.sh --from 3     # Stage 3만
# ============================================================

set -e   # 에러 즉시 중단

# ── 환경 설정 ──────────────────────────────────────────────────
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=2,3,7
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/dgist_shyo/sjLee/SWM
mkdir -p logs

N_GPUS=3
MASTER_PORT_S1=29520
MASTER_PORT_S2=29521
MASTER_PORT_S3=29522
MASTER_PORT_S3D=29523

# ── 시작 스테이지 파싱 ──────────────────────────────────────────
FROM_STAGE=1
if [[ "$1" == "--from" && -n "$2" ]]; then
    FROM_STAGE=$2
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX="logs/retrain_all_${TIMESTAMP}"

echo "============================================================"
echo " SWM 전체 재학습  GPU=${CUDA_VISIBLE_DEVICES}  from Stage ${FROM_STAGE}"
echo " 로그 prefix: ${LOG_PREFIX}"
echo "============================================================"


# ── 유틸 함수 ──────────────────────────────────────────────────
run_stage() {
    local label="$1"
    local port="$2"
    local script="$3"
    local config="$4"
    local extra_flags="$5"
    local log_file="$6"

    echo ""
    echo "------------------------------------------------------------"
    echo "[${label}] 시작  config=${config}"
    echo "[${label}] 로그 → ${log_file}"
    echo "------------------------------------------------------------"

    torchrun \
        --nproc_per_node=${N_GPUS} \
        --master_port=${port} \
        ${script} \
        --config ${config} \
        --no-wandb \
        ${extra_flags} \
        2>&1 | tee "${log_file}"

    local exit_code=${PIPESTATUS[0]}
    if [ ${exit_code} -ne 0 ]; then
        echo ""
        echo "!!! [${label}] 실패 (exit code=${exit_code}) → 중단 !!!"
        exit ${exit_code}
    fi
    echo "[${label}] 완료 ✓"
}


# ══════════════════════════════════════════════════════════════
# STAGE 1 — Encoder + Graph Head
# ══════════════════════════════════════════════════════════════
if [ ${FROM_STAGE} -le 1 ]; then

    # ── Stage 1 debug (데이터 subset, 1 epoch) ──────────────────
    run_stage \
        "Stage1-DEBUG" \
        ${MASTER_PORT_S1} \
        "scripts/train_stage1.py" \
        "configs/stage1_debug.yaml" \
        "--debug" \
        "${LOG_PREFIX}_stage1_debug.log"

fi


# ══════════════════════════════════════════════════════════════
# STAGE 2 — JEPA Transition
# ══════════════════════════════════════════════════════════════
if [ ${FROM_STAGE} -le 2 ]; then

    run_stage \
        "Stage2-DEBUG" \
        ${MASTER_PORT_S2} \
        "scripts/train_stage2.py" \
        "configs/stage2_debug.yaml" \
        "--debug" \
        "${LOG_PREFIX}_stage2_debug.log"

fi


# ══════════════════════════════════════════════════════════════
# STAGE 3 — GRPO VLA (Transition-L2 + LoRA + KL)
# ══════════════════════════════════════════════════════════════
if [ ${FROM_STAGE} -le 3 ]; then

    run_stage \
        "Stage3-DEBUG" \
        ${MASTER_PORT_S3D} \
        "scripts/train_stage3.py" \
        "configs/stage3_debug.yaml" \
        "--debug" \
        "${LOG_PREFIX}_stage3_debug.log"

fi


# ══════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo " 전체 debug 검증 완료! ✓"
echo " 이상 없으면 본학습 실행:"
echo "   bash run_stage1_retrain.sh"
echo "   bash run_stage2_retrain.sh"
echo "   bash run_stage3_tl2_lora.sh"
echo "============================================================"
