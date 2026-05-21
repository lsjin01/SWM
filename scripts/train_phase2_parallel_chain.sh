#!/bin/bash
# Phase 2 병렬 chain: 실험 2개씩 동시 학습 + 4개 eval 동시 실행
#
# 구조 (한 쌍당):
#   GPU 4,5 (N=2): exp_A 학습  ║  GPU 6,7 (N=2): exp_B 학습
#                   ↓ 둘 다 완료 대기
#   GPU4=best_A  GPU5=last_A  GPU6=best_B  GPU7=last_B  (동시 eval)
#
# 사용법: bash scripts/train_phase2_parallel_chain.sh [GPU_A] [GPU_B] [START_IDX]
# 예시:   bash scripts/train_phase2_parallel_chain.sh 4,5 6,7 0
#         bash scripts/train_phase2_parallel_chain.sh 4,5 6,7 6   ← 7번째부터 (0-indexed)

set -e

GPU_A="${1:-4,5}"
GPU_B="${2:-6,7}"
START_IDX="${3:-0}"
SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
EVAL_SCRIPT="${SWM}/eval/eval_swm_mimicgen.py"
VLA_BASE=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square
STAGE1_CKPT="${SWM}/outputs/stage1/multitask_dinosiglip/best.pt"
STAGE2_CKPT="${SWM}/outputs/stage2/robust_b/best.pt"
N_EPISODES=50
N_GPUS=2

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

mkdir -p "${SWM}/logs" "${SWM}/eval_results"

EXPERIMENTS=(
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
echo "  Phase 2 Parallel Chain  (GPU_A=${GPU_A}, GPU_B=${GPU_B}, N=2)"
echo "  START_IDX=${START_IDX}  N_EPISODES=${N_EPISODES}"
echo "  실험 ${TOTAL}개 → 2개씩 병렬 학습 + 4개 eval 동시"
echo "================================================================"

RESULTS=()

# eval 함수: 단일 GPU로 실행 (백그라운드 가능)
run_eval() {
    local EVAL_GPU="$1"
    local CKPT_PATH="$2"
    local JSON_OUT="$3"
    CUDA_VISIBLE_DEVICES=${EVAL_GPU} \
    TASK=square \
    CKPT_PATH=${CKPT_PATH} \
    STAGE1_CKPT=${STAGE1_CKPT} \
    STAGE2_CKPT=${STAGE2_CKPT} \
    VLA_BASE=${VLA_BASE} \
    N_EPISODES=${N_EPISODES} \
    SAVE_JSON=${JSON_OUT} \
      ${PYTHON} "${EVAL_SCRIPT}" > /dev/null 2>&1
}

# 2개씩 쌍으로 처리
i=${START_IDX}
while [ $i -lt $TOTAL ]; do
    EXP_A="${EXPERIMENTS[$i]}"
    TS=$(date +%Y%m%d_%H%M%S)
    NUM_A=$((i + 1))

    # B가 있는지 확인
    if [ $((i + 1)) -lt $TOTAL ]; then
        EXP_B="${EXPERIMENTS[$((i + 1))]}"
        NUM_B=$((i + 2))
        HAS_B=1
    else
        HAS_B=0
    fi

    CFG_A="${SWM}/configs/robust_b/${EXP_A}.yaml"
    LOG_A="${SWM}/logs/rb_${EXP_A}_${TS}.log"

    echo ""
    if [ $HAS_B -eq 1 ]; then
        CFG_B="${SWM}/configs/robust_b/${EXP_B}.yaml"
        LOG_B="${SWM}/logs/rb_${EXP_B}_${TS}.log"
        echo "[${NUM_A}&${NUM_B}/${TOTAL}] PARALLEL TRAIN"
        echo "  A (GPU ${GPU_A}): ${EXP_A}"
        echo "  B (GPU ${GPU_B}): ${EXP_B}"
    else
        echo "[${NUM_A}/${TOTAL}] SOLO TRAIN (마지막 홀수)"
        echo "  A (GPU ${GPU_A},${GPU_B} N=4): ${EXP_A}"
    fi

    # ── 학습 (병렬) ─────────────────────────────────────────────────────
    if [ $HAS_B -eq 1 ]; then
        # A, B 동시 시작
        CUDA_VISIBLE_DEVICES=${GPU_A} \
          ${TORCHRUN} --nproc_per_node=${N_GPUS} --master_port=29500 \
            "${SWM}/scripts/train_stage3.py" --config "${CFG_A}" \
            2>&1 | tee "${LOG_A}" &
        PID_A=$!

        CUDA_VISIBLE_DEVICES=${GPU_B} \
          ${TORCHRUN} --nproc_per_node=${N_GPUS} --master_port=29501 \
            "${SWM}/scripts/train_stage3.py" --config "${CFG_B}" \
            2>&1 | tee "${LOG_B}" &
        PID_B=$!

        echo "  Train PIDs: A=${PID_A} B=${PID_B}  (둘 다 완료 대기)"
        wait $PID_A
        wait $PID_B
        echo "  → 두 학습 완료"
    else
        # 마지막 홀수: 4 GPU 전체 사용
        CUDA_VISIBLE_DEVICES=${GPU_A},${GPU_B} \
          ${TORCHRUN} --nproc_per_node=4 --master_port=29500 \
            "${SWM}/scripts/train_stage3.py" --config "${CFG_A}" \
            2>&1 | tee "${LOG_A}"
    fi

    # ── output_dir 파싱 ─────────────────────────────────────────────────
    OUT_A=$( ${PYTHON} -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('${CFG_A}'); print(cfg.experiment.output_dir)" )
    if [ $HAS_B -eq 1 ]; then
        OUT_B=$( ${PYTHON} -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('${CFG_B}'); print(cfg.experiment.output_dir)" )
    fi

    # ── eval (4개 동시) ──────────────────────────────────────────────────
    # GPU 할당: best_A→4 last_A→5 best_B→6 last_B→7
    GPU4="${GPU_A%,*}"       # 4
    GPU5="${GPU_A#*,}"       # 5
    GPU6="${GPU_B%,*}"       # 6
    GPU7="${GPU_B#*,}"       # 7

    JSON_A_BEST="${SWM}/eval_results/rb_${EXP_A}_best_${TS}.json"
    JSON_A_LAST="${SWM}/eval_results/rb_${EXP_A}_last_${TS}.json"

    echo "  Eval 시작..."
    if [ $HAS_B -eq 1 ]; then
        JSON_B_BEST="${SWM}/eval_results/rb_${EXP_B}_best_${TS}.json"
        JSON_B_LAST="${SWM}/eval_results/rb_${EXP_B}_last_${TS}.json"

        [ -f "${SWM}/${OUT_A}/best.pt" ] && run_eval "${GPU4}" "${SWM}/${OUT_A}/best.pt" "${JSON_A_BEST}" &
        [ -f "${SWM}/${OUT_A}/last.pt" ] && run_eval "${GPU5}" "${SWM}/${OUT_A}/last.pt" "${JSON_A_LAST}" &
        [ -f "${SWM}/${OUT_B}/best.pt" ] && run_eval "${GPU6}" "${SWM}/${OUT_B}/best.pt" "${JSON_B_BEST}" &
        [ -f "${SWM}/${OUT_B}/last.pt" ] && run_eval "${GPU7}" "${SWM}/${OUT_B}/last.pt" "${JSON_B_LAST}" &
        wait
    else
        [ -f "${SWM}/${OUT_A}/best.pt" ] && run_eval "${GPU4}" "${SWM}/${OUT_A}/best.pt" "${JSON_A_BEST}" &
        [ -f "${SWM}/${OUT_A}/last.pt" ] && run_eval "${GPU5}" "${SWM}/${OUT_A}/last.pt" "${JSON_A_LAST}" &
        wait
    fi

    # ── 결과 수집 ────────────────────────────────────────────────────────
    for JSON in "${JSON_A_BEST}" "${JSON_A_LAST}"; do
        [ -f "$JSON" ] || continue
        SR=$( ${PYTHON} -c "import json; d=json.load(open('${JSON}')); print(f\"{d.get('success_rate',0):.2%}\")" 2>/dev/null || echo "?")
        LABEL=$(basename $JSON .json | sed "s/rb_${EXP_A}_//;s/_${TS}//")
        echo "  → A ${LABEL}: SR=${SR}"
        RESULTS+=("${EXP_A}/${LABEL}: ${SR}")
    done
    if [ $HAS_B -eq 1 ]; then
        for JSON in "${JSON_B_BEST}" "${JSON_B_LAST}"; do
            [ -f "$JSON" ] || continue
            SR=$( ${PYTHON} -c "import json; d=json.load(open('${JSON}')); print(f\"{d.get('success_rate',0):.2%}\")" 2>/dev/null || echo "?")
            LABEL=$(basename $JSON .json | sed "s/rb_${EXP_B}_//;s/_${TS}//")
            echo "  → B ${LABEL}: SR=${SR}"
            RESULTS+=("${EXP_B}/${LABEL}: ${SR}")
        done
    fi

    i=$((i + 2))
done

echo ""
echo "================================================================"
echo "  Parallel Chain 완료 — Results"
echo "================================================================"
for r in "${RESULTS[@]}"; do echo "  ${r}"; done
echo "================================================================"
