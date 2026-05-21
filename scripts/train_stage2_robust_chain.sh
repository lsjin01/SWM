#!/bin/bash
# Stage 2 Robust Training Chain: B → C
# 사용법: bash scripts/train_stage2_robust_chain.sh [GPU_LIST]
# 예시:   bash scripts/train_stage2_robust_chain.sh 4,5,6,7

set -e

GPUS="${1:-4,5,6,7}"
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCHRUN=/home/dgist_shyo/miniconda3/envs/WMPO/bin/torchrun
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "${SWM}/logs"

echo "================================================================"
echo "  Stage 2 Robust Training Chain  (B → C)"
echo "  GPUs: ${GPUS}  (N=${N_GPUS})"
echo "================================================================"

# ── Phase B: Multi-step loss ────────────────────────────────────────────────
LOG_B="${SWM}/logs/stage2_robust_b_${TS}.log"
echo ""
echo "[Phase B] Multi-step unrolled loss  (50 epochs)"
echo "  log → ${LOG_B}"

CUDA_VISIBLE_DEVICES=${GPUS} \
  ${TORCHRUN} --nproc_per_node=${N_GPUS} \
    "${SWM}/scripts/train_stage2.py" \
    --config "${SWM}/configs/stage2_robust_b.yaml" \
    --no-wandb \
  2>&1 | tee "${LOG_B}"

B_CKPT="${SWM}/outputs/stage2/robust_b/best.pt"
if [ ! -f "${B_CKPT}" ]; then
    echo "ERROR: Phase B checkpoint not found: ${B_CKPT}"
    exit 1
fi
echo ""
echo "[Phase B] Done.  best.pt → ${B_CKPT}"

# ── Phase C: DAgger fine-tune ────────────────────────────────────────────────
LOG_C="${SWM}/logs/stage2_robust_c_${TS}.log"
echo ""
echo "[Phase C] DAgger fine-tune  (30 epochs, resume from B best)"
echo "  log → ${LOG_C}"

CUDA_VISIBLE_DEVICES=${GPUS} \
  ${TORCHRUN} --nproc_per_node=${N_GPUS} \
    "${SWM}/scripts/train_stage2.py" \
    --config "${SWM}/configs/stage2_robust_c.yaml" \
    --resume "${B_CKPT}" \
    --no-wandb \
  2>&1 | tee "${LOG_C}"

C_CKPT="${SWM}/outputs/stage2/robust_c/best.pt"
echo ""
echo "================================================================"
echo "  Chain complete."
echo "  B best: ${B_CKPT}"
echo "  C best: ${C_CKPT}"
echo "================================================================"
