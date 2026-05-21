#!/bin/bash
# Stage 2 robust_b3 완료 대기 후 Stage 3 전체 19개 자동 실행
# 사용법: nohup bash scripts/wait_and_run_stage3_rb3.sh > logs/wait_rb3_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT="${SWM}/outputs/stage2/robust_b3/best.pt"
GPU_A="4,5"
GPU_B="6,7"

echo "================================================================"
echo "  Stage 2 robust_b3 완료 대기 중..."
echo "  CKPT: ${CKPT}"
echo "================================================================"

while [ ! -f "${CKPT}" ]; do
    echo "  [$(date '+%H:%M:%S')] 아직 학습 중... 5분 후 재확인"
    sleep 300
done

echo ""
echo "  [$(date '+%H:%M:%S')] best.pt 감지. 학습 프로세스 종료 대기..."
while pgrep -f "train_stage2.py.*robust_b3" > /dev/null 2>&1; do
    echo "  [$(date '+%H:%M:%S')] train_stage2.py 아직 실행 중... 2분 후 재확인"
    sleep 120
done

echo ""
echo "================================================================"
echo "  Stage 2 robust_b3 완료!"
echo "  Stage 3 parallel chain 시작 (GPU ${GPU_A} + ${GPU_B})"
echo "================================================================"

bash "${SWM}/scripts/train_phase2_parallel_chain_rb3.sh" "${GPU_A}" "${GPU_B}" 0
