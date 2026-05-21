#!/bin/bash
# Stage 2 robust_b2 완료 대기 후 Stage 3 전체 19개 자동 실행
# 사용법: nohup bash scripts/wait_and_run_stage3_rb2.sh > logs/wait_rb2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT="${SWM}/outputs/stage2/robust_b2/best.pt"
GPU_A="4,5"
GPU_B="6,7"

echo "================================================================"
echo "  Stage 2 robust_b2 완료 대기 중..."
echo "  CKPT: ${CKPT}"
echo "================================================================"

# best.pt가 생길 때까지 대기 (5분 간격)
while [ ! -f "${CKPT}" ]; do
    echo "  [$(date '+%H:%M:%S')] 아직 학습 중... 5분 후 재확인"
    sleep 300
done

# 파일이 생겼어도 학습이 완전히 끝난 게 아닐 수 있으므로 (save_interval=5)
# Stage 2가 실제 완료됐는지 PID로 확인
echo ""
echo "  [$(date '+%H:%M:%S')] best.pt 감지. 학습 프로세스 종료 대기..."
while pgrep -f "train_stage2.py.*robust_b2" > /dev/null 2>&1; do
    echo "  [$(date '+%H:%M:%S')] train_stage2.py 아직 실행 중... 2분 후 재확인"
    sleep 120
done

echo ""
echo "================================================================"
echo "  Stage 2 robust_b2 완료!"
echo "  Stage 3 parallel chain 시작 (GPU ${GPU_A} + ${GPU_B})"
echo "================================================================"

# Stage 3 parallel chain 실행 (robust_b2 configs 사용)
# train_phase2_parallel_chain.sh를 robust_b2용으로 실행
bash "${SWM}/scripts/train_phase2_parallel_chain_rb2.sh" "${GPU_A}" "${GPU_B}" 0
