#!/bin/bash
# fixtest 결과 보고 자동 분기
# - fixtest SR > 14% (원본 best) → Option A: 전체 처음부터 재학습
# - fixtest SR ≤ 14%            → Option B: 남은 chain 이어서 실행
# 사용법: bash scripts/decide_and_run.sh [GPU] [N_GPUS]

GPUS="${1:-4,5,6,7}"
N_GPUS="${2:-4}"
SWM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/home/dgist_shyo/miniconda3/envs/WMPO/bin/python3.11
BASELINE_SR=14  # 원본 spatial_pca_binary best SR (%)

echo "================================================================"
echo "  Fix Test 결과 대기 중..."
echo "  기준: SR > ${BASELINE_SR}% → Option A (전체 재학습)"
echo "        SR ≤ ${BASELINE_SR}% → Option B (남은 chain 이어서)"
echo "================================================================"

# fixtest best.pt eval 결과 JSON 대기
while true; do
    JSON=$(ls -t "${SWM}/eval_results/fixtest_pca_binary_best_"*.json 2>/dev/null | head -1)
    if [ -n "$JSON" ]; then
        break
    fi
    sleep 30
done

SR_PCT=$( ${PYTHON} -c "
import json
d = json.load(open('$JSON'))
print(int(round(d.get('success_rate', 0) * 100)))
" 2>/dev/null || echo "0")

SR_LAST_JSON=$(ls -t "${SWM}/eval_results/fixtest_pca_binary_last_"*.json 2>/dev/null | head -1)
SR_LAST=$( ${PYTHON} -c "
import json
d = json.load(open('$SR_LAST_JSON'))
print(int(round(d.get('success_rate', 0) * 100)))
" 2>/dev/null || echo "0")

echo ""
echo "================================================================"
echo "  Fixtest 결과:"
echo "    best SR = ${SR_PCT}%  (원본 14%)"
echo "    last SR = ${SR_LAST}%  (원본 14%)"
echo "================================================================"

if [ "$SR_PCT" -gt "$BASELINE_SR" ] || [ "$SR_LAST" -gt "$BASELINE_SR" ]; then
    echo ""
    echo "  → SR 향상 확인! Option A: 전체 처음부터 재학습"
    echo "================================================================"
    bash "${SWM}/scripts/train_phase2_robust_b_chain.sh" "${GPUS}" "${N_GPUS}"
else
    echo ""
    echo "  → SR 동일/하락. Option B: 남은 chain 이어서 실행"
    echo "================================================================"
    bash "${SWM}/scripts/option_b_continue_chain.sh" "${GPUS}" "${N_GPUS}"
fi
