#!/bin/bash
# =============================================================================
# baseline 다중 재평가: SFT / P_128 / P_1280 를 각 N회 재실행해 mean±std (CI) 추정.
# eval 은 temperature=1.6 샘플링이라 재실행마다 결과가 달라짐 → 경험적 분산 측정.
# GPU1 사용 (GPU0 의 ③④ 체인과 무간섭).
#
# 사용: REPEATS=3 bash scripts/reeval_multiseed.sh
# 결과: eval_results/reeval_baseline.csv  (model, run, SR)
# =============================================================================
set -u
SWM=/home/miplab1/sjLee/swm
WMPO=/home/miplab1/sjLee/WMPO
OFT=/home/mipstu/jiPark/openvla-oft/experiments/robot
ENVN=wmpo
TASK=square
REPEATS=${REPEATS:-3}
GPU=${GPU:-1}
RESULTS=$SWM/eval_results/reeval_baseline.csv

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SWM:$WMPO:$OFT
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=$GPU
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export CPATH=/opt/conda/envs/$ENVN/x86_64-conda-linux-gnu/sysroot/usr/include:/opt/conda/envs/$ENVN/include:${CPATH:-}
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p $SWM/logs $SWM/eval_results
[ -f "$RESULTS" ] || echo "model,run,SR" > "$RESULTS"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# model 이름 → WMPO 기준 상대경로
declare -A PATHS=(
  [SFT]="./checkpoint_files/SFT_models/square"
  [P_128]="./checkpoint_files/WMPO_models/square/P_128"
  [P_1280]="./checkpoint_files/WMPO_models/square/P_1280"
)

for M in ${MODELS:-SFT P_128 P_1280}; do
  for r in $(seq 1 $REPEATS); do
    EXP="reeval_${M}_r${r}"
    LOG="$SWM/logs/${EXP}.log"
    echo "[reeval $(ts)] $M run $r/$REPEATS → $LOG"
    ( cd $WMPO && conda run -n $ENVN bash $SWM/scripts/eval_one.sh "$TASK" "${PATHS[$M]}" "$EXP" > "$LOG" 2>&1 )
    SR=$(grep -aoE "val/test_score/${TASK}:[0-9.]+" "$LOG" 2>/dev/null | tail -1 | cut -d: -f2)
    [ -z "$SR" ] && SR="FAILED"
    echo "${M},${r},${SR}" >> "$RESULTS"
    echo "[reeval $(ts)] $M run $r → SR=$SR"
    # 디스크 절약: rollout 임시 산출물 정리
    rm -rf $WMPO/tmp_files/rollout_${EXP} 2>/dev/null || true
    rm -rf $WMPO/checkpoint_files/WMPO-mimicgen/${EXP} 2>/dev/null || true
  done
done

echo "=================================================================="
echo "[reeval $(ts)] ALL DONE. 결과 + 통계:"
cat "$RESULTS"
echo "------ mean±std ------"
$ENVN_PY 2>/dev/null
conda run -n $ENVN python - << 'PY'
import csv, statistics as st
from collections import defaultdict
d=defaultdict(list)
with open('/home/miplab1/sjLee/swm/eval_results/reeval_baseline.csv') as f:
    for row in csv.DictReader(f):
        try: d[row['model']].append(float(row['SR']))
        except: pass
for m,v in d.items():
    mean=sum(v)/len(v)
    sd=st.pstdev(v) if len(v)>1 else 0.0
    print(f"{m:8s}  n={len(v)}  mean={mean:.3f}  std={sd:.3f}  runs={['%.3f'%x for x in v]}")
PY
