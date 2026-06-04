#!/bin/bash
# =============================================================================
# Stage3 reward 실험 자동 체인:  각 변형마다  train → export → verl eval → 기록 → 정리
#
# 사용:
#   GPU=0 TASK=square VARIANTS="grounding_ema ema_only" bash run_chain.sh
#   (VARIANTS = configs/stage3_<variant>.yaml 의 <variant> 목록)
#
# 결과: eval_results/chain_results.csv  (variant, train_best_reward, eval_SR)
# =============================================================================
set -u

SWM=/home/miplab1/sjLee/swm
WMPO=/home/miplab1/sjLee/WMPO
OFT=/home/mipstu/jiPark/openvla-oft/experiments/robot
ENVN=wmpo
GPU=${GPU:-0}
TASK=${TASK:-square}
VARIANTS=${VARIANTS:-"grounding_ema"}
SFT_BASE=$SWM/ckpts/SFT_models/$TASK/checkpoint_files/SFT_models/$TASK
RESULTS=$SWM/eval_results/chain_results.csv

# 공통 환경
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SWM:$WMPO:$OFT
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export CPATH=/opt/conda/envs/$ENVN/x86_64-conda-linux-gnu/sysroot/usr/include:/opt/conda/envs/$ENVN/include:${CPATH:-}
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p $SWM/logs $SWM/eval_results
[ -f "$RESULTS" ] || echo "timestamp,variant,task,train_best_reward,eval_SR" > "$RESULTS"

run() { conda run -n $ENVN "$@"; }
ts()  { date '+%Y-%m-%d %H:%M:%S'; }

for V in $VARIANTS; do
  CFG=$SWM/configs/stage3_${V}.yaml
  if [ ! -f "$CFG" ]; then echo "[chain] config 없음: $CFG  → skip"; continue; fi

  OUTDIR=$SWM/outputs/stage3/${V}/${TASK}
  EXPDIR=$SWM/ckpts/exported/${V}_${TASK}
  TRAINLOG=$SWM/logs/chain_train_${V}.log
  EVALLOG=$SWM/logs/chain_eval_${V}.log

  echo "=================================================================="
  echo "[chain $(ts)] VARIANT=$V  TASK=$TASK"
  echo "=================================================================="

  # ── 1) train ────────────────────────────────────────────────────────────
  echo "[chain $(ts)] (1/4) train ..."
  run python $SWM/scripts/train_stage3.py --config "$CFG" --no-wandb > "$TRAINLOG" 2>&1
  if [ ! -f "$OUTDIR/best.pt" ]; then
    echo "[chain $(ts)] best.pt 없음 → train 실패. 로그: $TRAINLOG  → skip"
    continue
  fi
  TRAIN_BEST=$(grep -oE "Best reward=[-0-9.]+" "$OUTDIR/train.log" 2>/dev/null | tail -1 | cut -d= -f2)

  # ── 2) export → HF ───────────────────────────────────────────────────────
  echo "[chain $(ts)] (2/4) export → HF ..."
  rm -rf "$EXPDIR"
  run python $SWM/scripts/export_stage3_to_hf.py \
      --ckpt "$OUTDIR/best.pt" --base "$SFT_BASE" --out "$EXPDIR" >> "$TRAINLOG" 2>&1
  if [ ! -f "$EXPDIR/config.json" ]; then
    echo "[chain $(ts)] export 실패. 로그: $TRAINLOG  → skip"
    continue
  fi

  # ── 3) verl eval ─────────────────────────────────────────────────────────
  echo "[chain $(ts)] (3/4) verl eval ..."
  # 루트 커스텀 코드 복사 (evaluate.sh와 동일 동작)
  for f in modeling_prismatic.py preprocessor_config.json processing_prismatic.py; do
    cp $SWM/ckpts/SFT_models/$TASK/checkpoint_files/SFT_models/$TASK/$f "$EXPDIR/" 2>/dev/null || true
  done
  ( cd $WMPO && conda run -n $ENVN bash $SWM/scripts/eval_one.sh "$TASK" "$EXPDIR" "chain_${V}" > "$EVALLOG" 2>&1 )
  EVAL_SR=$(grep -oE "val/test_score/${TASK}:[0-9.]+" "$EVALLOG" 2>/dev/null | tail -1 | cut -d: -f2)
  [ -z "$EVAL_SR" ] && EVAL_SR="FAILED"

  # ── 4) 기록 + 정리 ────────────────────────────────────────────────────────
  echo "$(ts),$V,$TASK,${TRAIN_BEST:-NA},$EVAL_SR" >> "$RESULTS"
  echo "[chain $(ts)] (4/4) result: variant=$V  train_best=${TRAIN_BEST:-NA}  eval_SR=$EVAL_SR"

  # 디스크 절약: export 디렉토리 + last.pt 삭제 (best.pt만 보존)
  rm -rf "$EXPDIR"
  rm -f "$OUTDIR/last.pt"
done

echo "=================================================================="
echo "[chain $(ts)] ALL DONE.  결과:"
cat "$RESULTS"
