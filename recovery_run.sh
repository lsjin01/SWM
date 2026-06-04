#!/bin/bash
# =============================================================================
# 디스크full 복구 재실행:
#   (A) full_probe_add 는 best.pt 가 멀쩡 → eval 만 재실행 (export→verl eval→기록)
#   (B) full_pls_add 재학습 + ③④ 4변형 체인 (run_chain.sh)
# =============================================================================
set -u
SWM=/home/miplab1/sjLee/swm
WMPO=/home/miplab1/sjLee/WMPO
OFT=/home/mipstu/jiPark/openvla-oft/experiments/robot
ENVN=wmpo
TASK=square
SFT_BASE=$SWM/ckpts/SFT_models/$TASK/checkpoint_files/SFT_models/$TASK
RESULTS=$SWM/eval_results/chain_results.csv

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SWM:$WMPO:$OFT
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export CPATH=/opt/conda/envs/$ENVN/x86_64-conda-linux-gnu/sysroot/usr/include:/opt/conda/envs/$ENVN/include:${CPATH:-}
export TF_CPP_MIN_LOG_LEVEL=3
run() { conda run -n $ENVN "$@"; }
ts()  { date '+%Y-%m-%d %H:%M:%S'; }

# ── (A) full_probe_add : eval only ──────────────────────────────────────────
V=full_probe_add
OUTDIR=$SWM/outputs/stage3/$V/$TASK
EXPDIR=$SWM/ckpts/exported/${V}_${TASK}
EVALLOG=$SWM/logs/recover_eval_${V}.log
echo "[recover $(ts)] (A) $V eval-only 시작 (재학습 없음)"
if [ -f "$OUTDIR/best.pt" ]; then
  rm -rf "$EXPDIR"
  run python $SWM/scripts/export_stage3_to_hf.py \
      --ckpt "$OUTDIR/best.pt" --base "$SFT_BASE" --out "$EXPDIR" > "$EVALLOG" 2>&1
  for f in modeling_prismatic.py preprocessor_config.json processing_prismatic.py; do
    cp $SFT_BASE/$f "$EXPDIR/" 2>/dev/null || true
  done
  if [ -f "$EXPDIR/config.json" ]; then
    ( cd $WMPO && run bash $SWM/scripts/eval_one.sh "$TASK" "$EXPDIR" "recover_${V}" >> "$EVALLOG" 2>&1 )
    SR=$(grep -oE "val/test_score/${TASK}:[0-9.]+" "$EVALLOG" 2>/dev/null | tail -1 | cut -d: -f2)
    TB=$(grep -oE "Best reward=[-0-9.]+" "$OUTDIR/train.log" 2>/dev/null | tail -1 | cut -d= -f2)
    [ -z "$SR" ] && SR="FAILED"
    echo "$(ts),$V,$TASK,${TB:-NA},$SR  (recovered)" >> "$RESULTS"
    echo "[recover $(ts)] (A) $V eval_SR=$SR"
  else
    echo "[recover $(ts)] (A) $V export 실패 → skip"
  fi
  rm -rf "$EXPDIR"
else
  echo "[recover $(ts)] (A) $V best.pt 없음 → skip"
fi

# ── (B) pls_add 재학습 + ③④ 체인 ────────────────────────────────────────────
echo "[recover $(ts)] (B) pls_add + ③④ 체인 시작"
GPU=0 TASK=$TASK \
  VARIANTS="full_pls_add full_tcn_only full_goaldist_only full_tcn_add full_goaldist_add" \
  bash $SWM/run_chain.sh >> $SWM/logs/chain_recover.log 2>&1
echo "[recover $(ts)] ALL DONE"
