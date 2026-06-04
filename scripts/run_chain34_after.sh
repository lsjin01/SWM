#!/bin/bash
# ①② 체인(run_chain.sh, PID 인자)이 끝나면 ③④ 체인 자동 시작
WAIT_PID=${1:?need PID of running ①② chain}
SWM=/home/miplab1/sjLee/swm
cd $SWM
echo "[watcher $(date '+%F %T')] PID $WAIT_PID (①② chain) 종료 대기..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[watcher $(date '+%F %T')] ①② 종료 감지 → ③④ 체인 시작"
GPU=0 TASK=square \
  VARIANTS="full_tcn_only full_goaldist_only full_tcn_add full_goaldist_add" \
  bash $SWM/scripts/run_chain.sh > $SWM/logs/chain_full34.log 2>&1
echo "[watcher $(date '+%F %T')] ③④ 체인 종료"
