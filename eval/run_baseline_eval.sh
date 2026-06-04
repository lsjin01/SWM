#!/bin/bash
# Baseline evaluation: SFT / WMPO-P128 / WMPO-P1280
# GPU 3장 병렬 (SFT=GPU0, P128=GPU1, P1280=GPU2)
set -e

SWM=/home/miplab1/sjLee/swm
CKPTS=$SWM/ckpts
TASK=${TASK:-square}
N_EPISODES=${N_EPISODES:-50}

SFT_PATH=$CKPTS/SFT_models/$TASK/checkpoint_files/SFT_models/$TASK
P128_PATH=$CKPTS/WMPO_models/$TASK/P_128/checkpoint_files/WMPO_models/$TASK/P_128
P1280_PATH=$CKPTS/WMPO_models/$TASK/P_1280/checkpoint_files/WMPO_models/$TASK/P_1280

RESULTS=$SWM/eval_results/baseline
mkdir -p $RESULTS $SWM/logs

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

source /opt/conda/etc/profile.d/conda.sh
conda activate wmpo

cd $SWM

echo "====== Baseline Eval: TASK=$TASK  N_EPISODES=$N_EPISODES ======"
echo "  SFT    → GPU 0"
echo "  P128   → GPU 1"
echo "  P1280  → GPU 2"
echo ""

# ── SFT ──────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 \
TASK=$TASK \
VLA_BASE=$SFT_PATH \
MODEL_NAME=SFT \
N_EPISODES=$N_EPISODES \
SAVE_JSON=$RESULTS/${TASK}_sft.json \
python eval/eval_baseline.py \
  > $SWM/logs/eval_${TASK}_sft.log 2>&1 &
PID_SFT=$!

# ── WMPO P128 ────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 \
TASK=$TASK \
VLA_BASE=$P128_PATH \
MODEL_NAME=WMPO_P128 \
N_EPISODES=$N_EPISODES \
SAVE_JSON=$RESULTS/${TASK}_wmpo_p128.json \
python eval/eval_baseline.py \
  > $SWM/logs/eval_${TASK}_wmpo_p128.log 2>&1 &
PID_P128=$!

# ── WMPO P1280 ───────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 \
TASK=$TASK \
VLA_BASE=$P1280_PATH \
MODEL_NAME=WMPO_P1280 \
N_EPISODES=$N_EPISODES \
SAVE_JSON=$RESULTS/${TASK}_wmpo_p1280.json \
python eval/eval_baseline.py \
  > $SWM/logs/eval_${TASK}_wmpo_p1280.log 2>&1 &
PID_P1280=$!

echo "  PIDs: SFT=$PID_SFT  P128=$PID_P128  P1280=$PID_P1280"
echo "  Logs: logs/eval_${TASK}_*.log"
echo ""

wait $PID_SFT $PID_P128 $PID_P1280

echo "====== RESULTS ======"
for f in $RESULTS/${TASK}_sft.json $RESULTS/${TASK}_wmpo_p128.json $RESULTS/${TASK}_wmpo_p1280.json; do
    if [ -f "$f" ]; then
        name=$(python3 -c "import json; d=json.load(open('$f')); print(d['model_name'])")
        sr=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['success_rate']:.3f}\")")
        printf "  %-15s SR = %s\n" "$name" "$sr"
    fi
done
echo "====== DONE ======"
