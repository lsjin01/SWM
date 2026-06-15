#!/bin/bash
#SBATCH --job-name=swm_eval
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/eval_baseline_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
CKPTS=$SWM/ckpts/checkpoint_files
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210
MUJOCO_GEN=$SWM/containers/mujoco_py_generated
TASK=${TASK:-square}
N_EPISODES=${N_EPISODES:-50}

module purge
module load Singularity/4.3.4

echo "=== Baseline Eval: $(date) / Node: $(hostname) ==="
echo "  TASK=$TASK  N_EPISODES=$N_EPISODES"

# ── [1] states pkl 생성 (없을 경우만) ─────────────────────────────────────
STATES_PKL=$SWM/data/states/${TASK}_d0_states.pkl
if [ ! -f "$STATES_PKL" ]; then
    echo "[1] Generating states pkl ..."
    singularity exec \
        --bind /scratch/mip25:/scratch/mip25 \
        "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
python3 $SWM/scripts/gen_states.py --task $TASK --swm_root $SWM
"
else
    echo "[1] States pkl already exists: $STATES_PKL"
fi

# ── 공통 컨테이너 실행 함수 ────────────────────────────────────────────────
run_eval() {
    local MODEL_NAME=$1
    local VLA_BASE=$2
    local GPU=$3
    local SAVE_JSON=$SWM/eval_results/${TASK}_${MODEL_NAME}.json
    mkdir -p $SWM/eval_results

    echo ""
    echo "[eval] $MODEL_NAME on GPU $GPU → $SAVE_JSON"

    singularity exec --nv \
        --bind /scratch/mip25:/scratch/mip25 \
        --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
        "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_PATH
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:/opt/conda/lib:\$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# openvla-oft 전체 + experiments/robot 경로 추가
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot:\$PYTHONPATH

TASK=$TASK \
VLA_BASE=$VLA_BASE \
MODEL_NAME=$MODEL_NAME \
N_EPISODES=$N_EPISODES \
GPU=$GPU \
SAVE_JSON=$SAVE_JSON \
python3 $SWM/eval/eval_baseline.py \
    2>&1 | tee $LOGDIR/eval_${TASK}_${MODEL_NAME}_${SLURM_JOB_ID}.log
"
}

# ── [2] SFT 평가 (GPU 0) ───────────────────────────────────────────────────
run_eval "SFT" "$CKPTS/SFT_models/$TASK" "0" &
PID_SFT=$!

# ── [3] WMPO P128 평가 (GPU 1) ────────────────────────────────────────────
run_eval "WMPO_P128" "$CKPTS/WMPO_models/$TASK/P_128" "1" &
PID_P128=$!

wait $PID_SFT $PID_P128

# ── [4] WMPO P1280 평가 (GPU 0, 앞 작업 완료 후) ──────────────────────────
run_eval "WMPO_P1280" "$CKPTS/WMPO_models/$TASK/P_1280" "0"

# ── 결과 요약 ─────────────────────────────────────────────────────────────
echo ""
echo "====== RESULTS ======"
for MODEL in SFT WMPO_P128 WMPO_P1280; do
    JSON=$SWM/eval_results/${TASK}_${MODEL}.json
    if [ -f "$JSON" ]; then
        python3 -c "
import json
d = json.load(open('$JSON'))
print(f\"  {d['model_name']:<15} SR = {d['success_rate']:.3f}  ({d['n_success']}/{d['n_episodes']})\")
"
    else
        echo "  $MODEL: result not found"
    fi
done
echo "====== DONE: $(date) ======"
