#!/bin/bash
#SBATCH --job-name=swm_reeval
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/reeval_all_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
CKPTS=$SWM/ckpts/checkpoint_files
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210
N_EPISODES=128

module purge
module load Singularity/4.3.4

echo "=== Re-Eval All Tasks (official instructions): $(date) / Node: $(hostname) ==="
echo "  square instruction: 'Insert the square into the stick'"
echo "  coffee instruction: 'coffee'"

run_eval() {
    local TASK=$1
    local MODEL_NAME=$2
    local VLA_BASE=$3
    local GPU=$4
    local SAVE_JSON=$SWM/eval_results/${TASK}_${MODEL_NAME}.json
    mkdir -p $SWM/eval_results

    echo ""
    echo "[eval] $TASK / $MODEL_NAME  GPU=$GPU"

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

print_results() {
    local TASK=$1
    echo ""
    echo "====== $TASK RESULTS (official instruction, 128 episodes) ======"
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
}

# ── [1] SQUARE ────────────────────────────────────────────────────────────────
echo ""
echo "==============================="
echo "  TASK: square (128 episodes)"
echo "==============================="

run_eval "square" "SFT"       "$CKPTS/SFT_models/square"         "0" &
PID_SFT=$!
run_eval "square" "WMPO_P128" "$CKPTS/WMPO_models/square/P_128"  "1" &
PID_P128=$!
wait $PID_SFT $PID_P128

run_eval "square" "WMPO_P1280" "$CKPTS/WMPO_models/square/P_1280" "0"

print_results "square"

# ── [2] COFFEE ────────────────────────────────────────────────────────────────
echo ""
echo "==============================="
echo "  TASK: coffee (128 episodes)"
echo "==============================="

run_eval "coffee" "SFT"       "$CKPTS/SFT_models/coffee"         "0" &
PID_SFT=$!
run_eval "coffee" "WMPO_P128" "$CKPTS/WMPO_models/coffee/P_128"  "1" &
PID_P128=$!
wait $PID_SFT $PID_P128

run_eval "coffee" "WMPO_P1280" "$CKPTS/WMPO_models/coffee/P_1280" "0"

print_results "coffee"

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  FINAL SUMMARY (official WMPO instructions)"
echo "============================================"
for TASK in square coffee; do
    echo "  --- $TASK ---"
    for MODEL in SFT WMPO_P128 WMPO_P1280; do
        JSON=$SWM/eval_results/${TASK}_${MODEL}.json
        if [ -f "$JSON" ]; then
            python3 -c "
import json
d = json.load(open('$JSON'))
print(f\"    {d['model_name']:<15} SR = {d['success_rate']:.3f}  ({d['n_success']}/{d['n_episodes']})\")
"
        fi
    done
done
echo "====== DONE: $(date) ======"
