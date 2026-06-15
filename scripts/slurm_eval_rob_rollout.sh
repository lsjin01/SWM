#!/bin/bash
#SBATCH --job-name=swm_rob
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
exec 1>"$LOGDIR/eval_rob_rollout_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
CKPTS=$SWM/ckpts/checkpoint_files
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210
TASK=${TASK:-coffee}
N_EPISODES=${N_EPISODES:-128}

module purge
module load Singularity/4.3.4

echo "=== rob_rollout port (generate_action_verl, official logic): $(date) / Node: $(hostname) ==="
echo "  TASK=$TASK  N_EPISODES=$N_EPISODES"

run_eval() {
    local MODEL_NAME=$1
    local VLA_BASE=$2
    local GPU=$3
    local SAVE_JSON=$SWM/eval_results/rob_fp32_${TASK}_${MODEL_NAME}.json

    echo ""
    echo "[eval] $MODEL_NAME  GPU=$GPU"

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
export SWM_ROOT=$SWM
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot:\$PYTHONPATH

python3 $SWM/eval/eval_rob_rollout.py \
    --task $TASK \
    --ckpt $VLA_BASE \
    --model_name $MODEL_NAME \
    --runs $N_EPISODES \
    --gpu $GPU \
    --save_json $SAVE_JSON \
    2>&1 | tee $LOGDIR/eval_rob_${TASK}_${MODEL_NAME}_${SLURM_JOB_ID}.log
"
}

# SFT + P128 병렬
run_eval "SFT"       "$CKPTS/SFT_models/$TASK"         "0" &
PID_SFT=$!
run_eval "WMPO_P128" "$CKPTS/WMPO_models/$TASK/P_128"  "1" &
PID_P128=$!
wait $PID_SFT $PID_P128

# P1280 순차
run_eval "WMPO_P1280" "$CKPTS/WMPO_models/$TASK/P_1280" "0"

echo ""
echo "====== rob_rollout PORT RESULTS fp32 ($TASK, $N_EPISODES episodes) ======"
echo "  Method: generate_action_verl (float32)"
for MODEL in SFT WMPO_P128 WMPO_P1280; do
    JSON=$SWM/eval_results/rob_fp32_${TASK}_${MODEL}.json
    if [ -f "$JSON" ]; then
        python3 -c "
import json
d = json.load(open('$JSON'))
print(f\"  {d['model_name']:<15} SR = {d['success_rate']:.3f}  ({d['n_success']}/{d['n_episodes']})  instruction='{d.get('instruction','')}'\")"
    else
        echo "  $MODEL: not found"
    fi
done
echo "====== DONE: $(date) ======"
