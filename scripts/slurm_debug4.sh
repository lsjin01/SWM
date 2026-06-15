#!/bin/bash
#SBATCH --job-name=swm_dbg4
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/debug4_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210

module purge
module load Singularity/4.3.4

echo "=== Debug4 (verl vs predict): $(date) / $(hostname) ==="

singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
    "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:/opt/conda/lib:\$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot:$SWM/dependencies/robosuite:$SWM/dependencies/robomimic:$SWM/dependencies/mimicgen:\$PYTHONPATH
python3 /scratch/mip25/sjLee/SWM/scripts/verl_debug.py
" 2>&1

echo "=== Done: $(date) ==="
