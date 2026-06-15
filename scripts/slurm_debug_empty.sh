#!/bin/bash
#SBATCH --job-name=swm_debug_empty
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=0:15:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/debug_empty_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox

module purge
module load Singularity/4.3.4

echo "=== Debug empty token: $(date) ==="

singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot:\$PYTHONPATH
python3 $SWM/scripts/debug_empty_token.py
"

echo "=== Done: $(date) ==="
