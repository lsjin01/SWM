#!/bin/bash
#SBATCH --job-name=sft_diag
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/mip25/sjLee/SWM/logs/sft_diag_%j.log
#SBATCH --error=/scratch/mip25/sjLee/SWM/logs/sft_diag_%j.log

SWM=/scratch/mip25/sjLee/SWM
module purge; module load Singularity/4.3.4

singularity exec --nv \
  --bind /scratch/mip25:/scratch/mip25 \
  --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
  $SWM/containers/swm_sandbox bash -c "
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export SWM_ROOT=$SWM
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot
python3 /scratch/mip25/sjLee/SWM/eval/diag_sft.py
"
