#!/bin/bash
#SBATCH --job-name=swm_fixdeps
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/fix_deps_${SLURM_JOB_ID}.log"
exec 2>&1

SANDBOX=/scratch/mip25/sjLee/SWM/containers/swm_sandbox
SWM=/scratch/mip25/sjLee/SWM

module purge
module load Singularity/4.3.4

echo "=== Fix deps: $(date) / Node: $(hostname) ==="

# egl_probe stub 생성 (pip 빌드 불가 → get_available_devices()만 구현하면 됨)
singularity exec --writable \
    --bind /scratch/mip25:/scratch/mip25 \
    "$SANDBOX" bash -c '
export PATH=/opt/conda/bin:$PATH
PYSITE=/opt/conda/lib/python3.11/site-packages

echo "[1] egl_probe stub 설치 ..."
mkdir -p $PYSITE/egl_probe
cat > $PYSITE/egl_probe/__init__.py << "PYEOF"
def get_available_devices():
    """Stub: returns GPU device 0 (real EGL enumeration not needed with --nv)."""
    return [0]
PYEOF
python3 -c "import egl_probe; print(\"egl_probe OK:\", egl_probe.get_available_devices())"
'

# 전체 검증 (--nv 필요)
singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
    "$SANDBOX" bash -c '
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/mip25/.mujoco/mujoco210/bin:/opt/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/scratch/mip25/sjLee/SWM/dependencies/openvla-oft:/scratch/mip25/sjLee/SWM/dependencies/openvla-oft/experiments/robot:$PYTHONPATH

echo "[2] 전체 검증 ..."
python3 -c "
import cv2;      print(\"cv2:\", cv2.__version__)
import numba;    print(\"numba:\", numba.__version__)
import egl_probe; print(\"egl_probe:\", egl_probe.get_available_devices())
import robosuite; print(\"robosuite OK\")
import mimicgen;  print(\"mimicgen OK\")
import prismatic; print(\"prismatic OK\")
print(\"=== ALL OK ===\")
" 2>/dev/null
'

echo "=== Finished: $(date) ==="
