#!/bin/bash
#SBATCH --job-name=swm_fixegl
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
exec 1>"$LOGDIR/fix_egl_${SLURM_JOB_ID}.log"
exec 2>&1

SANDBOX=/scratch/mip25/sjLee/SWM/containers/swm_sandbox

module purge
module load Singularity/4.3.4

echo "=== Fix EGL: $(date) / Node: $(hostname) ==="

# [1] 호스트 NVIDIA libEGL → 샌드박스 /opt/conda/lib/ 에 복사
# PyOpenGL find_library("EGL")이 ldconfig 캐시만 보므로 ldconfig가 아는 경로에 배치
echo "[1] Copying NVIDIA libEGL into sandbox (via scratch) ..."
singularity exec --writable \
    --bind /scratch/mip25:/scratch/mip25 \
    "$SANDBOX" bash -c '
cp /scratch/mip25/sjLee/libEGL_nvidia.so.580.119.02 /opt/conda/lib/libEGL_nvidia.so.580.119.02
ln -sf /opt/conda/lib/libEGL_nvidia.so.580.119.02 /opt/conda/lib/libEGL.so
ln -sf /opt/conda/lib/libEGL_nvidia.so.580.119.02 /opt/conda/lib/libEGL.so.1
ls -la /opt/conda/lib/libEGL*

# ldconfig 캐시 업데이트 (/opt/conda/lib → find_library 인식)
echo "/opt/conda/lib" > /etc/ld.so.conf.d/conda.conf
ldconfig
echo "ldconfig done, EGL in cache:"
ldconfig -p | grep libEGL
'

# [2] EGL device 열거 검증 (--nv 환경)
echo "[2] Testing EGL device enumeration ..."
singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    "$SANDBOX" bash -c '
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:$LD_LIBRARY_PATH

python3 -c "
from mujoco.egl import egl_ext as EGL
devices = EGL.eglQueryDevicesEXT()
print(\"EGL devices:\", devices)
print(\"count:\", len(devices))

if len(devices) > 0:
    from OpenGL import EGL as OEGL
    display = OEGL.eglGetPlatformDisplayEXT(OEGL.EGL_PLATFORM_DEVICE_EXT, devices[0], None)
    print(\"display:\", display)
    ok = OEGL.eglInitialize(display, None, None)
    print(\"eglInitialize:\", ok)
    print(\"=== EGL OK ===\")
else:
    print(\"=== FAILED: 0 EGL devices ===\")
" 2>&1
'

echo "=== Finished: $(date) ==="
