#!/bin/bash
# =============================================================================
# WMPO/SWM 평가 환경 전체 셋업 (검증 완료 2026-06-02)
# GPU: RTX PRO 6000 Blackwell (sm_120) / 시스템 CUDA toolkit: 12.9 (변경 안 함)
#
# 핵심 교훈:
#  - 평가는 반드시 공식 verl 파이프라인(generate_action_verl)을 써야 함.
#    커스텀 predict_action 직접 호출은 SR=0% 나옴.
#  - Blackwell GPU라 torch 2.5.1(원본 스펙) 불가 → torch 2.8.0+cu128 사용.
#    cu128도 sm_120 지원하며 prebuilt flash-attn wheel 존재.
#  - 시스템 nvcc(12.9)는 건드리지 않음. torch 자체 번들 CUDA 런타임만 사용.
#  - 설치 순서 중요. PYTHONNOUSERSITE=1로 user-site shadowing 방지.
#
# 사전 준비:
#  - conda, native MuJoCo 2.1.0 at ~/.mujoco/mujoco210
#  - WMPO 레포: git clone https://github.com/WM-PO/WMPO.git  (여기서는 $WMPO_DIR)
# =============================================================================
set -e

ENV_NAME=wmpo
WMPO_DIR=/home/miplab1/sjLee/WMPO
SP=/opt/conda/envs/$ENV_NAME/lib/python3.11/site-packages

export PYTHONNOUSERSITE=1
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH
# mujoco-py 빌드용 GL 헤더 경로 (conda mesa-libgl-devel)
export CPATH=/opt/conda/envs/$ENV_NAME/x86_64-conda-linux-gnu/sysroot/usr/include:/opt/conda/envs/$ENV_NAME/include:$CPATH

run() { conda run -n $ENV_NAME "$@"; }

echo "=== [0] conda env 생성 (python 3.11) ==="
conda create -n $ENV_NAME python=3.11 -y || echo "env exists"

echo "=== [1] torch 2.8.0 + cu128 (Blackwell 지원) ==="
run pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== [2] core python packages ==="
run pip install numpy==1.26.4 scipy scikit-learn tqdm wandb \
    omegaconf==2.3.0 hydra-core==1.3.2 \
    huggingface-hub==0.31.1 tokenizers==0.19.1 \
    Pillow opencv-python einops h5py pytest json_numpy pandas

echo "=== [3] transformers (moojink OpenVLA-OFT fork, bidirectional attn) ==="
run pip install --no-deps "git+https://github.com/moojink/transformers-openvla-oft.git"

echo "=== [4] TF + timm + peft + diffusers ==="
run pip install tensorflow==2.19.0 tensorflow-datasets==4.9.10 tensorflow-graphics==2021.12.3
run pip install timm==0.9.16 peft==0.11.0 accelerate diffusers==0.33.0

echo "=== [5] mujoco / mujoco-py (Cython<3.0 + GL 헤더 필요) ==="
conda install -n $ENV_NAME -c conda-forge mesa-libgl-devel-cos6-x86_64 patchelf glew -y
run pip install mujoco==3.7.0 "cython==0.29.37"
run pip install "mujoco-py==2.1.2.14" --no-build-isolation

echo "=== [6] robosuite/robomimic/mimicgen/robosuite-task-zoo (editable, --no-deps) ==="
cd $WMPO_DIR/dependencies
# install.sh가 클론하지 않았다면 먼저 클론:
[ -d robosuite ] || git clone https://github.com/ARISE-Initiative/robosuite.git
[ -d robomimic ] || git clone https://github.com/ARISE-Initiative/robomimic.git
[ -d mimicgen ]  || git clone https://github.com/NVlabs/mimicgen.git
[ -d robosuite-task-zoo ] || git clone https://github.com/ARISE-Initiative/robosuite-task-zoo.git
( cd robosuite && git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa 2>/dev/null || true )
( cd robomimic && git checkout d0b37cf214bd24fb590d182edb6384333f67b661 2>/dev/null || true )
( cd robosuite-task-zoo && git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed 2>/dev/null || true )
for pkg in robosuite robomimic mimicgen robosuite-task-zoo; do
    run pip install --no-deps -e $pkg
done

echo "=== [7] openvla-oft (prismatic) + dlimp + LIBERO ==="
run pip install --no-deps -e openvla-oft
run pip install --no-deps "git+https://github.com/moojink/dlimp_openvla"
[ -d /tmp/LIBERO_src ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO_src
run pip install --no-deps -e /tmp/LIBERO_src
echo "/tmp/LIBERO_src" > $SP/libero_src.pth

echo "=== [8] prismatic __init__ 비우기 (체인 import 차단; train_utils/constants만 사용) ==="
for f in "" /training /vla /models; do
    echo "" > $SP/prismatic${f}/__init__.py
done

echo "=== [9] verl deps + opensora world-model 의존성 ==="
run pip install "ray[default]==2.55.1" codetiming==1.4.0 tensordict==0.12.4 torchdata==0.11.0 \
    webdataset==1.0.2 jsonlines==4.0.0 pylatexenc==2.10
run pip install --no-deps -e opensora
run pip install --no-deps colossalai==0.4.7 galore_torch bitsandbytes mmengine
run pip install av imageio imageio-ffmpeg

echo "=== [10] opensora torchvision.io 비디오 import 패치 (torchvision 0.23은 io.video 제거) ==="
python3 - << 'PYEOF'
import re
base = "/home/miplab1/sjLee/WMPO/dependencies/opensora/opensora/datasets"
# read_video.py
f = f"{base}/read_video.py"; s = open(f).read()
if "try:\n    from torchvision import get_video_backend" not in s:
    s = s.replace(
        "from torchvision import get_video_backend\nfrom torchvision.io.video import _check_av_available",
        "try:\n    from torchvision import get_video_backend\nexcept Exception:\n    def get_video_backend(): return 'pyav'\n"
        "try:\n    from torchvision.io.video import _check_av_available\nexcept Exception:\n    def _check_av_available():\n        import av  # noqa")
    open(f,'w').write(s)
# utils.py
f = f"{base}/utils.py"; s = open(f).read()
if "try:\n    from torchvision.io import write_video" not in s:
    s = s.replace(
        "from torchvision.io import write_video",
        "try:\n    from torchvision.io import write_video\nexcept Exception:\n    def write_video(*a, **k): raise RuntimeError('write_video unavailable')")
    open(f,'w').write(s)
print("opensora video imports patched")
PYEOF

echo "=== [11] stub: tensornvme (libaio 빌드 불가) ==="
mkdir -p $SP/tensornvme
echo "" > $SP/tensornvme/__init__.py
cat > $SP/tensornvme/async_file_io.py << 'PYEOF'
class AsyncFileWriter:
    def __init__(self, *a, **k):
        raise RuntimeError("tensornvme AsyncFileWriter unavailable (stub)")
PYEOF

echo "=== [12] flash-attn 2.8.3 prebuilt wheel (Blackwell, torch2.8 cu12 cxx11abiTRUE) ==="
# 학습용. 소스 빌드/nvcc 불필요. eval만 할 거면 생략 가능(stub로 import만 통과시켜도 됨).
run pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

echo "=== [13] numpy 재고정 (위 설치들이 2.x로 올릴 수 있음) ==="
run pip install "numpy==1.26.4"

echo ""
echo "=== 검증 ==="
run python -c "
import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'sm', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
import transformers, mujoco_py, robosuite, mimicgen, flash_attn
print('transformers', transformers.__version__, '| mujoco-py', mujoco_py.__version__, '| flash_attn', flash_attn.__version__)
import flash_attn_2_cuda; print('flash_attn CUDA ext OK')
" 2>&1 | grep -iE "torch|transformers|flash|OK"

echo ""
echo "=== DONE: WMPO/SWM 평가 환경 셋업 완료 ==="
echo "데이터/체크포인트 심볼릭 링크는 별도 (WMPO 루트에 data_files/, checkpoint_files/)."
echo "평가 실행: examples/mimicgen/square/evaluate_{sft,p128,p1280}.sh"
