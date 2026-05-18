#!/bin/bash
# eval pipeline 검증: WMPO-GRPO (알려진 20% SR) → pixel mode로 평가
# 이 eval이 ~20% SR을 재현하면 pipeline이 올바른 것
set -e
source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

cd /home/dgist_shyo/sjLee/SWM

WMPO_GRPO=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/checkpoints/model_files/WMPO_models/square/P_128
SFT_BASE=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO/checkpoint_files/SFT_models/square

mkdir -p eval_results logs

echo "====== [1/2] SFT baseline (기대값: ~24%) ======"
TASK=square \
CKPT_PATH="" \
STAGE1_CKPT=outputs/stage1/multitask_dinosiglip/best.pt \
STAGE2_CKPT=outputs/stage2/multitask_dinosiglip/best.pt \
VLA_BASE=$SFT_BASE \
VLA_DEVICE=cuda \
N_EPISODES=20 \
USE_Z_BYPASS=0 \
SAVE_JSON=eval_results/baseline_sft_pixel.json \
python eval/eval_swm_mimicgen.py 2>&1 | tee logs/eval_baseline_sft.log

echo ""
echo "====== [2/2] WMPO-GRPO baseline (기대값: ~20%) ======"
TASK=square \
CKPT_PATH="" \
STAGE1_CKPT=outputs/stage1/multitask_dinosiglip/best.pt \
STAGE2_CKPT=outputs/stage2/multitask_dinosiglip/best.pt \
VLA_BASE=$WMPO_GRPO \
VLA_DEVICE=cuda \
N_EPISODES=20 \
USE_Z_BYPASS=0 \
SAVE_JSON=eval_results/baseline_wmpo_grpo_pixel.json \
python eval/eval_swm_mimicgen.py 2>&1 | tee logs/eval_baseline_wmpo.log

echo ""
echo "====== 결과 ======"
python3 -c "
import json, glob
for f in sorted(glob.glob('eval_results/baseline_*.json')):
    d = json.load(open(f))
    print(f\"{f.split('/')[-1]:40s}  SR={d['success_rate']:.3f}  ({d['n_success']}/{d['n_episodes']})\")
"
