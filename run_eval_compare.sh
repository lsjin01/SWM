#!/bin/bash
# 4-way 병렬 평가: SFT pixel / WMPO P128 / WMPO P1280 / Ours (SWM z_bypass)
# GPU: 2=SFT, 3=P128, 4=P1280, 5=Ours

source /home/dgist_shyo/miniconda3/etc/profile.d/conda.sh
conda activate WMPO

SWM=/home/dgist_shyo/sjLee/SWM
WMPO=/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO
SFT_BASE="${WMPO}/checkpoint_files/SFT_models/square"
P128_BASE="${WMPO}/checkpoint_files/WMPO_models/square/P_128"
P1280_BASE="${WMPO}/checkpoint_files/WMPO_models/square/P_1280"
STAGE1="${SWM}/outputs/stage1/multitask_dinosiglip/best.pt"
STAGE2="${SWM}/outputs/stage2/multitask_dinosiglip/best.pt"
STAGE3="${SWM}/outputs/stage3/vs_p128_rm_lora/square/best.pt"

N_EPISODES=50
mkdir -p "${SWM}/logs" "${SWM}/eval_results"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="/home/dgist_shyo/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TASK=square
export N_EPISODES=${N_EPISODES}
export STAGE1_CKPT=${STAGE1}
export STAGE2_CKPT=${STAGE2}

echo "================================================================"
echo "  4-way evaluation  (N_EPISODES=${N_EPISODES})"
echo "  GPU 2: SFT pixel"
echo "  GPU 3: WMPO P128 pixel"
echo "  GPU 4: WMPO P1280 pixel"
echo "  GPU 5: Ours (SWM Stage3 z_bypass)"
echo "================================================================"

# ── 1. SFT pixel baseline ────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 CKPT_PATH="" USE_Z_BYPASS=0 \
VLA_BASE="${SFT_BASE}" SAVE_JSON="${SWM}/eval_results/compare_sft_pixel.json" \
python "${SWM}/eval/eval_swm_mimicgen.py" \
  > "${SWM}/logs/compare_sft_pixel.log" 2>&1 &
PID_SFT=$!

# ── 2. WMPO P128 pixel ───────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=3 CKPT_PATH="" USE_Z_BYPASS=0 \
VLA_BASE="${P128_BASE}" SAVE_JSON="${SWM}/eval_results/compare_wmpo_p128.json" \
python "${SWM}/eval/eval_swm_mimicgen.py" \
  > "${SWM}/logs/compare_wmpo_p128.log" 2>&1 &
PID_P128=$!

# ── 3. WMPO P1280 pixel ──────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=4 CKPT_PATH="" USE_Z_BYPASS=0 \
VLA_BASE="${P1280_BASE}" SAVE_JSON="${SWM}/eval_results/compare_wmpo_p1280.json" \
python "${SWM}/eval/eval_swm_mimicgen.py" \
  > "${SWM}/logs/compare_wmpo_p1280.log" 2>&1 &
PID_P1280=$!

# ── 4. Ours: SWM Stage3 z_bypass ────────────────────────────────────
CUDA_VISIBLE_DEVICES=5 CKPT_PATH="${STAGE3}" USE_Z_BYPASS=1 \
VLA_BASE="${SFT_BASE}" SAVE_JSON="${SWM}/eval_results/compare_ours_zbp.json" \
python "${SWM}/eval/eval_swm_mimicgen.py" \
  > "${SWM}/logs/compare_ours_zbp.log" 2>&1 &
PID_OURS=$!

echo "[$(date '+%H:%M:%S')] Launched:"
echo "  SFT pixel   PID=${PID_SFT}"
echo "  WMPO P128   PID=${PID_P128}"
echo "  WMPO P1280  PID=${PID_P1280}"
echo "  Ours zbp    PID=${PID_OURS}"

wait $PID_SFT $PID_P128 $PID_P1280 $PID_OURS
echo "[$(date '+%H:%M:%S')] All done."

parse_sr() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('success_rate','?'))" "$1" 2>/dev/null || echo "FAILED"
}

echo ""
echo "================================================================"
echo "  Results  (N=${N_EPISODES})"
printf "  %-30s  SR=%s\n" "SFT pixel baseline"    "$(parse_sr ${SWM}/eval_results/compare_sft_pixel.json)"
printf "  %-30s  SR=%s\n" "WMPO P128"             "$(parse_sr ${SWM}/eval_results/compare_wmpo_p128.json)"
printf "  %-30s  SR=%s\n" "WMPO P1280"            "$(parse_sr ${SWM}/eval_results/compare_wmpo_p1280.json)"
printf "  %-30s  SR=%s\n" "Ours (SWM Stage3 zbp)" "$(parse_sr ${SWM}/eval_results/compare_ours_zbp.json)"
echo "================================================================"
