#!/bin/bash
#SBATCH --job-name=swm_dbg2
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOGDIR=/scratch/mip25/sjLee/SWM/logs
mkdir -p "$LOGDIR"
exec 1>"$LOGDIR/debug2_${SLURM_JOB_ID}.log"
exec 2>&1

SWM=/scratch/mip25/sjLee/SWM
SANDBOX=$SWM/containers/swm_sandbox
MUJOCO_PATH=/home/mip25/.mujoco/mujoco210

module purge
module load Singularity/4.3.4

echo "=== Debug2: $(date) / Node: $(hostname) ==="

singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    --bind /home/mip25/.mujoco:/home/mip25/.mujoco \
    "$SANDBOX" bash -c "
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$MUJOCO_PATH/bin:/opt/conda/lib:\$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$SWM/dependencies/openvla-oft:$SWM/dependencies/openvla-oft/experiments/robot:\$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python3 - << 'PYEOF'
import os, sys, json
import numpy as np
import torch
from pathlib import Path

SWM = Path('/scratch/mip25/sjLee/SWM')
CKPTS = SWM / 'ckpts/checkpoint_files'

os.environ['CUDA_VISIBLE_DEVICES'] = ''
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except: pass
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, str(SWM / 'dependencies/openvla-oft'))
sys.path.insert(0, str(SWM / 'dependencies/openvla-oft/experiments/robot'))

from transformers import AutoModelForVision2Seq, AutoProcessor
from openvla_utils import update_auto_map
from PIL import Image
import tensorflow as tf2

device = torch.device('cuda')

def load_and_test(model_name, model_path):
    print(f'\\n{\"=\"*60}')
    print(f'Testing: {model_name}')
    print(f'Path: {model_path}')

    update_auto_map(str(model_path))
    vla = AutoModelForVision2Seq.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    ds_path = model_path / 'dataset_statistics.json'
    if ds_path.exists():
        vla.norm_stats.update(json.load(open(ds_path)))
    vla.eval()

    # dummy 224x224 PIL image (gray)
    pil = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128).convert('RGB')
    prompt = 'In: What action should the robot take to pick up the square nut and insert it onto the peg?\\nOut:'
    inputs = processor(prompt, pil, return_tensors='pt').to(device, dtype=torch.bfloat16)

    print(f'  input_ids shape: {inputs[\"input_ids\"].shape}')
    print(f'  last 5 token IDs: {inputs[\"input_ids\"][0,-5:].tolist()}')
    print(f'  pixel_values: {inputs[\"pixel_values\"].shape}')

    # 직접 forward pass로 logit 분포 확인
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = vla.predict_action(**inputs, unnorm_key='square_d0_300_demos', do_sample=False)

    actions = result[0] if isinstance(result, tuple) else result
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    print(f'  actions shape: {actions.shape}')
    print(f'  actions[0]: {np.round(actions[0], 4).tolist()}')

    # vocab_size 확인
    print(f'  vocab_size (effective): {vla.vocab_size}')
    print(f'  n_action_bins: {vla.bin_centers.shape[0]}')
    print(f'  action bin range in vocab: [{vla.vocab_size - 256}, {vla.vocab_size - 1}]')

    # 모델 내부 hook으로 실제 token IDs 확인
    # _regression_or_discrete_prediction 패치
    import types
    original_fn = vla._regression_or_discrete_prediction
    predicted_ids_store = []
    def patched_fn(self, input_embeddings, all_actions_mask, projected_patch_embeddings,
                   attention_mask, labels, NUM_PATCHES, NUM_PROMPT_TOKENS, action_head=None):
        # Zero out action tokens
        all_actions_mask_3d = all_actions_mask.unsqueeze(-1)
        input_embeddings2 = input_embeddings * ~all_actions_mask_3d
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings2, projected_patch_embeddings, attention_mask
        )
        lm_out = self.language_model(
            input_ids=None, attention_mask=multimodal_attention_mask,
            inputs_embeds=multimodal_embeddings, labels=None,
            use_cache=None, output_hidden_states=False, return_dict=True,
        )
        logits = lm_out.logits  # [1, seq_len, 32064]

        # 실제 action token positions에서 argmax
        ACT_DIM = 7
        ACT_CHUNK = 8
        NUM_ACT = ACT_DIM * ACT_CHUNK  # 56

        # full vocab argmax (predict_action 방식)
        action_logits = logits[:, NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + NUM_ACT, :]
        full_argmax = action_logits.argmax(dim=-1).cpu().numpy()  # [1, 56]
        predicted_ids_store.append(('full_argmax', full_argmax[0]))

        # restricted to last 256 tokens (verl 방식)
        action_logits_last256 = action_logits[..., -256-64:-64]
        restricted_argmax = action_logits_last256.argmax(dim=-1).cpu().numpy()  # [1, 56]
        predicted_ids_store.append(('restricted_argmax_in_256', restricted_argmax[0]))

        # Continue with original (using full argmax)
        predicted_action_token_ids = full_argmax
        print(f'\\n  [Hook] NUM_PATCHES={NUM_PATCHES}, NUM_PROMPT_TOKENS={NUM_PROMPT_TOKENS}')
        print(f'  [Hook] logits shape: {logits.shape}')
        print(f'  [Hook] action position: [{NUM_PATCHES+NUM_PROMPT_TOKENS}, {NUM_PATCHES+NUM_PROMPT_TOKENS+NUM_ACT})')
        print(f'  [Hook] full_argmax token IDs (first 7): {full_argmax[0,:7].tolist()}')
        action_bin_min = self.vocab_size - 256
        action_bin_max = self.vocab_size - 1
        in_range = ((full_argmax[0] >= action_bin_min) & (full_argmax[0] <= action_bin_max)).sum()
        print(f'  [Hook] in action range [{action_bin_min},{action_bin_max}]: {in_range}/56 tokens')
        print(f'  [Hook] restricted_argmax in [0,255] (first 7): {restricted_argmax[0,:7].tolist()}')
        # get top5 logits at first action position
        top5 = torch.topk(action_logits[0, 0], 5)
        print(f'  [Hook] top5 token IDs at pos0: {top5.indices.tolist()}')
        print(f'  [Hook] top5 logit values at pos0: {[f\"{v:.3f}\" for v in top5.values.float().tolist()]}')

        # Call the original's computation
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        return normalized_actions, None

    vla._regression_or_discrete_prediction = types.MethodType(patched_fn, vla)

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        result2 = vla.predict_action(**inputs, unnorm_key='square_d0_300_demos', do_sample=False)
    print(f'  patched actions[0]: {np.round(result2[0][0], 4).tolist()}')

    del vla
    torch.cuda.empty_cache()

# SFT (base model without LoRA)
load_and_test('SFT_base', CKPTS / 'SFT_models/square')

# P128 (merged model)
load_and_test('WMPO_P128', CKPTS / 'WMPO_models/square/P_128')

PYEOF
" 2>&1

echo "=== Done: $(date) ==="
