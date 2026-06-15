"""Quick test: verify that adding empty token 29871 fixes generate_action_verl action predictions."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

SWM = "/scratch/mip25/sjLee/SWM"
sys.path.insert(0, SWM)
sys.path.insert(0, f"{SWM}/dependencies/openvla-oft")
sys.path.insert(0, f"{SWM}/dependencies/openvla-oft/experiments/robot")

import numpy as np, torch, json
from pathlib import Path

device = torch.device("cuda")
vla_base = f"{SWM}/ckpts/checkpoint_files/WMPO_models/square/P_128"
unnorm_key = "square_d0_300_demos"

from transformers import AutoModelForVision2Seq, AutoProcessor
from openvla_utils import update_auto_map
update_auto_map(vla_base)

print("Loading model...")
vla = AutoModelForVision2Seq.from_pretrained(vla_base, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(vla_base, trust_remote_code=True)
ds_path = Path(vla_base) / "dataset_statistics.json"
vla.norm_stats.update(json.load(open(ds_path)))
vla.eval()

instruction = "pick up the square nut and insert it onto the peg"
prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

# Use a dummy white image
from PIL import Image
pil = Image.new("RGB", (224, 224), (128, 128, 128))
bf = processor(prompt, pil)

_PAD = 32000
_EMPTY = 29871

def run_with_empty(add_empty: bool):
    inputs = {k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
              for k, v in bf.items()}
    print(f"\n[add_empty={add_empty}] input_ids shape: {inputs['input_ids'].shape}, last5={inputs['input_ids'][0,-5:].tolist()}")
    if add_empty:
        ids = inputs["input_ids"]
        if not torch.all(ids[:, -1] == _EMPTY):
            extra = torch.full((ids.shape[0], 1), _EMPTY, dtype=ids.dtype, device=ids.device)
            inputs["input_ids"] = torch.cat([ids, extra], dim=-1)
            attn = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                [attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=attn.device)], dim=-1
            )
        print(f"  after add_empty: input_ids shape={inputs['input_ids'].shape}, last3={inputs['input_ids'][0,-3:].tolist()}")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actions, _, _ = vla.generate_action_verl(
            **inputs, unnorm_key=unnorm_key, do_sample=False, temperature=1.0, padding_idx=_PAD)
    a = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
    if a.ndim == 3: a = a[0]
    print(f"  chunk[0]: {np.round(a[0], 4).tolist()}")
    print(f"  gripper: {a[0,-1]:.4f}  (GT close=-1.0, open=1.0)")

run_with_empty(False)
run_with_empty(True)

# Also run predict_action for comparison
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    inputs2 = {k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
               for k, v in bf.items()}
    actions2, _ = vla.predict_action(**inputs2, unnorm_key=unnorm_key, do_sample=False)
if isinstance(actions2, torch.Tensor): actions2 = actions2.cpu().numpy()
if actions2.ndim == 3: actions2 = actions2[0]
print(f"\n[predict_action] chunk[0]: {np.round(actions2[0], 4).tolist()}")
print(f"  gripper: {actions2[0,-1]:.4f}")
