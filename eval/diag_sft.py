"""
SFT LoRA merge 진단: base model vs merged model 의 action token logit 비교
"""
import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, '/scratch/mip25/sjLee/SWM/dependencies/openvla-oft')

SFT = '/scratch/mip25/sjLee/SWM/ckpts/checkpoint_files/SFT_models/coffee'
P128 = '/scratch/mip25/sjLee/SWM/ckpts/checkpoint_files/WMPO_models/coffee/P_128'

from transformers import AutoModelForVision2Seq, AutoProcessor
try:
    from openvla_utils import update_auto_map
    update_auto_map(SFT); update_auto_map(P128)
except: pass

print("=== Loading SFT base (no LoRA) ===")
base = AutoModelForVision2Seq.from_pretrained(
    SFT, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
).cuda().eval()
lm_base = base.language_model.lm_head.weight.data.clone()  # (32064, 4096)
print(f"  base lm_head norm: {lm_base.norm().item():.4f}")
del base
torch.cuda.empty_cache()

print("\n=== Loading SFT + LoRA merged ===")
sft = AutoModelForVision2Seq.from_pretrained(
    SFT, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
).cuda()
from peft import PeftModel
sft = PeftModel.from_pretrained(sft, f"{SFT}/lora_adapter", torch_dtype=torch.float32)
sft = sft.merge_and_unload()
sft.eval()
lm_merged = sft.language_model.lm_head.weight.data.clone()
print(f"  merged lm_head norm: {lm_merged.norm().item():.4f}")

delta = lm_merged - lm_base
print(f"  delta norm (total): {delta.norm().item():.6f}")
# action token range: vocab 31744~31999
action_delta = delta[31744:32000, :]
print(f"  delta norm (action tokens 31744:32000): {action_delta.norm().item():.6f}")
print(f"  delta max (action tokens): {action_delta.abs().max().item():.6f}")
print(f"  delta nonzero ratio: {(action_delta.abs() > 1e-7).float().mean().item():.4f}")

# 실제 action 출력 테스트: dummy 입력으로 action 예측
print("\n=== Action prediction test (dummy input) ===")
import json
sft.norm_stats.update(json.load(open(f"{SFT}/dataset_statistics.json")))
processor = AutoProcessor.from_pretrained(SFT, trust_remote_code=True)

from PIL import Image
import numpy as np
img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to coffee?\nOut:"
feat = processor(prompt, img)
input_ids = feat["input_ids"].cuda()
attn = feat["attention_mask"].cuda()
pv = feat["pixel_values"].cuda()

# empty token
if not torch.all(input_ids[:, -1] == 29871):
    input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)], dim=1)
    attn = torch.cat([attn, torch.ones((1,1), dtype=attn.dtype, device=attn.device)], dim=1)

with torch.no_grad():
    actions, _, norm_actions = sft.generate_action_verl(
        input_ids=input_ids, pixel_values=pv, attention_mask=attn,
        padding_idx=processor.tokenizer.pad_token_id,
        do_sample=False, unnorm_key="coffee_d0_300_demos", temperature=1.0
    )

print(f"  SFT actions[0]: {actions[0].tolist()}")
print(f"  SFT norm_actions[0]: {norm_actions[0].tolist()}")

del sft; torch.cuda.empty_cache()

print("\n=== Loading P128 for comparison ===")
p = AutoModelForVision2Seq.from_pretrained(
    P128, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
).cuda().eval()
import json as _json
p.norm_stats.update(_json.load(open(f"{P128}/dataset_statistics.json")))
proc2 = AutoProcessor.from_pretrained(P128, trust_remote_code=True)
feat2 = proc2(prompt, img)
input_ids2 = feat2["input_ids"].cuda()
attn2 = feat2["attention_mask"].cuda()
pv2 = feat2["pixel_values"].cuda()
if not torch.all(input_ids2[:, -1] == 29871):
    input_ids2 = torch.cat([input_ids2, torch.tensor([[29871]], dtype=input_ids2.dtype, device=input_ids2.device)], dim=1)
    attn2 = torch.cat([attn2, torch.ones((1,1), dtype=attn2.dtype, device=attn2.device)], dim=1)

with torch.no_grad():
    actions2, _, norm_actions2 = p.generate_action_verl(
        input_ids=input_ids2, pixel_values=pv2, attention_mask=attn2,
        padding_idx=proc2.tokenizer.pad_token_id,
        do_sample=False, unnorm_key="coffee_d0_300_demos", temperature=1.0
    )
print(f"  P128 actions[0]: {actions2[0].tolist()}")
print(f"  P128 norm_actions[0]: {norm_actions2[0].tolist()}")
