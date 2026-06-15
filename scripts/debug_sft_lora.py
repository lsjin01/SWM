"""SFT LoRA 실제 적용 여부 검증"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SWM = "/scratch/mip25/sjLee/SWM"
sys.path.insert(0, f"{SWM}/dependencies/openvla-oft")
sys.path.insert(0, f"{SWM}/dependencies/openvla-oft/experiments/robot")

import torch, numpy as np, json
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor
from openvla_utils import update_auto_map

sft_base = f"{SWM}/ckpts/checkpoint_files/SFT_models/square"
p128_base = f"{SWM}/ckpts/checkpoint_files/WMPO_models/square/P_128"
device = torch.device("cuda")

# ── SFT: base만 (LoRA 없이) ──────────────────────────────────────────────
print("=== 1. Loading SFT BASE (no LoRA) ===")
update_auto_map(sft_base)
base_only = AutoModelForVision2Seq.from_pretrained(
    sft_base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device)
# lm_head weight 샘플
w_base = base_only.language_model.lm_head.weight.data[31744:31750].clone().cpu()
print(f"  lm_head[31744:31750]: {w_base.tolist()}")
del base_only
torch.cuda.empty_cache()

# ── SFT: base + LoRA merge ───────────────────────────────────────────────
print("\n=== 2. Loading SFT BASE + LoRA ===")
update_auto_map(sft_base)
sft_model = AutoModelForVision2Seq.from_pretrained(
    sft_base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device)
from peft import PeftModel
lora_dir = Path(sft_base) / "lora_adapter"
sft_model = PeftModel.from_pretrained(sft_model, str(lora_dir))

# merge 전 weight
w_before = sft_model.base_model.model.language_model.lm_head.weight.data[31744:31750].clone().cpu()
print(f"  lm_head[31744:31750] BEFORE merge: {w_before.tolist()}")

sft_model = sft_model.merge_and_unload()
w_after = sft_model.language_model.lm_head.weight.data[31744:31750].clone().cpu()
print(f"  lm_head[31744:31750] AFTER  merge: {w_after.tolist()}")
print(f"  CHANGED: {not torch.allclose(w_base.to(torch.float32), w_after.to(torch.float32))}")
print(f"  max diff: {(w_after.float() - w_base.float()).abs().max().item():.6f}")
del sft_model
torch.cuda.empty_cache()

# ── 액션 출력 비교 ──────────────────────────────────────────────────────
print("\n=== 3. Action output: SFT vs P128 ===")
from PIL import Image

def get_action(model_path, label):
    update_auto_map(model_path)
    m = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    lora_d = Path(model_path) / "lora_adapter"
    if lora_d.exists():
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(lora_d))
        m = m.merge_and_unload()
    m.norm_stats.update(json.load(open(f"{model_path}/dataset_statistics.json")))
    m.eval()
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 300 states pkl에서 state 0 로드하여 실제 env 이미지 얻기
    import pickle
    states = pickle.load(open(f"{SWM}/data/states/square_d0_states.pkl", "rb"))
    # state에서 reset하고 warmup 후 첫 obs의 agentview_image 사용
    # 대신 회색 더미 이미지로 테스트
    pil = Image.new("RGB", (224, 224), (100, 80, 70))  # 약간 다른 색
    prompt = "In: What action should the robot take to pick up the square nut and insert it onto the peg?\nOut:"
    bf = proc(prompt, pil)
    
    ids = bf["input_ids"].to(device)
    if not torch.all(ids[:, -1] == 29871):
        extra = torch.full((ids.shape[0], 1), 29871, dtype=ids.dtype, device=ids.device)
        ids = torch.cat([ids, extra], dim=-1)
        attn = bf["attention_mask"].to(device)
        attn = torch.cat([attn, torch.ones((attn.shape[0],1), dtype=attn.dtype, device=attn.device)], dim=-1)
    else:
        attn = bf["attention_mask"].to(device)
    
    inputs = {"input_ids": ids, "attention_mask": attn,
              "pixel_values": bf["pixel_values"].to(device, dtype=torch.bfloat16)}
    
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a, _, _ = m.generate_action_verl(**inputs, unnorm_key="square_d0_300_demos",
                                          do_sample=False, temperature=1.0, padding_idx=32000)
    a = a.cpu().numpy() if isinstance(a, torch.Tensor) else a
    if a.ndim == 3: a = a[0]
    print(f"  [{label}] chunk[0]: xyz={np.round(a[0,:3],3).tolist()}  rot={np.round(a[0,3:6],3).tolist()}  gripper={a[0,6]:.4f}")
    del m; torch.cuda.empty_cache()

get_action(sft_base, "SFT")
get_action(p128_base, "P128")
print("\nDone.")
