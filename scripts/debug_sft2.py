"""LoRA adapter 내용 직접 확인"""
import sys
sys.path.insert(0, '/scratch/mip25/sjLee/SWM/dependencies/openvla-oft')
sys.path.insert(0, '/scratch/mip25/sjLee/SWM/dependencies/openvla-oft/experiments/robot')

from safetensors import safe_open
import torch, numpy as np

adapter_path = "/scratch/mip25/sjLee/SWM/ckpts/checkpoint_files/SFT_models/square/lora_adapter/adapter_model.safetensors"

print("=== LoRA adapter 키 목록 ===")
with safe_open(adapter_path, framework="pt", device="cpu") as f:
    keys = sorted(f.keys())
    print(f"Total keys: {len(keys)}")
    # lm_head 관련
    lm_head_keys = [k for k in keys if 'lm_head' in k]
    print(f"\nlm_head keys: {lm_head_keys}")
    
    # 첫 5개 키
    print(f"\nFirst 5 keys:")
    for k in keys[:5]:
        t = f.get_tensor(k)
        print(f"  {k}: shape={t.shape}, norm={t.float().norm().item():.6f}, max_abs={t.float().abs().max().item():.6f}")
    
    # q_proj 관련
    q_keys = [k for k in keys if 'q_proj' in k][:4]
    print(f"\nq_proj keys (first 4): {q_keys}")
    for k in q_keys:
        t = f.get_tensor(k)
        print(f"  {k}: shape={t.shape}, norm={t.float().norm().item():.6f}")
    
    # lm_head weight stats if exists
    if lm_head_keys:
        for k in lm_head_keys:
            t = f.get_tensor(k)
            print(f"\n  {k}: shape={t.shape}, norm={t.float().norm():.6f}, max_abs={t.float().abs().max():.6f}")
    else:
        print("\n  lm_head NOT in adapter!")
    
    # 전체 norm 통계
    norms = []
    for k in keys:
        t = f.get_tensor(k)
        norms.append(t.float().norm().item())
    print(f"\nAll layer norms: min={min(norms):.4f}, max={max(norms):.4f}, mean={np.mean(norms):.4f}")
    zero_count = sum(1 for n in norms if n < 1e-6)
    print(f"Near-zero norm layers: {zero_count}/{len(norms)}")

