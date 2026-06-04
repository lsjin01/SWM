#!/usr/bin/env python3
"""
Stage 3 체크포인트(.pt, vla state_dict) → HF 포맷 디렉토리로 export.
verl 평가(examples/.../evaluate.sh)는 HF 모델 디렉토리(TARGET_MODEL_PATH)를 요구하므로
GRPO로 학습된 정책을 평가하려면 이 변환이 필요하다.

사용:
  python scripts/export_stage3_to_hf.py \
    --ckpt   outputs/stage3/<exp>/square/best.pt \
    --base   ckpts/SFT_models/square/checkpoint_files/SFT_models/square \
    --out    ckpts/exported/<exp>_square
"""
import os, sys, json, shutil, argparse
from pathlib import Path

sys.path.insert(0, '/opt/conda/envs/wmpo/lib/python3.11/site-packages')
sys.path.insert(0, '/home/miplab1/sjLee/swm')
sys.path.insert(0, '/home/miplab1/sjLee/WMPO')
sys.path.insert(0, '/home/mipstu/jiPark/openvla-oft/experiments/robot')

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# SFT 베이스에서 export 디렉토리로 복사할 보조 파일 (커스텀 코드/토크나이저/통계)
AUX_FILES = [
    "configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py",
    "preprocessor_config.json", "processor_config.json", "dataset_statistics.json",
    "generation_config.json", "added_tokens.json", "special_tokens_map.json",
    "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="stage3 best.pt (vla state_dict 포함)")
    ap.add_argument("--base", required=True, help="SFT 베이스 HF 디렉토리")
    ap.add_argument("--out",  required=True, help="export 대상 디렉토리")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[export] base VLA 로드: {args.base}")
    # update_auto_map: config.json의 auto_map을 OpenVLA 클래스로 교정
    try:
        from verl.utils.openvla_utils import update_auto_map
        update_auto_map(args.base)
    except Exception as e:
        print(f"[export] update_auto_map skip: {e}")

    vla = AutoModelForVision2Seq.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )

    print(f"[export] stage3 ckpt 로드: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("vla", ckpt)
    missing, unexpected = vla.load_state_dict(state, strict=False)
    print(f"[export] load_state_dict  missing={len(missing)}  unexpected={len(unexpected)}")

    print(f"[export] save_pretrained → {out}")
    vla.save_pretrained(out, safe_serialization=True)

    # 보조 파일 복사 (커스텀 코드/토크나이저/통계)
    for f in AUX_FILES:
        src = Path(args.base) / f
        if src.exists():
            shutil.copy2(src, out / f)
    print(f"[export] aux files copied")
    print(f"[export] DONE: {out}")


if __name__ == "__main__":
    main()
