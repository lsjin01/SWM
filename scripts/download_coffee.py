"""Coffee task 관련 파일 다운로드 (HuggingFace fangqi/WMPO)"""
from huggingface_hub import snapshot_download
import os

# 다운로드 대상 디렉토리
CKPTS_DIR = "/scratch/mip25/sjLee/SWM/ckpts/checkpoint_files"
DATA_DIR  = "/scratch/mip25/sjLee/SWM/data/wmpo_data"

os.makedirs(CKPTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("=== Downloading coffee SFT model ===")
snapshot_download(
    repo_id="fangqi/WMPO",
    repo_type="model",
    local_dir=CKPTS_DIR,
    local_dir_use_symlinks=False,
    allow_patterns=["checkpoint_files/SFT_models/coffee/**"],
)

print("=== Downloading coffee WMPO models ===")
snapshot_download(
    repo_id="fangqi/WMPO",
    repo_type="model",
    local_dir=CKPTS_DIR,
    local_dir_use_symlinks=False,
    allow_patterns=["checkpoint_files/WMPO_models/coffee/**"],
)

print("=== Downloading coffee data files ===")
snapshot_download(
    repo_id="fangqi/WMPO",
    repo_type="model",
    local_dir=DATA_DIR,
    local_dir_use_symlinks=False,
    allow_patterns=["data_files/core_datasets/coffee/**",
                    "data_files/core_train_configs/*coffee*",
                    "data_files/statistics/**"],
)

print("=== Download complete ===")
