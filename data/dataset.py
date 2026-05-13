# data/dataset.py
"""
SWM Dataset — MimicGen HDF5 기반
obs/object shape 기준:
  square:               (T, 14) = square(7) + peg(7)
  coffee:               (T, ?)
  stack_three:          (T, ?)
  three_piece_assembly: (T, ?)
각 객체: (x, y, z, qw, qx, qy, qz) → position만 사용 (x, y, z)
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# Task별 object 구성
# ─────────────────────────────────────────────────────────────────────────────

TASK_OBJECT_CONFIG = {
    "square": {
        "num_objects": 2,
        "object_dim": 7,
        "names": ["square", "peg"],
    },
    "coffee": {
        "num_objects": 2,
        "object_dim": 7,
        "names": ["coffee_pod", "machine"],
    },
    "stack_three": {
        "num_objects": 3,
        "object_dim": 7,
        "names": ["cube_bottom", "cube_middle", "cube_top"],
    },
    "three_piece_assembly": {
        "num_objects": 3,
        "object_dim": 7,
        "names": ["piece_a", "piece_b", "piece_c"],
    },
}

HDF5_PATH_TEMPLATE = (
    "{root}/demos/core_datasets/{task}/"
    "demo_src_{task}_task_D0/demo.hdf5"
)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def make_transforms(image_size: int, split: str = "train") -> T.Compose:
    if split == "train":
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Object state parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_object_positions(obj_vec: np.ndarray, task: str) -> np.ndarray:
    """
    obs/object 벡터 → (N, 3) position  float32
    각 객체: (x,y,z, qw,qx,qy,qz) → x,y,z만 사용
    """
    cfg = TASK_OBJECT_CONFIG.get(task, {"num_objects": len(obj_vec)//7, "object_dim": 7})
    N, dim = cfg["num_objects"], cfg["object_dim"]
    positions = []
    for i in range(N):
        xyz = obj_vec[i * dim : i * dim + 3].astype(np.float32)
        positions.append(xyz)
    return np.stack(positions, axis=0)   # (N, 3)


def build_node_features(positions: np.ndarray) -> np.ndarray:
    """
    (N,3) positions → (N,5) topology-only node features
    [normalized_class_id, 1.0, 0, 0, 0]
    (실제 위치 정보는 node_pos에만 담음 — topology/edge 정보만 포함)
    """
    N = len(positions)
    feat = np.zeros((N, 5), dtype=np.float32)
    feat[:, 0] = np.arange(N) / max(N - 1, 1)   # normalized class id
    feat[:, 1] = 1.0                              # valid flag
    return feat


def pad_to_max_nodes(
    positions: np.ndarray,
    node_feat: np.ndarray,
    max_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (padded_feat, padded_pos, mask)."""
    N = min(len(positions), max_nodes)
    pf   = np.zeros((max_nodes, 5),  dtype=np.float32)
    pp   = np.zeros((max_nodes, 3),  dtype=np.float32)
    mask = np.zeros(max_nodes,       dtype=bool)
    pf[:N]   = node_feat[:N]
    pp[:N]   = positions[:N]
    mask[:N] = True
    return pf, pp, mask


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Stage1Dataset(Dataset):
    def __init__(
        self,
        data_root: str,
        task: str = "square",
        split: str = "train",
        train_ratio: float = 0.9,
        image_size: int = 224,
        max_nodes: int = 8,
        grid_size: int = 32,
        **kwargs,
    ):
        self.task      = task
        self.split     = split
        self.max_nodes = max_nodes
        self.grid_size = grid_size
        self.transform = make_transforms(image_size, split)
        self.hdf5_path = HDF5_PATH_TEMPLATE.format(
            root=data_root.rstrip("/"), task=task)
        assert Path(self.hdf5_path).exists(), f"Not found: {self.hdf5_path}"
        self.index = self._build_index(train_ratio)
        print(f"[Stage1/{split}] {task}: {len(self.index)} frames")

    def _build_index(self, ratio: float) -> List[Tuple[str, int]]:
        pairs = []
        with h5py.File(self.hdf5_path, "r") as f:
            for dk in sorted(f["data"].keys()):
                T_len = f[f"data/{dk}/obs/agentview_image"].shape[0]
                for t in range(T_len):
                    pairs.append((dk, t))
        random.seed(42); random.shuffle(pairs)
        n = int(len(pairs) * ratio)
        return pairs[:n] if self.split == "train" else pairs[n:]

    def __len__(self): return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dk, t = self.index[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            img = f[f"data/{dk}/obs/agentview_image"][t]
            obj = f[f"data/{dk}/obs/object"][t]
            act = f[f"data/{dk}/actions"][t]

        image = self.transform(img)
        pos   = parse_object_positions(obj, self.task)
        pf, pp, mk = pad_to_max_nodes(pos, build_node_features(pos), self.max_nodes)

        return {
            "image":        image,
            "graph_feat":   torch.from_numpy(pf),
            "node_pos":     torch.from_numpy(pp),
            "node_mask":    torch.from_numpy(mk),
            "sparse_depth": torch.zeros(self.grid_size, self.grid_size),
            "action":       torch.tensor(act, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Stage2Dataset(Dataset):
    def __init__(
        self,
        data_root: str,
        task: str = "square",
        split: str = "train",
        train_ratio: float = 0.9,
        image_size: int = 224,
        max_nodes: int = 8,
        grid_size: int = 32,
        seq_len: int = 16,
        **kwargs,
    ):
        self.task      = task
        self.split     = split
        self.max_nodes = max_nodes
        self.grid_size = grid_size
        self.transform = make_transforms(image_size, split)
        self.hdf5_path = HDF5_PATH_TEMPLATE.format(
            root=data_root.rstrip("/"), task=task)
        assert Path(self.hdf5_path).exists(), f"Not found: {self.hdf5_path}"
        self.index = self._build_index(train_ratio)
        print(f"[Stage2/{split}] {task}: {len(self.index)} pairs")

    def _build_index(self, ratio: float) -> List[Tuple[str, int]]:
        pairs = []
        with h5py.File(self.hdf5_path, "r") as f:
            for dk in sorted(f["data"].keys()):
                T_len = f[f"data/{dk}/obs/agentview_image"].shape[0]
                for t in range(T_len - 1):
                    pairs.append((dk, t))
        random.seed(42); random.shuffle(pairs)
        n = int(len(pairs) * ratio)
        return pairs[:n] if self.split == "train" else pairs[n:]

    def __len__(self): return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dk, t = self.index[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            img_t  = f[f"data/{dk}/obs/agentview_image"][t]
            img_t1 = f[f"data/{dk}/obs/agentview_image"][t + 1]
            obj_t  = f[f"data/{dk}/obs/object"][t]
            obj_t1 = f[f"data/{dk}/obs/object"][t + 1]
            act    = f[f"data/{dk}/actions"][t]

        pos_t  = parse_object_positions(obj_t,  self.task)
        pos_t1 = parse_object_positions(obj_t1, self.task)
        nf_t,  pp_t,  mk_t  = pad_to_max_nodes(pos_t,  build_node_features(pos_t),  self.max_nodes)
        nf_t1, pp_t1, mk_t1 = pad_to_max_nodes(pos_t1, build_node_features(pos_t1), self.max_nodes)

        return {
            "image_t":         self.transform(img_t),
            "graph_feat_t":    torch.from_numpy(nf_t),
            "node_pos_t":      torch.from_numpy(pp_t),
            "node_mask_t":     torch.from_numpy(mk_t),
            "action":          torch.tensor(act, dtype=torch.float32),
            "image_t1":        self.transform(img_t1),
            "graph_feat_t1":   torch.from_numpy(nf_t1),
            "node_pos_t1":     torch.from_numpy(pp_t1),
            "node_mask_t1":    torch.from_numpy(mk_t1),
            "sparse_depth_t1": torch.zeros(self.grid_size, self.grid_size),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Goal Graph Dataset (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

class GoalGraphDataset(Dataset):
    """
    MimicGen demos는 모두 성공 trajectory.
    각 demo 마지막 프레임 → Goal Graph G*_T
    """

    def __init__(self, data_root: str, task: str = "square", max_nodes: int = 8):
        self.task      = task
        self.max_nodes = max_nodes
        self.hdf5_path = HDF5_PATH_TEMPLATE.format(
            root=data_root.rstrip("/"), task=task)
        assert Path(self.hdf5_path).exists()
        self.goals = self._preload()
        print(f"GoalGraphDataset: {len(self.goals)} goals  ({task})")

    def _preload(self) -> List[torch.Tensor]:
        goals = []
        with h5py.File(self.hdf5_path, "r") as f:
            for dk in sorted(f["data"].keys()):
                T_len = f[f"data/{dk}/obs/object"].shape[0]
                obj   = f[f"data/{dk}/obs/object"][T_len - 1]
                pos   = parse_object_positions(obj, self.task)
                _, pp, _ = pad_to_max_nodes(pos, build_node_features(pos), self.max_nodes)
                goals.append(torch.from_numpy(pp))
        return goals

    def __len__(self): return len(self.goals)
    def __getitem__(self, idx): return self.goals[idx]
    def random_goal(self): return self.goals[random.randint(0, len(self.goals)-1)]


# ─────────────────────────────────────────────────────────────────────────────
# Initial State Dataset (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

class InitialStateDataset(Dataset):
    """Stage 3 rollout의 초기 상태 (각 demo 첫 프레임)."""

    def __init__(
        self,
        data_root: str,
        task: str = "square",
        image_size: int = 224,
        max_nodes: int = 8,
    ):
        self.task      = task
        self.max_nodes = max_nodes
        self.transform = make_transforms(image_size, split="val")
        self.hdf5_path = HDF5_PATH_TEMPLATE.format(
            root=data_root.rstrip("/"), task=task)
        assert Path(self.hdf5_path).exists()
        with h5py.File(self.hdf5_path, "r") as f:
            self.demo_keys = sorted(f["data"].keys())
        print(f"InitialStateDataset: {len(self.demo_keys)} episodes  ({task})")

    def __len__(self): return len(self.demo_keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dk = self.demo_keys[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            img = f[f"data/{dk}/obs/agentview_image"][0]
            obj = f[f"data/{dk}/obs/object"][0]
        pos = parse_object_positions(obj, self.task)
        nf, pp, mk = pad_to_max_nodes(pos, build_node_features(pos), self.max_nodes)
        return {
            "image":      self.transform(img),
            "graph_feat": torch.from_numpy(nf),
            "node_pos":   torch.from_numpy(pp),
            "node_mask":  torch.from_numpy(mk),
            "demo_key":   dk,
        }
