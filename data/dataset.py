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
        "total_dim":  14,
        "names": ["square", "peg"],
    },
    "coffee": {
        # dim=57: 8개×7=56 + 1(나머지) → 실제 조작 관련 객체 앞 3개만 사용
        "num_objects": 3,
        "object_dim": 7,
        "total_dim":  57,
        "names": ["coffee_pod", "machine_body", "machine_lid"],
    },
    "stack_three": {
        # dim=39: 5개×7=35 + 4(나머지) → 실제 블럭 3개만 사용
        "num_objects": 3,
        "object_dim": 7,
        "total_dim":  39,
        "names": ["cube_bottom", "cube_middle", "cube_top"],
    },
    "three_piece_assembly": {
        # dim=42: 6개×7=42 → 실제 조립 파트 3개(짝수 index: 0,2,4)
        "num_objects": 3,
        "object_dim": 7,
        "total_dim":  42,
        "names": ["piece_a", "piece_b", "piece_c"],
        "indices": [0, 2, 4],   # 사용할 객체 index
    },
}

HDF5_PATH_TEMPLATE = (
    "{root}/demos/core_datasets/{task}/"
    "demo_src_{task}_task_D0/demo.hdf5"
)

# RoboMimic 추가 task 설정
ROBOMIMIC_TASK_CONFIG = {
    "lift": {
        "num_objects": 1,
        "object_dim": 7,
        "total_dim":  10,   # 7 + 3 (gripper related)
        "names": ["cube"],
    },
    "can": {
        "num_objects": 2,
        "object_dim": 7,
        "total_dim":  14,
        "names": ["can", "bin"],
    },
    "square": {
        "num_objects": 2,
        "object_dim": 7,
        "total_dim":  14,
        "names": ["square", "peg"],
    },
}

ROBOMIMIC_HDF5_TEMPLATE = (
    "{root}/{task}/ph/image_v141.hdf5"
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
            # Normalize는 encoder 내부에서 backbone별로 적용 (DINOv2=ImageNet, SigLIP=[0.5,0.5,0.5])
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            # Normalize는 encoder 내부에서 backbone별로 적용
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Object state parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_object_positions(obj_vec: np.ndarray, task: str) -> np.ndarray:
    """
    obs/object 벡터 → (N, 3) position  float32
    각 객체: (x,y,z, qw,qx,qy,qz) → x,y,z만 사용
    indices가 있으면 해당 객체만 선택
    """
    cfg = TASK_OBJECT_CONFIG.get(task, {"num_objects": len(obj_vec)//7, "object_dim": 7})
    dim     = cfg["object_dim"]
    indices = cfg.get("indices", list(range(cfg["num_objects"])))
    positions = []
    for i in indices:
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
        """Demo 단위로 train/val 분리 — 같은 demo의 프레임이 섞이지 않음."""
        with h5py.File(self.hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys())

        # Demo 단위 분리
        n_train = int(len(demo_keys) * ratio)
        train_demos = demo_keys[:n_train]
        val_demos   = demo_keys[n_train:]
        target_demos = train_demos if self.split == "train" else val_demos

        pairs = []
        with h5py.File(self.hdf5_path, "r") as f:
            for dk in target_demos:
                T_len = f[f"data/{dk}/obs/agentview_image"].shape[0]
                for t in range(T_len):
                    pairs.append((dk, t))

        if self.split == "train":
            random.seed(42)
            random.shuffle(pairs)
        return pairs

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
        """Demo 단위로 train/val 분리."""
        with h5py.File(self.hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys())

        n_train = int(len(demo_keys) * ratio)
        train_demos = demo_keys[:n_train]
        val_demos   = demo_keys[n_train:]
        target_demos = train_demos if self.split == "train" else val_demos

        pairs = []
        with h5py.File(self.hdf5_path, "r") as f:
            for dk in target_demos:
                T_len = f[f"data/{dk}/obs/agentview_image"].shape[0]
                for t in range(T_len - 1):
                    pairs.append((dk, t))

        if self.split == "train":
            random.seed(42)
            random.shuffle(pairs)
        return pairs

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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Dataset (4개 task 합산)
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskStage1Dataset(Dataset):
    """
    4개 task의 Stage1Dataset을 합친 멀티태스크 데이터셋.
    각 task의 max_nodes를 통일 (가장 큰 값으로 pad).
    """

    TASKS = ["square", "coffee", "stack_three", "three_piece_assembly"]

    def __init__(
        self,
        data_root: str,
        tasks: Optional[List[str]] = None,
        split: str = "train",
        train_ratio: float = 0.9,
        image_size: int = 224,
        max_nodes: int = 8,
        grid_size: int = 32,
    ):
        self.tasks     = tasks or self.TASKS
        self.max_nodes = max_nodes
        self.datasets  = []

        for task in self.tasks:
            ds = Stage1Dataset(
                data_root=data_root,
                task=task,
                split=split,
                train_ratio=train_ratio,
                image_size=image_size,
                max_nodes=max_nodes,
                grid_size=grid_size,
            )
            self.datasets.append(ds)

        # flat index: (dataset_idx, sample_idx)
        self.index = []
        for ds_idx, ds in enumerate(self.datasets):
            base = ds.index if hasattr(ds, 'index') else list(range(len(ds)))
            for s_idx in range(len(ds)):
                self.index.append((ds_idx, s_idx))

        if split == "train":
            random.seed(42)
            random.shuffle(self.index)

        total = sum(len(ds) for ds in self.datasets)
        print(f"[MultiTask/{split}] {len(self.tasks)} tasks  "
              f"{total} samples total")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, s_idx = self.index[idx]
        return self.datasets[ds_idx][s_idx]


class MultiTaskStage2Dataset(Dataset):
    """
    4개 task의 Stage2Dataset을 합친 멀티태스크 데이터셋.
    """

    TASKS = ["square", "coffee", "stack_three", "three_piece_assembly"]

    def __init__(
        self,
        data_root: str,
        tasks: Optional[List[str]] = None,
        split: str = "train",
        train_ratio: float = 0.9,
        image_size: int = 224,
        max_nodes: int = 8,
        grid_size: int = 32,
        seq_len: int = 16,
    ):
        self.tasks = tasks or self.TASKS
        self.datasets = []

        for task in self.tasks:
            ds = Stage2Dataset(
                data_root=data_root,
                task=task,
                split=split,
                train_ratio=train_ratio,
                image_size=image_size,
                max_nodes=max_nodes,
                grid_size=grid_size,
                seq_len=seq_len,
            )
            self.datasets.append(ds)

        self.index = []
        for ds_idx, ds in enumerate(self.datasets):
            for s_idx in range(len(ds)):
                self.index.append((ds_idx, s_idx))

        if split == "train":
            random.seed(42)
            random.shuffle(self.index)

        total = sum(len(ds) for ds in self.datasets)
        print(f"[MultiTask/{split}] {len(self.tasks)} tasks  "
              f"{total} pairs total")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, s_idx = self.index[idx]
        return self.datasets[ds_idx][s_idx]


# ─────────────────────────────────────────────────────────────────────────────
# RoboMimic Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RoboMimicStage1Dataset(Dataset):
    """Stage 1 dataset for RoboMimic HDF5 files."""

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

        self.hdf5_path = ROBOMIMIC_HDF5_TEMPLATE.format(
            root=data_root.rstrip("/"), task=task)
        assert Path(self.hdf5_path).exists(), f"Not found: {self.hdf5_path}"

        self.cfg   = ROBOMIMIC_TASK_CONFIG.get(task, {"num_objects": 1, "object_dim": 7, "total_dim": 10})
        self.index = self._build_index(train_ratio)
        print(f"[RoboMimic/Stage1/{split}] {task}: {len(self.index)} frames")

    def _build_index(self, ratio: float) -> List[Tuple[str, int]]:
        pairs = []
        with h5py.File(self.hdf5_path, "r") as f:
            # RoboMimic은 mask/train, mask/valid 키가 있을 수 있음
            # 없으면 demo 단위 분리
            demo_keys = sorted(f["data"].keys())

        n_train = int(len(demo_keys) * ratio)
        target  = demo_keys[:n_train] if self.split == "train" else demo_keys[n_train:]

        with h5py.File(self.hdf5_path, "r") as f:
            for dk in target:
                T_len = f[f"data/{dk}/obs/agentview_image"].shape[0]
                for t in range(T_len):
                    pairs.append((dk, t))

        if self.split == "train":
            random.seed(42)
            random.shuffle(pairs)
        return pairs

    def __len__(self): return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dk, t = self.index[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            img = f[f"data/{dk}/obs/agentview_image"][t]   # (256,256,3)
            obj = f[f"data/{dk}/obs/object"][t]
            act = f[f"data/{dk}/actions"][t]

        image = self.transform(img)
        pos   = self._parse_positions(obj)
        nf, pp, mk = pad_to_max_nodes(pos, build_node_features(pos), self.max_nodes)

        return {
            "image":        image,
            "graph_feat":   torch.from_numpy(nf),
            "node_pos":     torch.from_numpy(pp),
            "node_mask":    torch.from_numpy(mk),
            "sparse_depth": torch.zeros(self.grid_size, self.grid_size),
            "action":       torch.tensor(act, dtype=torch.float32),
        }

    def _parse_positions(self, obj_vec: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        N   = cfg["num_objects"]
        dim = cfg["object_dim"]
        return np.stack([obj_vec[i*dim:i*dim+3].astype(np.float32)
                         for i in range(N)], axis=0)


class RoboMimicStage2Dataset(RoboMimicStage1Dataset):
    """Stage 2 (t, t+1) pair dataset for RoboMimic."""

    def __init__(self, seq_len: int = 16, **kwargs):
        super().__init__(**kwargs)
        # t+1이 유효한 쌍만
        self.index = [(dk, t) for dk, t in self.index if t > 0]
        # rebuild: need consecutive pairs
        self._rebuild_pairs(kwargs.get("train_ratio", 0.9))

    def _rebuild_pairs(self, ratio: float):
        pairs = []
        with h5py.File(self.hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys())
        n_train = int(len(demo_keys) * ratio)
        target  = demo_keys[:n_train] if self.split == "train" else demo_keys[n_train:]

        with h5py.File(self.hdf5_path, "r") as f:
            for dk in target:
                T_len = f[f"data/{dk}/obs/agentview_image"].shape[0]
                for t in range(T_len - 1):
                    pairs.append((dk, t))

        if self.split == "train":
            random.seed(42)
            random.shuffle(pairs)
        self.index = pairs
        print(f"[RoboMimic/Stage2/{self.split}] {self.task}: {len(self.index)} pairs")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dk, t = self.index[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            img_t  = f[f"data/{dk}/obs/agentview_image"][t]
            img_t1 = f[f"data/{dk}/obs/agentview_image"][t + 1]
            obj_t  = f[f"data/{dk}/obs/object"][t]
            obj_t1 = f[f"data/{dk}/obs/object"][t + 1]
            act    = f[f"data/{dk}/actions"][t]

        pos_t  = self._parse_positions(obj_t)
        pos_t1 = self._parse_positions(obj_t1)
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
# Combined MultiTask + RoboMimic Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CombinedStage1Dataset(Dataset):
    """MimicGen(4 tasks) + RoboMimic(lift, can, square) 합산."""

    def __init__(
        self,
        mimicgen_root: str,
        robomimic_root: str,
        mimicgen_tasks: Optional[List[str]] = None,
        robomimic_tasks: Optional[List[str]] = None,
        split: str = "train",
        train_ratio: float = 0.9,
        image_size: int = 224,
        max_nodes: int = 8,
        grid_size: int = 32,
    ):
        mimicgen_tasks  = mimicgen_tasks  or ["square", "coffee", "stack_three", "three_piece_assembly"]
        robomimic_tasks = robomimic_tasks or ["lift", "can", "square"]

        self.datasets = []

        for task in mimicgen_tasks:
            self.datasets.append(Stage1Dataset(
                data_root=mimicgen_root, task=task,
                split=split, train_ratio=train_ratio,
                image_size=image_size, max_nodes=max_nodes, grid_size=grid_size,
            ))

        for task in robomimic_tasks:
            self.datasets.append(RoboMimicStage1Dataset(
                data_root=robomimic_root, task=task,
                split=split, train_ratio=train_ratio,
                image_size=image_size, max_nodes=max_nodes, grid_size=grid_size,
            ))

        self.index = []
        for ds_idx, ds in enumerate(self.datasets):
            for s_idx in range(len(ds)):
                self.index.append((ds_idx, s_idx))

        if split == "train":
            random.seed(42)
            random.shuffle(self.index)

        total = sum(len(ds) for ds in self.datasets)
        print(f"[Combined/Stage1/{split}]  "
              f"MimicGen({len(mimicgen_tasks)}) + RoboMimic({len(robomimic_tasks)})  "
              f"= {total} samples")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        ds_idx, s_idx = self.index[idx]
        return self.datasets[ds_idx][s_idx]


class CombinedStage2Dataset(Dataset):
    """MimicGen(4 tasks) + RoboMimic(lift, can, square) 합산 Stage 2."""

    def __init__(
        self,
        mimicgen_root: str,
        robomimic_root: str,
        mimicgen_tasks: Optional[List[str]] = None,
        robomimic_tasks: Optional[List[str]] = None,
        split: str = "train",
        train_ratio: float = 0.9,
        image_size: int = 224,
        max_nodes: int = 8,
        grid_size: int = 32,
        seq_len: int = 16,
    ):
        mimicgen_tasks  = mimicgen_tasks  or ["square", "coffee", "stack_three", "three_piece_assembly"]
        robomimic_tasks = robomimic_tasks or ["lift", "can", "square"]

        self.datasets = []

        for task in mimicgen_tasks:
            self.datasets.append(Stage2Dataset(
                data_root=mimicgen_root, task=task,
                split=split, train_ratio=train_ratio,
                image_size=image_size, max_nodes=max_nodes,
                grid_size=grid_size, seq_len=seq_len,
            ))

        for task in robomimic_tasks:
            self.datasets.append(RoboMimicStage2Dataset(
                data_root=robomimic_root, task=task,
                split=split, train_ratio=train_ratio,
                image_size=image_size, max_nodes=max_nodes,
                grid_size=grid_size, seq_len=seq_len,
            ))

        self.index = []
        for ds_idx, ds in enumerate(self.datasets):
            for s_idx in range(len(ds)):
                self.index.append((ds_idx, s_idx))

        if split == "train":
            random.seed(42)
            random.shuffle(self.index)

        total = sum(len(ds) for ds in self.datasets)
        print(f"[Combined/Stage2/{split}]  "
              f"MimicGen({len(mimicgen_tasks)}) + RoboMimic({len(robomimic_tasks)})  "
              f"= {total} pairs")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        ds_idx, s_idx = self.index[idx]
        return self.datasets[ds_idx][s_idx]