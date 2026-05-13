# data/dataset.py
"""
SWM Dataset — wraps MimicGen / Robosuite HDF5 trajectories.

Each sample for Stage 1:
    obs_t, graph_t, node_positions_t, sparse_depth_t

Each sample for Stage 2:
    obs_t, graph_t, action_t, obs_t1, graph_t1, node_positions_t1, sparse_depth_t1

Each sample for Stage 3:
    initial state z_0, instruction, goal_graph G*_T
"""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from data.graph_utils import detections_to_graph, pad_graph, SceneGraph


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
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Stage1Dataset(Dataset):
    """
    Single-frame dataset for Encoder training.
    Loads (image, graph_topology, node_positions, sparse_depth).
    """

    def __init__(
        self,
        data_root: str,
        task: str = "square",
        split: str = "train",
        train_ratio: float = 0.9,
        image_size: int = 224,
        max_nodes: int = 8,
        grid_size: int = 32,
    ):
        self.data_root = Path(data_root)
        self.task = task
        self.split = split
        self.image_size = image_size
        self.max_nodes = max_nodes
        self.grid_size = grid_size
        self.transform = make_transforms(image_size, split)

        self.frames = self._load_index(train_ratio)

    def _load_index(self, train_ratio: float) -> List[Dict]:
        """Build a flat list of (hdf5_path, traj_key, timestep) entries."""
        hdf5_files = sorted((self.data_root / self.task).glob("*.hdf5"))
        all_frames = []
        for f in hdf5_files:
            with h5py.File(f, "r") as h:
                for traj_key in h["data"].keys():
                    T_len = h["data"][traj_key]["obs"]["agentview_rgb"].shape[0]
                    for t in range(T_len):
                        all_frames.append({
                            "hdf5": str(f),
                            "traj": traj_key,
                            "t": t,
                        })

        random.seed(42)
        random.shuffle(all_frames)
        n_train = int(len(all_frames) * train_ratio)
        if self.split == "train":
            return all_frames[:n_train]
        else:
            return all_frames[n_train:]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.frames[idx]
        with h5py.File(entry["hdf5"], "r") as h:
            traj = h["data"][entry["traj"]]
            t = entry["t"]

            # Image
            img = traj["obs"]["agentview_rgb"][t]   # (H, W, 3) uint8

            # Object states (ground-truth positions as proxy for Graph)
            obj_pos = traj["obs"]["object"][t]       # varies per task

            # Actions (not used in Stage 1 but stored for Stage 2)
            action = traj["actions"][t]              # (7,)

        image = self.transform(img)                  # (3, H, W)

        # Build pseudo-graph from object positions
        # In real use: replace with actual detection outputs
        graph_feat, node_pos, node_mask = self._build_graph(obj_pos)

        # Build sparse depth (pseudo: zeros if depth sensor unavailable)
        sparse_depth = self._build_sparse_depth(img)

        return {
            "image":        image,                   # (3, H, W)
            "graph_feat":   graph_feat,              # (max_nodes, 5)
            "node_pos":     node_pos,                # (max_nodes, 3)
            "node_mask":    node_mask,               # (max_nodes,) bool
            "sparse_depth": sparse_depth,            # (grid_size, grid_size)
            "action":       torch.tensor(action, dtype=torch.float32),
        }

    def _build_graph(
        self, obj_pos: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build padded graph from raw object position array.
        Each object: (x, y, z, quat_w, quat_x, quat_y, quat_z) → use xyz only.
        """
        # Reshape to (N_objects, 7) — task-dependent
        try:
            positions = obj_pos.reshape(-1, 7)[:, :3]  # (N, 3)
        except Exception:
            positions = obj_pos.reshape(-1, 3)

        N = min(len(positions), self.max_nodes)
        node_feat = torch.zeros(self.max_nodes, 5)
        node_pos  = torch.zeros(self.max_nodes, 3)
        node_mask = torch.zeros(self.max_nodes, dtype=torch.bool)

        node_feat[:N, :3] = torch.tensor(positions[:N], dtype=torch.float32)
        node_pos[:N]      = torch.tensor(positions[:N], dtype=torch.float32)
        node_mask[:N]     = True

        return node_feat, node_pos, node_mask

    def _build_sparse_depth(self, img: np.ndarray) -> torch.Tensor:
        """Placeholder: returns zero depth grid. Replace with real depth."""
        return torch.zeros(self.grid_size, self.grid_size)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Stage2Dataset(Stage1Dataset):
    """
    Sequential dataset for Transition training.
    Returns (frame_t, action_t, frame_t+1) pairs.
    """

    def __init__(self, seq_len: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        # Re-index as (traj, t) pairs where t+1 is valid
        self.pairs = [(e["hdf5"], e["traj"], e["t"])
                      for e in self.frames if e["t"] > 0]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hdf5_path, traj_key, t = self.pairs[idx]
        with h5py.File(hdf5_path, "r") as h:
            traj = h["data"][traj_key]
            img_t   = traj["obs"]["agentview_rgb"][t]
            img_t1  = traj["obs"]["agentview_rgb"][t + 1]
            obj_t   = traj["obs"]["object"][t]
            obj_t1  = traj["obs"]["object"][t + 1]
            action  = traj["actions"][t]

        image_t  = self.transform(img_t)
        image_t1 = self.transform(img_t1)

        gf_t,  np_t,  nm_t  = self._build_graph(obj_t)
        gf_t1, np_t1, nm_t1 = self._build_graph(obj_t1)
        sd_t1 = self._build_sparse_depth(img_t1)

        return {
            # t
            "image_t":       image_t,
            "graph_feat_t":  gf_t,
            "node_pos_t":    np_t,
            "node_mask_t":   nm_t,
            # action
            "action": torch.tensor(action, dtype=torch.float32),
            # t+1  (supervision targets)
            "image_t1":        image_t1,
            "graph_feat_t1":   gf_t1,
            "node_pos_t1":     np_t1,
            "node_mask_t1":    nm_t1,
            "sparse_depth_t1": sd_t1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Goal Graph loader
# ─────────────────────────────────────────────────────────────────────────────

class GoalGraphDataset(Dataset):
    """
    Loads goal graphs from successful trajectories' last frames.
    Used in Stage 3 GRPO reward computation.
    """

    def __init__(
        self,
        success_traj_dir: str,
        task: str = "square",
        max_nodes: int = 8,
    ):
        self.dir = Path(success_traj_dir)
        self.task = task
        self.max_nodes = max_nodes
        self.entries = self._index()

    def _index(self) -> List[Dict]:
        entries = []
        for f in (self.dir / self.task).glob("*.hdf5"):
            with h5py.File(f, "r") as h:
                for traj_key in h["data"].keys():
                    T_len = h["data"][traj_key]["obs"]["object"].shape[0]
                    entries.append({
                        "hdf5": str(f),
                        "traj": traj_key,
                        "t_last": T_len - 1,
                    })
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> torch.Tensor:
        e = self.entries[idx]
        with h5py.File(e["hdf5"], "r") as h:
            obj = h["data"][e["traj"]]["obs"]["object"][e["t_last"]]
        try:
            positions = obj.reshape(-1, 7)[:, :3]
        except Exception:
            positions = obj.reshape(-1, 3)
        N = min(len(positions), self.max_nodes)
        goal = torch.zeros(self.max_nodes, 3)
        goal[:N] = torch.tensor(positions[:N], dtype=torch.float32)
        return goal   # (max_nodes, 3)
