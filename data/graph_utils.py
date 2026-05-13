# data/graph_utils.py
"""
Detection → Graph builder.
Converts object bounding boxes + class labels into a graph
with node features (position, size, class) and edges (spatial relations).
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SceneGraph:
    """Lightweight scene graph for a single frame."""
    # Node features: [cx, cy, w, h, class_id]  shape (N, 5)
    node_features: torch.Tensor
    # Edge index: shape (2, E)
    edge_index: torch.Tensor
    # Node positions (cx, cy, depth) in normalised coords  shape (N, 3)
    node_positions: torch.Tensor
    # Number of nodes
    num_nodes: int
    # Class labels  shape (N,)
    labels: List[str] = field(default_factory=list)


def detections_to_graph(
    boxes: np.ndarray,          # (N, 4)  xyxy format, pixel coords
    labels: List[str],
    depths: Optional[np.ndarray],   # (N,) depth at box center, None if unavailable
    image_hw: Tuple[int, int],
    class_vocab: Optional[dict] = None,
) -> SceneGraph:
    """
    Build a SceneGraph from raw detections.

    Args:
        boxes:      (N,4) bounding boxes in xyxy pixel coords
        labels:     list of N class label strings
        depths:     (N,) depth value at each box centre (can be None)
        image_hw:   (H, W) of the source image for normalisation
        class_vocab: optional {label: int} mapping; auto-built if None
    """
    H, W = image_hw
    N = len(boxes)

    if class_vocab is None:
        unique = sorted(set(labels))
        class_vocab = {c: i for i, c in enumerate(unique)}

    class_ids = np.array([class_vocab.get(l, 0) for l in labels], dtype=np.float32)

    # Normalised cx, cy, w, h
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H

    node_features = torch.tensor(
        np.stack([cx, cy, w, h, class_ids], axis=1),
        dtype=torch.float32
    )  # (N, 5)

    # Node positions: (cx, cy, depth)
    if depths is None:
        depths = np.zeros(N, dtype=np.float32)
    node_positions = torch.tensor(
        np.stack([cx, cy, depths], axis=1),
        dtype=torch.float32
    )  # (N, 3)

    # Fully-connected edges (excluding self-loops)
    if N > 1:
        src, dst = zip(*[(i, j) for i in range(N) for j in range(N) if i != j])
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return SceneGraph(
        node_features=node_features,
        edge_index=edge_index,
        node_positions=node_positions,
        num_nodes=N,
        labels=labels,
    )


def pad_graph(graph: SceneGraph, max_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad node_features and node_positions to max_nodes with zeros.
    Returns (padded_features, mask) where mask[i]=1 for real nodes.
    """
    N = graph.num_nodes
    feat_dim = graph.node_features.shape[1]
    pos_dim  = graph.node_positions.shape[1]

    pad_feat = torch.zeros(max_nodes, feat_dim)
    pad_pos  = torch.zeros(max_nodes, pos_dim)
    mask     = torch.zeros(max_nodes, dtype=torch.bool)

    n = min(N, max_nodes)
    pad_feat[:n] = graph.node_features[:n]
    pad_pos[:n]  = graph.node_positions[:n]
    mask[:n]     = True

    return pad_feat, pad_pos, mask


def graph_distance(
    pred_positions: torch.Tensor,   # (N, 3)
    goal_positions: torch.Tensor,   # (N, 3)
    mask: Optional[torch.Tensor] = None,  # (N,) bool
) -> torch.Tensor:
    """
    Dense reward: negative mean L2 distance between predicted and goal positions.
    Returns scalar tensor in [−∞, 0]; closer to 0 = better.
    """
    diff = (pred_positions - goal_positions) ** 2   # (N, 3)
    dist = diff.sum(dim=-1).sqrt()                  # (N,)
    if mask is not None:
        dist = dist[mask]
    return -dist.mean()
