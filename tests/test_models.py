# tests/test_models.py
"""Smoke tests — run without GPU or real data."""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_encoder_forward():
    from models.encoder import SWMEncoder, GraphEncoder
    # Test graph branch + projection only (no pretrained download)
    enc = SWMEncoder.__new__(SWMEncoder)
    nn.Module.__init__(enc)
    enc.use_graph = True
    enc.graph_enc = GraphEncoder(5, 64, 128)
    # Stub visual backbone
    class _FakeVisual(nn.Module):
        embed_dim = 256
        def forward(self, x): return torch.randn(x.shape[0], 256)
    enc.visual = _FakeVisual()
    enc.proj = nn.Sequential(nn.Linear(256+128, 256), nn.LayerNorm(256))
    enc.latent_dim = 256
    B, N = 2, 4
    image      = torch.randn(B, 3, 224, 224)
    graph_feat = torch.randn(B, N, 5)
    node_mask  = torch.ones(B, N, dtype=torch.bool)
    z = enc(image, graph_feat, node_mask)
    assert z.shape == (B, 256), f"Expected (2,256), got {z.shape}"


def test_heads_forward():
    from models.heads import SWMHeads
    heads = SWMHeads(latent_dim=256, max_nodes=4,
                     grid_size=8, hidden_dims=[128, 64])
    z = torch.randn(2, 256)
    gp, dp = heads(z)
    assert gp.shape == (2, 4, 3)
    assert dp.shape == (2, 8, 8)


def test_transition_forward():
    from models.transition import SWMTransition
    trans = SWMTransition(latent_dim=256, action_dim=7,
                          hidden_dim=256, num_layers=2, num_heads=4)
    z  = torch.randn(2, 256)
    a  = torch.randn(2, 7)
    z2 = trans(z, a)
    assert z2.shape == (2, 256)


def test_transition_rollout():
    from models.transition import SWMTransition
    trans = SWMTransition(latent_dim=256, action_dim=7,
                          hidden_dim=256, num_layers=2, num_heads=4)
    z0      = torch.randn(4, 256)
    actions = torch.randn(4, 8, 7)
    traj    = trans.rollout(z0, actions, tf_ratio=0.0)
    assert traj.shape == (4, 8, 256)


def test_grpo_reward():
    from utils.grpo import compute_graph_distance_reward
    pred  = torch.randn(8, 4, 3)
    goal  = torch.randn(4, 3)
    mask  = torch.ones(4, dtype=torch.bool)
    r     = compute_graph_distance_reward(pred, goal, mask)
    assert r.shape == (8,)
    assert (r <= 0).all(), "Reward should be non-positive"


def test_grpo_loss():
    from utils.grpo import compute_grpo_loss
    B, T, K = 8, 4, 7
    lp     = torch.randn(B, T, K)
    old_lp = torch.randn(B, T, K)
    r      = torch.randn(B)
    loss, info = compute_grpo_loss(lp, old_lp, r)
    assert loss.isfinite()
    assert "reward/mean" in info


def test_graph_utils():
    import numpy as np
    from data.graph_utils import detections_to_graph, graph_distance
    boxes  = np.array([[10,10,50,50],[60,60,100,100]], dtype=np.float32)
    labels = ["cube", "stick"]
    g = detections_to_graph(boxes, labels, None, (128, 128))
    assert g.num_nodes == 2
    assert g.node_positions.shape == (2, 3)

    pred = torch.randn(4, 2, 3)
    goal = torch.randn(2, 3)
    mask = torch.ones(2, dtype=torch.bool)
    from utils.grpo import compute_graph_distance_reward
    r = compute_graph_distance_reward(pred, goal, mask)
    assert r.shape == (4,)
