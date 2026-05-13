#!/usr/bin/env python3
# tests/test_dataset.py
"""
실제 MimicGen HDF5로 dataset 동작 검증.
서버에서만 실행 가능 (데이터 경로 필요).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/NHNHOME/WORKSPACE/0526040052_A/sjLee/WMPO-JEPA/data"


def test_stage1_dataset():
    from data.dataset import Stage1Dataset
    ds = Stage1Dataset(
        data_root=DATA_ROOT, task="square",
        split="train", train_ratio=0.9,
        image_size=224, max_nodes=8,
    )
    assert len(ds) > 0

    sample = ds[0]
    assert sample["image"].shape        == (3, 224, 224)
    assert sample["graph_feat"].shape   == (8, 5)
    assert sample["node_pos"].shape     == (8, 3)
    assert sample["node_mask"].shape    == (8,)
    assert sample["node_mask"].sum()    == 2,  "square has 2 objects"
    assert sample["action"].shape       == (7,)
    print(f"  Stage1 sample OK  |  node_pos[0]: {sample['node_pos'][0]}")


def test_stage2_dataset():
    from data.dataset import Stage2Dataset
    ds = Stage2Dataset(
        data_root=DATA_ROOT, task="square",
        split="train", train_ratio=0.9, image_size=224,
    )
    assert len(ds) > 0
    s = ds[0]
    assert s["image_t"].shape    == (3, 224, 224)
    assert s["image_t1"].shape   == (3, 224, 224)
    assert s["action"].shape     == (7,)
    assert s["node_pos_t1"].shape == (8, 3)
    print(f"  Stage2 pairs: {len(ds)}")


def test_goal_graph_dataset():
    from data.dataset import GoalGraphDataset
    ds = GoalGraphDataset(data_root=DATA_ROOT, task="square", max_nodes=8)
    assert len(ds) == 300, f"Expected 300 demos, got {len(ds)}"
    goal = ds.random_goal()
    assert goal.shape == (8, 3)
    print(f"  GoalGraph OK  |  goal[0]: {goal[0]}")


def test_initial_state_dataset():
    from data.dataset import InitialStateDataset
    ds = InitialStateDataset(data_root=DATA_ROOT, task="square", image_size=224)
    assert len(ds) == 300
    s = ds[0]
    assert s["image"].shape == (3, 224, 224)
    print(f"  InitialState OK  |  demo_key: {s['demo_key']}")


def test_parse_object_positions():
    import numpy as np
    from data.dataset import parse_object_positions
    # square: 14-dim = 2 objects × 7
    obj = np.random.randn(14).astype(np.float64)
    pos = parse_object_positions(obj, "square")
    assert pos.shape == (2, 3), f"Expected (2,3), got {pos.shape}"
    # 첫 번째 객체 xyz
    assert np.allclose(pos[0], obj[0:3].astype(np.float32))
    # 두 번째 객체 xyz
    assert np.allclose(pos[1], obj[7:10].astype(np.float32))
    print(f"  parse_object_positions OK  |  pos: {pos}")


if __name__ == "__main__":
    print("=== Dataset tests ===")
    test_parse_object_positions()
    test_stage1_dataset()
    test_stage2_dataset()
    test_goal_graph_dataset()
    test_initial_state_dataset()
    print("\n✓ All dataset tests passed")