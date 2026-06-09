import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import tempfile
import pytest
from src.data.dataset import BBoxPlannerDataset
from src.data.collate import collate_fn
from torch.utils.data import DataLoader


def make_sample(sample_id="s0", video_id="v0", n_shots=3, entities=None):
    if entities is None:
        entities = ["cat", "dog"]
    E = len(entities)
    shots = []
    for s in range(n_shots):
        active = entities
        shots.append({
            "shot_id": s,
            "prompt": f"A {' and '.join(active)} in a park.",
            "background": "park",
            "active_entities": active,
            "states": {e: "initial" if s == 0 else "stay" for e in active},
            "relations": [["cat", "beside", "dog"]] if len(active) > 1 else [],
            "boxes": {e: [0.1 + i*0.3, 0.2, 0.4 + i*0.3, 0.8]
                      for i, e in enumerate(active)},
            "quality": {f"{e}_det_score": 0.85 for e in active},
            "keyframe_path": None,
        })
    return {
        "sample_id": sample_id,
        "video_id": video_id,
        "entity_vocab": entities,
        "presence": [[1]*E]*n_shots,
        "shots": shots,
    }


def make_tmp_jsonl(n=5):
    samples = [make_sample(f"s{i}", f"v{i}") for i in range(n)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
        return f.name


def test_dataset_len():
    path = make_tmp_jsonl(10)
    ds = BBoxPlannerDataset(path)
    assert len(ds) == 10


def test_dataset_shapes():
    path = make_tmp_jsonl(5)
    ds = BBoxPlannerDataset(path, max_shots=5, max_entities=5)
    item = ds[0]
    assert item["presence"].shape == (5, 5)
    assert item["target_boxes_cxcywh"].shape == (5, 5, 4)
    assert item["target_mask"].shape == (5, 5)
    assert item["state_ids"].shape == (5, 5)
    assert item["relation_ids"].shape == (5, 5, 5)


def test_dataloader():
    path = make_tmp_jsonl(8)
    ds = BBoxPlannerDataset(path, max_shots=5, max_entities=5)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))
    assert batch["presence"].shape == (4, 5, 5)
    assert batch["target_boxes_cxcywh"].shape == (4, 5, 5, 4)


def test_boxes_normalized():
    path = make_tmp_jsonl(5)
    ds = BBoxPlannerDataset(path)
    for i in range(len(ds)):
        item = ds[i]
        boxes = item["target_boxes_xyxy"]
        mask = item["target_mask"].bool()
        if mask.any():
            valid = boxes[mask]
            assert (valid >= 0).all() and (valid <= 1).all()


if __name__ == "__main__":
    test_dataset_len()
    test_dataset_shapes()
    test_dataloader()
    test_boxes_normalized()
    print("All dataset tests passed!")
