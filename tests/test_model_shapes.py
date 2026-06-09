import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model.bbox_planner import PresenceAwareBBoxPlanner, MLPBaseline
from src.model.losses import masked_l1_loss, masked_iou_loss, overlap_loss


def make_batch(B=2, S=3, E=4, d_text=512):
    shot_emb = torch.randn(B, S, d_text)
    entity_emb = torch.randn(B, E, d_text)
    state_ids = torch.zeros(B, S, E, dtype=torch.long)
    presence = torch.ones(B, S, E, dtype=torch.long)
    relation_ids = torch.zeros(B, S, E, E, dtype=torch.long)
    return shot_emb, entity_emb, state_ids, presence, relation_ids


def test_planner_forward_shape():
    model = PresenceAwareBBoxPlanner(d_text=512, d_model=128, num_layers=2, num_heads=4)
    B, S, E = 2, 3, 4
    shot_emb, entity_emb, state_ids, presence, relation_ids = make_batch(B, S, E)
    out = model(shot_emb, entity_emb, state_ids, presence, relation_ids)
    assert out.shape == (B, S, E, 4), f"Expected {(B,S,E,4)}, got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Boxes should be in [0,1]"


def test_mlp_baseline_shape():
    model = MLPBaseline(d_text=512, d_model=128)
    B, S, E = 2, 3, 4
    shot_emb, entity_emb, state_ids, presence, relation_ids = make_batch(B, S, E)
    out = model(shot_emb, entity_emb, state_ids, presence, relation_ids)
    assert out.shape == (B, S, E, 4)


def test_partial_presence():
    """Test that absent entities don't affect active ones (presence masking)."""
    model = PresenceAwareBBoxPlanner(d_text=64, d_model=32, num_layers=1, num_heads=2)
    B, S, E = 1, 3, 3
    shot_emb = torch.randn(B, S, 64)
    entity_emb = torch.randn(B, E, 64)
    state_ids = torch.zeros(B, S, E, dtype=torch.long)
    presence = torch.tensor([[[1, 1, 0], [0, 1, 0], [1, 1, 1]]], dtype=torch.long)
    relation_ids = torch.zeros(B, S, E, E, dtype=torch.long)
    out = model(shot_emb, entity_emb, state_ids, presence, relation_ids)
    assert out.shape == (B, S, E, 4)


def test_l1_loss():
    pred = torch.rand(2, 3, 4, 4)
    tgt = torch.rand(2, 3, 4, 4)
    mask = torch.ones(2, 3, 4)
    loss = masked_l1_loss(pred, tgt, mask)
    assert loss.item() >= 0


def test_iou_loss():
    pred = torch.rand(2, 3, 4, 4)
    tgt = torch.rand(2, 3, 4, 4)
    mask = torch.ones(2, 3, 4)
    loss = masked_iou_loss(pred, tgt, mask)
    assert loss.item() >= 0


def test_overlap_loss():
    # Two boxes completely overlapping -> high overlap loss
    pred = torch.zeros(1, 1, 2, 4)
    pred[0, 0, 0] = torch.tensor([0.1, 0.1, 0.9, 0.9])  # cxcywh
    pred[0, 0, 1] = torch.tensor([0.1, 0.1, 0.9, 0.9])
    presence = torch.ones(1, 1, 2, dtype=torch.long)
    loss = overlap_loss(pred, presence, tau=0.25)
    assert loss.item() > 0


if __name__ == "__main__":
    test_planner_forward_shape()
    test_mlp_baseline_shape()
    test_partial_presence()
    test_l1_loss()
    test_iou_loss()
    test_overlap_loss()
    print("All model shape tests passed!")
