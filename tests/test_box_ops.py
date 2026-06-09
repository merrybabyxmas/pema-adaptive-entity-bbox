import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from src.utils.box_ops import xyxy_to_cxcywh, cxcywh_to_xyxy, box_iou, box_giou


def test_roundtrip_torch():
    boxes = torch.tensor([[0.1, 0.2, 0.5, 0.8]])
    cx = xyxy_to_cxcywh(boxes)
    back = cxcywh_to_xyxy(cx)
    assert torch.allclose(boxes, back, atol=1e-5)


def test_roundtrip_numpy():
    boxes = np.array([[0.1, 0.2, 0.5, 0.8]])
    cx = xyxy_to_cxcywh(boxes)
    back = cxcywh_to_xyxy(cx)
    assert np.allclose(boxes, back, atol=1e-5)


def test_iou_perfect():
    box = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    iou = box_iou(box, box)
    assert torch.allclose(iou, torch.tensor([1.0]), atol=1e-5)


def test_iou_no_overlap():
    b1 = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    b2 = torch.tensor([[0.5, 0.5, 0.9, 0.9]])
    iou = box_iou(b1, b2)
    assert torch.allclose(iou, torch.tensor([0.0]), atol=1e-5)


def test_giou_perfect():
    box = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    giou = box_giou(box, box)
    assert torch.allclose(giou, torch.tensor([1.0]), atol=1e-5)


def test_cxcywh_values():
    boxes = torch.tensor([[0.1, 0.2, 0.5, 0.8]])
    cx = xyxy_to_cxcywh(boxes)
    assert abs(cx[0, 0].item() - 0.3) < 1e-5   # cx
    assert abs(cx[0, 1].item() - 0.5) < 1e-5   # cy
    assert abs(cx[0, 2].item() - 0.4) < 1e-5   # w
    assert abs(cx[0, 3].item() - 0.6) < 1e-5   # h


if __name__ == "__main__":
    test_roundtrip_torch()
    test_roundtrip_numpy()
    test_iou_perfect()
    test_iou_no_overlap()
    test_giou_perfect()
    test_cxcywh_values()
    print("All box_ops tests passed!")
