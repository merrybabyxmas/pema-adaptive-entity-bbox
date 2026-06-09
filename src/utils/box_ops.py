import torch
import numpy as np


def xyxy_to_cxcywh(boxes):
    """[x1,y1,x2,y2] -> [cx,cy,w,h]"""
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    else:
        boxes = np.array(boxes)
        cx = (boxes[..., 0] + boxes[..., 2]) / 2
        cy = (boxes[..., 1] + boxes[..., 3]) / 2
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1]
        return np.stack([cx, cy, w, h], axis=-1)


def cxcywh_to_xyxy(boxes):
    """[cx,cy,w,h] -> [x1,y1,x2,y2]"""
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        boxes = np.array(boxes)
        x1 = boxes[..., 0] - boxes[..., 2] / 2
        y1 = boxes[..., 1] - boxes[..., 3] / 2
        x2 = boxes[..., 0] + boxes[..., 2] / 2
        y2 = boxes[..., 1] + boxes[..., 3] / 2
        return np.stack([x1, y1, x2, y2], axis=-1)


def box_area(boxes):
    """boxes: [..., 4] in xyxy"""
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    else:
        boxes = np.array(boxes)
        return np.maximum(0, boxes[..., 2] - boxes[..., 0]) * np.maximum(0, boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes (xyxy). Returns [N] if boxes1:[N,4], boxes2:[N,4]"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    if isinstance(boxes1, torch.Tensor):
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union = area1 + area2 - inter
        iou = inter / (union + 1e-6)
        return iou
    else:
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union = area1 + area2 - inter
        return inter / (union + 1e-6)


def box_giou(boxes1, boxes2):
    """Generalized IoU loss term. Returns [N]"""
    iou = box_iou(boxes1, boxes2)
    if isinstance(boxes1, torch.Tensor):
        enc_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
        enc_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
        enc_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
        enc_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
        enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union = area1 + area2 - inter
        giou = iou - (enc_area - union) / (enc_area + 1e-6)
        return giou
    else:
        boxes1, boxes2 = np.array(boxes1), np.array(boxes2)
        enc_x1 = np.minimum(boxes1[..., 0], boxes2[..., 0])
        enc_y1 = np.minimum(boxes1[..., 1], boxes2[..., 1])
        enc_x2 = np.maximum(boxes1[..., 2], boxes2[..., 2])
        enc_y2 = np.maximum(boxes1[..., 3], boxes2[..., 3])
        enc_area = np.maximum(0, enc_x2 - enc_x1) * np.maximum(0, enc_y2 - enc_y1)
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union = area1 + area2 - inter
        giou = iou - (enc_area - union) / (enc_area + 1e-6)
        return giou


def clamp_boxes(boxes, min_val=0.0, max_val=1.0):
    if isinstance(boxes, torch.Tensor):
        return boxes.clamp(min_val, max_val)
    return np.clip(boxes, min_val, max_val)


def box_center(box):
    """xyxy -> (cx, cy)"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
