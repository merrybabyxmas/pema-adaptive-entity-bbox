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


def deoverlap_boxes(boxes_dict: dict, max_iter: int = 30) -> dict:
    """
    Iteratively push overlapping bboxes apart until they no longer overlap.
    Moves boxes along the axis of smallest penetration depth.
    boxes_dict: {entity_name: [x1,y1,x2,y2]} normalized [0,1]
    """
    names = list(boxes_dict.keys())
    if len(names) < 2:
        return boxes_dict

    boxes = [list(boxes_dict[n]) for n in names]

    for _ in range(max_iter):
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1, b2 = boxes[i], boxes[j]
                ox1 = max(b1[0], b2[0])
                oy1 = max(b1[1], b2[1])
                ox2 = min(b1[2], b2[2])
                oy2 = min(b1[3], b2[3])
                if ox2 <= ox1 or oy2 <= oy1:
                    continue  # no overlap

                ov_w = ox2 - ox1
                ov_h = oy2 - oy1
                push = 0.02  # extra gap after separation

                if ov_w <= ov_h:
                    # push horizontally (smaller penetration axis)
                    c1x = (b1[0] + b1[2]) / 2
                    c2x = (b2[0] + b2[2]) / 2
                    half = ov_w / 2 + push
                    if c1x <= c2x:
                        boxes[i][0] -= half; boxes[i][2] -= half
                        boxes[j][0] += half; boxes[j][2] += half
                    else:
                        boxes[i][0] += half; boxes[i][2] += half
                        boxes[j][0] -= half; boxes[j][2] -= half
                else:
                    # push vertically
                    c1y = (b1[1] + b1[3]) / 2
                    c2y = (b2[1] + b2[3]) / 2
                    half = ov_h / 2 + push
                    if c1y <= c2y:
                        boxes[i][1] -= half; boxes[i][3] -= half
                        boxes[j][1] += half; boxes[j][3] += half
                    else:
                        boxes[i][1] += half; boxes[i][3] += half
                        boxes[j][1] -= half; boxes[j][3] -= half
                moved = True

        if not moved:
            break

    # clamp to valid image range
    for b in boxes:
        w = b[2] - b[0]
        h = b[3] - b[1]
        b[0] = max(0.02, min(0.98 - w, b[0]))
        b[2] = b[0] + w
        b[1] = max(0.02, min(0.98 - h, b[1]))
        b[3] = b[1] + h

    return {names[i]: boxes[i] for i in range(len(names))}
