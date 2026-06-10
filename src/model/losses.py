import torch
import torch.nn.functional as F

from src.utils.box_ops import box_iou, box_giou, cxcywh_to_xyxy


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """pred, target: [B,S,E,4]; mask: [B,S,E]"""
    loss = F.l1_loss(pred, target, reduction='none').sum(-1)  # [B,S,E]
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss


def masked_iou_loss(pred_cxcywh: torch.Tensor, target_cxcywh: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """GIoU loss on active pairs."""
    pred_xyxy = cxcywh_to_xyxy(pred_cxcywh).clamp(0, 1)
    tgt_xyxy = cxcywh_to_xyxy(target_cxcywh).clamp(0, 1)

    B, S, E, _ = pred_xyxy.shape
    pred_flat = pred_xyxy.view(-1, 4)
    tgt_flat = tgt_xyxy.view(-1, 4)
    mask_flat = mask.view(-1)

    giou = box_giou(pred_flat, tgt_flat)  # [B*S*E]
    loss = 1 - giou
    loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)
    return loss


def overlap_loss(pred_cxcywh: torch.Tensor, presence: torch.Tensor,
                 tau: float = 0.25) -> torch.Tensor:
    """
    Penalize overlapping bbox pairs in the same shot.
    pred_cxcywh: [B,S,E,4]; presence: [B,S,E]
    """
    pred_xyxy = cxcywh_to_xyxy(pred_cxcywh).clamp(0, 1)
    B, S, E, _ = pred_xyxy.shape
    total_loss = torch.tensor(0.0, device=pred_cxcywh.device)
    count = 0

    for e1 in range(E):
        for e2 in range(e1 + 1, E):
            both_active = presence[:, :, e1] * presence[:, :, e2]  # [B,S]
            if both_active.sum() == 0:
                continue
            b1 = pred_xyxy[:, :, e1, :]  # [B,S,4]
            b2 = pred_xyxy[:, :, e2, :]
            iou = box_iou(b1.reshape(-1, 4), b2.reshape(-1, 4)).view(B, S)  # [B,S]
            penalty = torch.clamp(iou - tau, min=0) * both_active
            total_loss = total_loss + penalty.sum()
            count += both_active.sum()

    if count > 0:
        total_loss = total_loss / (count + 1e-6)
    return total_loss


def depth_ranking_loss(pred_depth: torch.Tensor, target_depth: torch.Tensor,
                       presence: torch.Tensor, margin: float = 0.1,
                       eps: float = 0.05) -> torch.Tensor:
    """Pairwise margin ranking on occlusion depth for co-present entities.

    For each same-shot pair (i,j) with a confident target order
    (|target_i - target_j| > eps), enforce the predicted order with a margin:
        loss = max(0, margin - sign(t_i - t_j) * (p_i - p_j))
    pred_depth, target_depth, presence: [B,S,E].
    """
    B, S, E = pred_depth.shape
    if E < 2:
        return torch.tensor(0.0, device=pred_depth.device)
    pi = pred_depth.unsqueeze(-1)            # [B,S,E,1]
    pj = pred_depth.unsqueeze(-2)            # [B,S,1,E]
    ti = target_depth.unsqueeze(-1)
    tj = target_depth.unsqueeze(-2)
    dp = pi - pj                              # [B,S,E,E] predicted diff
    dt = ti - tj                              # target diff
    sign = torch.sign(dt)
    both = presence.unsqueeze(-1) * presence.unsqueeze(-2)   # [B,S,E,E]
    # only ordered pairs (skip ties / near-equal targets), upper triangle to avoid double count
    iu = torch.triu(torch.ones(E, E, device=pred_depth.device), diagonal=1)
    valid = both * (dt.abs() > eps).float() * iu
    if valid.sum() < 1:
        return torch.tensor(0.0, device=pred_depth.device)
    loss = torch.clamp(margin - sign * dp, min=0.0)
    return (loss * valid).sum() / (valid.sum() + 1e-6)


def temporal_consistency_loss(pred_cxcywh: torch.Tensor, state_ids: torch.Tensor,
                              presence: torch.Tensor) -> torch.Tensor:
    """
    Enforce entity consistency across shots.

    - stay  (2): consecutive shots, penalize center + size deviation
    - re_entry (5): penalize deviation from last active center before exit
    pred_cxcywh: [B,S,E,4]; state_ids: [B,S,E] int; presence: [B,S,E] int
    """
    STAY, RE_ENTRY = 2, 5
    B, S, E, _ = pred_cxcywh.shape
    device = pred_cxcywh.device
    total = torch.tensor(0.0, device=device)
    n = 0

    # --- stay: smooth center + size between consecutive shots ---
    for s in range(S - 1):
        both = (presence[:, s, :] * presence[:, s + 1, :]).float()  # [B,E]
        stay_mask = (state_ids[:, s + 1, :] == STAY).float()        # [B,E]
        mask = both * stay_mask                                       # [B,E]
        if mask.sum() == 0:
            continue
        # center L2
        dc = ((pred_cxcywh[:, s + 1, :, :2] - pred_cxcywh[:, s, :, :2]) ** 2).sum(-1)  # [B,E]
        # size L2
        ds = ((pred_cxcywh[:, s + 1, :, 2:] - pred_cxcywh[:, s, :, 2:]) ** 2).sum(-1)  # [B,E]
        total = total + ((dc + 0.5 * ds) * mask).sum() / (mask.sum() + 1e-6)
        n += 1

    # --- re_entry: return to last known center before exit ---
    # last_center[b,e] = center of entity e at last active shot before current
    last_center = torch.zeros(B, E, 2, device=device)
    for s in range(S):
        re_mask = (state_ids[:, s, :] == RE_ENTRY).float() * presence[:, s, :].float()  # [B,E]
        if re_mask.sum() > 0:
            dc = ((pred_cxcywh[:, s, :, :2] - last_center) ** 2).sum(-1)  # [B,E]
            total = total + (dc * re_mask).sum() / (re_mask.sum() + 1e-6)
            n += 1
        # update last_center for active entities
        active = presence[:, s, :].float().unsqueeze(-1)  # [B,E,1]
        last_center = active * pred_cxcywh[:, s, :, :2].detach() + (1 - active) * last_center

    return total / max(n, 1)


def compute_metrics(pred_cxcywh: torch.Tensor, target_cxcywh: torch.Tensor,
                    mask: torch.Tensor) -> dict:
    pred_xyxy = cxcywh_to_xyxy(pred_cxcywh).clamp(0, 1)
    tgt_xyxy = cxcywh_to_xyxy(target_cxcywh).clamp(0, 1)

    B, S, E, _ = pred_xyxy.shape
    mask_flat = mask.view(-1).bool()

    pred_flat = pred_xyxy.view(-1, 4)[mask_flat]
    tgt_flat = tgt_xyxy.view(-1, 4)[mask_flat]
    pred_cxcy_flat = pred_cxcywh.view(-1, 4)[mask_flat]
    tgt_cxcy_flat = target_cxcywh.view(-1, 4)[mask_flat]

    if mask_flat.sum() == 0:
        return {"l1": 0.0, "iou": 0.0, "giou": 0.0, "center_err": 0.0, "area_err": 0.0}

    l1 = F.l1_loss(pred_flat, tgt_flat).item()
    iou = box_iou(pred_flat, tgt_flat).mean().item()
    giou_val = box_giou(pred_flat, tgt_flat).mean().item()

    center_err = (pred_cxcy_flat[:, :2] - tgt_cxcy_flat[:, :2]).norm(dim=-1).mean().item()
    area_pred = (pred_cxcy_flat[:, 2] * pred_cxcy_flat[:, 3]).mean().item()
    area_tgt = (tgt_cxcy_flat[:, 2] * tgt_cxcy_flat[:, 3]).mean().item()
    area_err = abs(area_pred - area_tgt)

    return {
        "l1": l1,
        "iou": iou,
        "giou": giou_val,
        "center_err": center_err,
        "area_err": area_err,
    }
