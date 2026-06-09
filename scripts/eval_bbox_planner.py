import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

from src.data.dataset import BBoxPlannerDataset
from src.data.collate import collate_fn
from src.model.bbox_planner import build_model
from src.model.embeddings import CLIPTextEncoder
from src.model.losses import compute_metrics, overlap_loss
from src.utils.box_ops import cxcywh_to_xyxy, box_iou
from src.utils.logging import get_logger

logger = get_logger("eval")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_eval(model, encoder, loader, device):
    model.eval()

    all_metrics = {"l1": [], "iou": [], "giou": [], "center_err": [], "area_err": []}
    overlap_rates = []

    # per-state breakdown
    from src.data.schema import STATE_VOCAB
    per_state_iou = {s: [] for s in STATE_VOCAB}
    per_nentity_iou = {1: [], 2: [], "3+": []}

    for batch in loader:
        presence = batch["presence"].to(device)
        state_ids = batch["state_ids"].to(device)
        relation_ids = batch["relation_ids"].to(device)
        target_cx = batch["target_boxes_cxcywh"].to(device)
        target_mask = batch["target_mask"].to(device)

        with torch.no_grad():
            shot_emb = encoder.encode_batch_shots(batch["shot_prompts"], device)
            entity_emb = encoder.encode_batch_entities(batch["entity_names"], device)

        pred_boxes = model(shot_emb.float(), entity_emb.float(),
                           state_ids, presence, relation_ids)

        metrics = compute_metrics(pred_boxes, target_cx, target_mask)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        # overlap rate
        B, S, E, _ = pred_boxes.shape
        pred_xy = cxcywh_to_xyxy(pred_boxes).clamp(0, 1)
        for b in range(B):
            for s in range(S):
                act = presence[b, s].sum().item()
                if act < 2:
                    continue
                for e1 in range(E):
                    for e2 in range(e1+1, E):
                        if presence[b,s,e1] and presence[b,s,e2]:
                            iou_pair = box_iou(
                                pred_xy[b,s,e1:e1+1],
                                pred_xy[b,s,e2:e2+1]
                            ).item()
                            overlap_rates.append(iou_pair > 0.1)

        # per-state IoU
        pred_xy_flat = pred_xy.view(-1, 4)
        tgt_xy_flat = cxcywh_to_xyxy(target_cx).clamp(0, 1).view(-1, 4)
        mask_flat = target_mask.view(-1).bool()
        state_flat = state_ids.view(-1)
        iou_flat = box_iou(pred_xy_flat, tgt_xy_flat)

        for idx in range(mask_flat.shape[0]):
            if mask_flat[idx]:
                st = state_flat[idx].item()
                st_name = STATE_VOCAB[st] if st < len(STATE_VOCAB) else "absent"
                iou_val = iou_flat[idx].item()
                per_state_iou[st_name].append(iou_val)

        # per entity count IoU
        for b in range(B):
            for s in range(S):
                n_act = presence[b, s].sum().item()
                for e in range(E):
                    if target_mask[b, s, e]:
                        iou_val = box_iou(
                            pred_xy[b,s,e:e+1],
                            cxcywh_to_xyxy(target_cx)[b,s,e:e+1].clamp(0,1)
                        ).item()
                        if n_act == 1:
                            per_nentity_iou[1].append(iou_val)
                        elif n_act == 2:
                            per_nentity_iou[2].append(iou_val)
                        else:
                            per_nentity_iou["3+"].append(iou_val)

    avg = {k: sum(v)/len(v) if v else 0.0 for k, v in all_metrics.items()}
    avg["overlap_rate"] = sum(overlap_rates)/len(overlap_rates) if overlap_rates else 0.0

    avg["per_state_iou"] = {k: sum(v)/len(v) if v else 0.0 for k,v in per_state_iou.items()}
    avg["per_nentity_iou"] = {str(k): sum(v)/len(v) if v else 0.0 for k,v in per_nentity_iou.items()}

    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/runs/bbox_planner_v1/checkpoints/best.pt")
    parser.add_argument("--data", default="data/splits/test.jsonl")
    parser.add_argument("--out", default="outputs/eval/bbox_planner_v1_test.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-shots", type=int, default=5)
    parser.add_argument("--max-entities", type=int, default=5)
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint to get model cfg
    ckpt_path = base / args.checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # load config from run dir
    cfg_path = ckpt_path.parent.parent / "config.yaml"
    cfg = load_config(str(cfg_path))

    encoder = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True).to(device)

    model_cfg = cfg["model"]
    model_cfg["d_text"] = encoder.d_out
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    ds = BBoxPlannerDataset(str(base / args.data),
                            max_shots=args.max_shots, max_entities=args.max_entities)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn)
    logger.info(f"Test samples: {len(ds)}")

    metrics = run_eval(model, encoder, loader, device)
    logger.info("=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            logger.info(f"  {k}:")
            for kk, vv in v.items():
                logger.info(f"    {kk}: {vv:.4f}")
        else:
            logger.info(f"  {k}: {v:.4f}")

    out_path = base / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
