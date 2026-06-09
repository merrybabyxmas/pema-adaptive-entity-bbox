import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

from src.data.dataset import BBoxPlannerDataset
from src.data.collate import collate_fn
from src.model.bbox_planner import build_model
from src.model.embeddings import CLIPTextEncoder
from src.model.losses import masked_l1_loss, masked_iou_loss, overlap_loss, temporal_consistency_loss, compute_metrics
from src.utils.box_ops import cxcywh_to_xyxy
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger("train")

ENTITY_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def _rule_based_plan(shot_prompts):
    common = ["cat", "dog", "bird", "horse", "person", "man", "woman",
              "child", "car", "bicycle", "rabbit", "cow", "sheep"]
    all_found, per_shot = [], []
    for p in shot_prompts:
        tokens = re.findall(r'\b\w+\b', p.lower())
        found = [e for e in common if e in tokens]
        per_shot.append(found)
        all_found.extend(found)
    entities = list(dict.fromkeys(all_found)) or ["entity"]
    shots = []
    for i, (p, active) in enumerate(zip(shot_prompts, per_shot)):
        shots.append({"shot_id": i, "prompt": p, "background": "scene",
                      "active_entities": active or entities[:1], "relations": []})
    return {"entities": entities, "shots": shots}


def _plan_to_tensors(plan, max_shots, max_entities, device):
    from src.lm_planner.validator import build_presence_matrix, compute_states
    from src.data.schema import STATE2ID, RELATION2ID

    entities = plan["entities"][:max_entities]
    E = len(entities)
    S_use = min(len(plan["shots"]), max_shots)

    P = build_presence_matrix({"entities": entities, "shots": plan["shots"][:S_use]})
    states = compute_states(P, entities)

    import numpy as np
    presence = np.zeros((max_shots, max_entities), dtype=np.int64)
    state_ids = np.zeros((max_shots, max_entities), dtype=np.int64)
    relation_ids = np.zeros((max_shots, max_entities, max_entities), dtype=np.int64)

    presence[:S_use, :E] = P
    for s in range(S_use):
        for ei, e in enumerate(entities):
            state_ids[s, ei] = STATE2ID.get(states[s][ei], 0)

    shot_prompts = [shot["prompt"] for shot in plan["shots"][:S_use]]
    shot_prompts += [""] * (max_shots - len(shot_prompts))
    entity_names = entities + [""] * (max_entities - E)

    return {
        "shot_prompts": [shot_prompts],
        "entity_names": [entity_names],
        "presence": torch.from_numpy(presence).unsqueeze(0).to(device),
        "state_ids": torch.from_numpy(state_ids).unsqueeze(0).to(device),
        "relation_ids": torch.from_numpy(relation_ids).unsqueeze(0).to(device),
        "num_shots": S_use,
        "num_entities": E,
    }


@torch.no_grad()
def viz_epoch_sample(model, encoder, story_path, out_dir, epoch, device,
                     max_shots=5, max_entities=5):
    """Generate a layout diagram for user story to track training progress."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    if not story_path.exists():
        return

    model.eval()
    with open(story_path) as f:
        user_input = json.load(f)
    shot_prompts = user_input["shots"]

    plan = _rule_based_plan(shot_prompts)
    for i, shot in enumerate(plan["shots"]):
        shot["prompt"] = shot_prompts[i] if i < len(shot_prompts) else ""

    tensors = _plan_to_tensors(plan, max_shots, max_entities, device)
    S_use = tensors["num_shots"]
    E_use = tensors["num_entities"]
    entities = plan["entities"][:E_use]

    shot_emb = encoder.encode_batch_shots(tensors["shot_prompts"], device)
    entity_emb = encoder.encode_batch_entities(tensors["entity_names"], device)
    pred_boxes = model(shot_emb.float(), entity_emb.float(),
                       tensors["state_ids"], tensors["presence"], tensors["relation_ids"])
    pred_xyxy = cxcywh_to_xyxy(pred_boxes).clamp(0, 1)[0].cpu().numpy()  # [S,E,4]
    presence_np = tensors["presence"][0].cpu().numpy()

    W, H = 256, 256
    PAD = 8
    canvas_w = S_use * W + (S_use + 1) * PAD
    canvas_h = H + 60

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except Exception:
        font = font_sm = ImageFont.load_default()

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    for s in range(S_use):
        ox = PAD + s * (W + PAD)
        oy = 30

        # shot background
        draw.rectangle([ox, oy, ox + W, oy + H], fill=(255, 255, 255), outline=(180, 180, 180), width=1)

        # shot label
        label = f"Shot {s}: {shot_prompts[s][:30]}..." if len(shot_prompts[s]) > 30 else f"Shot {s}: {shot_prompts[s]}"
        draw.text((ox, oy - 20), label, fill=(50, 50, 50), font=font_sm)

        for ei, e in enumerate(entities):
            if presence_np[s, ei] == 0:
                continue
            x1, y1, x2, y2 = pred_xyxy[s, ei]
            bx1 = int(ox + x1 * W)
            by1 = int(oy + y1 * H)
            bx2 = int(ox + x2 * W)
            by2 = int(oy + y2 * H)
            color = hex_to_rgb(ENTITY_COLORS[ei % len(ENTITY_COLORS)])
            draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
            # translucent fill
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rectangle([bx1, by1, bx2, by2], fill=color + (30,))
            canvas = canvas.convert("RGBA")
            canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            draw.text((bx1 + 3, by1 + 2), e, fill=color, font=font)

    # epoch title
    draw.text((PAD, 4), f"Epoch {epoch:03d}", fill=(80, 80, 80), font=font)

    gen_dir = out_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = gen_dir / f"epoch_{epoch:03d}.png"
    canvas.save(str(out_path))
    logger.info(f"  -> layout viz saved: {out_path.name}")

    # Save infer result JSON for SD generation
    infer_json = gen_dir / f"epoch_{epoch:03d}_boxes.json"
    infer_data = {
        "entities": entities,
        "presence": presence_np[:S_use, :E_use].tolist(),
        "shots": []
    }
    for s in range(S_use):
        shot_out = {"shot_id": s, "prompt": shot_prompts[s] if s < len(shot_prompts) else "", "boxes": {}}
        for ei, e in enumerate(entities):
            if presence_np[s, ei] == 1:
                x1, y1, x2, y2 = [round(float(v), 4) for v in pred_xyxy[s, ei]]
                shot_out["boxes"][e] = [x1, y1, x2, y2]
        infer_data["shots"].append(shot_out)
    with open(infer_json, "w") as f:
        json.dump(infer_data, f)

    # Launch GLIGEN layout-conditioned generation on GPU 1 (non-blocking)
    import subprocess
    gligen_out = gen_dir / f"epoch_{epoch:03d}_gligen"
    ref_dir = out_dir / "entity_refs"
    subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "generate_with_layout.py"),
         "--layout", str(infer_json),
         "--out", str(gligen_out),
         "--use-ref",
         "--ref-dir", str(ref_dir)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    logger.info(f"  -> GLIGEN+ref generation launched (GPU 1): epoch_{epoch:03d}_gligen/")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def encode_texts(encoder, shot_prompts_batch, entity_names_batch, device):
    with torch.no_grad():
        shot_emb = encoder.encode_batch_shots(shot_prompts_batch, device)
        entity_emb = encoder.encode_batch_entities(entity_names_batch, device)
    return shot_emb, entity_emb


def train_epoch(model, encoder, loader, optimizer, scaler, cfg, device):
    model.train()
    lambda_iou = cfg["loss"]["lambda_iou"]
    lambda_ov = cfg["loss"]["lambda_overlap"]
    lambda_temp = cfg["loss"]["lambda_temp"]
    overlap_tau = cfg["loss"]["overlap_tau"]

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        presence = batch["presence"].to(device)
        state_ids = batch["state_ids"].to(device)
        relation_ids = batch["relation_ids"].to(device)
        target_cx = batch["target_boxes_cxcywh"].to(device)
        target_mask = batch["target_mask"].to(device)

        shot_emb, entity_emb = encode_texts(
            encoder, batch["shot_prompts"], batch["entity_names"], device
        )

        with torch.cuda.amp.autocast(enabled=cfg["training"].get("mixed_precision", True)):
            pred_boxes = model(shot_emb, entity_emb, state_ids, presence, relation_ids)
            l_box = masked_l1_loss(pred_boxes, target_cx, target_mask)
            l_iou = masked_iou_loss(pred_boxes, target_cx, target_mask)
            l_ov = overlap_loss(pred_boxes, presence, tau=overlap_tau)
            l_temp = temporal_consistency_loss(pred_boxes, state_ids, presence)
            loss = l_box + lambda_iou * l_iou + lambda_ov * l_ov + lambda_temp * l_temp

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, encoder, loader, cfg, device):
    model.eval()
    lambda_iou = cfg["loss"]["lambda_iou"]
    lambda_ov = cfg["loss"]["lambda_overlap"]
    lambda_temp = cfg["loss"]["lambda_temp"]
    overlap_tau = cfg["loss"]["overlap_tau"]

    all_metrics = {"l1": [], "iou": [], "giou": [], "center_err": [], "area_err": []}
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        presence = batch["presence"].to(device)
        state_ids = batch["state_ids"].to(device)
        relation_ids = batch["relation_ids"].to(device)
        target_cx = batch["target_boxes_cxcywh"].to(device)
        target_mask = batch["target_mask"].to(device)

        shot_emb, entity_emb = encode_texts(
            encoder, batch["shot_prompts"], batch["entity_names"], device
        )
        with torch.cuda.amp.autocast(enabled=False):
            pred_boxes = model(shot_emb.float(), entity_emb.float(),
                               state_ids, presence, relation_ids)
            l_box = masked_l1_loss(pred_boxes, target_cx, target_mask)
            l_iou = masked_iou_loss(pred_boxes, target_cx, target_mask)
            l_ov = overlap_loss(pred_boxes, presence, tau=overlap_tau)
            l_temp = temporal_consistency_loss(pred_boxes, state_ids, presence)
            loss = l_box + lambda_iou * l_iou + lambda_ov * l_ov + lambda_temp * l_temp

        metrics = compute_metrics(pred_boxes, target_cx, target_mask)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        total_loss += loss.item()
        n_batches += 1

    avg = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
    avg["loss"] = total_loss / max(n_batches, 1)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_bbox_planner.yaml")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    cfg_path = base / args.config
    cfg = load_config(str(cfg_path))

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = base / cfg["paths"]["output_dir"]
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    story_path = base / "examples" / "user_story_001.json"
    gen_every = cfg.get("gen_every", 1)

    # save config
    import shutil
    shutil.copy(str(cfg_path), str(out_dir / "config.yaml"))

    # build encoder
    logger.info("Loading CLIP text encoder...")
    encoder = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True)
    encoder = encoder.to(device)

    # build model
    model_cfg = cfg["model"]
    model_cfg["d_text"] = encoder.d_out
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    # datasets
    max_shots = cfg["training"]["max_shots"]
    max_entities = cfg["training"]["max_entities"]
    train_ds = BBoxPlannerDataset(str(base / cfg["paths"]["train_jsonl"]),
                                  max_shots=max_shots, max_entities=max_entities)
    val_ds = BBoxPlannerDataset(str(base / cfg["paths"]["val_jsonl"]),
                                max_shots=max_shots, max_entities=max_entities)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=cfg["training"]["num_workers"],
                              collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=cfg["training"]["num_workers"],
                            collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("mixed_precision", True))

    best_iou = -1.0
    all_metrics = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, encoder, train_loader, optimizer, scaler, cfg, device)
        scheduler.step()

        if epoch % cfg["eval"]["eval_every"] == 0:
            val_metrics = eval_epoch(model, encoder, val_loader, cfg, device)
            dt = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | IoU={val_metrics['iou']:.4f} | "
                f"GIoU={val_metrics['giou']:.4f} | L1={val_metrics['l1']:.4f} | "
                f"dt={dt:.1f}s"
            )
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = train_loss
            all_metrics.append(val_metrics)

            with open(metrics_dir / f"val_epoch_{epoch:03d}.json", "w") as f:
                json.dump(val_metrics, f, indent=2)

            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "metrics": val_metrics}, str(ckpt_dir / "best.pt"))
                logger.info(f"  -> new best IoU: {best_iou:.4f}")

        if epoch % cfg["eval"]["save_every"] == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       str(ckpt_dir / f"epoch_{epoch:03d}.pt"))

        if epoch % gen_every == 0:
            viz_epoch_sample(model, encoder, story_path, out_dir, epoch, device,
                             max_shots=max_shots, max_entities=max_entities)

    # save final
    torch.save({"epoch": cfg["training"]["epochs"], "model": model.state_dict()},
               str(ckpt_dir / "final.pt"))

    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Training done. Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
