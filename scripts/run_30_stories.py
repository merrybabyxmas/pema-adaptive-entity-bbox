"""
30-story end-to-end test: NEURAL bbox-planner -> image-LISA, with bbox debug viz.

For each story (distinctive entities, 3 shots, presence pattern):
  1) PLANNER: presence (explicit per shot) -> states -> trained bbox-planner
     predicts per-shot per-entity boxes (cxcywh -> xyxy).
  2) DEBUG VIZ: draw the predicted boxes per shot (color per entity) so we can
     see HOW the planner places/learns boxes — saved as bbox_debug.png.
  3) LISA: single-object white-bg anchors -> Gaussian masks from the PREDICTED
     boxes -> SDXL native ip_adapter_masks (presence-aware, adaptive sigma/scale).
  4) Per-story montage: anchors | bbox-debug | 3 generated shots (with overlay).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_30_stories.py \
    --planner outputs/runs/bbox_planner_v2/checkpoints/best.pt \
    --stories examples/stories30.json --out outputs/lisa/stories30 [--limit N]
"""
import sys, os, argparse, gc, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch, yaml
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model.bbox_planner import build_model
from src.model.embeddings import CLIPTextEncoder
from src.lm_planner.validator import build_presence_matrix, compute_states
from src.data.schema import STATE2ID
from src.utils.box_ops import cxcywh_to_xyxy, deoverlap_boxes
from src.generation.lisa.build_anchors import build_all_anchors
from src.generation.lisa.lisa_pipeline import LISAPipeline

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
MAXS, MAXE = 5, 5


def adaptive(n):
    if n <= 1: return 20.0, 0.6
    if n == 2: return 12.0, 0.7
    return 8.0, 0.8


def enforce_gap(boxes: dict, gap: float = 0.12, shrink: float = 0.9):
    """Push horizontally-ordered boxes apart so there is a background strip
    between them — prevents large same-body-plan entities (quadrupeds, vehicles)
    from bridging across touching boxes into one fused body. Boxes are xyxy [0,1].

    Slots are allocated PROPORTIONAL to each entity's predicted box width (not
    equal), so the planner's learned relative size survives — a duck keeps a
    narrower slot than an elephant. Vertical extent is left as predicted (the
    planner's height already carries size). 'Go by what's learned.'"""
    if len(boxes) < 2:
        return boxes
    items = sorted(boxes.items(), key=lambda kv: (kv[1][0] + kv[1][2]) / 2)
    n = len(items)
    widths = [max(1e-3, (b[2] - b[0]) * shrink) for _, b in items]
    total = sum(widths) + gap * (n - 1)
    # only scale DOWN if the learned widths + gaps overflow the row; otherwise
    # keep the planner's predicted absolute widths (a duck stays duck-sized).
    if total > 1.0:
        s = 1.0 / total
        widths = [w * s for w in widths]
        gap = gap * s
        total = 1.0
    x = (1.0 - total) / 2.0                              # center the group
    out = {}
    for (e, b), w in zip(items, widths):
        cx = x + w / 2
        out[e] = [round(cx - w / 2, 4), b[1], round(cx + w / 2, 4), b[3]]
        x += w + gap
    return out


def plan_boxes(model, encoder, story, device):
    """Run the trained planner -> per-shot per-entity predicted xyxy boxes."""
    entities = [e["name"] for e in story["entities"]]
    E = len(entities)
    shots = [{"prompt": sh["prompt"], "active_entities": sh["present"], "relations": []}
             for sh in story["shots"]]
    S = len(shots)
    P = build_presence_matrix({"entities": entities, "shots": shots})  # [S,E]
    states = compute_states(P, entities)

    presence = np.zeros((MAXS, MAXE), np.int64)
    state_ids = np.zeros((MAXS, MAXE), np.int64)
    relation_ids = np.zeros((MAXS, MAXE, MAXE), np.int64)
    presence[:S, :E] = P
    for s in range(S):
        for ei in range(E):
            state_ids[s, ei] = STATE2ID.get(states[s][ei], 0)

    shot_prompts = [[sh["prompt"] for sh in shots] + [""] * (MAXS - S)]
    entity_names = [entities + [""] * (MAXE - E)]
    with torch.no_grad():
        se = encoder.encode_batch_shots(shot_prompts, device).float()
        ee = encoder.encode_batch_entities(entity_names, device).float()
        pb = model(se, ee,
                   torch.from_numpy(state_ids).unsqueeze(0).to(device),
                   torch.from_numpy(presence).unsqueeze(0).to(device),
                   torch.from_numpy(relation_ids).unsqueeze(0).to(device))
        xyxy = cxcywh_to_xyxy(pb).clamp(0, 1)[0].cpu().numpy()  # [S,E,4]
    # per shot: {entity: [x1,y1,x2,y2]} for present entities
    out = []
    for s in range(S):
        boxes = {entities[ei]: xyxy[s, ei].tolist()
                 for ei in range(E) if P[s, ei] == 1}
        out.append(boxes)
    return entities, out


def draw_bbox_debug(entities, per_shot_boxes, save_path, title):
    S = len(per_shot_boxes)
    fig, axes = plt.subplots(1, S, figsize=(4 * S, 4))
    if S == 1: axes = [axes]
    ecolor = {e: COLORS[i % len(COLORS)] for i, e in enumerate(entities)}
    for s, (ax, boxes) in enumerate(zip(axes, per_shot_boxes)):
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        ax.set_title(f"shot{s}  ({'+'.join(boxes.keys()) or 'none'})", fontsize=9)
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        for e, b in boxes.items():
            x1, y1, x2, y2 = b
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         fill=False, edgecolor=ecolor[e], linewidth=2.5))
            ax.text(x1 + 0.01, y1 + 0.05, e, color=ecolor[e], fontsize=8, weight="bold")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="outputs/runs/bbox_planner_v2/checkpoints/best.pt")
    ap.add_argument("--stories", default="examples/stories30.json")
    ap.add_argument("--config", default="configs/lisa_default.yaml")
    ap.add_argument("--out", default="outputs/lisa/stories30")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    dev = "cuda"
    out = base / args.out; out.mkdir(parents=True, exist_ok=True)
    stories = json.loads((base / args.stories).read_text())
    if args.limit: stories = stories[:args.limit]

    # --- load trained planner ---
    ck = torch.load(str(base / args.planner), map_location="cpu")
    pcfg = yaml.safe_load(open((base / args.planner).parent.parent / "config.yaml"))
    encoder = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True).to(dev)
    mcfg = pcfg["model"]; mcfg["d_text"] = encoder.d_out
    model = build_model(mcfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    print(f"[planner] loaded epoch {ck.get('epoch','?')}")

    # --- planner pass for all stories (+ debug viz) ---
    planned = []
    for st in stories:
        entities, per_shot = plan_boxes(model, encoder, st, dev)
        sdir = out / st["name"]; sdir.mkdir(parents=True, exist_ok=True)
        draw_bbox_debug(entities, per_shot, sdir / "bbox_debug.png", st["name"])
        planned.append((st, entities, per_shot))
    del model, encoder; gc.collect(); torch.cuda.empty_cache()
    print(f"[planner] predicted boxes + debug viz for {len(planned)} stories")

    # --- LISA config ---
    config = yaml.safe_load(open(base / args.config))
    config["generation"]["device"] = "cuda:0"

    # --- anchors (one plain SDXL load for all stories) ---
    print("[anchors] building single-object white-bg anchors...")
    from diffusers import StableDiffusionXLPipeline
    bp = StableDiffusionXLPipeline.from_pretrained(
        config["models"]["sdxl"], torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True).to("cuda:0"); bp.vae.enable_slicing()
    anchor_banks = {}
    for st, entities, _ in planned:
        sdir = out / st["name"]
        cfg = {**config, "anchors": {**config["anchors"], "save_dir": str(sdir / "anchor_bank")}}
        anchor_banks[st["name"]] = build_all_anchors(
            bp, st["entities"], {"name": "bg", "prompt": st["background"]}, cfg)
    del bp; gc.collect(); torch.cuda.empty_cache()

    # --- LISA generation ---
    print("[LISA] generating shots...")
    lisa = LISAPipeline(config); lisa.load_models()
    L = config["layout"]["latent_size"]
    for st, entities, per_shot in planned:
        sdir = out / st["name"]
        # build a layout_plan with PREDICTED boxes (xyxy norm -> latent)
        shots_lp = []
        for s, boxes in enumerate(per_shot):
            # safeguard: separate overlapping predicted boxes before LISA, then
            # enforce a background gap so large same-body-plan entities (quadrupeds,
            # vehicles) don't bridge across touching boxes into one fused body.
            if len(boxes) > 1:
                # gap-separated slots: each entity gets its own clear region so
                # neither fuses NOR gets suppressed/missing (both entities present)
                boxes = enforce_gap(deoverlap_boxes(dict(boxes)))
            ents = []
            for e, b in boxes.items():
                cx = (b[0] + b[2]) / 2
                pos = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
                ents.append({"name": e, "position": pos,
                             "bbox": [int(b[0]*L), int(b[1]*L), int(b[2]*L), int(b[3]*L)]})
            shots_lp.append({"shot_index": s, "description": st["shots"][s]["prompt"], "entities": ents})
        layout_plan = {"entity_definitions": st["entities"],
                       "background": {"prompt": st["background"]}, "shots": shots_lp}
        config["evaluation"]["output_dir"] = str(sdir)
        lisa.config["evaluation"]["output_dir"] = str(sdir)
        for s, shot in enumerate(shots_lp):
            n = len(shot["entities"]); sig, ips = adaptive(n)
            lisa.generate_shot(anchor_banks[st["name"]], layout_plan, shot_index=s,
                               seed=42 + s, sigma_override=sig, ip_scale_override=ips)
            # overlay predicted bbox on generated shot
            img = Image.open(sdir / f"shot_{s:03d}.png").convert("RGB")
            dr = ImageDraw.Draw(img); W, H = img.size
            for i, ent in enumerate(shot["entities"]):
                b = ent["bbox"]
                dr.rectangle([b[0]/L*W, b[1]/L*H, b[2]/L*W, b[3]/L*H],
                             outline=COLORS[entities.index(ent["name"]) % len(COLORS)], width=4)
            img.save(sdir / f"shot_{s:03d}_bbox.png")
        print(f"  {st['name']} done")

    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
