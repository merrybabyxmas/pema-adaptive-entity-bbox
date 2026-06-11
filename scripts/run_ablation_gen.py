"""
Ablation keyframe generation (A/B/C) on the AAAI-120 prompt set.

Renderer is FIXED (SDXL + regional IP-Adapter, shared white-bg anchors); only the
LAYOUT SOURCE and the RENDER MODE change. Two phases:

  --mode anchors : build the shared white-bg anchor cache (one per story), shardable.
  --mode render  : run ONE job (= layout source + render mode) over all stories.

Jobs:
  A (planner ablation, render=full):  A_full A_wo_state A_wo_relation A_wo_entityint A_wo_depth
  B (layout source,    render=full):  B_template B_retrieval B_llm   (B_ours = A_full)
  C (render mode, layout=A_full):     C_nodepth C_gaponly C_depthown (C_full = A_full)

render modes (box arrangement + depth-mask):
  full     : resolve_for_occlusion (partial seam) + depth occlusion        [our method]
  nodepth  : raw predicted boxes (overlap kept), NO depth occlusion        [fusion baseline]
  gaponly  : enforce_gap separation, NO depth occlusion                    [separate, no occ]
  depthown : keep overlap (deoverlap only) + depth occlusion               [ownership on overlap]

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation_gen.py --mode anchors --shard 0/4
  CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation_gen.py --mode render --job A_full
"""
import sys, os, argparse, gc, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch, yaml
from PIL import Image, ImageDraw

from src.utils.box_ops import deoverlap_boxes, cxcywh_to_xyxy
from src.model.bbox_planner import build_model
from src.model.embeddings import CLIPTextEncoder
from src.generation.lisa.build_anchors import build_all_anchors
from src.generation.lisa.lisa_pipeline import LISAPipeline
from scripts.run_30_stories import (plan_boxes, resolve_for_occlusion, enforce_gap,
                                     draw_bbox_debug, adaptive, COLORS)

JOBS = {
    "A_full":        dict(layout="planner", ckpt="abl_full",       render="full"),
    "A_wo_state":    dict(layout="planner", ckpt="abl_wo_state",   render="full"),
    "A_wo_relation": dict(layout="planner", ckpt="abl_wo_relation", render="full"),
    "A_wo_entityint":dict(layout="planner", ckpt="abl_wo_shotattn", render="full"),
    "A_wo_depth":    dict(layout="planner", ckpt="abl_wo_depth",   render="full"),
    "B_template":    dict(layout="template",  render="full"),
    "B_retrieval":   dict(layout="retrieval", render="full"),
    "B_llm":         dict(layout="llm",       render="full"),
    "B_center":      dict(layout="center",    render="full"),
    "C_nodepth":     dict(layout="planner", ckpt="abl_full", render="nodepth"),
    "C_gaponly":     dict(layout="planner", ckpt="abl_full", render="gaponly"),
    "C_depthown":    dict(layout="planner", ckpt="abl_full", render="depthown"),
    # final chosen config: state + depth kept, relation + entity-interaction removed
    "FINAL_combo":   dict(layout="planner", ckpt="planner_v6_combo", render="full"),
}
MEAN_AREA = 0.223


def slug(s):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)


def anchor_bank_from_cache(story, cache):
    sd = cache / story["name"]
    return {"entities": [{"entity_name": e["name"], "image_path": str(sd / f"{slug(e['name'])}.png")}
                         for e in story["entities"]],
            "background": {"bg_name": "bg", "image_path": str(sd / "bg.png")}}


# ---------- layout sources -> (per_shot boxes dict, per_shot depth dict) ----------
def layout_planner(model, enc, story, dev):
    _, per, perd = plan_boxes(model, enc, story, dev)
    return per, perd


def _template(present, idx_order):
    n = len(present); w = min(0.4, 0.85 / max(n, 1))
    order = sorted(present, key=lambda e: idx_order[e])
    boxes = {}
    for r, e in enumerate(order):
        cx = (r + 0.5) / n
        boxes[e] = [round(cx - w / 2, 4), 0.2, round(cx + w / 2, 4), 0.8]
    return boxes


def layout_template(story, dev):
    idx = {e["name"]: i for i, e in enumerate(story["entities"])}
    per, perd = [], []
    for sh in story["shots"]:
        b = _template(sh["present"], idx)
        per.append(b); perd.append({e: 0.5 for e in b})
    return per, perd


def layout_retrieval(story, idx_by_cats, dev):
    idx = {e["name"]: i for i, e in enumerate(story["entities"])}
    per, perd = [], []
    for sh in story["shots"]:
        cats = frozenset(sh["present"])
        ref = idx_by_cats.get(cats)
        if ref:
            b = {e: ref[e] for e in sh["present"] if e in ref}
            for e in sh["present"]:
                if e not in b:
                    b[e] = [0.3, 0.3, 0.7, 0.9]
        else:
            b = _template(sh["present"], idx)
        per.append(b); perd.append({e: 0.5 for e in b})
    return per, perd


def layout_center(story, dev):
    side = float(np.sqrt(MEAN_AREA))
    box = [round(0.5 - side / 2, 4), round(0.5 - side / 2, 4),
           round(0.5 + side / 2, 4), round(0.5 + side / 2, 4)]
    per, perd = [], []
    for sh in story["shots"]:
        per.append({e: list(box) for e in sh["present"]})
        perd.append({e: 0.5 for e in sh["present"]})
    return per, perd


def layout_llm(story, llm, dev):
    rows = llm.get(story["name"], [])
    per, perd = [], []
    for i, sh in enumerate(story["shots"]):
        b = rows[i] if i < len(rows) else {e: [0.3, 0.3, 0.7, 0.9] for e in sh["present"]}
        b = {e: b[e] for e in sh["present"] if e in b}
        for e in sh["present"]:
            if e not in b:
                b[e] = [0.3, 0.3, 0.7, 0.9]
        per.append(b); perd.append({e: 0.5 for e in b})
    return per, perd


def build_retrieval_index(train_path):
    idx = {}
    for line in open(train_path):
        s = json.loads(line)
        for shot in s["shots"]:
            bx = {e: b for e, b in shot.get("boxes", {}).items() if b}
            k = frozenset(bx.keys())
            if k and k not in idx:
                idx[k] = {e: [round((b[0]+b[2])/2, 4), round((b[1]+b[3])/2, 4),
                              round(b[2]-b[0], 4), round(b[3]-b[1], 4)] for e, b in bx.items()}
                # store as xyxy for renderer
                idx[k] = {e: b for e, b in bx.items()}
    return idx


# ---------- render-mode box arrangement ----------
def arrange(boxes, depths, mode):
    if len(boxes) < 2:
        return dict(boxes), True
    if mode == "full":
        return resolve_for_occlusion(deoverlap_boxes(dict(boxes)), depths), True
    if mode == "nodepth":
        return {k: [max(0, min(1, v)) for v in b] for k, b in boxes.items()}, False
    if mode == "gaponly":
        return enforce_gap(deoverlap_boxes(dict(boxes))), False
    if mode == "depthown":
        return deoverlap_boxes(dict(boxes)), True
    return dict(boxes), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["anchors", "render"], required=True)
    ap.add_argument("--job", default="")
    ap.add_argument("--stories", default="data/captions/stories_aaai_eval_120.json")
    ap.add_argument("--config", default="configs/lisa_default.yaml")
    ap.add_argument("--anchor-cache", default="outputs/lisa/_aaai_anchors")
    ap.add_argument("--out-root", default="outputs/lisa/aaai_ablation")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--llm-layout", default="outputs/layouts/llm_aaai.json")
    ap.add_argument("--train", default="data/splits/train.jsonl")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="", help="comma-separated story names")
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    dev = "cuda"
    stories = json.loads((base / args.stories).read_text())
    if args.only:
        keep = set(args.only.split(",")); stories = [s for s in stories if s["name"] in keep]
    config = yaml.safe_load(open(base / args.config)); config["generation"]["device"] = "cuda:0"
    cache = base / args.anchor_cache

    if args.mode == "anchors":
        i, n = map(int, args.shard.split("/"))
        mine = [s for k, s in enumerate(stories) if k % n == i]
        from diffusers import StableDiffusionXLPipeline
        bp = StableDiffusionXLPipeline.from_pretrained(
            config["models"]["sdxl"], torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True).to("cuda:0"); bp.vae.enable_slicing()
        for st in mine:
            sd = cache / st["name"]
            if (sd / "bg.png").exists() and all((sd / f"{slug(e['name'])}.png").exists() for e in st["entities"]):
                continue
            cfg = {**config, "anchors": {**config["anchors"], "save_dir": str(sd)}}
            build_all_anchors(bp, st["entities"], {"name": "bg", "prompt": st["background"]}, cfg)
            print(f"  anchors {st['name']}", flush=True)
        print(f"[anchors] shard {args.shard} done"); return

    # ---- render ----
    job = JOBS[args.job]
    out = base / args.out_root / args.job; out.mkdir(parents=True, exist_ok=True)
    enc = model = llm = retr = None
    if job["layout"] == "planner":
        rdir = base / args.runs_dir / job["ckpt"]
        ck = torch.load(str(rdir / "checkpoints" / "best.pt"), map_location="cpu")
        pcfg = yaml.safe_load(open(rdir / "config.yaml"))
        enc = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True).to(dev)
        mcfg = pcfg["model"]; mcfg["d_text"] = enc.d_out
        model = build_model(mcfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    elif job["layout"] == "llm":
        llm = json.loads((base / args.llm_layout).read_text())
    elif job["layout"] == "retrieval":
        retr = build_retrieval_index(str(base / args.train))

    lisa = LISAPipeline(config); lisa.load_models()
    L = config["layout"]["latent_size"]

    for st in stories:
        sdir = out / st["name"]; sdir.mkdir(parents=True, exist_ok=True)
        if job["layout"] == "planner":
            per, perd = layout_planner(model, enc, st, dev)
        elif job["layout"] == "template":
            per, perd = layout_template(st, dev)
        elif job["layout"] == "retrieval":
            per, perd = layout_retrieval(st, retr, dev)
        elif job["layout"] == "llm":
            per, perd = layout_llm(st, llm, dev)
        elif job["layout"] == "center":
            per, perd = layout_center(st, dev)
        draw_bbox_debug([e["name"] for e in st["entities"]], per, sdir / "bbox_debug.png", st["name"], perd)

        anchor_bank = anchor_bank_from_cache(st, cache)
        shots_lp = []
        for s, boxes in enumerate(per):
            depths = perd[s]
            ar, _ = arrange(boxes, depths, job["render"])
            ents = []
            for e, b in ar.items():
                cx = (b[0] + b[2]) / 2
                pos = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
                ents.append({"name": e, "position": pos, "depth": float(depths.get(e, 0.5)),
                             "bbox": [int(b[0]*L), int(b[1]*L), int(b[2]*L), int(b[3]*L)]})
            shots_lp.append({"shot_index": s, "description": st["shots"][s]["prompt"], "entities": ents})
        layout_plan = {"entity_definitions": st["entities"],
                       "background": {"prompt": st["background"]}, "shots": shots_lp}
        config["evaluation"]["output_dir"] = str(sdir); lisa.config["evaluation"]["output_dir"] = str(sdir)
        use_occ = job["render"] in ("full", "depthown")
        enames = [e["name"] for e in st["entities"]]
        for s, shot in enumerate(shots_lp):
            n = len(shot["entities"]); sig, ips = adaptive(n)
            lisa.generate_shot(anchor_bank, layout_plan, shot_index=s, seed=42 + s,
                               sigma_override=sig, ip_scale_override=ips,
                               use_depth_occlusion=use_occ)
            img = Image.open(sdir / f"shot_{s:03d}.png").convert("RGB")
            dr = ImageDraw.Draw(img); W, H = img.size
            for ent in shot["entities"]:
                b = ent["bbox"]
                dr.rectangle([b[0]/L*W, b[1]/L*H, b[2]/L*W, b[3]/L*H],
                             outline=COLORS[enames.index(ent["name"]) % len(COLORS)], width=4)
            img.save(sdir / f"shot_{s:03d}_bbox.png")
        print(f"  {args.job} {st['name']} done", flush=True)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
