"""
v14 multi-shot: presence-aware, bbox-localized entity identity across a story.

Winning config (training-free): GLIGEN layout + per-entity IP-Adapter-Plus image
cross-attention, SAM patch-token bg-masked anchors, feathered bbox masks, strong
all-step IP, full text prompt. FIXED per-entity anchors are injected into every
shot where that entity is present (presence matrix) and localized to that shot's
predicted bbox → same individual across shots.

Story (user_story_001_boxes.json): shot0 cat+dog, shot1 dog, shot2 cat+dog.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_v14_multishot.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --cat-anchor outputs/generations/v13e/a0.5/anchors/anchor_cat_crop.png \
    --dog-anchor outputs/generations/v13e/a0.5/anchors/anchor_dog_crop.png \
    --out outputs/generations/v14_multishot
"""
import sys, os, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, StableDiffusionGLIGENPipeline
from transformers import AutoModel, AutoImageProcessor
from src.generation.pema_pipeline import GLIGEN_MODEL, deoverlap_boxes
from src.generation.entity_ip_adapter import (
    EntityIPController, extract_ip_adapter, entity_tokens, install_entity_ip,
)
from src.generation.anchor_segment import AnchorSegmenter


def crop(img, b):
    W, H = img.size
    return img.crop((int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--cat-anchor", required=True)
    ap.add_argument("--dog-anchor", required=True)
    ap.add_argument("--out", default="outputs/generations/v14_multishot")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--feather", type=float, default=0.10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ip", action="store_true", help="baseline: GLIGEN only")
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    dev = "cuda"
    out = base / args.out; out.mkdir(parents=True, exist_ok=True)
    anchors = {"cat": Image.open(args.cat_anchor).convert("RGB"),
               "dog": Image.open(args.dog_anchor).convert("RGB")}

    # trained IP-Adapter-Plus weights + per-entity (bg-masked) tokens
    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
        safety_checker=None).to(dev)
    sd.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                       weight_name="ip-adapter-plus_sd15.bin")
    ip = extract_ip_adapter(sd)
    seg = AnchorSegmenter(device=dev)
    toks = {e: entity_tokens(ip, img, dev, obj_mask=seg.mask(img))
            for e, img in anchors.items()}

    g = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to(dev)
    g.set_progress_bar_config(disable=True)
    ctrl = EntityIPController(scale=args.scale, cfg=True, t_apply_below=1000.0,
                             feather=args.feather, text_suppress=1.0)
    if not args.no_ip:
        install_entity_ip(g.unet, ctrl, ip)

    plan = json.loads((base / args.layout).read_text())
    layouts = sorted(plan["shots"], key=lambda s: s["shot_id"])

    shot_imgs = {}
    for shot in layouts:
        sid = shot["shot_id"]
        boxes = deoverlap_boxes(dict(shot["boxes"]))
        names = list(boxes.keys())
        # presence-aware: only entities present in this shot inject their anchor
        ctrl.set_active([(n, toks[n], boxes[n]) for n in names if n in toks])
        img = g(prompt=shot["prompt"], gligen_phrases=names,
                gligen_boxes=[boxes[n] for n in names],
                gligen_scheduled_sampling_beta=1.0, num_inference_steps=args.steps,
                guidance_scale=7.5, height=512, width=512,
                generator=torch.Generator(dev).manual_seed(args.seed + sid)).images[0]
        img.save(out / f"shot_{sid:03d}.png")
        ov = img.copy(); d = ImageDraw.Draw(ov); W, H = ov.size
        cols = {"cat": "red", "dog": "blue"}
        for n in names:
            b = boxes[n]; c = cols.get(n, "green")
            d.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline=c, width=3)
            d.text((b[0]*W+3, b[1]*H+3), n, fill=c)
        ov.save(out / f"shot_{sid:03d}_bbox.png")
        shot_imgs[sid] = (img, boxes)
        print(f"shot {sid}: active={names}", flush=True)

    # ── metrics: cross-shot consistency + anchor fidelity (DINOv2) ──────────
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    @torch.no_grad()
    def feat(im):
        x = proc(images=im, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(dino(**x).last_hidden_state[:, 0], dim=-1).squeeze(0)
    def sim(a, b): return float((feat(a)*feat(b)).sum())

    # per entity: which shots it appears in
    appears = {}
    for sid, (im, boxes) in shot_imgs.items():
        for n in boxes:
            appears.setdefault(n, []).append(sid)
    print("\n── cross-shot consistency (same entity region, shot-pair DINOv2) ──")
    for e, sids in appears.items():
        regions = {s: crop(shot_imgs[s][0], shot_imgs[s][1][e]) for s in sids}
        pairs = list(itertools.combinations(sids, 2))
        cs = [sim(regions[a], regions[b]) for a, b in pairs]
        anc = [sim(regions[s], anchors[e]) for s in sids]
        pstr = ", ".join(f"{a}-{b}:{sim(regions[a],regions[b]):.3f}" for a, b in pairs)
        print(f"  {e}: cross-shot[{pstr}] mean={sum(cs)/len(cs):.3f} | vs-anchor mean={sum(anc)/len(anc):.3f}")
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
