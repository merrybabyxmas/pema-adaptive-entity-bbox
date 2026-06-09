"""
v13e: anchor-visible attention sharing.

CRITICAL DEBUG FIX: the canonical anchor (its bbox crop's self-attn K/V) is the
single most important driver of v13 — but it was never saved, so collapse causes
could not be isolated (bad anchor? bad sharing? bad bbox? weak base model?).

This script:
  - generates N anchor candidates per entity, saves ALL (for human inspection),
  - scores each (CLIP text-match + sharpness + not-collapsed) and picks best,
  - saves the chosen anchor img + bbox overlay + bbox CROP (what actually feeds
    the bank) + meta.json,
  - captures the chosen anchor's K/V into the bank,
  - generates the story shots (+ bbox overlays) into the same run folder.

Run folder:
  out/anchors/{entity}_cand_NN.png, anchor_{entity}.png, anchor_{entity}_bbox.png,
              anchor_{entity}_crop.png, anchor_{entity}_meta.json, scores.json
  out/shots/shot_NNN.png, shot_NNN_bbox.png
  out/<alpha runs>/...

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_v13e.py \
    --layout outputs/eval/user_story_001_boxes.json --out outputs/generations/v13e \
    --alphas 0.5 --n-candidates 4
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from diffusers import StableDiffusionGLIGENPipeline
from src.generation.pema_pipeline import GLIGEN_MODEL, deoverlap_boxes
from src.generation.entity_attention_sharing import (
    SharingController, install_sharing, attach_timestep_hook,
)

ANCHOR_BBOX = [0.28, 0.20, 0.72, 0.85]


def draw_bbox(img, boxes_named, color="red"):
    im = img.copy(); d = ImageDraw.Draw(im); W, H = im.size
    for name, b in boxes_named:
        d.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline=color, width=3)
        d.text((b[0]*W+3, b[1]*H+3), name, fill=color)
    return im


def crop_box(img, b):
    W, H = img.size
    return img.crop((int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)))


class ClipScorer:
    def __init__(self, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

    @torch.no_grad()
    def match(self, img, text):
        x = self.p(text=[f"a photo of a {text}"], images=img, return_tensors="pt",
                   padding=True).to(self.device)
        o = self.m(**x)
        return float(o.logits_per_image[0, 0].item())


def score_candidate(crop, clip_score):
    g = np.asarray(crop.convert("L"), dtype=np.float64)
    bright = float(g.mean())
    blur = float(ndimage.laplace(g).var())
    collapsed = bool(bright < 18 or bright > 238 or blur < 30)
    # combined: clip dominates, penalize collapse
    s = float(clip_score + 0.002 * min(blur, 2000) - (100 if collapsed else 0))
    return dict(clip=round(float(clip_score), 2), blur=round(blur, 1),
                bright=round(bright, 1), collapsed=collapsed, score=round(s, 3))


def build_anchors(pipe, ctrl, entities, anchor_dir, steps, gscale, seed,
                  n_cand, clip, alpha, share_layers):
    anchor_dir.mkdir(parents=True, exist_ok=True)
    ctrl.bank = {}; ctrl.mode = "off"; ctrl.freeze_bank = False
    chosen = {}
    all_scores = {}
    for ei, e in enumerate(sorted(entities)):
        prompt = f"a {e.replace('/', ' ')}, full body, centered, plain neutral background"
        cands = []
        for ci in range(n_cand):
            g = torch.Generator(str(pipe.device)).manual_seed(seed + 1000 + ei*10 + ci)
            ctrl.reset_active()
            img = pipe(prompt=prompt, gligen_phrases=[e], gligen_boxes=[ANCHOR_BBOX],
                       gligen_scheduled_sampling_beta=1.0, num_inference_steps=steps,
                       guidance_scale=gscale, height=512, width=512, generator=g).images[0]
            img.save(anchor_dir / f"{e.replace('/','_')}_cand_{ci:02d}.png")
            crop = crop_box(img, ANCHOR_BBOX)
            sc = score_candidate(crop, clip.match(crop, e) if clip else 0.0)
            sc["seed"] = seed + 1000 + ei*10 + ci
            cands.append((img, crop, sc))
        cands.sort(key=lambda c: -c[2]["score"])
        best_img, best_crop, best_sc = cands[0]
        slug = e.replace('/', '_')
        best_img.save(anchor_dir / f"anchor_{slug}.png")
        draw_bbox(best_img, [(e, ANCHOR_BBOX)]).save(anchor_dir / f"anchor_{slug}_bbox.png")
        best_crop.save(anchor_dir / f"anchor_{slug}_crop.png")
        meta = dict(entity=e, prompt=prompt, anchor_bbox=ANCHOR_BBOX, seed=best_sc["seed"],
                    share_layers=share_layers, alpha=alpha, steps=steps,
                    image_size=512, scores=[c[2] for c in cands])
        (anchor_dir / f"anchor_{slug}_meta.json").write_text(json.dumps(meta, indent=2))
        all_scores[e] = best_sc
        chosen[e] = best_sc["seed"]
        print(f"  anchor[{e}] best seed={best_sc['seed']} {best_sc}", flush=True)
    (anchor_dir / "scores.json").write_text(json.dumps(all_scores, indent=2))

    # capture the chosen anchors into the bank
    ctrl.bank = {}; ctrl.mode = "on"; ctrl.freeze_bank = False
    for ei, e in enumerate(sorted(entities)):
        prompt = f"a {e.replace('/', ' ')}, full body, centered, plain neutral background"
        g = torch.Generator(str(pipe.device)).manual_seed(chosen[e])
        ctrl.reset_active(); ctrl.active = [(e, ANCHOR_BBOX)]
        pipe(prompt=prompt, gligen_phrases=[e], gligen_boxes=[ANCHOR_BBOX],
             gligen_scheduled_sampling_beta=1.0, num_inference_steps=steps,
             guidance_scale=gscale, height=512, width=512, generator=g)
    ctrl.freeze_bank = True
    print(f"  bank captured for {sorted(entities)}", flush=True)


def run_story(pipe, ctrl, layouts, out_dir, steps, gscale, seed, share):
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl.mode = "on" if share else "off"
    for shot in layouts:
        sid = shot["shot_id"]
        boxes = deoverlap_boxes(dict(shot["boxes"]))
        names = list(boxes.keys())
        ctrl.reset_active(); ctrl.active = [(n, boxes[n]) for n in names]
        g = torch.Generator(str(pipe.device)).manual_seed(seed + sid)
        img = pipe(prompt=shot["prompt"], gligen_phrases=names,
                   gligen_boxes=[boxes[n] for n in names],
                   gligen_scheduled_sampling_beta=1.0, num_inference_steps=steps,
                   guidance_scale=gscale, height=512, width=512, generator=g).images[0]
        img.save(out_dir / f"shot_{sid:03d}.png")
        draw_bbox(img, [(n, boxes[n]) for n in names]).save(out_dir / f"shot_{sid:03d}_bbox.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", default="outputs/generations/v13e")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5])
    ap.add_argument("--n-candidates", type=int, default=1)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--share-layers", nargs="*",
                    default=["mid_block", "up_blocks.1", "up_blocks.2"])
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    ctrl = SharingController(share_layers=args.share_layers, cfg=True)
    install_sharing(pipe.unet, ctrl)
    attach_timestep_hook(pipe.unet, ctrl)
    clip = None if args.no_clip else ClipScorer("cuda")

    plan = json.loads((base / args.layout).read_text())
    layouts = sorted(plan["shots"], key=lambda s: s["shot_id"])
    entities = sorted({n for s in layouts for n in s["boxes"]})
    out = base / args.out

    # baseline (no sharing)
    ctrl.bank = {}; ctrl.freeze_bank = False
    run_story(pipe, ctrl, layouts, out / "off", args.steps, args.guidance_scale,
              args.seed, share=False)
    print("baseline (off) done", flush=True)

    for a in args.alphas:
        ctrl.alpha_max = a
        build_anchors(pipe, ctrl, entities, out / f"a{a}" / "anchors",
                      args.steps, args.guidance_scale, args.seed,
                      args.n_candidates, clip, a, args.share_layers)
        run_story(pipe, ctrl, layouts, out / f"a{a}" / "shots", args.steps,
                  args.guidance_scale, args.seed, share=True)
        print(f"alpha={a} done → {out}/a{a}", flush=True)
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
