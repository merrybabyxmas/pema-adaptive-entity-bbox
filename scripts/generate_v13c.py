"""
v13c: canonical-anchor attention sharing + alpha sweep, targeting GT identity
ceiling (DINOv2 same-individual ~0.557).

Improvements over v13:
  - canonical anchor: generate one clean single-entity image per entity (centered
    bbox), capture its self-attn K/V → inject into ALL story shots (incl. shot 0),
    so every shot aligns to the SAME stable reference (raises pairwise consistency).
  - alpha-max knob + extra share layers, swept to push toward the GT ceiling.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_v13c.py \
    --layout outputs/eval/user_story_001_boxes.json --out outputs/generations/v13c \
    --alphas 0.55 0.75 0.9 --share-layers mid_block up_blocks.1 up_blocks.2
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from PIL import Image

from diffusers import StableDiffusionGLIGENPipeline
from src.generation.pema_pipeline import GLIGEN_MODEL, deoverlap_boxes
from src.generation.entity_attention_sharing import (
    SharingController, install_sharing, attach_timestep_hook,
)

ANCHOR_BBOX = [0.28, 0.20, 0.72, 0.85]   # centered box for canonical anchor gen


def build_canonical_anchors(pipe, ctrl, entities, steps, gscale, seed):
    """Generate one clean image per entity and capture its bbox K/V as the anchor."""
    ctrl.bank = {}
    ctrl.mode = "on"; ctrl.freeze_bank = False
    for i, e in enumerate(sorted(entities)):
        ctrl.reset_active()
        ctrl.active = [(e, ANCHOR_BBOX)]
        g = torch.Generator(str(pipe.device)).manual_seed(seed + 1000 + i)
        pipe(prompt=f"a {e}, full body, centered, neutral background",
             gligen_phrases=[e], gligen_boxes=[ANCHOR_BBOX],
             gligen_scheduled_sampling_beta=1.0, num_inference_steps=steps,
             guidance_scale=gscale, height=512, width=512, generator=g)
    ctrl.freeze_bank = True   # lock: story shots only inject


def run_story(pipe, ctrl, layouts, out_dir, steps, gscale, seed, share):
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl.mode = "on" if share else "off"
    for shot in layouts:
        sid = shot["shot_id"]
        boxes = deoverlap_boxes(dict(shot["boxes"]))
        names = list(boxes.keys())
        ctrl.reset_active()
        ctrl.active = [(n, boxes[n]) for n in names]
        g = torch.Generator(str(pipe.device)).manual_seed(seed + sid)
        res = pipe(prompt=shot["prompt"], gligen_phrases=names,
                   gligen_boxes=[boxes[n] for n in names],
                   gligen_scheduled_sampling_beta=1.0, num_inference_steps=steps,
                   guidance_scale=gscale, height=512, width=512, generator=g)
        res.images[0].save(out_dir / f"shot_{sid:03d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", default="outputs/generations/v13c")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.55, 0.75, 0.9])
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

    plan = json.loads((base / args.layout).read_text())
    layouts = sorted(plan["shots"], key=lambda s: s["shot_id"])
    entities = sorted({n for s in layouts for n in s["boxes"]})

    out = base / args.out
    # baseline (no sharing) once
    ctrl.bank = {}; ctrl.freeze_bank = False
    run_story(pipe, ctrl, layouts, out / "off", args.steps, args.guidance_scale,
              args.seed, share=False)
    print("baseline (off) done", flush=True)
    # canonical anchors + each alpha
    for a in args.alphas:
        ctrl.alpha_max = a
        build_canonical_anchors(pipe, ctrl, entities, args.steps,
                                args.guidance_scale, args.seed)
        run_story(pipe, ctrl, layouts, out / f"a{a}", args.steps,
                  args.guidance_scale, args.seed, share=True)
        print(f"alpha={a} done", flush=True)
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
