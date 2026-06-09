"""
v13: BBox-Localized Same-Entity Attention Sharing (training-free).

Generates a multi-shot story with GLIGEN; the ONLY cross-shot identity
mechanism is self-attention sharing (attn1) localized to each active entity's
predicted bbox. First occurrence captures the entity anchor; later shots inject
it. Runs sharing ON and OFF (same seeds) for A/B.

No adapter / no memory EMA / no pixel blend / no identity loss — isolates the
sharing mechanism.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_v13.py \
    --layout outputs/eval/user_story_001_boxes.json --out outputs/generations/v13
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


def run_story(pipe, ctrl, layouts, out_dir, steps, gscale, seed0, share):
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl.bank = {}                       # fresh bank per run
    ctrl.mode = "on" if share else "off"
    for li, shot in enumerate(layouts):
        sid = shot["shot_id"]
        boxes = dict(shot["boxes"])
        boxes = deoverlap_boxes(boxes)
        names = list(boxes.keys())
        ctrl.reset_active()
        ctrl.active = [(n, boxes[n]) for n in names]   # presence-aware: only this shot's entities
        g = torch.Generator(str(pipe.device)).manual_seed(seed0 + sid)
        res = pipe(
            prompt=shot["prompt"],
            gligen_phrases=names,
            gligen_boxes=[boxes[n] for n in names],
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=steps, guidance_scale=gscale,
            height=512, width=512, generator=g,
        )
        res.images[0].save(out_dir / f"shot_{sid:03d}.png")
        print(f"  shot {sid} ({'share' if share else 'base'}) entities={names}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", default="outputs/generations/v13")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--share-layers", nargs="*",
                    default=["mid_block", "up_blocks.1"])
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    device = "cuda"
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)

    ctrl = SharingController(share_layers=args.share_layers, cfg=True)
    install_sharing(pipe.unet, ctrl)
    attach_timestep_hook(pipe.unet, ctrl)

    plan = json.loads((base / args.layout).read_text())
    layouts = sorted(plan["shots"], key=lambda s: s["shot_id"])

    out = base / args.out
    print("=== sharing OFF (baseline) ===")
    run_story(pipe, ctrl, layouts, out / "off", args.steps,
              args.guidance_scale, args.seed, share=False)
    print("=== sharing ON ===")
    run_story(pipe, ctrl, layouts, out / "on", args.steps,
              args.guidance_scale, args.seed, share=True)
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
