"""
v14 showcase: ~10 examples chosen to suit the method — visually DISTINCTIVE
entities (clear markings/colors), large well-separated bboxes, static scenes
(no motion priors fighting identity).

Per example: generate a clean canonical anchor per entity (GLIGEN centered box)
→ SAM bg-mask → IP-Adapter-Plus tokens → v14 scene (GLIGEN layout + per-entity
bbox-localized identity, feathered, full prompt). Saves anchors + scene + bbox
overlay, and a final montage.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/make_showcase.py --out outputs/generations/showcase
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, StableDiffusionGLIGENPipeline
from src.generation.pema_pipeline import GLIGEN_MODEL
from src.generation.entity_ip_adapter import (
    EntityIPController, extract_ip_adapter, entity_tokens, install_entity_ip,
)
from src.generation.anchor_segment import AnchorSegmenter

L = [0.05, 0.28, 0.46, 0.92]
R = [0.54, 0.22, 0.96, 0.95]
ACENTER = [0.24, 0.14, 0.76, 0.92]

# (id, left_entity, right_entity, scene_prompt)
EXAMPLES = [
    ("tabby_retriever", "an orange tabby cat", "a golden retriever dog",
     "an orange tabby cat and a golden retriever sitting on a lawn, photo"),
    ("dalmatian_tuxedo", "a dalmatian dog", "a black and white tuxedo cat",
     "a dalmatian and a tuxedo cat on a wooden floor, photo"),
    ("rabbit_duck", "a fluffy white rabbit", "a brown mallard duck",
     "a white rabbit and a brown duck on green grass, photo"),
    ("horse_sheep", "a brown horse", "a fluffy white sheep",
     "a brown horse and a white sheep in a green meadow, photo"),
    ("redcar_taxi", "a red vintage car", "a yellow taxi cab",
     "a red vintage car and a yellow taxi parked on a street, photo"),
    ("parrot_cockatoo", "a green parrot", "a white cockatoo",
     "a green parrot and a white cockatoo perched on a branch, photo"),
    ("teddy_robot", "a brown teddy bear", "a silver toy robot",
     "a brown teddy bear and a silver toy robot on a shelf, photo"),
    ("elephant_zebra", "a grey elephant", "a striped zebra",
     "an elephant and a zebra in the savanna, photo"),
    ("penguin_polarbear", "a penguin", "a white polar bear",
     "a penguin and a polar bear on the snow, photo"),
    ("panda_redpanda", "a giant panda", "a red panda",
     "a giant panda and a red panda among green bamboo, photo"),
]


def draw(img, pairs):
    im = img.copy(); d = ImageDraw.Draw(im); W, H = im.size
    for b, c in pairs:
        d.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline=c, width=3)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/generations/showcase")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    dev = "cuda"
    out = base / args.out; out.mkdir(parents=True, exist_ok=True)

    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
        safety_checker=None).to(dev)
    sd.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                       weight_name="ip-adapter-plus_sd15.bin")
    ip = extract_ip_adapter(sd)
    seg = AnchorSegmenter(device=dev)
    g = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to(dev)
    g.set_progress_bar_config(disable=True)

    def gen_anchor(desc, sd_seed):
        # clean centered single-entity anchor via GLIGEN (no IP yet)
        return g(prompt=f"{desc}, full body, centered, plain neutral background, photo",
                 gligen_phrases=[desc], gligen_boxes=[ACENTER],
                 gligen_scheduled_sampling_beta=1.0, num_inference_steps=args.steps,
                 guidance_scale=7.5, height=512, width=512,
                 generator=torch.Generator(dev).manual_seed(sd_seed)).images[0]

    # Phase A: anchors (before installing IP)
    anchors = {}
    for i, (eid, le, re, _) in enumerate(EXAMPLES):
        for j, ent in enumerate([le, re]):
            img = gen_anchor(ent, args.seed + 100*i + j)
            tok = entity_tokens(ip, img, dev, obj_mask=seg.mask(img))
            anchors[(eid, j)] = (img, tok)
            (out / eid).mkdir(exist_ok=True)
            img.save(out / eid / f"anchor_{j}.png")
        print(f"anchors[{eid}] done", flush=True)

    # Phase B: install per-entity IP, generate scenes
    ctrl = EntityIPController(scale=args.scale, cfg=True, t_apply_below=1000.0,
                             feather=0.10, text_suppress=1.0)
    install_entity_ip(g.unet, ctrl, ip)
    for i, (eid, le, re, scene) in enumerate(EXAMPLES):
        _, ltok = anchors[(eid, 0)]; _, rtok = anchors[(eid, 1)]
        ctrl.set_active([(le, ltok, L), (re, rtok, R)])
        img = g(prompt=scene, gligen_phrases=[le, re], gligen_boxes=[L, R],
                gligen_scheduled_sampling_beta=1.0, num_inference_steps=args.steps,
                guidance_scale=7.5, height=512, width=512,
                generator=torch.Generator(dev).manual_seed(args.seed + i)).images[0]
        img.save(out / eid / "scene.png")
        draw(img, [(L, "red"), (R, "blue")]).save(out / eid / "scene_bbox.png")
        print(f"scene[{eid}] done", flush=True)

    # montage: per example -> [anchorL | anchorR | scene]
    W = H = 150; bar = 14; rows = len(EXAMPLES)
    grid = Image.new("RGB", (3*W, rows*(H+bar)), "white"); d = ImageDraw.Draw(grid)
    for i, (eid, le, re, _) in enumerate(EXAMPLES):
        y = i*(H+bar)
        grid.paste(anchors[(eid, 0)][0].resize((W, H)), (0, y+bar))
        grid.paste(anchors[(eid, 1)][0].resize((W, H)), (W, y+bar))
        grid.paste(Image.open(out / eid / "scene.png").convert("RGB").resize((W, H)), (2*W, y+bar))
        d.text((3, y+2), f"{eid}: anchorL | anchorR | scene", fill="black")
    grid.save(out / "_montage.png")
    print(f"saved montage → {out}/_montage.png")


if __name__ == "__main__":
    main()
