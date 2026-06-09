"""
v14 ablation: does the entity CLASS text prior ("dog") override the anchor
identity (black dog)? Same seed / anchors / boxes / strong IP (no gating).
Vary only the class text in BOTH the prompt and the GLIGEN grounding phrases.

Variants:
  full     : prompt "a cat and a dog ...", gligen ["cat","dog"]
  neutral  : prompt "two animals ...",     gligen ["animal","animal"]
  scene    : prompt "... on a sidewalk ..." (no entity noun), gligen ["animal","animal"]

If the dog turns BLACK (anchor) as the class text is removed → text prior was
suppressing the anchor identity (user's hypothesis). If it stays brown → the
anchor branch isn't strong enough on class semantics (→ needs training).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_v14_ablation.py \
    --cat-anchor outputs/generations/v13e/a0.5/anchors/anchor_cat_crop.png \
    --dog-anchor outputs/generations/v13e/a0.5/anchors/anchor_dog_crop.png \
    --out outputs/generations/v14_ablation
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, StableDiffusionGLIGENPipeline
from transformers import AutoModel, AutoImageProcessor
from src.generation.pema_pipeline import GLIGEN_MODEL
from src.generation.entity_ip_adapter import (
    EntityIPController, extract_ip_adapter, entity_tokens, install_entity_ip,
)
from src.generation.anchor_segment import AnchorSegmenter

CAT_BBOX = [0.06, 0.30, 0.46, 0.92]
DOG_BBOX = [0.54, 0.22, 0.96, 0.95]

VARIANTS = {
    "full":    dict(prompt="a cat and a dog sitting on a sidewalk, photo",
                    phrases=["cat", "dog"]),
    "neutral": dict(prompt="two animals sitting on a sidewalk, photo",
                    phrases=["animal", "animal"]),
    "scene":   dict(prompt="sitting on a sidewalk in front of a wall, photo",
                    phrases=["animal", "animal"]),
}


def crop(img, b):
    W, H = img.size
    return img.crop((int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-anchor", required=True)
    ap.add_argument("--dog-anchor", required=True)
    ap.add_argument("--out", default="outputs/generations/v14_ablation")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    dev = "cuda"
    out = base / args.out; out.mkdir(parents=True, exist_ok=True)
    cat_img = Image.open(args.cat_anchor).convert("RGB")
    dog_img = Image.open(args.dog_anchor).convert("RGB")

    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
        safety_checker=None).to(dev)
    sd.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                       weight_name="ip-adapter-plus_sd15.bin")
    ip = extract_ip_adapter(sd)
    seg = AnchorSegmenter(device=dev)
    cat_m, dog_m = seg.mask(cat_img), seg.mask(dog_img)
    cat_tok = entity_tokens(ip, cat_img, dev, obj_mask=cat_m)
    dog_tok = entity_tokens(ip, dog_img, dev, obj_mask=dog_m)

    g = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to(dev)
    g.set_progress_bar_config(disable=True)
    ctrl = EntityIPController(scale=args.scale, cfg=True,
                              t_apply_below=1000.0, feather=0.06)  # strong, all steps
    install_entity_ip(g.unet, ctrl, ip)
    ctrl.set_active([("cat", cat_tok, CAT_BBOX), ("dog", dog_tok, DOG_BBOX)])

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    @torch.no_grad()
    def feat(im):
        x = proc(images=im, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(dino(**x).last_hidden_state[:, 0], dim=-1).squeeze(0)
    def sim(a, b): return float((feat(a)*feat(b)).sum())

    print(f"{'variant':9} | catReg-vs-CAT  dogReg-vs-DOG")
    for name, cfg in VARIANTS.items():
        img = g(prompt=cfg["prompt"], gligen_phrases=cfg["phrases"],
                gligen_boxes=[CAT_BBOX, DOG_BBOX], gligen_scheduled_sampling_beta=1.0,
                num_inference_steps=args.steps, guidance_scale=7.5,
                height=512, width=512,
                generator=torch.Generator(dev).manual_seed(args.seed)).images[0]
        img.save(out / f"{name}.png")
        ov = img.copy(); d = ImageDraw.Draw(ov); W, H = ov.size
        for b, c in [(CAT_BBOX, "red"), (DOG_BBOX, "blue")]:
            d.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline=c, width=3)
        ov.save(out / f"{name}_bbox.png")
        print(f"{name:9} | {sim(crop(img,CAT_BBOX),cat_img):.3f}        "
              f"{sim(crop(img,DOG_BBOX),dog_img):.3f}   ({cfg['prompt']})")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
