"""
v14 demo: Presence-aware BBox-localized Entity Image Adapter (training-free,
reuses trained IP-Adapter SD1.5 weights).

Compares on a 2-entity scene (cat=left bbox, dog=right bbox):
  text  : no image cond
  global: vanilla IP-Adapter, both refs global (expected: cross-entity leakage)
  ours  : per-entity bbox-localized image cross-attention

Metric: region-wise DINOv2 of generated cat-bbox / dog-bbox vs each anchor.
Good = cat-region matches cat-anchor AND dog-region matches dog-anchor, with
low cross terms (no leakage).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_v14.py \
    --cat-anchor outputs/generations/v13e/a0.5/anchors/anchor_cat_crop.png \
    --dog-anchor outputs/generations/v13e/a0.5/anchors/anchor_dog_crop.png \
    --out outputs/generations/v14 --scale 1.0
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch, numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline
from transformers import AutoModel, AutoImageProcessor
from src.generation.entity_ip_adapter import (
    EntityIPController, extract_ip_adapter, entity_tokens, install_entity_ip,
)

CAT_BBOX = [0.05, 0.28, 0.46, 0.92]
DOG_BBOX = [0.54, 0.18, 0.97, 0.95]
PROMPT = "a cat on the left and a dog on the right, sitting on pavement, photo"


def crop(img, b):
    W, H = img.size
    return img.crop((int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-anchor", required=True)
    ap.add_argument("--dog-anchor", required=True)
    ap.add_argument("--out", default="outputs/generations/v14")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    dev = "cuda"
    out = base / args.out; out.mkdir(parents=True, exist_ok=True)
    cat_img = Image.open(args.cat_anchor).convert("RGB")
    dog_img = Image.open(args.dog_anchor).convert("RGB")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
        safety_checker=None).to(dev)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.bin")

    def gen(seed):
        return torch.Generator(dev).manual_seed(seed)

    # (1) text-only
    pipe.set_ip_adapter_scale(0.0)
    img_text = pipe(prompt=PROMPT, ip_adapter_image=cat_img, num_inference_steps=args.steps,
                    guidance_scale=7.5, generator=gen(args.seed)).images[0]
    img_text.save(out / "text.png")

    # (2) vanilla GLOBAL IP-Adapter, both refs (expect leakage)
    pipe.set_ip_adapter_scale(args.scale)
    img_glob = pipe(prompt=PROMPT, ip_adapter_image=[[cat_img, dog_img]],
                    num_inference_steps=args.steps, guidance_scale=7.5,
                    generator=gen(args.seed)).images[0]
    img_glob.save(out / "global_ip.png")

    # (3) OURS: per-entity bbox-localized
    ip = extract_ip_adapter(pipe)
    cat_tok = entity_tokens(ip, cat_img, dev)
    dog_tok = entity_tokens(ip, dog_img, dev)
    ctrl = EntityIPController(scale=args.scale, cfg=True)
    # disable the standard IP path so only our processors inject
    pipe.unet.encoder_hid_proj = None
    pipe.unet.config.encoder_hid_dim_type = None
    install_entity_ip(pipe.unet, ctrl, ip)
    ctrl.set_active([("cat", cat_tok, CAT_BBOX), ("dog", dog_tok, DOG_BBOX)])
    img_ours = pipe(prompt=PROMPT, num_inference_steps=args.steps, guidance_scale=7.5,
                    generator=gen(args.seed)).images[0]
    img_ours.save(out / "ours.png")
    # bbox overlay
    ov = img_ours.copy(); d = ImageDraw.Draw(ov); W, H = ov.size
    for nm, b, c in [("cat", CAT_BBOX, "red"), ("dog", DOG_BBOX, "blue")]:
        d.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline=c, width=3)
    ov.save(out / "ours_bbox.png")

    # ── metric: region-wise DINOv2 vs anchors ──────────────────────────────
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    @torch.no_grad()
    def feat(im):
        x = proc(images=im, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(dino(**x).last_hidden_state[:, 0], dim=-1).squeeze(0)
    def sim(a, b): return float((feat(a)*feat(b)).sum())

    print(f"\n{'run':8} | catReg-vs-CATanc  catReg-vs-DOGanc | dogReg-vs-DOGanc  dogReg-vs-CATanc")
    for lab, im in [("text", img_text), ("global", img_glob), ("ours", img_ours)]:
        cr, dr = crop(im, CAT_BBOX), crop(im, DOG_BBOX)
        print(f"{lab:8} | {sim(cr,cat_img):.3f}            {sim(cr,dog_img):.3f}        "
              f"| {sim(dr,dog_img):.3f}            {sim(dr,cat_img):.3f}")
    print(f"\nsaved → {out} (want: ours catReg~CATanc high, dogReg~DOGanc high, cross low)")


if __name__ == "__main__":
    main()
