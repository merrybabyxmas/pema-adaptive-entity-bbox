"""
v14 + GLIGEN: layout grounding (GLIGEN) + per-entity bbox-localized IP identity.

Diagnosis showed the EntityIPAttnProcessor faithfully reproduces IP-Adapter
identity transfer (single-entity full-frame DINOv2 0.467 ~ global IP 0.489), but
WITHOUT layout the entity isn't drawn in its bbox so identity lands on empty
background. Fix: GLIGEN places each entity in its bbox; our processor injects
that entity's anchor identity localized to the same bbox.

IP-Adapter trained weights (to_k_ip/to_v_ip + image_proj) are extracted from a
StableDiffusionPipeline (which supports load_ip_adapter) and installed onto the
GLIGEN UNet (same SD1.x attn2 layer structure → keys match).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_v14_gligen.py \
    --cat-anchor outputs/generations/v13e/a0.5/anchors/anchor_cat_crop.png \
    --dog-anchor outputs/generations/v13e/a0.5/anchors/anchor_dog_crop.png \
    --out outputs/generations/v14g --scale 1.0
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

CAT_BBOX = [0.06, 0.30, 0.46, 0.92]
DOG_BBOX = [0.54, 0.22, 0.96, 0.95]
PROMPT = "a cat and a dog sitting on pavement, photo"


def crop(img, b):
    W, H = img.size
    return img.crop((int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-anchor", required=True)
    ap.add_argument("--dog-anchor", required=True)
    ap.add_argument("--out", default="outputs/generations/v14g")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--weight-name", default="ip-adapter_sd15.bin",
                    help="ip-adapter_sd15.bin (4 tok) or ip-adapter-plus_sd15.bin (16 tok, stronger id)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    dev = "cuda"
    out = base / args.out; out.mkdir(parents=True, exist_ok=True)
    cat_img = Image.open(args.cat_anchor).convert("RGB")
    dog_img = Image.open(args.dog_anchor).convert("RGB")

    # 1) extract trained IP-Adapter weights from an SD pipe
    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
        safety_checker=None).to(dev)
    sd.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                       weight_name=args.weight_name)
    ip = extract_ip_adapter(sd)
    cat_tok = entity_tokens(ip, cat_img, dev)
    dog_tok = entity_tokens(ip, dog_img, dev)

    # 2) GLIGEN pipe for layout
    g = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to(dev)
    g.set_progress_bar_config(disable=True)

    names = ["cat", "dog"]; boxes = [CAT_BBOX, DOG_BBOX]

    def run_gligen(gen):
        return g(prompt=PROMPT, gligen_phrases=names, gligen_boxes=boxes,
                 gligen_scheduled_sampling_beta=1.0, num_inference_steps=args.steps,
                 guidance_scale=7.5, height=512, width=512, generator=gen).images[0]

    # baseline: GLIGEN layout only (no IP identity)
    img_base = run_gligen(torch.Generator(dev).manual_seed(args.seed))
    img_base.save(out / "gligen_only.png")

    # 3) install our per-entity IP processors onto the GLIGEN unet
    ctrl = EntityIPController(scale=args.scale, cfg=True)
    install_entity_ip(g.unet, ctrl, ip)
    ctrl.set_active([("cat", cat_tok, CAT_BBOX), ("dog", dog_tok, DOG_BBOX)])
    img_ours = run_gligen(torch.Generator(dev).manual_seed(args.seed))
    img_ours.save(out / "gligen_ours.png")
    ov = img_ours.copy(); d = ImageDraw.Draw(ov); W, H = ov.size
    for nm, b, c in [("cat", CAT_BBOX, "red"), ("dog", DOG_BBOX, "blue")]:
        d.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline=c, width=3)
    ov.save(out / "gligen_ours_bbox.png")

    # metric
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    @torch.no_grad()
    def feat(im):
        x = proc(images=im, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(dino(**x).last_hidden_state[:, 0], dim=-1).squeeze(0)
    def sim(a, b): return float((feat(a)*feat(b)).sum())
    print(f"\n{'run':12} | catReg-vs-CAT  dogReg-vs-DOG | cross(cat-vs-DOG, dog-vs-CAT)")
    for lab, im in [("gligen_only", img_base), ("gligen+ours", img_ours)]:
        cr, dr = crop(im, CAT_BBOX), crop(im, DOG_BBOX)
        print(f"{lab:12} | {sim(cr,cat_img):.3f}        {sim(dr,dog_img):.3f}      "
              f"| {sim(cr,dog_img):.3f}, {sim(dr,cat_img):.3f}")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
