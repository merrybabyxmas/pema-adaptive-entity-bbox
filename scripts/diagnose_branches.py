"""
Diagnostic: isolate which adapter branch causes background blurring.

For each validation sample, generate 4 variants with the SAME seed/prompt/bbox:
  (1) text-only    : adapter off (pure GLIGEN + text)
  (2) entity-only  : entity branch on, style branch off
  (3) style-only   : style branch on, entity branch off
  (4) full         : both branches on

Saves a horizontal strip per sample so the background regions can be compared
directly.  Background degradation that appears in (3)/(4) but NOT (2) implicates
the global (unmasked) style branch.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/diagnose_branches.py \
      --ckpt outputs/runs/phase4_adapter_v4/adapter_epoch0025.pt \
      --out outputs/runs/phase4_adapter_v4/diagnosis
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from diffusers import StableDiffusionGLIGENPipeline
from transformers import CLIPTokenizer, CLIPTextModel

from src.memory.entity_encoder import EntityEncoder
from src.model.style_encoder import StyleEncoder
from src.model.entity_style_adapter import EntityStyleAdapter
from src.generation.pema_pipeline import GLIGEN_MODEL

# reuse dataset + val builder from training script
from scripts.train_phase4 import Phase4Dataset, build_val_samples


def label_strip(images, labels):
    """Stack images horizontally with a text label bar above each."""
    w, h = images[0].size
    bar = 22
    strip = Image.new("RGB", (w * len(images), h + bar), "white")
    draw = ImageDraw.Draw(strip)
    for i, (img, lab) in enumerate(zip(images, labels)):
        strip.paste(img, (i * w, bar))
        draw.text((i * w + 4, 5), lab, fill="black")
    return strip


def gen(pipe, adapter, vs, device, ent_on, sty_on, seed, n_steps):
    ent_toks = None
    if ent_on:
        ent_toks = vs["entity_tokens"].unsqueeze(0).to(device).float()
    sty_toks = None
    if sty_on and vs["style_tokens"] is not None:
        sty_toks = vs["style_tokens"].unsqueeze(0).to(device).float()

    adapter.set_conditions(ent_toks, vs["entity_bboxes"], sty_toks)
    try:
        res = pipe(
            prompt=vs["prompt"],
            gligen_phrases=vs["entity_names"],
            gligen_boxes=vs["entity_bboxes"],
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=n_steps,
            height=512, width=512,
            generator=torch.Generator(str(device)).manual_seed(seed),
        )
        img = res.images[0]
    finally:
        adapter.clear_conditions()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  default="outputs/runs/phase4_adapter_v4/diagnosis")
    ap.add_argument("--data-dir", default="data/phase4_train")
    ap.add_argument("--style-encoder-path",
                    default="outputs/runs/pema_style_encoder/style_encoder_best.pt")
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    out_dir = base / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16

    print(f"Loading GLIGEN pipeline...")
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=dtype).to(str(device))
    pipe.set_progress_bar_config(disable=True)
    unet, vae = pipe.unet, pipe.vae
    text_encoder, tokenizer = pipe.text_encoder, pipe.tokenizer
    for m in [unet, vae, text_encoder]:
        for p in m.parameters():
            p.requires_grad_(False)

    print("Loading encoders...")
    entity_encoder = EntityEncoder(device=str(device))
    se_path = base / args.style_encoder_path
    ckpt_se = torch.load(str(se_path), map_location=device, weights_only=False)
    n_tok = ckpt_se.get("n_tokens", 4)
    style_encoder = StyleEncoder(n_tokens=n_tok).to(device)
    style_encoder.load_state_dict(ckpt_se["model"])
    style_encoder.eval()

    print("Pre-encoding dataset (for val samples)...")
    dataset = Phase4Dataset(
        data_dir=str(base / args.data_dir), vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, entity_encoder=entity_encoder,
        style_encoder=style_encoder, device=str(device),
        vae_scale=vae.config.get("scaling_factor", 0.18215),
    )
    val_samples = build_val_samples(base / args.data_dir, dataset,
                                    n_val=args.n_val, seed=42)

    print(f"Loading adapter: {args.ckpt}")
    adapter = EntityStyleAdapter.load(str(base / args.ckpt), unet, str(device))
    adapter.register_to_unet()
    adapter.processors.to(device)
    unet.eval()

    variants = [
        ("text-only",   False, False),
        ("entity-only", True,  False),
        ("style-only",  False, True),
        ("full",        True,  True),
    ]

    with torch.no_grad():
        for j, vs in enumerate(val_samples):
            imgs, labs = [], []
            for lab, ent_on, sty_on in variants:
                img = gen(pipe, adapter, vs, device, ent_on, sty_on,
                          seed=j, n_steps=args.steps)
                imgs.append(img)
                labs.append(lab)
            strip = label_strip(imgs, labs)
            strip.save(str(out_dir / f"sample_{j:02d}_strip.png"))
            print(f"  saved sample_{j:02d}_strip.png  | {vs['prompt'][:50]}")

    print(f"Done. -> {out_dir}")


if __name__ == "__main__":
    main()
