"""
Ref-fidelity eval: generate the fixed val samples with a trained adapter, crop
each entity's bbox region from the generated image, and measure CLIP-sim to the
entity's reference image. Auto-detects pooled vs patch entity mode from the
adapter checkpoint (entity_to_k.in_features: 1024=pooled, 1280=patch).

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/eval_ref_fidelity.py \
    --ckpt outputs/runs/phase4_adapter_v10/adapter_best.pt
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionGLIGENPipeline
from src.memory.entity_encoder import EntityEncoder
from src.model.style_encoder import StyleEncoder
from src.model.entity_style_adapter import EntityStyleAdapter
from src.generation.pema_pipeline import GLIGEN_MODEL
from scripts.train_phase4 import Phase4Dataset, build_val_samples


def crop_box(img, box):
    W, H = img.size
    x1, y1, x2, y2 = box
    return img.crop((int(x1*W), int(y1*H), int(x2*W), int(y2*H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/phase4_train")
    ap.add_argument("--style-encoder-path",
                    default="outputs/runs/pema_style_encoder/style_encoder_best.pt")
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--grid", type=int, default=4)
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    device = "cuda"
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)
    unet, vae = pipe.unet, pipe.vae
    for m in [unet, vae, pipe.text_encoder]:
        for p in m.parameters():
            p.requires_grad_(False)

    enc = EntityEncoder(device=device)
    se = torch.load(str(base/args.style_encoder_path), map_location=device, weights_only=False)
    style_encoder = StyleEncoder(n_tokens=se.get("n_tokens", 4)).to(device)
    style_encoder.load_state_dict(se["model"]); style_encoder.eval()

    dataset = Phase4Dataset(
        data_dir=str(base/args.data_dir), vae=vae, text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer, entity_encoder=enc, style_encoder=style_encoder,
        device=device, vae_scale=vae.config.get("scaling_factor", 0.18215))
    val = build_val_samples(base/args.data_dir, dataset, n_val=args.n_val, seed=42)

    adapter = EntityStyleAdapter.load(str(base/args.ckpt), unet, device=device)
    adapter.register_to_unet(); adapter.processors.to(device)
    unet.eval()
    is_patch = adapter.processors[next(iter(adapter.processors))].entity_to_k.in_features == 1280
    print(f"adapter mode: {'PATCH' if is_patch else 'POOLED'}")

    def encode_entity(img):
        if is_patch:
            t = enc.encode_patches(img, grid=args.grid)        # (K_e,1280)
        else:
            t = enc.encode(img)                                 # (1024,)
        return (t / (t.norm(dim=-1, keepdim=True) + 1e-8)).cpu()

    sims = []
    with torch.no_grad():
        for j, vs in enumerate(val):
            # encode entity tokens from the ref images actually used
            refs = []
            for rp in vs["ref_paths"]:
                p = Path(rp)
                if not p.is_absolute():
                    p = Path(vs["scene_path"]).parent / rp
                refs.append(Image.open(str(p)).convert("RGB") if p.exists() else None)
            if any(r is None for r in refs):
                continue
            toks = torch.stack([encode_entity(r) for r in refs]).unsqueeze(0).to(device).float()
            adapter.set_conditions(toks, vs["entity_bboxes"],
                                   vs["style_tokens"].unsqueeze(0).to(device).float()
                                   if vs["style_tokens"] is not None else None)
            try:
                img = pipe(prompt=vs["prompt"], gligen_phrases=vs["entity_names"],
                           gligen_boxes=vs["entity_bboxes"],
                           gligen_scheduled_sampling_beta=1.0,
                           num_inference_steps=args.steps, height=512, width=512,
                           generator=torch.Generator(device).manual_seed(j)).images[0]
            finally:
                adapter.clear_conditions()
            # crop each entity bbox, compare to its ref
            for box, ref in zip(vs["entity_bboxes"], refs):
                gen_crop = crop_box(img, box)
                sims.append(enc.similarity(enc.encode(gen_crop), enc.encode(ref)))

    print(f"REF-FIDELITY (entity bbox crop vs ref CLIP-sim): "
          f"mean={np.mean(sims):.4f}  n={len(sims)}")


if __name__ == "__main__":
    main()
