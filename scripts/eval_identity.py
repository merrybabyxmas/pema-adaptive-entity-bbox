"""
Identity-preservation eval on held-out SAME-INSTANCE pairs (phase4_identity).

For each held-out pair (ref crop + target frame/bbox of the SAME individual):
  - encode ref crop as entity tokens (pooled or patch, auto from adapter)
  - generate with prompt + bbox
  - crop the generated entity bbox region
  - measure CLIP-sim(generated crop, ref crop)         → ref fidelity
            CLIP-sim(generated crop, target crop)      → identity-to-true-instance

Compares an adapter trained on different-instance data (v10) vs same-instance
data (v12a) on the SAME held-out pairs → isolates the data fix.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/eval_identity.py \
    --ckpt outputs/runs/phase4_adapter_v12a/adapter_best.pt \
    --data-dir data/phase4_identity --val-index identity_val.json
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionGLIGENPipeline
from src.memory.entity_encoder import EntityEncoder
from src.model.entity_style_adapter import EntityStyleAdapter
from src.generation.pema_pipeline import GLIGEN_MODEL


def crop_box(img, box):
    W, H = img.size
    x1, y1, x2, y2 = box
    return img.crop((int(x1*W), int(y1*H), int(x2*W), int(y2*H)))


class DinoSim:
    """DINOv2 instance-level similarity (CLS token cosine). More discriminative
    of 'same individual' than CLIP, which keys on semantic category."""
    def __init__(self, device="cuda"):
        from transformers import AutoModel, AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.device = device

    @torch.no_grad()
    def feat(self, img):
        x = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.model(**x).last_hidden_state[:, 0]   # CLS
        return torch.nn.functional.normalize(out, dim=-1).squeeze(0)

    def sim(self, a, b):
        return float((self.feat(a) * self.feat(b)).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/phase4_identity")
    ap.add_argument("--val-index", default="identity_val.json")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--grid", type=int, default=4)
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    dd = base / args.data_dir
    device = "cuda"
    val = json.loads((dd/args.val_index).read_text())[:args.n]

    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet
    for p in unet.parameters(): p.requires_grad_(False)
    enc = EntityEncoder(device=device)
    adapter = EntityStyleAdapter.load(str(base/args.ckpt), unet, device=device)
    adapter.register_to_unet(); adapter.processors.to(device); unet.eval()
    is_patch = adapter.processors[next(iter(adapter.processors))].entity_to_k.in_features == 1280
    print(f"mode={'PATCH' if is_patch else 'POOLED'}  val_pairs={len(val)}")

    def enc_ent(img):
        t = enc.encode_patches(img, grid=args.grid) if is_patch else enc.encode(img)
        return (t/(t.norm(dim=-1, keepdim=True)+1e-8)).cpu()

    dino = DinoSim(device=device)
    ref_sims, tgt_sims = [], []
    dref_sims, dtgt_sims = [], []
    with torch.no_grad():
        for k, s in enumerate(val):
            sdir = dd/s
            m = json.loads((sdir/"metadata.json").read_text())
            e = m["entities"][0]; box = e["box_xyxy"]
            ref_img = Image.open(sdir/e["ref_image"]).convert("RGB")
            tgt_crop = crop_box(Image.open(sdir/"scene.png").convert("RGB"), box)
            # shape to (1, n_ent=1, dim) pooled or (1, n_ent=1, K_e, dim) patch
            toks = enc_ent(ref_img).unsqueeze(0).unsqueeze(0).to(device).float()
            adapter.set_conditions(toks, [box], None)
            try:
                img = pipe(prompt=m["prompt"], gligen_phrases=[e["name"]],
                           gligen_boxes=[box], gligen_scheduled_sampling_beta=1.0,
                           num_inference_steps=args.steps, height=512, width=512,
                           generator=torch.Generator(device).manual_seed(k)).images[0]
            finally:
                adapter.clear_conditions()
            gen_crop = crop_box(img, box)
            gf = enc.encode(gen_crop)
            ref_sims.append(enc.similarity(gf, enc.encode(ref_img)))
            tgt_sims.append(enc.similarity(gf, enc.encode(tgt_crop)))
            dref_sims.append(dino.sim(gen_crop, ref_img))
            dtgt_sims.append(dino.sim(gen_crop, tgt_crop))

    print(f"gen-vs-REF    CLIP={np.mean(ref_sims):.4f}  DINOv2={np.mean(dref_sims):.4f}  (n={len(ref_sims)})")
    print(f"gen-vs-TARGET CLIP={np.mean(tgt_sims):.4f}  DINOv2={np.mean(dtgt_sims):.4f}  (same individual)")


if __name__ == "__main__":
    main()
