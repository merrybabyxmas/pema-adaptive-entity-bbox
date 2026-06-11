"""
Image-quality GUARD metrics (secondary; not the main metric). Per generated keyframe:
  clip_t    : CLIP image-text cosine (open_clip ViT-B/32) vs the shot prompt.
  aesthetic : LAION aesthetic predictor (open_clip ViT-L/14 + linear MSE head). Best-effort;
              skipped (NaN) if the head weights cannot be fetched.
  sharpness : variance of Laplacian (cv2) — higher = sharper (blur guard).
(ImageReward is incompatible with the installed transformers; omitted and noted.)

Output CSV: method,story,shot,clip_t,aesthetic,sharpness

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_image_quality.py \
    --jobs FINAL_combo,B_template --root outputs/lisa/aaai_ablation --out outputs/eval_120/metrics/quality.csv
"""
import sys, os, argparse, json, csv, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import cv2
from PIL import Image
import open_clip

AES_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"


class AesMLP(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default="outputs/lisa/aaai_ablation")
    ap.add_argument("--stories", default="data/captions/stories_aaai_eval_120.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(__file__).parent.parent; dev = "cuda"
    stories = json.loads((base / args.stories).read_text())

    clip_b, _, prep_b = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_b = clip_b.to(dev).eval(); tok = open_clip.get_tokenizer("ViT-B-32")

    # aesthetic (best-effort)
    aes = aes_l = prep_l = None
    try:
        cache = base / "outputs/eval_120/aes_head.pth"
        if not cache.exists():
            urllib.request.urlretrieve(AES_URL, str(cache))
        clip_l, _, prep_l = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_l = clip_l.to(dev).eval()
        aes = AesMLP(768).to(dev).eval()
        sd = torch.load(str(cache), map_location="cpu")
        aes.load_state_dict(sd); aes_l = clip_l
        print("[quality] aesthetic predictor loaded")
    except Exception as e:
        print(f"[quality] aesthetic unavailable ({repr(e)[:80]}) -> NaN")

    def clip_t(img, prompt):
        im = prep_b(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            ie = clip_b.encode_image(im); te = clip_b.encode_text(tok([prompt]).to(dev))
        ie = ie / ie.norm(dim=-1, keepdim=True); te = te / te.norm(dim=-1, keepdim=True)
        return float((ie * te).sum())

    def aesthetic(img):
        if aes is None:
            return float("nan")
        im = prep_l(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            f = aes_l.encode_image(im); f = f / f.norm(dim=-1, keepdim=True)
            return float(aes(f.float())[0, 0])

    def sharp(img):
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    rows = []
    for job in args.jobs.split(","):
        jdir = base / args.root / job
        for st in stories:
            for t, sh in enumerate(st["shots"]):
                p = jdir / st["name"] / f"shot_{t:03d}.png"
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                rows.append([job, st["name"], t, round(clip_t(img, sh["prompt"]), 4),
                             round(aesthetic(img), 4), round(sharp(img), 2)])
        print(f"  {job} done", flush=True)
    outp = base / args.out; outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "story", "shot", "clip_t", "aesthetic", "sharpness"]); w.writerows(rows)
    print(f"saved -> {outp} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
