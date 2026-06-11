"""
Multi-shot consistency metrics (ViStoryBench-style, extended for our task).

For each story (a multi-shot sequence) and each generation job (planner variant):
  identity_consistency : an entity that appears in >=2 shots should look like the SAME
      individual. We localize the entity per shot with OWLv2 (open-vocab), crop it, embed
      with DINOv2 (instance-discriminative CLS), and average the cross-shot pairwise cosine.
  background_consistency: the scene/background should persist across shots. We mask out the
      detected entity boxes (gray) in each shot, embed the background-only image with DINOv2,
      and average the cross-shot pairwise cosine.

Aggregated per job. Higher = more consistent.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_multishot.py \
    --jobs CABL_full,CABL_wo_state --root outputs/lisa/aaai_cablation --out outputs/cabl_logs/ms_0.json
"""
import sys, os, argparse, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import (Owlv2Processor, Owlv2ForObjectDetection,
                          AutoImageProcessor, AutoModel)

LABELMAP = {"bus/truck": "bus", "sheep/goat": "sheep", "ball/sports_ball": "ball", "cattle": "cow"}
DET_THR = 0.15


def lab(n):
    return LABELMAP.get(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default="outputs/lisa/aaai_cablation")
    ap.add_argument("--stories", default="data/captions/stories_aaai_eval_120.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    dev = "cuda"
    stories = json.loads((base / args.stories).read_text())

    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    dinop = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()

    def dino_emb(img):
        inp = dinop(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            v = dino(**inp).last_hidden_state[:, 0]      # CLS [1,768]
        return torch.nn.functional.normalize(v, dim=-1)[0]

    def detect(img, label):
        inp = owlp(text=[[f"a photo of a {lab(label)}"]], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = owl(**inp)
        r = owlp.post_process_grounded_object_detection(
            o, target_sizes=torch.tensor([img.size[::-1]]).to(dev), threshold=DET_THR)[0]
        if len(r["scores"]) == 0:
            return None
        i = int(r["scores"].argmax())
        return [int(v) for v in r["boxes"][i].tolist()]

    def pairwise_cos(embs):
        if len(embs) < 2:
            return None
        sims = [float(torch.dot(a, b)) for a, b in itertools.combinations(embs, 2)]
        return sum(sims) / len(sims)

    results = {}
    for job in args.jobs.split(","):
        jdir = base / args.root / job
        id_scores, bg_scores = [], []
        for st in stories:
            nsh = len(st["shots"])
            imgs, dets = [], []
            for s in range(nsh):
                p = jdir / st["name"] / f"shot_{s:03d}.png"
                imgs.append(Image.open(p).convert("RGB") if p.exists() else None)
            # ---- identity: per entity present in >=2 shots ----
            for e in st["entities"]:
                name = e["name"]
                crops = []
                for s in range(nsh):
                    if imgs[s] is None or name not in st["shots"][s]["present"]:
                        continue
                    box = detect(imgs[s], name)
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    crops.append(dino_emb(imgs[s].crop((x1, y1, x2, y2))))
                c = pairwise_cos(crops)
                if c is not None:
                    id_scores.append(c)
            # ---- background: mask detected entities, embed background, cross-shot ----
            bg_embs = []
            for s in range(nsh):
                if imgs[s] is None:
                    continue
                im = imgs[s].copy()
                for name in st["shots"][s]["present"]:
                    box = detect(imgs[s], name)
                    if box:
                        im.paste((128, 128, 128), tuple(box))
                bg_embs.append(dino_emb(im))
            b = pairwise_cos(bg_embs)
            if b is not None:
                bg_scores.append(b)
        results[job] = {
            "identity_consistency": round(float(np.mean(id_scores)), 4) if id_scores else None,
            "background_consistency": round(float(np.mean(bg_scores)), 4) if bg_scores else None,
            "_n_id": len(id_scores), "_n_bg": len(bg_scores),
        }
        print(f"  {job}: {results[job]}", flush=True)
    json.dump(results, open(base / args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
