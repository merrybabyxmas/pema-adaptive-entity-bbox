"""
Layout-fidelity + text-alignment metrics (layout-to-image + story literature standard).

Per job (planner variant) over the AAAI-120 set:
  grounding_mIoU : for each present entity, IoU between the planner's INTENDED box
       (what we conditioned on, after deoverlap+resolve) and the box where the entity
       was actually rendered (localized by OWLv2). Mean over present entities.
       (undetected entity -> IoU 0). This is the core layout-fidelity metric.
  SR@0.5         : fraction of present entities with IoU >= 0.5 (Success Rate).
  CLIP_T         : cosine(CLIP image emb of the shot, CLIP text emb of the shot prompt),
       averaged over shots (image-text alignment).

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_layout.py --jobs CABL_full --out outputs/cabl_logs/lay_0.json
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch, yaml
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import open_clip

from src.model.bbox_planner import build_model
from src.model.embeddings import CLIPTextEncoder
from src.utils.box_ops import deoverlap_boxes
from scripts.run_30_stories import plan_boxes, resolve_for_occlusion
from scripts.run_ablation_gen import (JOBS, layout_template, layout_retrieval,
                                      layout_center, layout_llm, build_retrieval_index)

LABELMAP = {"bus/truck": "bus", "sheep/goat": "sheep", "ball/sports_ball": "ball", "cattle": "cow"}
DET_THR = 0.15
SR_IOU = 0.5


def lab(n):
    return LABELMAP.get(n, n)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default="outputs/lisa/aaai_cablation")
    ap.add_argument("--runs", default="outputs/runs")
    ap.add_argument("--stories", default="data/captions/stories_aaai_eval_120.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(__file__).parent.parent; dev = "cuda"
    stories = json.loads((base / args.stories).read_text())

    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    enc = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True).to(dev)

    def detect(img, label):
        inp = owlp(text=[[f"a photo of a {lab(label)}"]], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = owl(**inp)
        r = owlp.post_process_grounded_object_detection(
            o, target_sizes=torch.tensor([img.size[::-1]]).to(dev), threshold=DET_THR)[0]
        if len(r["scores"]) == 0:
            return None
        i = int(r["scores"].argmax()); W, H = img.size
        b = r["boxes"][i].tolist()
        return [b[0]/W, b[1]/H, b[2]/W, b[3]/H]   # normalized xyxy

    def clip_t(img, prompt):
        im = preprocess(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            ie = clip_model.encode_image(im); te = clip_model.encode_text(tok([prompt]).to(dev))
        ie = ie / ie.norm(dim=-1, keepdim=True); te = te / te.norm(dim=-1, keepdim=True)
        return float((ie * te).sum())

    # shared resources for non-planner layout sources
    retr_idx = build_retrieval_index(str(base / "data/splits/train.jsonl"))
    llm_path = base / "outputs/layouts/llm_aaai.json"
    llm = json.loads(llm_path.read_text()) if llm_path.exists() else {}

    def intended_boxes(job, st, model):
        """per-shot intended boxes + depths for the job's layout source."""
        lt = JOBS[job]["layout"]
        if lt == "planner":
            _, per, perd = plan_boxes(model, enc, st, dev); return per, perd
        if lt == "template":
            return layout_template(st, dev)
        if lt == "retrieval":
            return layout_retrieval(st, retr_idx, dev)
        if lt == "center":
            return layout_center(st, dev)
        if lt == "llm":
            return layout_llm(st, llm, dev)
        return [], []

    results = {}
    for job in args.jobs.split(","):
        model = None
        if JOBS[job]["layout"] == "planner":
            rdir = base / args.runs / JOBS[job]["ckpt"]
            ck = torch.load(str(rdir / "checkpoints" / "best.pt"), map_location="cpu")
            mcfg = yaml.safe_load(open(rdir / "config.yaml"))["model"]; mcfg["d_text"] = enc.d_out
            model = build_model(mcfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
        jdir = base / args.root / job

        ious, srs, clipts = [], [], []
        for st in stories:
            per, perd = intended_boxes(job, st, model)
            for s, (boxes, depths) in enumerate(zip(per, perd)):
                p = jdir / st["name"] / f"shot_{s:03d}.png"
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                clipts.append(clip_t(img, st["shots"][s]["prompt"]))
                intended = (resolve_for_occlusion(deoverlap_boxes(dict(boxes)), depths)
                            if len(boxes) > 1 else dict(boxes))
                for e in st["shots"][s]["present"]:
                    if e not in intended:
                        continue
                    det = detect(img, e)
                    iv = iou(det, intended[e]) if det else 0.0
                    ious.append(iv); srs.append(1.0 if iv >= SR_IOU else 0.0)
        results[job] = {
            "grounding_mIoU": round(float(np.mean(ious)), 4) if ious else None,
            "SR@0.5": round(float(np.mean(srs)), 4) if srs else None,
            "CLIP_T": round(float(np.mean(clipts)), 4) if clipts else None,
            "_n": len(ious),
        }
        print(f"  {job}: {results[job]}", flush=True)
    json.dump(results, open(base / args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
