"""
Generation-side quantitative eval of ablation keyframes via OWLv2 open-vocab detection.
Measures, per job (layout-source / render-mode), whether the entities the story says
are PRESENT actually get rendered (and not dropped / duplicated / fused).

Metrics (averaged over all shots of the AAAI-120 set):
  presence_recall : fraction of present entities the detector finds (>=1 box)   [up]
  all_present     : fraction of shots where ALL present entities are found       [up]
  dup_rate        : fraction of present entities detected >=2 times (duplication/fusion split) [down]
  count_mae       : |#distinct entities detected - #present|                      [down]

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_keyframes.py --jobs A_full,B_template --out outputs/abl_logs/det_0.json
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

LABELMAP = {"bus/truck": "bus", "sheep/goat": "sheep", "ball/sports_ball": "ball", "cattle": "cow"}
THR = 0.20


def lab(name):
    return LABELMAP.get(name, name)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)


def count_instances(boxes):
    """light NMS to count distinct instances."""
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        if all(iou(b, k) < 0.5 for k in keep):
            keep.append(b)
    return len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default="outputs/lisa/aaai_ablation")
    ap.add_argument("--stories", default="data/captions/stories_aaai_eval_120.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    dev = "cuda"
    stories = json.loads((base / args.stories).read_text())
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()

    results = {}
    for job in args.jobs.split(","):
        jdir = base / args.root / job
        pres_hits = pres_tot = 0
        allp_hits = allp_tot = 0
        dup_hits = 0
        count_abs = count_tot = 0
        for st in stories:
            for s, sh in enumerate(st["shots"]):
                img_p = jdir / st["name"] / f"shot_{s:03d}.png"
                if not img_p.exists():
                    continue
                present = sh["present"]
                queries = [f"a photo of a {lab(e)}" for e in present]
                img = Image.open(img_p).convert("RGB")
                inp = proc(text=[queries], images=img, return_tensors="pt").to(dev)
                with torch.no_grad():
                    out = model(**inp)
                tgt = torch.tensor([img.size[::-1]]).to(dev)
                res = proc.post_process_grounded_object_detection(out, target_sizes=tgt, threshold=THR)[0]
                boxes = res["boxes"].cpu().tolist(); scores = res["scores"].cpu().tolist(); labels = res["labels"].cpu().tolist()
                per_ent = {i: [] for i in range(len(present))}
                for bx, sc, lb in zip(boxes, scores, labels):
                    per_ent[lb].append([bx[0], bx[1], bx[2], bx[3], sc])
                ndist = 0
                all_found = True
                for i in range(len(present)):
                    cnt = count_instances(per_ent[i])
                    pres_tot += 1
                    if cnt >= 1:
                        pres_hits += 1; ndist += 1
                    else:
                        all_found = False
                    if cnt >= 2:
                        dup_hits += 1
                allp_tot += 1
                if all_found:
                    allp_hits += 1
                count_abs += abs(ndist - len(present)); count_tot += 1
        results[job] = {
            "presence_recall": round(pres_hits / max(pres_tot, 1), 4),
            "all_present": round(allp_hits / max(allp_tot, 1), 4),
            "dup_rate": round(dup_hits / max(pres_tot, 1), 4),
            "count_mae": round(count_abs / max(count_tot, 1), 4),
            "_shots": allp_tot,
        }
        print(f"  {job}: {results[job]}", flush=True)
    json.dump(results, open(base / args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
