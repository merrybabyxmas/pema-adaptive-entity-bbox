"""
Generation-level STATE-COMPLIANCE diagnostics (Entity State Accuracy / Transition Accuracy
/ Missing / Leakage), per group (A-H) and overall. Renderer-fixed; one job = one layout source.

S*    : prescribed binary presence matrix [T,N] from the story (present lists).
S_obs : observed presence [T,N] from OWLv2 (entity detected above threshold in shot t).
We query EVERY story entity in EVERY shot (incl. prescribed-absent) to catch leakage.

  ESA = 1 - ||S_obs - S*||_0 / (T*N)
  TA  : over prescribed transitions M=1[dS*!=0]:  1 - ||(dS_obs - dS*)*M||_0 / ||M||_0
  Missing  = mean over (S*==1) of (S_obs==0)
  Leakage  = mean over (S*==0) of (S_obs==1)

Diagnostic metrics only (not a proposed benchmark). Raw S_obs saved for inspection.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_generation_state.py \
    --jobs FINAL_combo,B_template --root outputs/lisa/aaai_ablation --out outputs/eval_120/metrics/state_0.json
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

LABELMAP = {"bus/truck": "bus", "sheep/goat": "sheep", "ball/sports_ball": "ball", "cattle": "cow"}
DET_THR = 0.20


def lab(n):
    return LABELMAP.get(n, n)


def group_of(name):
    # "P01_relay_..." -> 'P01' ; "aaai_A00_..." -> 'A'
    if name.startswith("P0"):
        return name.split("_")[0]
    try:
        return name.split("_")[1][0]
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default="outputs/lisa/aaai_ablation")
    ap.add_argument("--stories", default="data/captions/stories_aaai_eval_120.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="outputs/eval_120/detections")
    args = ap.parse_args()
    base = Path(__file__).parent.parent; dev = "cuda"
    stories = json.loads((base / args.stories).read_text())
    Path(base / args.raw).mkdir(parents=True, exist_ok=True)

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()

    def present_set(img, labels):
        q = [f"a photo of a {lab(e)}" for e in labels]
        inp = proc(text=[q], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = owl(**inp)
        r = proc.post_process_grounded_object_detection(
            o, target_sizes=torch.tensor([img.size[::-1]]).to(dev), threshold=DET_THR)[0]
        det = set(int(l) for l in r["labels"].cpu().tolist())
        return [1 if i in det else 0 for i in range(len(labels))]

    out = {}
    for job in args.jobs.split(","):
        jdir = base / args.root / job
        raw = {}
        from collections import defaultdict
        acc = defaultdict(lambda: {"esa": [], "ta": [], "miss": [], "leak": []})
        for st in stories:
            names = [e["name"] for e in st["entities"]]
            T, N = len(st["shots"]), len(names)
            Sstar = np.zeros((T, N), int)
            for t, sh in enumerate(st["shots"]):
                for n_i, nm in enumerate(names):
                    if nm in sh["present"]:
                        Sstar[t, n_i] = 1
            Sobs = np.zeros((T, N), int)
            for t in range(T):
                p = jdir / st["name"] / f"shot_{t:03d}.png"
                if p.exists():
                    Sobs[t] = present_set(Image.open(p).convert("RGB"), names)
            raw[st["name"]] = {"entities": names, "Sstar": Sstar.tolist(), "Sobs": Sobs.tolist()}
            esa = 1.0 - (Sobs != Sstar).sum() / (T * N)
            dM = (Sstar[1:] - Sstar[:-1])
            mask = (dM != 0)
            if mask.sum() > 0:
                ta = 1.0 - (((Sobs[1:] - Sobs[:-1]) - dM) * mask != 0).sum() / mask.sum()
            else:
                ta = None
            miss = float((Sobs[Sstar == 1] == 0).mean()) if (Sstar == 1).any() else None
            leak = float((Sobs[Sstar == 0] == 1).mean()) if (Sstar == 0).any() else None
            g = group_of(st["name"])
            for key, val in [("esa", esa), ("ta", ta), ("miss", miss), ("leak", leak)]:
                if val is None:
                    continue
                acc["ALL"][key].append(val)
                acc[g][key].append(val)
        def agg(d):
            return {k: (round(float(np.mean(v)), 4) if v else None) for k, v in d.items()}
        groups = sorted(k for k in acc if k != "ALL")
        out[job] = {"overall": agg(acc["ALL"]),
                    "per_group": {g: agg(acc[g]) for g in groups}}
        json.dump(raw, open(base / args.raw / f"state_{job}.json", "w"))
        print(f"  {job}: ESA={out[job]['overall']['esa']} TA={out[job]['overall']['ta']} "
              f"miss={out[job]['overall']['miss']} leak={out[job]['overall']['leak']}", flush=True)
    json.dump(out, open(base / args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
