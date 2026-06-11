"""
Depth-order accuracy comparison on VidOR/VidSTG test: ours (learned planner, predicts depth
from narrative) vs non-learned depth baselines. Only methods that produce a front/back signal
are comparable (template/retrieval/center predict NO depth -> chance).

Baselines (per co-present ordered pair with |GT depth diff|>eps):
  chance        : 0.5 by definition
  geom_area     : front = entity with the LARGER GT box area   (classic size prior)
  geom_bottom   : front = entity with the LOWER bottom edge y2 (classic 2.5D ground prior)
  ours          : sign(pred_depth_i - pred_depth_j) from the planner depth head

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_depth_baselines.py \
    --planner outputs/runs/planner_v6_combo/checkpoints/best.pt --out outputs/eval_120/metrics/depth_baselines.json
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch, yaml
from torch.utils.data import DataLoader
from src.data.dataset import BBoxPlannerDataset
from src.data.collate import collate_fn
from src.model.bbox_planner import build_model
from src.model.embeddings import CLIPTextEncoder

EPS = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/splits/test.jsonl")
    ap.add_argument("--planner", default="outputs/runs/planner_v6_combo/checkpoints/best.pt")
    ap.add_argument("--out", default="outputs/eval_120/metrics/depth_baselines.json")
    args = ap.parse_args()
    base = Path(__file__).parent.parent; dev = "cuda"
    ds = BBoxPlannerDataset(str(base / args.test), 5, 5)
    dl = DataLoader(ds, batch_size=64, collate_fn=collate_fn)

    ck = torch.load(str(base / args.planner), map_location="cpu")
    pcfg = yaml.safe_load(open((base / args.planner).parent.parent / "config.yaml"))
    enc = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True).to(dev)
    mcfg = pcfg["model"]; mcfg["d_text"] = enc.d_out
    m = build_model(mcfg).to(dev); m.load_state_dict(ck["model"]); m.eval()

    n = {"ours": 0, "geom_area": 0, "geom_bottom": 0}
    c = {"ours": 0, "geom_area": 0, "geom_bottom": 0}
    tot = 0
    with torch.no_grad():
        for b in dl:
            se = enc.encode_batch_shots(b["shot_prompts"], dev).float()
            ee = enc.encode_batch_entities(b["entity_names"], dev).float()
            out = m(se, ee, b["state_ids"].to(dev), b["presence"].to(dev), b["relation_ids"].to(dev))
            pd = out[..., 4].cpu().numpy()                       # [B,S,E]
            td = b["target_depth"].numpy()                       # [B,S,E]
            tb = b["target_boxes_cxcywh"].numpy()                # [B,S,E,4] cx,cy,w,h
            pres = b["presence"].numpy()
            B, S, E = pres.shape
            for bi in range(B):
                for s in range(S):
                    idx = [e for e in range(E) if pres[bi, s, e] > 0]
                    for a in range(len(idx)):
                        for d2 in range(a + 1, len(idx)):
                            i, j = idx[a], idx[d2]
                            gt = td[bi, s, i] - td[bi, s, j]
                            if abs(gt) <= EPS:
                                continue
                            tot += 1
                            sg = np.sign(gt)
                            # ours
                            n["ours"] += 1
                            c["ours"] += (np.sign(pd[bi, s, i] - pd[bi, s, j]) == sg)
                            # geom area (larger area = front = higher depth)
                            ai = tb[bi, s, i, 2] * tb[bi, s, i, 3]; aj = tb[bi, s, j, 2] * tb[bi, s, j, 3]
                            n["geom_area"] += 1
                            c["geom_area"] += (np.sign(ai - aj) == sg)
                            # geom bottom (lower bottom edge cy+h/2 = closer = front)
                            byi = tb[bi, s, i, 1] + tb[bi, s, i, 3] / 2; byj = tb[bi, s, j, 1] + tb[bi, s, j, 3] / 2
                            n["geom_bottom"] += 1
                            c["geom_bottom"] += (np.sign(byi - byj) == sg)
    res = {"chance": 0.5}
    for k in n:
        res[k] = round(c[k] / max(n[k], 1), 4)
    res["_pairs"] = tot
    json.dump(res, open(base / args.out, "w"), indent=2)
    print(f"depth-order accuracy on {tot} ordered co-present pairs (VidOR test):")
    for k in ["chance", "geom_area", "geom_bottom", "ours"]:
        print(f"  {k:12s} {res[k]}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
