"""
Planner-only ablation/baseline evaluation on the VidOR/VidSTG test split.
Produces the main quantitative table (Table 1 + Table 3 planner part).

Methods:
  learned   : abl_full / abl_wo_state / abl_wo_relation / abl_wo_shotattn /
              abl_wo_tempattn / abl_wo_depth   (each loads its own saved config)
  baselines : center, template, gt_dist_random, retrieval, rule_based
              (LLM-direct requires an API key -> reported as omitted)

Metrics (masked to present entities):
  L1 down, GIoU up, center_err down, area_err down,
  overlap_rate down (mean pairwise IoU of co-present boxes = fusion risk),
  size_order_acc up (predicted vs GT relative-size ordering),
  depth_order_acc up (predicted vs GT front/back; learned-with-depth only)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_ablation.py --out outputs/abl_logs/table.md
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
from src.utils.box_ops import cxcywh_to_xyxy, box_iou, box_giou

EPS = 0.03
LEFT_OF, RIGHT_OF, ABOVE, BELOW = 1, 2, 3, 4


# ---------- metrics (operate on padded [B,S,E,*] tensors with masks) ----------
def _pairwise(pred_xyxy, presence):
    """yield (b,s,i,j, iou) for co-present pairs."""
    B, S, E, _ = pred_xyxy.shape
    out = []
    for b in range(B):
        for s in range(S):
            idx = [e for e in range(E) if presence[b, s, e] > 0]
            for a in range(len(idx)):
                for c in range(a + 1, len(idx)):
                    i, j = idx[a], idx[c]
                    iou = box_iou(pred_xyxy[b, s, i:i+1], pred_xyxy[b, s, j:j+1]).item()
                    out.append((b, s, i, j, iou))
    return out


def compute_all(pred5, batch, device):
    pb = pred5[..., :4]
    pd = pred5[..., 4]
    tgt = batch["target_boxes_cxcywh"].to(device)
    tdep = batch["target_depth"].to(device)
    mask = batch["target_mask"].to(device)
    pres = batch["presence"].to(device)
    m = mask.bool()
    px = cxcywh_to_xyxy(pb).clamp(0, 1)
    tx = cxcywh_to_xyxy(tgt).clamp(0, 1)

    pf = px.reshape(-1, 4)[m.reshape(-1)]
    tf = tx.reshape(-1, 4)[m.reshape(-1)]
    pcx = pb.reshape(-1, 4)[m.reshape(-1)]
    tcx = tgt.reshape(-1, 4)[m.reshape(-1)]
    res = {}
    res["L1"] = torch.nn.functional.l1_loss(pcx, tcx).item()
    res["GIoU"] = box_giou(pf, tf).mean().item()
    res["center"] = (pcx[:, :2] - tcx[:, :2]).norm(dim=-1).mean().item()
    res["area"] = abs((pcx[:, 2] * pcx[:, 3]).mean().item() - (tcx[:, 2] * tcx[:, 3]).mean().item())

    # pairwise overlap (fusion risk) + size-order + depth-order
    pa = pb[..., 2] * pb[..., 3]
    ta = tgt[..., 2] * tgt[..., 3]
    both = pres.unsqueeze(-1) * pres.unsqueeze(-2)
    E = pres.shape[-1]
    iu = torch.triu(torch.ones(E, E, device=device), 1)
    valid = both * iu
    # overlap rate
    pxx = px
    ov_num, ov_den = 0.0, 0.0
    dsize = pa.unsqueeze(-1) - pa.unsqueeze(-2)
    tsize = ta.unsqueeze(-1) - ta.unsqueeze(-2)
    so_corr, so_den = 0.0, 0.0
    ddep = pd.unsqueeze(-1) - pd.unsqueeze(-2)
    tdp = tdep.unsqueeze(-1) - tdep.unsqueeze(-2)
    do_corr, do_den = 0.0, 0.0
    B, S = pres.shape[0], pres.shape[1]
    for b in range(B):
        for s in range(S):
            idx = (pres[b, s] > 0).nonzero(as_tuple=True)[0].tolist()
            for a in range(len(idx)):
                for c in range(a + 1, len(idx)):
                    i, j = idx[a], idx[c]
                    iou = box_iou(pxx[b, s, i:i+1], pxx[b, s, j:j+1]).item()
                    ov_num += iou; ov_den += 1
                    if abs(tsize[b, s, i, j].item()) > 0.01:
                        so_den += 1
                        if np.sign(dsize[b, s, i, j].item()) == np.sign(tsize[b, s, i, j].item()):
                            so_corr += 1
                    if abs(tdp[b, s, i, j].item()) > 0.05:
                        do_den += 1
                        if np.sign(ddep[b, s, i, j].item()) == np.sign(tdp[b, s, i, j].item()):
                            do_corr += 1
    res["overlap"] = ov_num / max(ov_den, 1)
    res["size_acc"] = so_corr / max(so_den, 1)
    # depth only meaningful if the method actually predicts a (non-constant) depth
    depth_const = (pd.max() - pd.min()).abs().item() < 1e-6
    res["depth_acc"] = float("nan") if (depth_const or do_den == 0) else do_corr / do_den
    res["_n"] = int(m.sum().item())
    return res


def agg(dicts):
    keys = [k for k in dicts[0] if not k.startswith("_")]
    n = sum(d["_n"] for d in dicts)
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if not (isinstance(d[k], float) and np.isnan(d[k]))]
        ws = [d["_n"] for d in dicts if not (isinstance(d[k], float) and np.isnan(d[k]))]
        out[k] = float(np.average(vals, weights=ws)) if vals else float("nan")
    out["_n"] = n
    return out


# ---------- non-learned baselines: build pred5 [B,S,E,5] ----------
def make_baseline(name, batch, train_stats, device, rng):
    pres = batch["presence"]
    rel = batch["relation_ids"]
    names = batch["entity_names"]
    B, S, E = pres.shape
    pred = torch.zeros(B, S, E, 5)
    pred[..., 4] = 0.5
    A = train_stats["mean_area"]
    side = float(np.sqrt(A))
    pool = train_stats["box_pool"]
    idx_by_cats = train_stats["idx_by_cats"]
    for b in range(B):
        for s in range(S):
            idx = [e for e in range(E) if pres[b, s, e] > 0]
            n = len(idx)
            if n == 0:
                continue
            if name == "center":
                for e in idx:
                    pred[b, s, e, :4] = torch.tensor([0.5, 0.5, side, side])
            elif name in ("template", "rule_based"):
                order = list(idx)
                if name == "rule_based":
                    order = _rule_order(idx, rel[b, s])
                w = min(0.4, 0.85 / n)
                for r, e in enumerate(order):
                    cx = (r + 0.5) / n
                    pred[b, s, e, :4] = torch.tensor([cx, 0.5, w, 0.6])
            elif name == "gt_dist_random":
                for e in idx:
                    pred[b, s, e, :4] = torch.tensor(pool[rng.randint(len(pool))])
            elif name == "retrieval":
                cats = frozenset(names[b][e] for e in idx)
                ref = idx_by_cats.get(cats)
                if ref is not None:
                    cat2box = ref
                    for e in idx:
                        c = names[b][e]
                        if c in cat2box:
                            pred[b, s, e, :4] = torch.tensor(cat2box[c])
                        else:
                            pred[b, s, e, :4] = torch.tensor([0.5, 0.5, side, side])
                else:
                    w = min(0.4, 0.85 / n)
                    for r, e in enumerate(idx):
                        pred[b, s, e, :4] = torch.tensor([(r + 0.5) / n, 0.5, w, 0.6])
    return pred.to(device)


def _rule_order(idx, rel_se):
    """order present entities left->right using left_of/right_of GT relations."""
    order = list(idx)
    # simple bubble by pairwise constraints
    changed = True; it = 0
    while changed and it < 10:
        changed = False; it += 1
        for a in range(len(order)):
            for c in range(a + 1, len(order)):
                i, j = order[a], order[c]
                r_ij = int(rel_se[i, j].item()); r_ji = int(rel_se[j, i].item())
                want_after = (r_ij == RIGHT_OF) or (r_ji == LEFT_OF)  # i should be right of j
                if want_after:
                    order[a], order[c] = order[c], order[a]; changed = True
    return order


def build_train_stats(train_path):
    areas, pool = [], []
    idx_by_cats = {}
    for line in open(train_path):
        s = json.loads(line)
        for shot in s["shots"]:
            bx = {e: b for e, b in shot.get("boxes", {}).items() if b}
            cm = {}
            for e, b in bx.items():
                cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2
                w = b[2] - b[0]; h = b[3] - b[1]
                areas.append(w * h); pool.append([cx, cy, w, h]); cm[e] = [cx, cy, w, h]
            k = frozenset(bx.keys())
            if k and k not in idx_by_cats:
                idx_by_cats[k] = cm
    return {"mean_area": float(np.mean(areas)), "box_pool": pool, "idx_by_cats": idx_by_cats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/splits/test.jsonl")
    ap.add_argument("--train", default="data/splits/train.jsonl")
    ap.add_argument("--runs", default="abl_full,abl_wo_state,abl_wo_relation,abl_wo_shotattn,abl_wo_tempattn,abl_wo_depth")
    ap.add_argument("--baselines", default="center,template,gt_dist_random,retrieval,rule_based")
    ap.add_argument("--out", default="outputs/abl_logs/table.md")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    dev = "cuda"

    ds = BBoxPlannerDataset(str(base / args.test), 5, 5)
    if args.limit:
        ds.samples = ds.samples[:args.limit]
    dl = DataLoader(ds, batch_size=64, collate_fn=collate_fn)
    batches = list(dl)
    print(f"[eval] {len(ds)} test samples, {len(batches)} batches")
    train_stats = build_train_stats(str(base / args.train))
    print(f"[eval] train mean_area={train_stats['mean_area']:.3f}, retrieval keys={len(train_stats['idx_by_cats'])}")

    rows = {}
    enc = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True).to(dev)

    # baselines
    for name in args.baselines.split(","):
        rng = np.random.RandomState(0)
        ds_list = [compute_all(make_baseline(name, b, train_stats, dev, rng), b, dev) for b in batches]
        rows[name] = agg(ds_list)
        print(f"  baseline {name}: {rows[name]}")

    # learned variants
    for run in args.runs.split(","):
        rdir = base / "outputs" / "runs" / run
        ckp = rdir / "checkpoints" / "best.pt"
        if not ckp.exists():
            print(f"  ! {run}: no best.pt yet, skip"); continue
        ck = torch.load(str(ckp), map_location="cpu")
        pcfg = yaml.safe_load(open(rdir / "config.yaml"))
        mcfg = pcfg["model"]; mcfg["d_text"] = enc.d_out
        m = build_model(mcfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
        ds_list = []
        with torch.no_grad():
            for b in batches:
                se = enc.encode_batch_shots(b["shot_prompts"], dev).float()
                ee = enc.encode_batch_entities(b["entity_names"], dev).float()
                out = m(se, ee, b["state_ids"].to(dev), b["presence"].to(dev), b["relation_ids"].to(dev))
                ds_list.append(compute_all(out, b, dev))
        rows[run] = agg(ds_list)
        rows[run]["_epoch"] = ck.get("epoch", "?")
        print(f"  {run} (ep{ck.get('epoch','?')}): {rows[run]}")

    # markdown table
    cols = ["L1", "GIoU", "center", "area", "overlap", "size_acc", "depth_acc"]
    arrow = {"L1": "↓", "GIoU": "↑", "center": "↓", "area": "↓", "overlap": "↓", "size_acc": "↑", "depth_acc": "↑"}
    lines = ["| method | " + " | ".join(f"{c}{arrow[c]}" for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    order = args.baselines.split(",") + args.runs.split(",")
    for name in order:
        if name not in rows:
            continue
        r = rows[name]
        cells = []
        for c in cols:
            v = r.get(c, float("nan"))
            cells.append("—" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    outp = base / args.out
    outp.write_text(table + "\nLLM-direct: omitted (requires LLM API)\n")
    print("\n" + table)
    print(f"\nsaved -> {outp}")
    json.dump(rows, open(str(outp).replace(".md", ".json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
