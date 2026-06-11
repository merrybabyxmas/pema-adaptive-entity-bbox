"""
Fair depth-order comparison on a FIXED VidOR-test subset: every predictor scored on the
SAME shots/pairs. Predictors: chance, geom_area, geom_bottom, retrieval, LLM (gpt-4o-mini),
ours (planner depth head).

Subset: N multi-entity test shots that contain >=1 ordered co-present pair
(|GT depth diff| > eps), sampled with a fixed seed and cached to
outputs/eval_120/depth_subset.json so all methods use identical pairs.

Usage:
  OPENAI_API_KEY=... CUDA_VISIBLE_DEVICES=0 python scripts/eval_depth_subset.py \
    --planner outputs/runs/planner_v6_combo/checkpoints/best.pt --n 500
"""
import sys, os, argparse, json, re
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


def build_retrieval_depth(train_path):
    idx = {}
    for line in open(train_path):
        s = json.loads(line)
        for shot in s["shots"]:
            dep = shot.get("depth", {})
            pres = [e for e in shot.get("active_entities", []) if e in dep]
            k = frozenset(pres)
            if len(k) >= 2 and k not in idx:
                idx[k] = {e: dep[e] for e in pres}
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/splits/test.jsonl")
    ap.add_argument("--train", default="data/splits/train.jsonl")
    ap.add_argument("--planner", default="outputs/runs/planner_v6_combo/checkpoints/best.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval_120/metrics/depth_subset.json")
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    args = ap.parse_args()
    base = Path(__file__).parent.parent; dev = "cuda"

    ds = BBoxPlannerDataset(str(base / args.test), 5, 5)
    enc = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True).to(dev)
    ck = torch.load(str(base / args.planner), map_location="cpu")
    mcfg = yaml.safe_load(open((base / args.planner).parent.parent / "config.yaml"))["model"]
    mcfg["d_text"] = enc.d_out
    m = build_model(mcfg).to(dev); m.load_state_dict(ck["model"]); m.eval()

    # ---- run planner over all test samples; collect per-(sample,shot) info ----
    dl = DataLoader(ds, batch_size=64, collate_fn=collate_fn, shuffle=False)
    samples = []  # dict per (di, s) with names/pres/td/tb/pred_depth/prompt
    di = 0
    with torch.no_grad():
        for b in dl:
            se = enc.encode_batch_shots(b["shot_prompts"], dev).float()
            ee = enc.encode_batch_entities(b["entity_names"], dev).float()
            pd = m(se, ee, b["state_ids"].to(dev), b["presence"].to(dev),
                   b["relation_ids"].to(dev))[..., 4].cpu().numpy()
            pres = b["presence"].numpy(); td = b["target_depth"].numpy(); tb = b["target_boxes_cxcywh"].numpy()
            B = pres.shape[0]
            for bi in range(B):
                names = b["entity_names"][bi]; prompts = b["shot_prompts"][bi]
                meta = b["metadata"][bi]
                for s in range(pres.shape[1]):
                    idxs = [e for e in range(pres.shape[2]) if pres[bi, s, e] > 0]
                    pairs = [(i, j) for a, i in enumerate(idxs) for j in idxs[a+1:]
                             if abs(td[bi, s, i] - td[bi, s, j]) > EPS]
                    if not pairs:
                        continue
                    samples.append(dict(sid=meta["sample_id"], s=s,
                        names={e: names[e] for e in idxs},
                        td={e: float(td[bi, s, e]) for e in idxs},
                        area={e: float(tb[bi, s, e, 2]*tb[bi, s, e, 3]) for e in idxs},
                        bottom={e: float(tb[bi, s, e, 1]+tb[bi, s, e, 3]/2) for e in idxs},
                        pred={e: float(pd[bi, s, e]) for e in idxs},
                        prompt=prompts[s], pairs=[(i, j) for i, j in pairs]))
            di += B

    # ---- fixed subset ----
    rng = np.random.RandomState(args.seed)
    sub_path = base / "outputs/eval_120/depth_subset.json"
    if sub_path.exists():
        keep = set(tuple(x) for x in json.load(open(sub_path)))
        subset = [r for r in samples if (r["sid"], r["s"]) in keep]
    else:
        order = rng.permutation(len(samples))[:args.n]
        subset = [samples[i] for i in order]
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump([[r["sid"], r["s"]] for r in subset], open(sub_path, "w"))
    print(f"[subset] {len(subset)} shots, {sum(len(r['pairs']) for r in subset)} ordered pairs")

    # ---- retrieval index ----
    retr = build_retrieval_depth(str(base / args.train))

    # ---- LLM depth (gpt-4o-mini), cached/resumable ----
    llm_cache = base / "outputs/eval_120/metrics/llm_depth_cache.json"
    cache = json.loads(llm_cache.read_text()) if llm_cache.exists() else {}
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        cli = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        for k, r in enumerate(subset):
            key = f"{r['sid']}|{r['s']}"
            if key in cache:
                continue
            ents = list(r["names"].values())
            q = (f"Scene: '{r['prompt']}'. Entities present: {ents}. Assign each entity a depth "
                 f"in [0,1] where 1 = closest to camera (front), 0 = farthest (back), using typical "
                 f"scene/relations. Return JSON only: {{entity: depth}}.")
            try:
                resp = cli.chat.completions.create(model=args.llm_model, temperature=0, max_tokens=120,
                    messages=[{"role": "user", "content": q}]).choices[0].message.content
                mt = re.search(r"\{.*\}", resp, re.S); cache[key] = json.loads(mt.group(0)) if mt else {}
            except Exception as e:
                cache[key] = {}
            if k % 25 == 0:
                llm_cache.write_text(json.dumps(cache)); print(f"  llm {k}/{len(subset)}", flush=True)
        llm_cache.write_text(json.dumps(cache))
    else:
        print("[llm] no OPENAI_API_KEY -> LLM-depth skipped")

    # ---- score all predictors on identical pairs ----
    methods = ["geom_area", "geom_bottom", "retrieval", "llm", "ours"]
    cor = {x: 0 for x in methods}; n = {x: 0 for x in methods}
    for r in subset:
        names = r["names"]
        rk = frozenset(names.values()); rdep = retr.get(rk, {})
        ldep = cache.get(f"{r['sid']}|{r['s']}", {})
        for i, j in r["pairs"]:
            sg = np.sign(r["td"][i] - r["td"][j])
            def score(name, pi, pj):
                n[name] += 1; cor[name] += int(np.sign(pi - pj) == sg)
            score("geom_area", r["area"][i], r["area"][j])
            score("geom_bottom", r["bottom"][i], r["bottom"][j])
            ni, nj = names[i], names[j]
            score("retrieval", rdep.get(ni, 0.5), rdep.get(nj, 0.5))
            try:
                score("llm", float(ldep.get(ni, 0.5)), float(ldep.get(nj, 0.5)))
            except Exception:
                n["llm"] += 1
            score("ours", r["pred"][i], r["pred"][j])
    res = {"_shots": len(subset), "_pairs": n["ours"]}
    res["chance"] = 0.5
    for x in ["geom_area", "geom_bottom", "retrieval", "llm", "ours"]:
        res[x] = round(cor[x] / max(n[x], 1), 4)
    json.dump(res, open(base / args.out, "w"), indent=2)
    print("\ndepth-order accuracy (same fixed subset):")
    for x in ["chance", "geom_area", "geom_bottom", "retrieval", "llm", "ours"]:
        print(f"  {x:12s} {res[x]}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
