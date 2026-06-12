"""Component-ablation 5-metric table (ESA / Missing / mIoU / Fusion / CLIP-T) for the 300 run.
Reads outputs/eval_300/metrics/{state_,lay_}*.json + detections; writes tables/ablation_5metric.{md,tex}."""
import json, glob
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent.parent
ED = BASE / "outputs/eval_300"
state, lay = {}, {}
for f in glob.glob(str(ED / "metrics/state_*.json")):
    state.update(json.load(open(f)))
for f in glob.glob(str(ED / "metrics/lay_*.json")):
    lay.update(json.load(open(f)))


def fusion_rate(job):
    raw = ED / "detections" / f"state_{job}.json"
    if not raw.exists():
        return None
    d = json.load(open(raw)); n = bad = 0
    for st, v in d.items():
        S = np.array(v["Sstar"]); O = np.array(v["Sobs"])
        for t in range(S.shape[0]):
            p = S[t].sum()
            if p >= 2:
                n += 1
                if int((O[t] & S[t]).sum()) < p:
                    bad += 1
    return round(bad / max(n, 1), 4)


ROWS = [("CABL_full", "full (combo / Ours)"), ("CABL_wo_shotemb", "w/o shot_emb"),
        ("CABL_wo_entityemb", "w/o entity_emb"), ("CABL_wo_state", "w/o state"),
        ("CABL_wo_temporal", "w/o temporal"), ("CABL_wo_depth", "w/o depth")]
hdr = ["variant", "ESA↑", "Missing↓", "mIoU↑", "Fusion↓", "CLIP-T↑"]
md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
tex = ["\\begin{table}[t]\\centering", "\\caption{Component ablation (300 stories, fixed renderer).}",
       "\\begin{tabular}{lccccc}", "\\toprule",
       "Variant & ESA $\\uparrow$ & Missing $\\downarrow$ & mIoU $\\uparrow$ & Fusion $\\downarrow$ & CLIP-T $\\uparrow$ \\\\", "\\midrule"]
for jk, labn in ROWS:
    s = state.get(jk, {}).get("overall", {}); g = lay.get(jk, {})
    vals = [s.get("esa"), s.get("miss"), g.get("grounding_mIoU"), fusion_rate(jk), g.get("CLIP_T")]
    md.append(f"| {labn} | " + " | ".join(str(v) for v in vals) + " |")
    tex.append(f"{labn} & " + " & ".join(str(v) for v in vals) + " \\\\")
tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(ED / "tables").mkdir(parents=True, exist_ok=True)
(ED / "tables/ablation_5metric.md").write_text("\n".join(md))
(ED / "tables/ablation_5metric.tex").write_text("\n".join(tex))
print("\n".join(md))
print(f"\nsaved -> {ED}/tables/ablation_5metric.{{md,tex}}")
