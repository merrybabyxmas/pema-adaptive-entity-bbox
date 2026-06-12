"""
Aggregate all evaluation outputs into paper-ready Tables 1-4 (CSV + Markdown + LaTeX),
per-group (A-H) breakdowns, bootstrap 95% CIs over stories, and summary figures.

Reads (best-effort; missing inputs are skipped with a note):
  outputs/abl_logs/table_clean.json        -> Table 1 (planner-only, VidOR test)
  outputs/eval_120/metrics/lay_*.json       -> grounding mIoU / SR / CLIP-T (layout sources)
  outputs/eval_120/metrics/state_*.json     -> ESA / TA / Missing / Leakage (overall + per-group)
  outputs/eval_120/detections/state_<job>.json -> raw S*/S_obs (per-story bootstrap)
  outputs/eval_120/metrics/quality_*.csv    -> aesthetic / clip_t / sharpness
  outputs/abl_logs/gen_table.json + det_final.json -> presence / dup
  outputs/cabl_logs/FULL_ablation_table.md  -> Table 3 (component ablation, passthrough)

Writes to outputs/eval_120/tables/ and outputs/eval_120/figures/.

Usage: python scripts/aggregate_eval_tables.py
"""
import sys, os, json, glob, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--evaldir", default="outputs/eval_120")
_ap.add_argument("--ours", default="FINAL_combo")  # job key used as "Ours"
_args, _ = _ap.parse_known_args()
ED = _args.evaldir
T = BASE / ED / "tables"; T.mkdir(parents=True, exist_ok=True)
F = BASE / ED / "figures"; F.mkdir(parents=True, exist_ok=True)
LAYOUT_ROWS = [("B_template", "Template"), ("B_retrieval", "Retrieval"),
               ("B_llm", "LLM-direct"), ("B_center", "Center"), (_args.ours, "Ours")]


def load_merge(pattern):
    d = {}
    for f in glob.glob(str(BASE / pattern)):
        d.update(json.load(open(f)))
    return d


def boot_ci(vals, n=2000, seed=0):
    if len(vals) < 2:
        return (None, None)
    rng = np.random.RandomState(seed); v = np.array(vals)
    means = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return (round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3))


def per_story_esa_ta(raw):
    """recompute per-story ESA/TA from raw S*/S_obs for bootstrap."""
    esa, ta = [], []
    for st, d in raw.items():
        S = np.array(d["Sstar"]); O = np.array(d["Sobs"])
        Tn, Nn = S.shape
        esa.append(1.0 - (O != S).sum() / (Tn * Nn))
        dM = S[1:] - S[:-1]; m = (dM != 0)
        if m.sum() > 0:
            ta.append(1.0 - (((O[1:] - O[:-1]) - dM) * m != 0).sum() / m.sum())
    return esa, ta


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def latex_table(headers, rows, caption):
    cols = "l" + "c" * (len(headers) - 1)
    L = ["\\begin{table}[t]\\centering", f"\\caption{{{caption}}}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
         " & ".join(headers) + " \\\\", "\\midrule"]
    for r in rows:
        L.append(" & ".join(str(x) for x in r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)


def write(name, headers, rows, caption):
    (T / f"{name}.md").write_text(md_table(headers, rows))
    with open(T / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)
    (T / f"{name}.tex").write_text(latex_table(headers, rows, caption))
    print(f"  wrote {name}.{{md,csv,tex}}")


def main():
    lay = load_merge(f"{ED}/metrics/lay_*.json")
    state = load_merge(f"{ED}/metrics/state_*.json")
    det = load_merge(f"{ED}/metrics/det_*.json")
    if not det:  # fall back to the 120 run's detection sources
        det = load_merge("outputs/abl_logs/det_*.json")
        det.update(load_merge("outputs/eval_120/metrics/det_final.json"))
        det.update(load_merge("outputs/eval_120/metrics/det_bcenter.json"))
    # quality means
    q = {}
    for f in glob.glob(str(BASE / f"{ED}/metrics/quality_*.csv")):
        for r in csv.DictReader(open(f)):
            m = r["method"]; q.setdefault(m, {"clip_t": [], "aesthetic": [], "sharpness": []})
            for k in q[m]:
                try:
                    val = float(r[k]); q[m][k].append(val) if not np.isnan(val) else None
                except Exception:
                    pass

    # ---- Table 2: controlled layout-source comparison (generation) ----
    headers = ["LayoutSource", "ESA↑", "TA↑", "Missing↓", "Leakage↓",
               "mIoU↑", "SR@.5↑", "Presence↑", "Fusion(dup)↓", "CLIP-T↑", "Aesth↑"]
    rows, boot = [], {}
    for jk, lab in LAYOUT_ROWS:
        s = state.get(jk, {}).get("overall", {})
        g = lay.get(jk, {}); d = det.get(jk, {})
        qa = q.get(jk, {})
        aes = np.mean(qa["aesthetic"]) if qa.get("aesthetic") else float("nan")
        clt = np.mean(qa["clip_t"]) if qa.get("clip_t") else float("nan")
        rows.append([lab, s.get("esa"), s.get("ta"), s.get("miss"), s.get("leak"),
                     g.get("grounding_mIoU"), g.get("SR@0.5"), d.get("presence_recall"),
                     d.get("dup_rate"), round(clt, 3), round(aes, 3)])
        raw_p = BASE / f"{ED}/detections/state_{jk}.json"
        if raw_p.exists():
            esa_v, ta_v = per_story_esa_ta(json.load(open(raw_p)))
            boot[lab] = {"ESA_CI": boot_ci(esa_v), "TA_CI": boot_ci(ta_v)}
    write("table2_layout_source", headers, rows,
          "Controlled comparison: fixed SDXL+IP renderer, varying layout source (AAAI-120).")
    json.dump(boot, open(T / "table2_bootstrap_CI.json", "w"), indent=2)

    # ---- Table 4: failure breakdown ----
    h4 = ["LayoutSource", "Missing↓", "Leakage↓", "Fusion(dup)↓", "1-SR@.5↓"]
    r4 = []
    for jk, lab in LAYOUT_ROWS:
        s = state.get(jk, {}).get("overall", {}); d = det.get(jk, {}); g = lay.get(jk, {})
        sr = g.get("SR@0.5"); r4.append([lab, s.get("miss"), s.get("leak"), d.get("dup_rate"),
                                         round(1 - sr, 3) if sr is not None else None])
    write("table4_failure", h4, r4, "Failure breakdown by layout source (AAAI-120).")

    # ---- Per-group ESA/TA ----
    groups = list("ABCDEFGH")
    hg = ["LayoutSource"] + [f"{g}-ESA" for g in groups]
    rg = []
    for jk, lab in LAYOUT_ROWS:
        pg = state.get(jk, {}).get("per_group", {})
        rg.append([lab] + [pg.get(g, {}).get("esa") for g in groups])
    write("table_pergroup_ESA", hg, rg, "Per-group Entity State Accuracy (A-H).")

    # ---- Table 1 passthrough (planner-only) ----
    pc = BASE / "outputs/abl_logs/table_clean.md"
    if pc.exists():
        (T / "table1_planner.md").write_text(pc.read_text())
        print("  copied table1_planner.md (planner-only VidOR)")
    # ---- Table 3 passthrough (component ablation) ----
    ab = BASE / "outputs/cabl_logs/FULL_ablation_table.md"
    if ab.exists():
        (T / "table3_ablation.md").write_text(ab.read_text())
        print("  copied table3_ablation.md (component ablation)")

    # ---- Figures ----
    labs = [l for _, l in LAYOUT_ROWS]
    # layout fidelity
    miou = [lay.get(jk, {}).get("grounding_mIoU") or 0 for jk, _ in LAYOUT_ROWS]
    sr = [lay.get(jk, {}).get("SR@0.5") or 0 for jk, _ in LAYOUT_ROWS]
    x = np.arange(len(labs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, miou, w, label="grounding mIoU", color="#2980b9")
    ax.bar(x + w/2, sr, w, label="SR@0.5", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.legend(); ax.set_title("Layout fidelity by source")
    fig.tight_layout(); fig.savefig(F / "fig_layout_fidelity.png", dpi=110); plt.close(fig)
    # ESA/TA
    esa = [state.get(jk, {}).get("overall", {}).get("esa") or 0 for jk, _ in LAYOUT_ROWS]
    ta = [state.get(jk, {}).get("overall", {}).get("ta") or 0 for jk, _ in LAYOUT_ROWS]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, esa, w, label="ESA", color="#27ae60")
    ax.bar(x + w/2, ta, w, label="TA", color="#8e44ad")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0.8, 1.0); ax.legend()
    ax.set_title("State compliance (ESA/TA) by source")
    fig.tight_layout(); fig.savefig(F / "fig_state.png", dpi=110); plt.close(fig)
    # quality guard
    aes = [np.mean(q.get(jk, {}).get("aesthetic", [np.nan])) for jk, _ in LAYOUT_ROWS]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, aes, color="#16a085"); ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylim(5.0, 6.0); ax.set_title("Aesthetic (quality guard)")
    fig.tight_layout(); fig.savefig(F / "fig_quality.png", dpi=110); plt.close(fig)
    print(f"  figures -> {F}")
    print("\n=== Table 2 (controlled layout-source) ===")
    print(md_table(headers, rows))


if __name__ == "__main__":
    main()
