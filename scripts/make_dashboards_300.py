"""4-page dashboard (montage) per job for the 300-story run -> outputs/eval_300/dashboards/<job>/."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).parent.parent
R = BASE / "outputs/lisa/aaai300"
OUT = BASE / "outputs/eval_300/dashboards"; OUT.mkdir(parents=True, exist_ok=True)
STORIES = [s["name"] for s in json.loads((BASE / "data/captions/stories_aaai_eval_300_6patterns_no_reverse.json").read_text())]
JOBS = ["B_template", "B_retrieval", "B_llm", "B_center", "CABL_full", "CABL_wo_shotemb",
        "CABL_wo_entityemb", "CABL_wo_state", "CABL_wo_temporal", "CABL_wo_depth"]
TH, lbl, MAXC, NCOL, per_page = 120, 120, 3, 2, 30  # 300 stories -> 10 pages? keep 4 by sampling

# to keep "4 pages like before", sample 120 of 300 (every ~2.5th) -> 30/page x 4
sample = STORIES[::2][:120]
pages = [sample[i:i+per_page] for i in range(0, len(sample), per_page)]


def build(job):
    J = R / job; od = OUT / job; od.mkdir(parents=True, exist_ok=True)
    if not J.exists():
        return
    for pi, page in enumerate(pages, 1):
        rows = (len(page) + NCOL - 1) // NCOL
        W = NCOL * (lbl + MAXC * TH); H = 24 + rows * TH
        c = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(c)
        d.text((4, 6), f"{job}  (sample page {pi}/{len(pages)} of 300)", fill="black")
        for k, st in enumerate(page):
            r, col = divmod(k, NCOL); y = 24 + r * TH; x0 = col * (lbl + MAXC * TH)
            d.text((x0 + 2, y + 2), st[:26], fill="black")
            shots = [p for p in sorted((J / st).glob("shot_[0-9][0-9][0-9].png")) if "bbox" not in p.name][:MAXC]
            for ci, p in enumerate(shots):
                c.paste(Image.open(p).convert("RGB").resize((TH, TH)), (x0 + lbl + ci * TH, y))
        c.save(od / f"page_{pi}.png")
    print(f"  {job}: {len(pages)} pages")


for j in JOBS:
    build(j)
print(f"dashboards -> {OUT}")
