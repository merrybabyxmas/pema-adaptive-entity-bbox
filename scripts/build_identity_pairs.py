"""
Build SAME-INSTANCE identity training pairs from VidOR tracks.

Root cause fix for PEMA-v12: prior phase4_train paired ref=one cat with
target=a DIFFERENT cat (same category) → model learned category/style transfer,
never identity. VidOR trajectories give per-object tracks (tid) across frames →
the SAME individual at different poses/times. We crop the same tid from two
different frames: ref frame (conditioning) and target frame (to reconstruct).

Output mirrors phase4_train so train_phase4.py can consume it directly:
  out/sample_XXXXXX/
    scene.png             ← target frame (full image, encoded to latent)
    entity_<slug>.png     ← ref crop of SAME entity from a DIFFERENT frame
    style_bg.png          ← target frame background (entity bbox blanked) [optional]
    metadata.json         ← {prompt, entities:[{name, box_xyxy(target), ref_image}]}
  out/scene_index.json

Only videos whose .mp4 exists under --video-dir are processed (lets us download
VidOR in parts and build incrementally).

Usage:
  python scripts/build_identity_pairs.py \
    --ann-dir data/vidor_annotations/training \
    --video-dir data/vidor_videos/videos \
    --out data/phase4_identity \
    --max-per-cat 80 --pairs-per-track 2
"""
import sys, os, json, argparse, glob, re, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

slug = lambda n: re.sub(r"[^a-z0-9_]", "_", n.lower().strip())


def track_bboxes(ann):
    """tid -> {frame_idx: [x1,y1,x2,y2] normalized}, and tid -> category."""
    W, H = ann.get("width", 1), ann.get("height", 1)
    tid2cat = {o["tid"]: o["category"] for o in ann.get("subject/objects", [])}
    tid2fr = defaultdict(dict)
    for f_idx, fobjs in enumerate(ann.get("trajectories", [])):
        for o in fobjs:
            bx = o.get("bbox", {})
            if not isinstance(bx, dict):
                continue
            x1 = max(0., bx["xmin"]/W); y1 = max(0., bx["ymin"]/H)
            x2 = min(1., bx["xmax"]/W); y2 = min(1., bx["ymax"]/H)
            if x2 > x1 + 0.02 and y2 > y1 + 0.02:
                tid2fr[o["tid"]][f_idx] = [round(x1,4), round(y1,4), round(x2,4), round(y2,4)]
    return tid2cat, tid2fr


def pick_frames(frames, n=3, min_gap=15):
    """Pick n well-separated frame indices spanning the track."""
    fs = sorted(frames)
    if len(fs) < 2 or (fs[-1] - fs[0]) < min_gap:
        return []
    # evenly spaced across the track span
    picks = [fs[int(round(k*(len(fs)-1)/(n-1)))] for k in range(n)]
    return sorted(set(picks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="data/vidor_annotations/training")
    ap.add_argument("--video-dir", default="data/vidor_videos/videos")
    ap.add_argument("--out", default="data/phase4_identity")
    ap.add_argument("--entity-list", default="data/pema_train/entity_list.txt")
    ap.add_argument("--max-per-cat", type=int, default=80)
    ap.add_argument("--pairs-per-track", type=int, default=2)
    ap.add_argument("--min-crop-frac", type=float, default=0.05,
                    help="skip entity bbox smaller than this fraction of frame area")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import decord
    decord.bridge.set_bridge("native")

    base = Path(__file__).parent.parent
    out_dir = base / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ents = set(l.strip() for l in open(base/args.entity_list) if l.strip())
    video_root = base / args.video_dir

    ann_files = glob.glob(str(base/args.ann_dir/"**/*.json"), recursive=True)
    rng.shuffle(ann_files)
    print(f"annotations: {len(ann_files)} | video root: {video_root}")

    cat_count = defaultdict(int)
    index, sid = [], 0
    n_vid_used = 0

    for fp in ann_files:
        if all(cat_count[c] >= args.max_per_cat for c in ents):
            break
        try:
            ann = json.load(open(fp))
        except Exception:
            continue
        vpath = ann.get("video_path", "")
        vfile = video_root / vpath
        if not vfile.exists():
            continue  # video not downloaded yet
        tid2cat, tid2fr = track_bboxes(ann)
        good = [(tid, tid2cat[tid], fr) for tid, fr in tid2fr.items()
                if tid2cat.get(tid) in ents and cat_count[tid2cat[tid]] < args.max_per_cat]
        if not good:
            continue
        try:
            vr = decord.VideoReader(str(vfile))
            nfr = len(vr)
        except Exception:
            continue
        used_this_video = False
        for tid, cat, fr in good:
            if cat_count[cat] >= args.max_per_cat:
                continue
            picks = pick_frames(fr, n=args.pairs_per_track + 1)
            picks = [p for p in picks if p < nfr]
            if len(picks) < 2:
                continue
            # ref = first pick; targets = the rest
            ref_fi = picks[0]
            for tgt_fi in picks[1:]:
                if cat_count[cat] >= args.max_per_cat:
                    break
                bb_t = fr.get(tgt_fi); bb_r = fr.get(ref_fi)
                if bb_t is None or bb_r is None:
                    continue
                if (bb_t[2]-bb_t[0])*(bb_t[3]-bb_t[1]) < args.min_crop_frac:
                    continue
                try:
                    tgt_img = Image.fromarray(vr[tgt_fi].asnumpy()).convert("RGB")
                    ref_img = Image.fromarray(vr[ref_fi].asnumpy()).convert("RGB")
                except Exception:
                    continue
                W, H = ref_img.size
                rc = ref_img.crop((int(bb_r[0]*W), int(bb_r[1]*H),
                                   int(bb_r[2]*W), int(bb_r[3]*H)))
                sdir = out_dir / f"sample_{sid:06d}"
                sdir.mkdir(parents=True, exist_ok=True)
                tgt_img.save(sdir/"scene.png")
                ref_name = f"entity_{slug(cat)}.png"
                rc.save(sdir/ref_name)
                meta = {
                    "prompt": f"a {cat.replace('/', ' ')}",
                    "entities": [{"name": cat, "box_xyxy": bb_t, "ref_image": ref_name}],
                    "source": {"video": vpath, "tid": tid,
                               "ref_frame": ref_fi, "tgt_frame": tgt_fi},
                }
                (sdir/"metadata.json").write_text(json.dumps(meta))
                index.append(f"sample_{sid:06d}")
                sid += 1
                cat_count[cat] += 1
                used_this_video = True
        if used_this_video:
            n_vid_used += 1

    (out_dir/"scene_index.json").write_text(json.dumps(index))
    print(f"\nBUILT {sid} identity pairs from {n_vid_used} videos")
    print("per-category:", dict(sorted(cat_count.items(), key=lambda x:-x[1])))


if __name__ == "__main__":
    main()
