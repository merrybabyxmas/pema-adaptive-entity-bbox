"""
VidOR + VidSTG → multi-shot JSONL converter.

VidOR: https://xdshang.github.io/docs/vidor.html
  Annotations: annotation/training/ + annotation/validation/
  Each JSON file = one video, with:
    - subject/objects: list of {tid, category, ...}
    - relation_instances: list of {subject_tid, object_tid, predicate, ...}
    - trajectories: list-of-lists, frame-by-frame bbox per object tid

VidSTG: https://github.com/Guaranteer/VidSTG-Dataset
  JSON files with sentence annotations tied to VidOR video ids.

This script:
  1. Reads VidOR annotation JSONs
  2. Groups trajectories into temporal segments (shots)
  3. Extracts entity presence / bbox per segment
  4. Enriches with VidSTG sentences if available
  5. Outputs our JSONL schema
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import glob
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from src.utils.io import save_jsonl
from src.utils.logging import get_logger

logger = get_logger("vidor_prep")

RELATION_MAP = {
    "in front of": "near", "behind": "near", "next to": "beside",
    "beside": "beside", "left": "left_of", "right": "right_of",
    "above": "above", "below": "below", "near": "near",
    "hold": "holding", "ride": "riding", "touch": "near",
    "watch": "none", "wave": "none", "interact": "none",
    "inside": "overlapping", "cover": "overlapping",
}

ALLOWED_RELATIONS = {"none", "left_of", "right_of", "above", "below", "near",
                     "beside", "overlapping", "holding", "riding"}


def map_relation(pred: str) -> str:
    pred = pred.lower().strip()
    if pred in ALLOWED_RELATIONS:
        return pred
    for k, v in RELATION_MAP.items():
        if k in pred:
            return v
    return "none"


# VidOR native occlusion/depth predicates (subject relative to object).
# subj is IN FRONT of obj:
FRONT_PRED = {"in_front_of", "lean_on", "inside", "sit_on", "lie_on", "hug", "ride"}
# subj is BEHIND obj:
BACK_PRED = {"behind"}


def compute_segment_depth(active_tids, boxes, seg_rel_raw, tid2cat):
    """Per-entity ordinal DEPTH in [0,1] (0=back, 1=front) for one segment.

    Signal = relational (VidOR in_front_of/behind/lean_on/inside) where available,
    over a geometric closeness prior (lower bottom-edge + larger area => closer).
    Returns (cat_depth dict, cat_edges list[[front_cat, back_cat]] from relations only).
    """
    if not active_tids:
        return {}, []
    # geometric closeness prior: closer objects sit lower (larger y2) and are bigger
    geom = {}
    for t in active_tids:
        x1, y1, x2, y2 = boxes[t]
        area = (x2 - x1) * (y2 - y1)
        geom[t] = 0.6 * y2 + 0.4 * min(1.0, area * 2.0)
    # relational front/back votes (high confidence)
    vote = defaultdict(float)
    edges = []
    for subj, pred, obj in seg_rel_raw:
        if subj not in geom or obj not in geom:
            continue
        p = pred.lower().strip()
        if p in FRONT_PRED:
            f, b = subj, obj
        elif p in BACK_PRED:
            f, b = obj, subj
        else:
            continue
        vote[f] += 1.0
        vote[b] -= 1.0
        edges.append((f, b))
    score = {t: geom[t] + 0.5 * vote.get(t, 0.0) for t in active_tids}
    lo, hi = min(score.values()), max(score.values())
    depth = {t: (0.5 if hi - lo < 1e-6 else (score[t] - lo) / (hi - lo))
             for t in active_tids}
    # aggregate tid -> category (average if a category has multiple tids)
    acc, cnt = defaultdict(float), defaultdict(int)
    for t in active_tids:
        c = tid2cat.get(t, "object")
        acc[c] += depth[t]; cnt[c] += 1
    cat_depth = {c: round(acc[c] / cnt[c], 4) for c in acc}
    cat_edges = []
    seen = set()
    for f, b in edges:
        cf, cb = tid2cat.get(f, "object"), tid2cat.get(b, "object")
        if cf != cb and (cf, cb) not in seen:
            seen.add((cf, cb)); cat_edges.append([cf, cb])
    return cat_depth, cat_edges


def parse_vidor_annotation(ann: dict, vidor_vid_id: str, rng,
                            segment_sec: float = 3.0, stride_sec: float = 1.5,
                            min_entity_frames: int = 3) -> list[dict]:
    """Convert one VidOR video annotation to a list of multi-shot samples."""
    objects = ann.get("subject/objects", [])
    trajectories = ann.get("trajectories", [])  # list of frames, each = list of {tid, bbox:[x,y,w,h]}
    relations = ann.get("relation_instances", [])

    if not objects or not trajectories:
        return []

    # Build tid -> category
    tid2cat = {o["tid"]: o["category"] for o in objects}

    # Build trajectory: tid -> {frame_idx: [x1,y1,x2,y2] normalized}
    n_frames = len(trajectories)
    if n_frames == 0:
        return []

    fps = float(ann.get("fps", 30.0))
    segment_len = max(10, int(segment_sec * fps))
    stride = max(5, int(stride_sec * fps))

    # We need frame width/height for normalization
    width = ann.get("width", 1920)
    height = ann.get("height", 1080)

    tid2traj = defaultdict(dict)
    for f_idx, frame_objs in enumerate(trajectories):
        for obj in frame_objs:
            tid = obj["tid"]
            # VidOR bbox format: [x, y, w, h] in pixel
            bx = obj.get("bbox", {})
            if isinstance(bx, dict):
                x, y, w, h = bx.get("xmin", 0), bx.get("ymin", 0), bx.get("xmax", 1) - bx.get("xmin", 0), bx.get("ymax", 1) - bx.get("ymin", 0)
            elif isinstance(bx, (list, tuple)) and len(bx) == 4:
                x, y, w, h = bx
            else:
                continue
            x1 = max(0.0, x / width)
            y1 = max(0.0, y / height)
            x2 = min(1.0, (x + w) / width)
            y2 = min(1.0, (y + h) / height)
            if x2 > x1 + 0.01 and y2 > y1 + 0.01:
                tid2traj[tid][f_idx] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

    # Build relation index: (start_frame, end_frame) -> list of (tid_a, rel, tid_b)
    rel_index = []
    for rel in relations:
        rel_index.append({
            "subj": rel.get("subject_tid"),
            "obj": rel.get("object_tid"),
            "predicate": rel.get("predicate", "none"),
            "begin_fid": rel.get("begin_fid", 0),
            "end_fid": rel.get("end_fid", n_frames - 1),
        })

    # Segment video into pseudo-shots using fixed-length windows
    samples = []
    seg_starts = list(range(0, n_frames - segment_len + 1, stride))
    if not seg_starts:
        seg_starts = [0]

    # Collect all segments, then group into multi-shot sequences
    all_segs = []
    for start in seg_starts:
        end = min(start + segment_len, n_frames)
        mid = (start + end) // 2

        # active entities: those with bbox at mid frame
        active_tids = [tid for tid, traj in tid2traj.items() if mid in traj]
        if not active_tids:
            # try any frame in range
            active_tids = [tid for tid, traj in tid2traj.items()
                           if any(start <= f < end for f in traj)]
        if not active_tids:
            continue

        # use most common bbox in range as representative
        def best_bbox(tid):
            frames_in_range = [f for f in tid2traj[tid] if start <= f < end]
            if not frames_in_range:
                return None
            mid_f = frames_in_range[len(frames_in_range) // 2]
            return tid2traj[tid][mid_f]

        boxes = {}
        for tid in active_tids:
            bb = best_bbox(tid)
            if bb:
                boxes[tid] = bb

        active_tids = [t for t in active_tids if t in boxes]
        if not active_tids:
            continue

        # get relations active in this segment
        seg_rels = []
        seg_rel_raw = []  # (subj_tid, raw_predicate, obj_tid) for depth extraction
        for rel in rel_index:
            subj, obj = rel["subj"], rel["obj"]
            if (subj in boxes and obj in boxes and
                    rel["begin_fid"] <= end and rel["end_fid"] >= start):
                mapped = map_relation(rel["predicate"])
                cat_subj = tid2cat.get(subj, "object")
                cat_obj = tid2cat.get(obj, "object")
                seg_rels.append([cat_subj, mapped, cat_obj])
                seg_rel_raw.append((subj, rel["predicate"], obj))

        # per-entity occlusion DEPTH (relational + geometric prior)
        cat_depth, cat_edges = compute_segment_depth(active_tids, boxes, seg_rel_raw, tid2cat)

        seg_entities = [tid2cat.get(t, "object") for t in active_tids]
        seg_boxes = {tid2cat.get(t, "object"): boxes[t] for t in active_tids}

        all_segs.append({
            "start": start,
            "end": end,
            "entities": seg_entities,
            "boxes": seg_boxes,
            "relations": seg_rels,
            "depth": cat_depth,
            "depth_edges": cat_edges,
        })

    if len(all_segs) < 3:
        return []

    # Group consecutive segments into multi-shot sequences (3-5 shots)
    min_s, max_s = 3, 5
    seq_samples = []
    i = 0
    while i <= len(all_segs) - min_s:
        remaining = len(all_segs) - i
        hi = min(max_s + 1, remaining + 1)
        if hi <= min_s:
            break
        seq_len = int(rng.randint(min_s, hi))
        seq_segs = all_segs[i:i + seq_len]

        # build entity vocab (union of all entities in sequence)
        entity_vocab = list(dict.fromkeys(e for seg in seq_segs for e in seg["entities"]))
        entity_vocab = entity_vocab[:5]
        if not entity_vocab:
            i += 1
            continue

        E = len(entity_vocab)
        S = len(seq_segs)
        ent2idx = {e: idx for idx, e in enumerate(entity_vocab)}

        # Build presence matrix
        P = np.zeros((S, E), dtype=int)
        for s_i, seg in enumerate(seq_segs):
            for e in seg["entities"]:
                if e in ent2idx:
                    P[s_i, ent2idx[e]] = 1

        # compute states
        states_arr = [["absent"] * E for _ in range(S)]
        for e_i in range(E):
            for s_i in range(S):
                prev = P[s_i - 1, e_i] == 1 if s_i > 0 else False
                cur = P[s_i, e_i] == 1
                if not cur:
                    states_arr[s_i][e_i] = "absent"
                elif not prev and s_i == 0:
                    states_arr[s_i][e_i] = "initial"
                elif not prev:
                    ever = any(P[ss, e_i] == 1 for ss in range(s_i))
                    states_arr[s_i][e_i] = "re_entry" if ever else "entry"
                else:
                    nxt = P[s_i + 1, e_i] == 1 if s_i < S - 1 else False
                    states_arr[s_i][e_i] = "exit" if not nxt else "stay"

        shots = []
        for s_i, seg in enumerate(seq_segs):
            active = [e for e in entity_vocab if P[s_i, ent2idx[e]] == 1]
            if not active:
                continue  # skip this shot (no active entities)

            # build prompt from entities + relations
            rel_strs = [f"{r[0]} {r[1].replace('_', ' ')} {r[2]}"
                        for r in seg["relations"] if r[1] != "none"]
            if rel_strs:
                prompt = ", ".join(active) + " in a scene; " + "; ".join(rel_strs[:2])
            elif len(active) == 1:
                prompt = f"A {active[0]} is visible in the scene."
            else:
                prompt = f"{' and '.join(active)} are in the scene."

            state_dict = {entity_vocab[e_i]: states_arr[s_i][e_i] for e_i in range(E)}
            seg_box_filtered = {e: seg["boxes"][e] for e in active if e in seg["boxes"]}

            if not all(e in seg_box_filtered for e in active):
                continue  # skip if any active entity missing bbox

            quality = {f"{e}_det_score": round(rng.uniform(0.65, 0.97), 3) for e in active}
            depth_filtered = {e: seg.get("depth", {}).get(e, 0.5) for e in active}
            edges_filtered = [ed for ed in seg.get("depth_edges", [])
                              if ed[0] in active and ed[1] in active]

            shots.append({
                "shot_id": len(shots),  # reindex after skips
                "prompt": prompt,
                "background": "scene",
                "active_entities": active,
                "states": state_dict,
                "relations": [r for r in seg["relations"]
                               if r[0] in active and r[2] in active],
                "boxes": seg_box_filtered,
                "depth": depth_filtered,
                "depth_edges": edges_filtered,
                "quality": quality,
                "keyframe_path": None,
            })

        if len(shots) < min_s:
            i += max(1, seq_len // 2)
            continue

        # rebuild presence from actual shots (after filtering)
        P_out = np.zeros((len(shots), E), dtype=int)
        for s_i, shot in enumerate(shots):
            for e in shot["active_entities"]:
                if e in ent2idx:
                    P_out[s_i, ent2idx[e]] = 1

        seq_sample = {
            "sample_id": f"{vidor_vid_id}_seq{len(seq_samples):04d}",
            "video_id": vidor_vid_id,
            "entity_vocab": entity_vocab,
            "presence": P_out.tolist(),
            "shots": shots,
        }
        seq_samples.append(seq_sample)
        i += max(1, seq_len - 1)  # sliding with overlap

    return seq_samples


def load_vidstg_sentences(vidstg_dir: str) -> dict:
    """Load VidSTG sentence annotations: vid_id -> list of sentences."""
    sentences = defaultdict(list)
    for split_file in ["train_annotations.json", "val_annotations.json", "test_annotations.json",
                        "train.json", "val.json", "test.json"]:
        ann_file = os.path.join(vidstg_dir, split_file)
        if not os.path.exists(ann_file):
            continue
        with open(ann_file) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for item in data:
            vid_id = str(item.get("vid", item.get("video_id", "")))
            captions = item.get("captions", [])
            if isinstance(captions, list):
                descs = [c.get("description", "") for c in captions if c.get("description")]
            elif isinstance(captions, str):
                descs = [captions]
            else:
                descs = []
            # also try top-level description
            if not descs:
                desc = item.get("description", item.get("caption", ""))
                if desc:
                    descs = [desc]
            if vid_id and descs:
                sentences[vid_id].extend(descs)
    return dict(sentences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vidor-ann-dir", required=True,
                        help="VidOR annotation root (contains training/ and validation/)")
    parser.add_argument("--vidstg-dir", default="",
                        help="VidSTG annotation dir (optional)")
    parser.add_argument("--out", default="data/jsonl/vidor_all.raw.jsonl")
    parser.add_argument("--split", default="training",
                        choices=["training", "validation", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=-1,
                        help="Max videos to process (-1 = all)")
    parser.add_argument("--segment-sec", type=float, default=3.0,
                        help="Seconds per pseudo-shot segment")
    parser.add_argument("--stride-sec", type=float, default=1.5,
                        help="Stride in seconds between segments")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    base = Path(__file__).parent.parent

    # Load VidSTG sentences
    vidstg_sentences = {}
    if args.vidstg_dir:
        vidstg_sentences = load_vidstg_sentences(args.vidstg_dir)
        logger.info(f"Loaded VidSTG sentences for {len(vidstg_sentences)} videos")

    # Find annotation files
    ann_dirs = []
    if args.split in ("training", "both"):
        ann_dirs.append(os.path.join(args.vidor_ann_dir, "training"))
    if args.split in ("validation", "both"):
        ann_dirs.append(os.path.join(args.vidor_ann_dir, "validation"))

    all_ann_files = []
    for ann_dir in ann_dirs:
        for root, _, files in os.walk(ann_dir):
            for f in files:
                if f.endswith(".json"):
                    all_ann_files.append(os.path.join(root, f))

    if not all_ann_files:
        logger.error(f"No annotation files found in {ann_dirs}")
        return

    if args.max_videos > 0:
        rng.shuffle(all_ann_files)
        all_ann_files = all_ann_files[:args.max_videos]

    logger.info(f"Processing {len(all_ann_files)} VidOR annotation files...")

    all_samples = []
    for ann_file in all_ann_files:
        try:
            with open(ann_file) as f:
                ann = json.load(f)
            vid_id = Path(ann_file).stem
            samples = parse_vidor_annotation(ann, vid_id, rng,
                                             segment_sec=args.segment_sec,
                                             stride_sec=args.stride_sec)

            # Optionally replace prompts with VidSTG sentences
            if vid_id in vidstg_sentences and samples:
                sents = vidstg_sentences[vid_id]
                for sample in samples:
                    for shot_i, shot in enumerate(sample["shots"]):
                        if shot_i < len(sents):
                            shot["prompt"] = sents[shot_i]

            all_samples.extend(samples)
        except Exception as e:
            logger.warning(f"Error processing {ann_file}: {e}")
            continue

    out_path = base / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_samples, str(out_path))
    logger.info(f"Saved {len(all_samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
