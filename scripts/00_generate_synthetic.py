"""
Synthetic multi-shot dataset generator.
Produces realistic JSONL samples without needing raw videos or detectors.
Bboxes are generated from rule-based layout templates + noise.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
from pathlib import Path
from src.utils.io import save_jsonl
from src.utils.logging import get_logger

logger = get_logger("synthetic")

ENTITY_VOCAB = [
    "cat", "dog", "bird", "horse", "cow", "sheep",
    "person", "child", "man", "woman",
    "car", "bicycle", "motorcycle",
    "rabbit", "fox", "bear", "elephant",
]

BACKGROUNDS = [
    "park", "garden", "forest", "beach", "street", "room", "field",
    "yard", "lake", "mountain", "playground",
]

RELATION_TYPES = ["left_of", "right_of", "beside", "near", "above", "below"]

PROMPT_TEMPLATES_1 = [
    "A {e0} is in the {bg}.",
    "The {e0} stands alone in the {bg}.",
    "A {e0} walks through the {bg}.",
    "The {e0} sits in the {bg}.",
    "A {e0} runs across the {bg}.",
]

PROMPT_TEMPLATES_2 = [
    "A {e0} and a {e1} are in the {bg}.",
    "The {e0} sits beside the {e1} in the {bg}.",
    "A {e0} and a {e1} play together in the {bg}.",
    "The {e0} stands near the {e1} in the {bg}.",
    "A {e0} chases a {e1} in the {bg}.",
    "The {e0} and the {e1} face each other in the {bg}.",
]

PROMPT_TEMPLATES_3 = [
    "A {e0}, a {e1}, and a {e2} are gathered in the {bg}.",
    "The {e0} stands between the {e1} and the {e2} in the {bg}.",
    "A {e0} watches a {e1} and a {e2} in the {bg}.",
]

FOCUS_ONLY_TEMPLATES = [
    "Only the {e0} remains in the {bg}.",
    "The {e0} moves alone in the {bg}.",
    "The {e0} runs away in the {bg}.",
]

RE_ENTRY_TEMPLATES = [
    "The {e0} reappears in the {bg}.",
    "The {e0} returns to the {bg}.",
    "The {e0} comes back in the {bg}.",
]


def sample_layout_1entity(rng) -> np.ndarray:
    """Generate single entity bbox: center-biased."""
    cx = rng.uniform(0.3, 0.7)
    cy = rng.uniform(0.25, 0.75)
    w = rng.uniform(0.25, 0.55)
    h = rng.uniform(0.30, 0.65)
    x1 = max(0.01, cx - w / 2)
    y1 = max(0.01, cy - h / 2)
    x2 = min(0.99, cx + w / 2)
    y2 = min(0.99, cy + h / 2)
    return np.array([x1, y1, x2, y2])


def sample_layout_2entities(rng, relation=None) -> tuple[np.ndarray, np.ndarray]:
    """Generate two non-overlapping entity bboxes."""
    if relation in ("left_of", "right_of", None):
        # horizontal split
        split = rng.uniform(0.35, 0.65)
        margin = rng.uniform(0.02, 0.08)
        cx1 = rng.uniform(0.12, split - margin)
        cx2 = rng.uniform(split + margin, 0.88)
        cy = rng.uniform(0.3, 0.7)
        w = rng.uniform(0.18, 0.38)
        h = rng.uniform(0.25, 0.55)
        b1 = np.clip([cx1 - w/2, cy - h/2, cx1 + w/2, cy + h/2], 0.01, 0.99)
        b2 = np.clip([cx2 - w/2, cy - h/2, cx2 + w/2, cy + h/2], 0.01, 0.99)
        if relation == "right_of":
            b1, b2 = b2, b1
    elif relation in ("above", "below"):
        cx = rng.uniform(0.3, 0.7)
        split = rng.uniform(0.35, 0.60)
        margin = 0.05
        cy1 = rng.uniform(0.1, split - margin)
        cy2 = rng.uniform(split + margin, 0.85)
        w = rng.uniform(0.2, 0.45)
        h = rng.uniform(0.18, 0.35)
        b1 = np.clip([cx - w/2, cy1 - h/2, cx + w/2, cy1 + h/2], 0.01, 0.99)
        b2 = np.clip([cx - w/2, cy2 - h/2, cx + w/2, cy2 + h/2], 0.01, 0.99)
        if relation == "below":
            b1, b2 = b2, b1
    else:
        # near/beside: close together
        cx1 = rng.uniform(0.15, 0.42)
        cx2 = cx1 + rng.uniform(0.15, 0.35)
        cy = rng.uniform(0.25, 0.70)
        w = rng.uniform(0.15, 0.32)
        h = rng.uniform(0.20, 0.50)
        b1 = np.clip([cx1 - w/2, cy - h/2, cx1 + w/2, cy + h/2], 0.01, 0.99)
        b2 = np.clip([cx2 - w/2, cy - h/2, cx2 + w/2, cy + h/2], 0.01, 0.99)

    # add noise
    noise = rng.normal(0, 0.02, size=4)
    b1 = np.clip(b1 + noise, 0.01, 0.99)
    noise = rng.normal(0, 0.02, size=4)
    b2 = np.clip(b2 + noise, 0.01, 0.99)
    # ensure x2>x1, y2>y1
    b1 = ensure_valid_box(b1)
    b2 = ensure_valid_box(b2)
    return b1, b2


def sample_layout_3entities(rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ws = [rng.uniform(0.15, 0.28) for _ in range(3)]
    hs = [rng.uniform(0.20, 0.45) for _ in range(3)]
    cxs = [0.18, 0.50, 0.82]
    cy = rng.uniform(0.30, 0.65)
    boxes = []
    for i in range(3):
        cx = cxs[i] + rng.normal(0, 0.04)
        b = np.clip([cx - ws[i]/2, cy - hs[i]/2, cx + ws[i]/2, cy + hs[i]/2], 0.01, 0.99)
        boxes.append(ensure_valid_box(b))
    return tuple(boxes)


def ensure_valid_box(b, min_size=0.05):
    x1, y1, x2, y2 = b
    if x2 - x1 < min_size:
        x2 = x1 + min_size
    if y2 - y1 < min_size:
        y2 = y1 + min_size
    return np.clip([x1, y1, x2, y2], 0.01, 0.99)


def generate_presence_pattern(S: int, E: int, rng) -> np.ndarray:
    """Generate a plausible presence matrix [S, E]."""
    P = np.ones((S, E), dtype=int)
    # randomly make some entities absent from some shots
    for e in range(E):
        # decide on a pattern
        pattern_type = rng.choice(["always", "enter_late", "exit_early", "middle_only", "re_entry"])
        if pattern_type == "always":
            pass
        elif pattern_type == "enter_late":
            start = rng.randint(1, max(2, S - 1))
            P[:start, e] = 0
        elif pattern_type == "exit_early":
            end = rng.randint(1, max(2, S))
            P[end:, e] = 0
        elif pattern_type == "middle_only" and S >= 3:
            P[0, e] = 0
            P[-1, e] = 0
        elif pattern_type == "re_entry" and S >= 4:
            mid = S // 2
            P[mid, e] = 0

    # ensure at least one entity is always present in each shot
    for s in range(S):
        if P[s].sum() == 0:
            P[s, 0] = 1
    return P


def build_shot(shot_id: int, active_entities: list[str], background: str,
               all_entities: list[str], state_map: dict, P: np.ndarray, rng) -> dict:
    n_active = len(active_entities)
    relation = None
    relations = []

    if n_active == 1:
        e0 = active_entities[0]
        templates = PROMPT_TEMPLATES_1
        st = state_map.get(e0, "stay")
        if st in ("re_entry", "entry"):
            templates = RE_ENTRY_TEMPLATES + PROMPT_TEMPLATES_1
        tmpl = rng.choice(templates)
        prompt = tmpl.format(e0=e0, bg=background)
        boxes = {e0: sample_layout_1entity(rng).tolist()}

    elif n_active == 2:
        e0, e1 = active_entities[0], active_entities[1]
        relation = rng.choice(RELATION_TYPES)
        relations = [[e0, relation, e1]]
        tmpl = rng.choice(PROMPT_TEMPLATES_2)
        prompt = tmpl.format(e0=e0, e1=e1, bg=background)
        b0, b1 = sample_layout_2entities(rng, relation)
        boxes = {e0: b0.tolist(), e1: b1.tolist()}

    elif n_active == 3:
        e0, e1, e2 = active_entities[0], active_entities[1], active_entities[2]
        tmpl = rng.choice(PROMPT_TEMPLATES_3)
        prompt = tmpl.format(e0=e0, e1=e1, e2=e2, bg=background)
        b0, b1, b2 = sample_layout_3entities(rng)
        boxes = {e0: b0.tolist(), e1: b1.tolist(), e2: b2.tolist()}

    else:
        # 4 entities: two rows
        prompt = "Several animals are gathered in the {bg}.".format(bg=background)
        boxes = {}
        cxs = [0.15, 0.45, 0.65, 0.85]
        for i, e in enumerate(active_entities[:4]):
            cx = cxs[i % 4] + rng.normal(0, 0.03)
            cy = 0.35 + (i // 4) * 0.35 + rng.normal(0, 0.03)
            w = rng.uniform(0.12, 0.24)
            h = rng.uniform(0.15, 0.35)
            b = ensure_valid_box(np.clip([cx - w/2, cy - h/2, cx + w/2, cy + h/2], 0.01, 0.99))
            boxes[e] = b.tolist()

    quality = {f"{e}_det_score": round(rng.uniform(0.55, 0.98), 3) for e in active_entities}

    states = {}
    for e in all_entities:
        states[e] = state_map.get(e, "absent")

    return {
        "shot_id": shot_id,
        "prompt": prompt,
        "background": background,
        "active_entities": active_entities,
        "states": states,
        "relations": relations,
        "boxes": boxes,
        "quality": quality,
        "keyframe_path": None,
    }


def compute_state_map(P: np.ndarray, entities: list[str], shot_id: int) -> dict:
    state_map = {}
    S = P.shape[0]
    for ei, e in enumerate(entities):
        s = shot_id
        prev = P[s - 1, ei] == 1 if s > 0 else False
        cur = P[s, ei] == 1
        if not cur:
            state_map[e] = "absent"
        elif not prev and s == 0:
            state_map[e] = "initial"
        elif not prev:
            ever = any(P[ss, ei] == 1 for ss in range(s))
            state_map[e] = "re_entry" if ever else "entry"
        elif prev and cur:
            nxt = P[s + 1, ei] == 1 if s < S - 1 else False
            state_map[e] = "exit" if not nxt else "stay"
    return state_map


def generate_sample(video_id: str, seq_id: int, rng, min_shots=3, max_shots=5,
                    min_entities=1, max_entities=4) -> dict:
    S = rng.randint(min_shots, max_shots + 1)
    E = rng.randint(min_entities, max_entities + 1)

    entities = rng.choice(ENTITY_VOCAB, size=E, replace=False).tolist()
    background = rng.choice(BACKGROUNDS)
    P = generate_presence_pattern(S, E, rng)

    shots = []
    for s in range(S):
        state_map = compute_state_map(P, entities, s)
        active = [entities[ei] for ei in range(E) if P[s, ei] == 1]
        shot = build_shot(s, active, background, entities, state_map, P, rng)
        shots.append(shot)

    presence = P.tolist()

    return {
        "sample_id": f"{video_id}_seq{seq_id:04d}",
        "video_id": video_id,
        "entity_vocab": entities,
        "presence": presence,
        "shots": shots,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/jsonl/all_samples.raw.jsonl")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-videos", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)

    base = Path(__file__).parent.parent
    out_path = base / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    video_ids = [f"video{i:05d}" for i in range(args.n_videos)]
    per_video = args.n_samples // args.n_videos

    for vid in video_ids:
        for seq_id in range(per_video):
            try:
                s = generate_sample(vid, seq_id, rng)
                samples.append(s)
            except Exception as e:
                logger.warning(f"skip {vid}_{seq_id}: {e}")

    # fill remaining
    extra = args.n_samples - len(samples)
    for i in range(extra):
        vid = rng.choice(video_ids)
        try:
            s = generate_sample(vid, 9000 + i, rng)
            samples.append(s)
        except Exception:
            pass

    save_jsonl(samples, str(out_path))
    logger.info(f"Saved {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
