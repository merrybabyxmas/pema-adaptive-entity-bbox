import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import Counter
from src.utils.io import load_jsonl, save_jsonl
from src.data.quality import check_box_valid, check_sample_quality, has_multiple_same_class
from src.utils.logging import get_logger

logger = get_logger("filter")


def filter_sample(sample: dict, cfg: dict) -> bool:
    min_score = cfg.get("min_det_score", 0.45)
    min_area = cfg.get("min_box_area", 0.01)
    max_area = cfg.get("max_box_area", 0.85)
    min_asp = cfg.get("min_aspect_ratio", 0.15)
    max_asp = cfg.get("max_aspect_ratio", 6.0)
    remove_multi = cfg.get("remove_multiple_same_class", True)

    for shot in sample["shots"]:
        if remove_multi and has_multiple_same_class(shot):
            return False
        for entity in shot["active_entities"]:
            score = shot["quality"].get(f"{entity}_det_score", 1.0)
            if score < min_score:
                return False
            if entity not in shot["boxes"]:
                return False
            box = shot["boxes"][entity]
            if not check_box_valid(box, min_area, max_area, min_asp, max_asp):
                return False
            # ensure x2>x1, y2>y1
            if box[2] <= box[0] or box[3] <= box[1]:
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/jsonl/all_samples.raw.jsonl")
    parser.add_argument("--output", default="data/jsonl/all_samples.filtered.jsonl")
    parser.add_argument("--min-det-score", type=float, default=0.45)
    parser.add_argument("--min-box-area", type=float, default=0.01)
    parser.add_argument("--max-box-area", type=float, default=0.85)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.15)
    parser.add_argument("--max-aspect-ratio", type=float, default=6.0)
    parser.add_argument("--remove-multiple-same-class", type=str, default="true")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inp = os.path.join(base, args.input)
    out = os.path.join(base, args.output)

    cfg = {
        "min_det_score": args.min_det_score,
        "min_box_area": args.min_box_area,
        "max_box_area": args.max_box_area,
        "min_aspect_ratio": args.min_aspect_ratio,
        "max_aspect_ratio": args.max_aspect_ratio,
        "remove_multiple_same_class": args.remove_multiple_same_class.lower() == "true",
    }

    samples = load_jsonl(inp)
    logger.info(f"Loaded {len(samples)} samples")

    kept = [s for s in samples if filter_sample(s, cfg)]
    logger.info(f"Kept {len(kept)}/{len(samples)} samples after filtering")

    save_jsonl(kept, out)
    logger.info(f"Saved to {out}")


if __name__ == "__main__":
    main()
