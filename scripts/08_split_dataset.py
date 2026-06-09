import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from collections import defaultdict
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import get_logger

logger = get_logger("split")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/jsonl/all_samples.filtered.jsonl")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-key", default="video_id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inp = os.path.join(base, args.input)
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    samples = load_jsonl(inp)
    logger.info(f"Loaded {len(samples)} samples")

    # group by split key
    groups = defaultdict(list)
    for s in samples:
        groups[s[args.split_key]].append(s)

    keys = sorted(groups.keys())
    random.seed(args.seed)
    random.shuffle(keys)

    n = len(keys)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])

    train = [s for s in samples if s[args.split_key] in train_keys]
    val = [s for s in samples if s[args.split_key] in val_keys]
    test = [s for s in samples if s[args.split_key] in test_keys]

    save_jsonl(train, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(out_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(out_dir, "test.jsonl"))

    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    logger.info(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
