import json
import jsonlines
from pathlib import Path


def load_jsonl(path):
    samples = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            samples.append(obj)
    return samples


def save_jsonl(samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode='w') as writer:
        writer.write_all(samples)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(obj, path, indent=2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
