#!/usr/bin/env bash
# Full dataset build pipeline
# Usage: bash scripts/build_dataset_all.sh [--real-data]
set -e

BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"

source /home/dongwoo39/.venv/bin/activate

USE_REAL=${1:-""}

if [ "$USE_REAL" = "--real-data" ]; then
    echo "=== Using VidOR + VidSTG (real data) ==="
    if [ ! -d "data/vidor_annotations" ]; then
        echo "ERROR: data/vidor_annotations not found."
        echo "Please run: bash scripts/download_vidor.sh first"
        exit 1
    fi
    python scripts/01_prepare_vidor_vidstg.py \
        --vidor-ann-dir data/vidor_annotations \
        --vidstg-dir data/vidstg_annotations \
        --out data/jsonl/all_samples.raw.jsonl \
        --split both
else
    echo "=== Using synthetic data (VidOR not yet downloaded) ==="
    python scripts/00_generate_synthetic.py \
        --out data/jsonl/all_samples.raw.jsonl \
        --n-samples 8000 \
        --n-videos 800
fi

echo "=== Step 7: Filter ==="
python scripts/07_filter_dataset.py \
    --input data/jsonl/all_samples.raw.jsonl \
    --output data/jsonl/all_samples.filtered.jsonl \
    --min-det-score 0.45 \
    --min-box-area 0.01 \
    --max-box-area 0.85

echo "=== Step 8: Split ==="
python scripts/08_split_dataset.py \
    --input data/jsonl/all_samples.filtered.jsonl \
    --out-dir data/splits \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

echo "=== Dataset build complete ==="
wc -l data/splits/train.jsonl data/splits/val.jsonl data/splits/test.jsonl
