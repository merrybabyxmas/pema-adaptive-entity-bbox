#!/usr/bin/env bash
# Download VidOR and VidSTG annotation files (no videos - annotations only)
set -e

BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"

echo "=== Downloading VidOR annotations ==="
mkdir -p data/vidor_annotations

# VidOR annotation download via official source
# Training annotations
VIDOR_TRAIN_URL="https://xdshang.github.io/docs/vidor/vidor_annotation_training.zip"
VIDOR_VAL_URL="https://xdshang.github.io/docs/vidor/vidor_annotation_validation.zip"

if command -v wget &>/dev/null; then
    DL="wget -q --show-progress -O"
else
    DL="curl -L -o"
fi

echo "Downloading VidOR training annotations..."
$DL data/vidor_annotations/training.zip "$VIDOR_TRAIN_URL" || {
    echo "Direct download failed - trying alternative..."
    # Alternative: check if already cached
    if [ -d "data/vidor_annotations/training" ]; then
        echo "Training annotations already exist, skipping."
    else
        echo "Please manually download VidOR annotations from:"
        echo "  https://xdshang.github.io/docs/vidor.html"
        echo "and extract to: data/vidor_annotations/"
    fi
}

if [ -f "data/vidor_annotations/training.zip" ]; then
    echo "Extracting training annotations..."
    unzip -q -o data/vidor_annotations/training.zip -d data/vidor_annotations/
fi

echo "Downloading VidOR validation annotations..."
$DL data/vidor_annotations/validation.zip "$VIDOR_VAL_URL" || true

if [ -f "data/vidor_annotations/validation.zip" ]; then
    echo "Extracting validation annotations..."
    unzip -q -o data/vidor_annotations/validation.zip -d data/vidor_annotations/
fi

echo "=== Downloading VidSTG annotations ==="
mkdir -p data/vidstg_annotations
VIDSTG_URL="https://github.com/Guaranteer/VidSTG-Dataset/archive/refs/heads/master.zip"
$DL data/vidstg_annotations/vidstg.zip "$VIDSTG_URL" || {
    echo "VidSTG download failed - will proceed without VidSTG sentences"
}

if [ -f "data/vidstg_annotations/vidstg.zip" ]; then
    unzip -q -o data/vidstg_annotations/vidstg.zip -d data/vidstg_annotations/
    # Find JSON files
    find data/vidstg_annotations/ -name "*.json" | head -5
fi

echo "=== Download complete ==="
ls -la data/vidor_annotations/
