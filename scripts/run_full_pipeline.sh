#!/usr/bin/env bash
# Full pipeline: eval -> infer -> generate
# Run after training is complete
set -e

BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"
source /home/dongwoo39/.venv/bin/activate

CKPT="outputs/runs/bbox_planner_v1/checkpoints/best.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT"
    exit 1
fi

echo "=== Step 1: Evaluation on test set ==="
python scripts/eval_bbox_planner.py \
    --checkpoint "$CKPT" \
    --data data/splits/test.jsonl \
    --out outputs/eval/bbox_planner_v1_test.json

echo ""
echo "=== Evaluation Results ==="
python3 -c "
import json
with open('outputs/eval/bbox_planner_v1_test.json') as f:
    m = json.load(f)
print(f'IoU:        {m[\"iou\"]:.4f}')
print(f'GIoU:       {m[\"giou\"]:.4f}')
print(f'L1:         {m[\"l1\"]:.4f}')
print(f'Center Err: {m[\"center_err\"]:.4f}')
print(f'Area Err:   {m[\"area_err\"]:.4f}')
print(f'Overlap %:  {m[\"overlap_rate\"]*100:.2f}%')
print()
print('Per entity count IoU:')
for k, v in m.get(\"per_nentity_iou\", {}).items():
    print(f'  {k} entities: {v:.4f}')
print()
print('Per state IoU:')
for k, v in m.get(\"per_state_iou\", {}).items():
    if v > 0:
        print(f'  {k}: {v:.4f}')
"

echo ""
echo "=== Step 2: Inference on user story ==="
python scripts/infer_bbox_plan.py \
    --checkpoint "$CKPT" \
    --input examples/user_story_001.json \
    --out outputs/eval/user_story_001_boxes.json

echo ""
echo "=== Predicted Layout ==="
python3 -c "
import json
with open('outputs/eval/user_story_001_boxes.json') as f:
    result = json.load(f)
print('Entities:', result['entities'])
print('Presence:', result['presence'])
for shot in result['shots']:
    print(f'Shot {shot[\"shot_id\"]}: {shot[\"prompt\"]}')
    for e, box in shot['boxes'].items():
        print(f'  {e}: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]')
"

echo ""
echo "=== Step 3: Generate images with predicted layout ==="
python scripts/generate_with_layout.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --out outputs/generations/user_story_001 \
    --no-sd

echo ""
echo "=== Pipeline Complete ==="
echo "Results in outputs/"
ls outputs/eval/ outputs/generations/ 2>/dev/null
