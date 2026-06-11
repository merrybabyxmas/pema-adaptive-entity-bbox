#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $PY scripts/run_ablation_gen.py --mode anchors --shard $i/4 \
    > outputs/abl_logs/anchors_$i.log 2>&1 &
done
wait
echo "ANCHORS ALL DONE"
