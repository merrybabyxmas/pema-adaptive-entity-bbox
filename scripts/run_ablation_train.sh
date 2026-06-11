#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
run(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/train_bbox_planner.py --config configs/ablation/$2.yaml > outputs/abl_logs/$2.log 2>&1; echo "DONE $2"; }
# wave 1 (4 GPUs)
run 0 full &
run 1 wo_state &
run 2 wo_relation &
run 3 wo_shotattn &
wait
# wave 2
run 0 wo_tempattn &
run 1 wo_depth &
wait
echo "ALL ABLATIONS DONE"
