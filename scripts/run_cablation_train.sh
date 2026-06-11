#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
run(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/train_bbox_planner.py --config configs/cablation/$2.yaml > outputs/cabl_logs/$2.log 2>&1; echo "DONE $2"; }
mkdir -p outputs/cabl_logs
run 0 cabl_full & run 1 cabl_wo_shotemb & run 2 cabl_wo_entityemb & run 3 cabl_wo_state &
wait
run 0 cabl_wo_temporal & run 1 cabl_wo_depth &
wait
echo "ALL CABL TRAIN DONE"
