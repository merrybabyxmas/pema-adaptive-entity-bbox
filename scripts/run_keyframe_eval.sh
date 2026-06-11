#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_keyframes.py --jobs A_full,A_wo_state,A_wo_relation     --out outputs/abl_logs/det_0.json > outputs/abl_logs/det_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_keyframes.py --jobs A_wo_entityint,A_wo_depth,B_template --out outputs/abl_logs/det_1.json > outputs/abl_logs/det_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY scripts/eval_keyframes.py --jobs B_retrieval,B_llm,C_nodepth          --out outputs/abl_logs/det_2.json > outputs/abl_logs/det_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY scripts/eval_keyframes.py --jobs C_gaponly,C_depthown                 --out outputs/abl_logs/det_3.json > outputs/abl_logs/det_3.log 2>&1 &
wait
echo "ALL DET DONE"
