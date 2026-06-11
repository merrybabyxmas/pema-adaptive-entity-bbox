#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
until grep -q "ALL CABL GEN DONE" outputs/cabl_logs/_gen_master.log 2>/dev/null; do sleep 30; done
echo "[ms] generation done, running multishot consistency eval"
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_multishot.py --jobs CABL_full,CABL_wo_temporal --out outputs/cabl_logs/ms_0.json > outputs/cabl_logs/ms_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_multishot.py --jobs CABL_wo_shotemb,CABL_wo_depth --out outputs/cabl_logs/ms_1.json > outputs/cabl_logs/ms_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY scripts/eval_multishot.py --jobs CABL_wo_entityemb --out outputs/cabl_logs/ms_2.json > outputs/cabl_logs/ms_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY scripts/eval_multishot.py --jobs CABL_wo_state --out outputs/cabl_logs/ms_3.json > outputs/cabl_logs/ms_3.log 2>&1 &
wait
echo "ALL MULTISHOT EVAL DONE"
