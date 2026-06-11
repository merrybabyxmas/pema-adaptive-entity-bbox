#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
# wait for training to finish
until grep -q "ALL CABL TRAIN DONE" outputs/cabl_logs/_master.log 2>/dev/null; do sleep 30; done
echo "[gen] training done, starting generation"
for job in CABL_full CABL_wo_shotemb CABL_wo_entityemb CABL_wo_state CABL_wo_temporal CABL_wo_depth; do
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PY scripts/run_ablation_gen.py --mode render --job $job \
      --only "$(cat /tmp/cabl_shard_$i.txt)" --out-root outputs/lisa/aaai_cablation \
      > outputs/cabl_logs/gen_${job}_$i.log 2>&1 &
  done
  wait
  echo "[gen] DONE $job ($(find outputs/lisa/aaai_cablation/$job -name shot_000.png 2>/dev/null | wc -l) stories)"
done
echo "ALL CABL GEN DONE"
