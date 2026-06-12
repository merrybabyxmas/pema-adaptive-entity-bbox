#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
# OPENAI_API_KEY must be set in the environment by the caller (used only for LLM-direct layout)
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in env}"
S=data/captions/stories_aaai_eval_300_6patterns_no_reverse.json
AC=outputs/lisa/_aaai300_anchors
R=outputs/lisa/aaai300
LLM=outputs/layouts/llm_300.json
ED=outputs/eval_300
M=$ED/metrics
L=outputs/e300_logs
JOBS="B_template B_retrieval B_llm B_center CABL_full CABL_wo_shotemb CABL_wo_entityemb CABL_wo_state CABL_wo_temporal CABL_wo_depth"
sh(){ cat /tmp/e300_shard_$1.txt; }

echo "[A] LLM-300 layouts"; $PY scripts/gen_llm_layout.py --stories $S --out $LLM > $L/llm.log 2>&1 || true

echo "[B] anchors (4-GPU)"
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $PY scripts/run_ablation_gen.py --mode anchors --stories $S --anchor-cache $AC --shard $i/4 > $L/anch_$i.log 2>&1 & done; wait
echo "[B] anchors done: $(ls $AC | wc -l) stories"

echo "[C] generate 10 jobs x 300 (story-sharded 4-GPU)"
for J in $JOBS; do
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PY scripts/run_ablation_gen.py --mode render --job $J --stories $S \
      --anchor-cache $AC --out-root $R --llm-layout $LLM --only "$(sh $i)" > $L/gen_${J}_$i.log 2>&1 &
  done; wait
  echo "[C] DONE $J ($(find $R/$J -name shot_000.png 2>/dev/null | wc -l)/300)"
done

echo "[D] evaluations (per job, 4-GPU round-robin)"
run_eval(){ # $1=script $2=out-prefix $3=extra
  local g=0
  for J in $JOBS; do
    CUDA_VISIBLE_DEVICES=$g $PY $1 --jobs $J --root $R --stories $S --out $M/$2_$J.json $3 > $L/$2_$J.log 2>&1 &
    g=$(( (g+1)%4 )); [ $g -eq 0 ] && wait
  done; wait
}
g=0; for J in $JOBS; do CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_generation_state.py --jobs $J --root $R --stories $S --raw $ED/detections --out $M/state_$J.json > $L/state_$J.log 2>&1 & g=$(( (g+1)%4 )); [ $g -eq 0 ] && wait; done; wait
run_eval scripts/eval_layout.py lay "--llm-layout $LLM"
run_eval scripts/eval_keyframes.py det ""
# image quality writes csv (different out flag)
g=0; for J in $JOBS; do CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_image_quality.py --jobs $J --root $R --stories $S --out $M/quality_$J.csv > $L/q_$J.log 2>&1 & g=$(( (g+1)%4 )); [ $g -eq 0 ] && wait; done; wait

echo "[E] aggregate tables + figures + ablation 5-metric"
$PY scripts/aggregate_eval_tables.py --evaldir $ED --ours CABL_full > $L/aggregate.log 2>&1 || true
$PY scripts/make_ablation_table_300.py > $L/ablation.log 2>&1 || true

echo "[F] dashboards (4-page montage per job)"
$PY scripts/make_dashboards_300.py > $L/dash.log 2>&1 || true
echo "ALL EVAL300 DONE"
