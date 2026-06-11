#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
run(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/run_ablation_gen.py --mode render --job $2 \
       > outputs/abl_logs/gen_$2.log 2>&1; echo "DONE $2"; }
# GPU0
( run 0 A_full;        run 0 B_template;  run 0 C_depthown ) &
# GPU1
( run 1 A_wo_state;    run 1 B_retrieval; run 1 C_nodepth  ) &
# GPU2
( run 2 A_wo_relation; run 2 B_llm;       run 2 C_gaponly  ) &
# GPU3
( run 3 A_wo_entityint; run 3 A_wo_depth ) &
wait
echo "ALL RENDER JOBS DONE"
