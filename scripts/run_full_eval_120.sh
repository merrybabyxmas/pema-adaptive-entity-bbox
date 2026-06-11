#!/usr/bin/env bash
# Full controlled evaluation on the AAAI-120 set. Renderer is FIXED; only layout source varies.
# Assumes layout-source generations already exist under outputs/lisa/aaai_ablation/<job>/
#   jobs: B_template B_retrieval B_llm B_center FINAL_combo(ours)   (generate via run_ablation_gen.py)
# Canonical script name mapping:
#   eval_planner_layout.py     = eval_ablation.py   (Table 1, VidOR planner)
#   eval_generation_layout.py  = eval_layout.py     (mIoU/SR/CLIP-T)
#   eval_generation_state.py   (ESA/TA/Missing/Leakage, per-group A-H)
#   eval_keyframes.py          (presence/dup/count = detection success / co-present recall / fusion)
#   eval_multishot.py          (identity_consistency = DINO-FG, background_consistency)
#   eval_image_quality.py      (aesthetic / CLIP-T / sharpness guards)
#   eval_occlusion_vlm.py      (VLM occlusion accuracy, group D)
#   aggregate_eval_tables.py   (Tables 1-4 + per-group + bootstrap + figures)
set -e
cd "$(dirname "$0")/.."
PY=/home/dongwoo39/.venv/bin/python
R=outputs/lisa/aaai_ablation
JOBS="B_template B_retrieval B_llm B_center FINAL_combo"
M=outputs/eval_120/metrics; mkdir -p $M

echo "[1/6] planner-level (VidOR test)"; CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_ablation.py --out outputs/abl_logs/table_clean.md || true
echo "[2/6] generation layout (mIoU/SR/CLIP-T)"; for j in $JOBS; do CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_layout.py --jobs $j --root $R --out $M/lay_$j.json; done
echo "[3/6] state (ESA/TA/miss/leak)";          for j in $JOBS; do CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_generation_state.py --jobs $j --root $R --out $M/state_$j.json; done
echo "[4/6] detection (presence/dup/fusion)";   for j in $JOBS; do CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_keyframes.py --jobs $j --root $R --out $M/det_$j.json; done
echo "[5/6] image quality + occlusion VLM";     for j in $JOBS; do CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_image_quality.py --jobs $j --root $R --out $M/quality_$j.csv; done
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_occlusion_vlm.py --jobs "$(echo $JOBS | tr ' ' ',')" --root $R --out $M/occlusion.json || true
echo "[6/6] aggregate tables + figures";        $PY scripts/aggregate_eval_tables.py
echo "DONE -> outputs/eval_120/tables, outputs/eval_120/figures"
