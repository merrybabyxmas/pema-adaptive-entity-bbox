# Presence-Aware Adaptive Entity BBox Planner → Multi-Shot Image Generation

From a multi-shot story prompt, infer **which entities appear in each shot** (presence
matrix) and **where** (per-shot per-entity bounding boxes) with a small learned layout
transformer, then render coherent multi-shot images where the same entity stays
consistent across shots, each entity is localized to its box, and entities neither fuse
nor go missing.

```
multi-shot prompt
   → presence matrix + entry/exit/stay/re-entry states
   → Presence-Aware BBox Planner (layout transformer)   [predicts cxcywh per shot/entity]
   → deoverlap + learned-size gap separation
   → LISA-image generator (SDXL + regional IP-Adapter)  [identity + localization]
   → per-shot images (+ bbox overlay, + bbox debug viz)
```

The research contribution is the **planner** (predict entity boxes from narrative
context), not a new generator. Generation reuses SDXL + IP-Adapter regional masking.

---

## Pipeline (the only path — everything else was removed)

### 1. Data — `scripts/build_dataset_all.sh`
VidOR (+ VidSTG) annotations → per-shot samples with presence, states, relations, and
GT boxes. Produces `data/splits/{train,val,test}.jsonl`.
```bash
bash scripts/build_dataset_all.sh --real-data   # needs data/vidor_annotations (download_vidor.sh)
```
Steps: `01_prepare_vidor_vidstg.py` → `07_filter_dataset.py` → `08_split_dataset.py`
(`00_generate_synthetic.py` is a synthetic fallback when VidOR is not present).

### 2. Train the planner — `scripts/train_bbox_planner.py`
Small transformer (`src/model/bbox_planner.py`): conditions on CLIP ViT-B/32 text
embeddings of (shot prompt, entity name) + presence + states + relations → `[S,E,4]`
cxcywh boxes. Loss = masked L1 + GIoU + overlap + (temporal, off by default), all masked
by presence.
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_bbox_planner.py --config configs/train_bbox_planner.yaml
```
Current checkpoint: `outputs/runs/bbox_planner_v3/checkpoints/best.pt`.

### 3. Generate multi-shot images — `scripts/run_30_stories.py`  ← **main entry point**
Planner → boxes → `bbox_debug.png` (predicted boxes per shot) → `deoverlap_boxes` +
`enforce_gap` → LISA-image (SDXL native `ip_adapter_masks`) → shots + bbox overlay.
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_30_stories.py \
  --planner outputs/runs/bbox_planner_v3/checkpoints/best.pt \
  --stories examples/stories30_invocab.json \
  --out outputs/lisa/stories30_sizeaware
```

### 4. Evaluate the planner — `scripts/eval_bbox_planner.py`
L1 / IoU / GIoU / center / area error + overlap on the val/test split.

### 5. (extension) Multi-shot video — `scripts/run_video_i2v.py`
Core stays ours (planner → per-shot keyframes). Video synthesis is the **official**
`THUDM/CogVideoX-5b-I2V` (diffusers): each keyframe is animated into a clip and the
shots are concatenated into one multi-shot video. Nothing about the generator is
custom — only the planner is.

## Occlusion depth (planner predicts who is in front)
The planner also predicts a per-entity **occlusion depth** (5th head channel, 0=back,
1=front), supervised by a pairwise ranking loss against order derived from VidOR
`in_front_of`/`behind` relations (96.9% of co-present pairs have an edge; geometric
bottom-edge/area prior fills the rest). `01_prepare_vidor_vidstg.py` extracts it,
`train_bbox_planner.py` adds `depth_ranking_loss` (val depth-order accuracy ≈ 0.77).
At generation, `run_30_stories.enforce` arranges co-present boxes side-by-side with a
partial overlap seam and `mask_utils.apply_depth_occlusion` lets the **front** entity
own the shared region — natural occlusion instead of fusion. Checkpoint:
`outputs/runs/bbox_planner_v4_depth/`.

---

## Key design decisions

- **GPU 0 only.** All generation/training pins `cuda:0`.
- **Identity, not just style.** Per-entity identity comes from SDXL **regional
  IP-Adapter** (`ip_adapter_masks`, native diffusers): each entity's white-background
  anchor is injected only inside its (soft Gaussian) box region; self-attention stays
  global. Complexity-adaptive: 1 entity → σ20/scale0.6, 2 → σ12/0.7, 3+ → σ8/0.8
  (`adaptive(n)` in `run_30_stories.py`).
- **Anti-fusion via learned-size gap, no forced layout.** `enforce_gap` separates boxes
  with a background strip so large same-body-plan entities (quadrupeds, vehicles) can't
  bridge into one fused body — but it **preserves the planner's predicted box widths**
  (proportional, only scaled down on overflow) and vertical extent. It does **not**
  equalize slots. "Go by what's learned." Fusion is further suppressed by a strong
  negative prompt; no added loss, no hard box constraint.
- **The planner does predict size.** On in-distribution data its predicted box area
  spans 0.03–0.84 and correlates with GT (r≈0.57); relative ordering is learned
  (elephant > horse > car, sofa > cup). Note VidOR box size encodes *camera framing* as
  much as intrinsic object size, so absolute sizes are framing-influenced.

---

## Layout

```
scripts/
  build_dataset_all.sh  download_vidor.sh
  00_generate_synthetic.py  01_prepare_vidor_vidstg.py  07_filter_dataset.py  08_split_dataset.py
  train_bbox_planner.py     eval_bbox_planner.py        run_30_stories.py   ← main
src/
  model/        bbox_planner.py  attention.py  heads.py  embeddings.py  losses.py
  data/         dataset.py  collate.py  schema.py  quality.py
  lm_planner/   validator.py            (presence matrix + state computation)
  utils/        box_ops.py (incl. deoverlap_boxes)  io.py  seed.py  logging.py
  generation/lisa/  lisa_pipeline.py  build_anchors.py  mask_utils.py  attention_injection.py
configs/        lisa_default.yaml  train_bbox_planner.yaml  data_build.yaml  eval_bbox.yaml
examples/       stories30_invocab.json (VidOR-vocab)  stories30.json
outputs/
  runs/bbox_planner_v3/           trained planner
  lisa/stories30_sizeaware/       current 30-story result (per story: bbox_debug, anchors, shots)
```

## Generator backend note
GLIGEN (box-filling grounding) was evaluated to force objects to *fill* their boxes, but
it is SD1.x-only and the SDXL→SD1.5 quality drop was not worth it. The kept path is
SDXL + regional IP-Adapter; box size flows from the planner through the size-preserving
`enforce_gap`.
