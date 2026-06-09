"""
Phase 4 training data generation.

Generates synthetic (scene, entity_refs, style_ref, bbox, prompt) tuples
using GLIGEN.  For each sample:

  1. Sample 2 entities from entity list
  2. Sample a random non-overlapping layout (from bbox templates)
  3. Generate scene with GLIGEN
  4. Crop entity regions → entity reference images
  5. Extract background (entity regions masked) → style reference
  6. Save metadata + images to data/phase4_train/

Output structure:
  data/phase4_train/
    samples/
      sample_000/
        scene.png
        style_bg.png
        entity_<name>.png      (reference crop per entity)
        metadata.json
    scene_index.json           (list of all sample dirs)

Usage:
  python scripts/generate_phase4_data.py --n-samples 60 --out data/phase4_train
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import torch
from PIL import Image, ImageFilter, ImageDraw

from src.generation.pema_pipeline import GLIGEN_MODEL
from src.utils.logging import get_logger

logger = get_logger("gen_phase4_data")


# ── Fixed bbox layout templates ───────────────────────────────────────────────
# Each template is a list of [x1,y1,x2,y2] boxes (normalized [0,1]) for 2 entities.
# Non-overlapping, varied positions and sizes.
BBOX_TEMPLATES_2 = [
    [[0.05, 0.1,  0.50, 0.9 ], [0.52, 0.1,  0.95, 0.9 ]],  # left/right split
    [[0.1,  0.05, 0.45, 0.95], [0.55, 0.05, 0.90, 0.95]],  # wide L/R
    [[0.05, 0.05, 0.48, 0.90], [0.52, 0.10, 0.95, 0.90]],  # left heavy
    [[0.1,  0.1,  0.90, 0.48], [0.1,  0.52, 0.90, 0.90]],  # top/bottom split
    [[0.05, 0.05, 0.45, 0.45], [0.55, 0.55, 0.95, 0.95]],  # diagonal
    [[0.05, 0.55, 0.45, 0.95], [0.55, 0.05, 0.95, 0.45]],  # anti-diagonal
    [[0.1,  0.1,  0.55, 0.88], [0.60, 0.15, 0.95, 0.85]],  # slight size diff
    [[0.05, 0.1,  0.42, 0.88], [0.48, 0.12, 0.95, 0.86]],  # wide subjects
    [[0.15, 0.05, 0.50, 0.95], [0.52, 0.05, 0.85, 0.95]],  # center balanced
    [[0.05, 0.05, 0.94, 0.42], [0.05, 0.58, 0.94, 0.95]],  # horizontal strips
]

BBOX_TEMPLATES_1 = [
    [[0.1,  0.05, 0.90, 0.95]],  # full frame
    [[0.15, 0.10, 0.85, 0.90]],  # centered
    [[0.05, 0.05, 0.55, 0.95]],  # left-anchored
    [[0.45, 0.05, 0.95, 0.95]],  # right-anchored
]

SCENE_PROMPTS = [
    "a {e1} and a {e2} in a bright living room",
    "a {e1} and a {e2} in a garden with flowers",
    "a {e1} playing next to a {e2} on a beach",
    "a {e1} and a {e2} in a cozy kitchen setting",
    "a {e1} and a {e2} resting in a park",
    "a {e1} looking at a {e2} in a sunny backyard",
    "a photo of a {e1} and a {e2} indoors",
    "a {e1} and a {e2} sitting together outside",
    "a {e1} near a {e2} in a minimalist room",
    "a {e1} and a {e2} on a wooden floor",
]

SINGLE_PROMPTS = [
    "a {e1} in a cozy living room",
    "a {e1} outdoors in sunlight",
    "a {e1} on a wooden floor",
    "a {e1} in a garden",
    "a {e1} resting indoors",
]


def crop_bbox(image: Image.Image, box: list[float]) -> Image.Image:
    """Crop image at normalized bbox [x1,y1,x2,y2]."""
    W, H = image.size
    x1 = int(box[0] * W); y1 = int(box[1] * H)
    x2 = int(box[2] * W); y2 = int(box[3] * H)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    return image.crop((x1, y1, x2, y2))


def extract_background(image: Image.Image, bboxes: list[list[float]],
                        blur_radius: int = 40) -> Image.Image:
    """Blur entity regions to isolate background style."""
    W, H = image.size
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for box in bboxes:
        pad = int(0.05 * min(W, H))
        x1 = max(0, int(box[0] * W) - pad); y1 = max(0, int(box[1] * H) - pad)
        x2 = min(W, int(box[2] * W) + pad); y2 = min(H, int(box[3] * H) + pad)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
    return Image.composite(blurred, image, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples",   type=int,   default=60)
    parser.add_argument("--n-single",    type=int,   default=20,
                        help="Additional single-entity samples (better bbox coverage)")
    parser.add_argument("--out",         default="data/phase4_train")
    parser.add_argument("--gligen-model",default=GLIGEN_MODEL)
    parser.add_argument("--steps",       type=int,   default=25)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--entity-list", default="data/pema_train/entity_list.txt")
    args = parser.parse_args()

    random.seed(args.seed)
    base    = Path(__file__).parent.parent
    out_dir = base / args.out
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    entity_list = (base / args.entity_list).read_text().splitlines()
    logger.info(f"Entities available: {len(entity_list)}")

    # ── Load GLIGEN pipeline ─────────────────────────────────────────────────
    logger.info(f"Loading GLIGEN pipeline: {args.gligen_model}")
    from diffusers import StableDiffusionGLIGENPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        args.gligen_model, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    sample_idx = 0
    index = []

    # ── Two-entity samples ───────────────────────────────────────────────────
    logger.info(f"Generating {args.n_samples} two-entity samples...")
    for _ in range(args.n_samples):
        e1, e2   = random.sample(entity_list, 2)
        template = random.choice(BBOX_TEMPLATES_2)
        boxes    = template  # [[x1,y1,x2,y2], [x1,y1,x2,y2]]
        prompt   = random.choice(SCENE_PROMPTS).format(e1=e1, e2=e2)

        gen = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, cartoon, text, watermark",
            gligen_phrases=[e1, e2],
            gligen_boxes=boxes,
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=args.steps,
            height=512, width=512,
            generator=torch.manual_seed(sample_idx),
        )
        scene = gen.images[0]

        sdir = samples_dir / f"sample_{sample_idx:04d}"
        sdir.mkdir(exist_ok=True)

        scene.save(str(sdir / "scene.png"))
        bg = extract_background(scene, boxes)
        bg.save(str(sdir / "style_bg.png"))

        import re
        def safe(n): return re.sub(r"[^a-z0-9_]", "_", n.lower().strip())
        entity_info = []
        for name, box in [(e1, boxes[0]), (e2, boxes[1])]:
            crop = crop_bbox(scene, box)
            fname = f"entity_{safe(name)}.png"
            crop.save(str(sdir / fname))
            entity_info.append({"name": name, "box_xyxy": box, "ref_image": fname})

        meta = {
            "sample_id": sample_idx,
            "prompt": prompt,
            "entities": entity_info,
            "n_entities": 2,
        }
        (sdir / "metadata.json").write_text(json.dumps(meta, indent=2))
        index.append(str(sdir.relative_to(out_dir)))
        sample_idx += 1

        if sample_idx % 10 == 0:
            logger.info(f"  {sample_idx} samples done")

    # ── Single-entity samples (larger bbox coverage) ─────────────────────────
    logger.info(f"Generating {args.n_single} single-entity samples...")
    for _ in range(args.n_single):
        e1      = random.choice(entity_list)
        template = random.choice(BBOX_TEMPLATES_1)
        boxes   = template
        prompt  = random.choice(SINGLE_PROMPTS).format(e1=e1)

        gen = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, cartoon, text, watermark",
            gligen_phrases=[e1],
            gligen_boxes=boxes,
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=args.steps,
            height=512, width=512,
            generator=torch.manual_seed(sample_idx),
        )
        scene = gen.images[0]

        sdir = samples_dir / f"sample_{sample_idx:04d}"
        sdir.mkdir(exist_ok=True)

        scene.save(str(sdir / "scene.png"))
        bg = extract_background(scene, boxes)
        bg.save(str(sdir / "style_bg.png"))

        import re
        def safe(n): return re.sub(r"[^a-z0-9_]", "_", n.lower().strip())
        fname1 = f"entity_{safe(e1)}.png"
        crop = crop_bbox(scene, boxes[0])
        crop.save(str(sdir / fname1))

        meta = {
            "sample_id": sample_idx,
            "prompt": prompt,
            "entities": [{"name": e1, "box_xyxy": boxes[0], "ref_image": fname1}],
            "n_entities": 1,
        }
        (sdir / "metadata.json").write_text(json.dumps(meta, indent=2))
        index.append(str(sdir.relative_to(out_dir)))
        sample_idx += 1

    # ── Also include pema_train entity images as single-entity samples ───────
    pema_dir = base / "data/pema_train/images"
    if pema_dir.exists():
        import re
        slug = lambda n: re.sub(r"[^a-z0-9_]", "_", n.lower().strip())
        logger.info("Adding pema_train entity images as single-entity samples...")
        added = 0
        for ename in entity_list:
            edir = pema_dir / slug(ename)
            for img_path in sorted(edir.glob("*.png"))[:5]:
                sdir = samples_dir / f"sample_{sample_idx:04d}"
                sdir.mkdir(exist_ok=True)

                scene = Image.open(str(img_path)).convert("RGB")
                scene.save(str(sdir / "scene.png"))
                bg = extract_background(scene, [[0.1, 0.1, 0.9, 0.9]])
                bg.save(str(sdir / "style_bg.png"))
                scene.save(str(sdir / f"entity_{slug(ename)}.png"))

                meta = {
                    "sample_id": sample_idx,
                    "prompt": f"a {ename}",
                    "entities": [{"name": ename,
                                   "box_xyxy": [0.1, 0.1, 0.9, 0.9],
                                   "ref_image": f"entity_{slug(ename)}.png"}],
                    "n_entities": 1,
                    "source": "pema_train",
                }
                (sdir / "metadata.json").write_text(json.dumps(meta, indent=2))
                index.append(str(sdir.relative_to(out_dir)))
                sample_idx += 1
                added += 1

        logger.info(f"  Added {added} pema_train samples")

    # ── Write index ───────────────────────────────────────────────────────────
    (out_dir / "scene_index.json").write_text(json.dumps(index, indent=2))
    logger.info(f"Done. {sample_idx} total samples → {out_dir}")


if __name__ == "__main__":
    main()
