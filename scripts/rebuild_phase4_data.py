"""
Phase 4 training data rebuild: cross-image entity reference pairing.

Problem with original data:
  ref_image = entity crop from the SAME target scene
  → trivial solution: adapter just copies appearance from self

Fix:
  ref_image = pema_train image of the SAME entity (different scene/view)
  → adapter must generalize entity appearance across scenes

Strategy:
  1. GLIGEN-generated 2-entity scenes (80):
       For each entity E in scene, ref = random pema_train image of E
       (different from scene itself, gives proper multi-entity bbox training)

  2. pema_train single-entity pairs (300 → 1200):
       For each of 60 entities × 5 images:
         target = img_k,  ref = random other img_j (j ≠ k) of same entity
       Expand each existing sample to all 5 target images × 1 ref each
       → 60 × 5 = 300 pairs (keep same count, but refs are now cross-image)
       OR expand to full combinatorial: 60 × 5 × 4 = 1200 pairs (more diversity)

Default: full expansion = 1200 single-entity + 80 gligen = 1280 total samples

Usage:
  python scripts/rebuild_phase4_data.py --out data/phase4_train
  python scripts/rebuild_phase4_data.py --out data/phase4_train --single-pairs 1  # minimal (300)
"""
import sys, os, json, random, re, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger("rebuild_phase4")
slug = lambda n: re.sub(r"[^a-z0-9_]", "_", n.lower().strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",          default="data/phase4_train")
    parser.add_argument("--pema-dir",     default="data/pema_train")
    parser.add_argument("--single-pairs", type=int, default=4,
                        help="Number of ref images to pair with each target "
                             "(1=minimal, 4=full combinatorial, max=n_imgs-1)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    base       = Path(__file__).parent.parent
    out_dir    = base / args.out
    pema_dir   = base / args.pema_dir / "images"
    samples_dir = out_dir / "samples"

    # Pre-load pema_train image paths per entity
    entity_list = (base / args.pema_dir / "entity_list.txt").read_text().splitlines()
    pema_imgs: dict[str, list[Path]] = {}
    for ename in entity_list:
        paths = sorted((pema_dir / slug(ename)).glob("*.png"))
        if paths:
            pema_imgs[ename] = paths
    logger.info(f"pema_train: {len(pema_imgs)} entities, "
                f"{sum(len(v) for v in pema_imgs.values())} images")

    index_path = out_dir / "scene_index.json"
    old_index  = json.loads(index_path.read_text())

    new_index  = []
    sample_id  = 0

    # ── Pass 1: GLIGEN-generated scenes (keep scene, update refs) ─────────────
    gligen_count = 0
    for rel_dir in old_index:
        sdir = out_dir / rel_dir
        meta = json.loads((sdir / "metadata.json").read_text())

        if meta.get("source") == "pema_train":
            continue  # handled in pass 2

        # For each entity, point ref_image to a pema_train image
        updated = False
        for e in meta["entities"]:
            ename = e["name"]
            if ename not in pema_imgs:
                continue  # keep old ref if no pema_train match
            ref = random.choice(pema_imgs[ename])
            # Store as absolute path so Phase4Dataset can find it unambiguously
            e["ref_image"] = str(ref)
            e["ref_cross"]  = True
            updated = True

        if updated:
            meta["sample_id"] = sample_id
            (sdir / "metadata.json").write_text(json.dumps(meta, indent=2))
            new_index.append(rel_dir)
            sample_id += 1
            gligen_count += 1

    logger.info(f"Updated {gligen_count} GLIGEN scenes with cross-image refs")

    # ── Pass 2: pema_train pairs — expand to cross-image target/ref pairs ─────
    # Remove old pema_train-sourced sample dirs (we'll recreate them)
    old_pema_dirs = [d for d in old_index
                     if json.loads((out_dir / d / "metadata.json").read_text()).get("source") == "pema_train"]
    logger.info(f"Removing {len(old_pema_dirs)} old pema_train sample dirs...")
    for rel_dir in old_pema_dirs:
        shutil.rmtree(str(out_dir / rel_dir), ignore_errors=True)

    # Rebuild with proper cross-image pairs
    pema_count = 0
    for ename, paths in pema_imgs.items():
        n = len(paths)
        for t_idx, target_path in enumerate(paths):
            # ref candidates: all OTHER images of same entity
            other = [p for i, p in enumerate(paths) if i != t_idx]
            # Sample `single_pairs` refs per target
            n_pairs = min(args.single_pairs, len(other))
            refs = random.sample(other, n_pairs) if len(other) >= n_pairs else other

            for ref_path in refs:
                sdir = samples_dir / f"sample_{sample_id:05d}"
                sdir.mkdir(parents=True, exist_ok=True)

                # Copy target image as scene.png
                shutil.copy(str(target_path), str(sdir / "scene.png"))

                # Copy ref image
                shutil.copy(str(ref_path), str(sdir / "entity_ref.png"))

                # Background = lightly blurred version of target (entity fills frame)
                scene = Image.open(str(target_path)).convert("RGB")
                from PIL import ImageFilter
                bg = scene.filter(ImageFilter.GaussianBlur(radius=30))
                bg.save(str(sdir / "style_bg.png"))

                meta = {
                    "sample_id": sample_id,
                    "prompt":    f"a {ename}",
                    "entities": [{
                        "name":      ename,
                        "box_xyxy":  [0.1, 0.1, 0.9, 0.9],
                        "ref_image": str(ref_path),  # absolute path to pema_train ref
                        "ref_cross": True,
                    }],
                    "n_entities": 1,
                    "source":    "pema_train_cross",
                    "target_img": str(target_path),
                    "ref_img":    str(ref_path),
                }
                (sdir / "metadata.json").write_text(json.dumps(meta, indent=2))

                rel = str(sdir.relative_to(out_dir))
                new_index.append(rel)
                sample_id += 1
                pema_count += 1

    logger.info(f"Created {pema_count} cross-image pema_train pairs")

    # ── Write new index ────────────────────────────────────────────────────────
    index_path.write_text(json.dumps(new_index, indent=2))
    logger.info(f"Total samples: {len(new_index)} "
                f"({gligen_count} GLIGEN + {pema_count} pema_train cross-pairs)")
    logger.info(f"New index → {index_path}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    cross_count = 0
    for rel_dir in new_index[:10]:
        meta = json.loads((out_dir / rel_dir / "metadata.json").read_text())
        for e in meta["entities"]:
            if e.get("ref_cross"):
                ref = Path(e["ref_image"])
                scene = out_dir / rel_dir / "scene.png"
                assert ref.exists(), f"ref not found: {ref}"
                assert ref.resolve() != scene.resolve(), "ref == target (trivial!)"
                cross_count += 1
    logger.info(f"Sanity check OK: {cross_count}/checked pairs are cross-image")


if __name__ == "__main__":
    main()
