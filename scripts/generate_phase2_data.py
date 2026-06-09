"""
Generate synthetic entity variation images for Phase 2 training.

For each entity type sampled from the dataset:
  - Generate N_VAR images with different backgrounds / poses / angles
  - These form (anchor, positive) pairs for the same entity
  - Pairs from different entities form (anchor, negative) pairs

Output:
  data/pema_train/images/{entity_slug}/{0..N_VAR-1}.png
  data/pema_train/entity_list.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, re
from pathlib import Path
import torch
from diffusers import StableDiffusionGLIGENPipeline
from src.utils.logging import get_logger

logger = get_logger("phase2_data")

VARIATION_TEMPLATES = [
    "A single {entity}, full body, plain white background, photorealistic",
    "A {entity} sitting on grass in a sunny park, photorealistic",
    "A {entity} standing, looking at camera, natural lighting",
    "A close-up portrait of a {entity}, detailed fur/features, sharp focus",
    "A {entity} in motion, outdoors, natural environment",
]

NEG_PROMPT = (
    "watermark, text, logo, blurry, low quality, cartoon, "
    "multiple animals, humans, duplicate"
)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-jsonl", default="data/splits/train.jsonl")
    parser.add_argument("--out-dir", default="data/pema_train")
    parser.add_argument("--n-entities", type=int, default=60,
                        help="Number of entity types to generate for")
    parser.add_argument("--n-var", type=int, default=5,
                        help="Variations per entity")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    out_dir = base / args.out_dir
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Sample entity types from the dataset
    entity_counts: dict[str, int] = {}
    with open(base / args.data_jsonl) as f:
        for line in f:
            d = json.loads(line)
            for e in d.get("entity_vocab", []):
                entity_counts[e] = entity_counts.get(e, 0) + 1

    # Pick top-N most frequent concrete entities (skip vague ones)
    skip = {"adult", "child", "person", "man", "woman", "people", "scene", "background"}
    candidates = [
        e for e, _ in sorted(entity_counts.items(), key=lambda x: -x[1])
        if e not in skip and len(e) >= 3
    ][:args.n_entities]
    logger.info(f"Selected {len(candidates)} entity types")

    # Load GLIGEN
    logger.info("Loading GLIGEN...")
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-generation-text-box",
        torch_dtype=torch.float16
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    entity_list = []
    for entity in candidates:
        entity_dir = img_dir / slug(entity)
        entity_dir.mkdir(exist_ok=True)

        generated = 0
        for i, template in enumerate(VARIATION_TEMPLATES[:args.n_var]):
            out_path = entity_dir / f"{i}.png"
            if out_path.exists():
                generated += 1
                continue
            prompt = template.format(entity=entity)
            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEG_PROMPT,
                    gligen_phrases=[entity],
                    gligen_boxes=[[0.1, 0.1, 0.9, 0.9]],
                    gligen_scheduled_sampling_beta=1.0,
                    num_inference_steps=args.steps,
                    height=512, width=512,
                )
                result.images[0].save(str(out_path))
                generated += 1
            except Exception as e:
                logger.warning(f"  [{entity}] var {i} failed: {e}")

        if generated > 0:
            entity_list.append(entity)
            logger.info(f"  {entity}: {generated}/{args.n_var} images")

    # Save entity list
    list_path = out_dir / "entity_list.txt"
    list_path.write_text("\n".join(entity_list))
    logger.info(f"\nDone. {len(entity_list)} entities → {img_dir}")
    logger.info(f"Entity list saved to {list_path}")


if __name__ == "__main__":
    main()
