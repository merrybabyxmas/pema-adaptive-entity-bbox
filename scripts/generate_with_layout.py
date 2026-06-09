import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from src.utils.io import load_json
from src.generation.pipeline import generate_with_layout
from src.utils.logging import get_logger

logger = get_logger("generate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="outputs/eval/user_story_001_boxes.json")
    parser.add_argument("--refs", default="examples/refs")
    parser.add_argument("--out", default="outputs/generations/user_story_001")
    parser.add_argument("--no-sd", action="store_true", help="Skip SD generation (bbox viz only)")
    parser.add_argument("--sd-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--gligen", action="store_true", help="Use GLIGEN layout-conditioned generation")
    parser.add_argument("--gligen-model", default="masterful/gligen-1-4-generation-text-box")
    parser.add_argument("--use-ref", action="store_true", help="Generate/use per-entity reference images for identity consistency")
    parser.add_argument("--ref-dir", default=None, help="Directory to cache entity reference images")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    plan_output = load_json(str(base / args.layout))
    out_dir = str(base / args.out)

    images = generate_with_layout(
        plan_output=plan_output,
        output_dir=out_dir,
        use_sd=not args.no_sd and not args.gligen and not args.use_ref,
        sd_model_id=args.sd_model,
        use_gligen=args.gligen or args.use_ref,
        gligen_model_id=args.gligen_model,
        use_ref=args.use_ref,
        ref_dir=str(base / args.ref_dir) if args.ref_dir else None,
    )
    logger.info(f"Generated {len(images)} shots -> {out_dir}")


if __name__ == "__main__":
    main()
