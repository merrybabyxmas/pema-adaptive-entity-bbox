"""
PEMA generation script — Phase 1 & 2.

Phase 1 (default):  GLIGEN layout + IP-Adapter identity (image-based)
Phase 2 (--projector-path):  EntityProjector enhances CLIP embeddings
                             → injected as ip_adapter_image_embeds

Usage (bootstrap, Phase 1):
  python scripts/generate_pema.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --out outputs/generations/pema_004

Usage (Phase 2 with trained projector):
  python scripts/generate_pema.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --out outputs/generations/pema_004 \
    --projector-path outputs/runs/pema_phase2/projector_best.pt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
from PIL import Image

from src.memory.entity_encoder import EntityEncoder
from src.memory.memory_bank import EntityMemoryBank
from src.model.entity_projector import EntityProjector
from src.generation.pema_pipeline import (
    generate_with_pema, bootstrap_entity_image, GlobalStyleMemory,
    GLIGEN_MODEL, SD15_MODEL, IP_ADAPTER_REPO, IP_ADAPTER_WEIGHT,
    _load_gligen, _load_ip_adapter_img2img,
)
from src.generation.layout_adapter import plan_to_layout
from src.utils.io import load_json
from src.utils.logging import get_logger

logger = get_logger("pema")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--refs-dir", default=None,
                        help="Directory with {entity_name}.png user reference images")
    parser.add_argument("--bootstrap-dir", default=None)
    parser.add_argument("--gligen-model", default=GLIGEN_MODEL)
    parser.add_argument("--sd-model", default=SD15_MODEL)
    parser.add_argument("--no-memory-update", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    parser.add_argument("--can-weight", type=float, default=0.7)
    parser.add_argument("--style-blend-alpha", type=float, default=0.45,
                        help="Weight of shot-0 style anchor in background blend (0=off, 1=full copy)")
    parser.add_argument("--gligen-steps", type=int, default=30)
    parser.add_argument("--refine-steps", type=int, default=25)
    parser.add_argument("--refine-strength", type=float, default=0.75)
    parser.add_argument("--ip-scale", type=float, default=0.85)
    parser.add_argument("--projector-path", default=None,
                        help="Path to trained EntityProjector checkpoint "
                             "(outputs/runs/pema_phase2/projector_best.pt). "
                             "If provided, enables Phase 2 enhanced embedding injection.")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    plan_output = load_json(str(base / args.layout))
    out_path = base / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    bootstrap_dir = (
        base / args.bootstrap_dir if args.bootstrap_dir
        else out_path / "bootstrap"
    )
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    # Load GLIGEN (Pass 1 layout)
    logger.info(f"Loading GLIGEN: {args.gligen_model}")
    gligen_pipe = _load_gligen(args.gligen_model, device)

    # Load IP-Adapter img2img (Pass 2 identity)
    logger.info(f"Loading IP-Adapter on {args.sd_model}")
    ip_pipe = _load_ip_adapter_img2img(
        args.sd_model, device,
        ip_scale=args.ip_scale,
    )

    # Entity encoder + memory bank
    encoder = EntityEncoder(device=device)
    memory = EntityMemoryBank(
        encoder,
        alpha=args.alpha,
        quality_threshold=args.quality_threshold,
        can_weight=args.can_weight,
        rec_weight=1.0 - args.can_weight,
    )

    # Collect all entities in plan
    layouts = plan_to_layout(plan_output)
    all_entities = sorted({e["name"] for l in layouts for e in l["entities"]})
    logger.info(f"Entities: {all_entities}")

    # Initialize entity memories
    refs_dir = base / args.refs_dir if args.refs_dir else None
    for entity in all_entities:
        ref_path = refs_dir / f"{entity}.png" if refs_dir else None
        if ref_path and ref_path.exists():
            ref_img = Image.open(str(ref_path)).convert("RGB")
            logger.info(f"  [{entity}] init from user ref")
        else:
            ref_img = bootstrap_entity_image(entity, gligen_pipe, bootstrap_dir,
                                             steps=args.gligen_steps + 10)
        memory.initialize(entity, ref_img)

    # Phase 2: EntityProjector (optional)
    projector = None
    if args.projector_path:
        proj_path = base / args.projector_path if not Path(args.projector_path).is_absolute() \
                    else Path(args.projector_path)
        if proj_path.exists():
            ckpt = torch.load(str(proj_path), map_location=device)
            dim = ckpt.get("dim", 1024)
            projector = EntityProjector(dim=dim).to(device)
            projector.load_state_dict(ckpt["model"])
            projector.eval()
            logger.info(f"Loaded EntityProjector from {proj_path} "
                        f"(epoch {ckpt.get('epoch','?')}, loss {ckpt.get('loss','?'):.4f})")
        else:
            logger.warning(f"Projector not found at {proj_path} — running Phase 1")

    # Global Style Memory
    style_memory = GlobalStyleMemory(blend_alpha=args.style_blend_alpha)

    phase_str = "Phase 2 (EntityProjector)" if projector else "Phase 1"
    logger.info(f"=== PEMA Generation ({phase_str}: 2-pass + GlobalStyle) ===")
    results = generate_with_pema(
        plan_output=plan_output,
        memory_bank=memory,
        output_dir=str(out_path),
        gligen_pipe=gligen_pipe,
        ip_pipe=ip_pipe,
        style_memory=style_memory,
        device=device,
        update_memory=not args.no_memory_update,
        gligen_steps=args.gligen_steps,
        refine_steps=args.refine_steps,
        refine_strength=args.refine_strength,
        ip_scale=args.ip_scale,
        projector=projector,
    )

    logger.info(f"\nDone. {len(results)} shots → {out_path}")
    for r in results:
        logger.info(
            f"  shot {r['shot_id']:02d}: active={r['active_entities']} "
            f"memory={r['memory_updates']}"
        )
        logger.info(f"    base: {r['base_path']}")
        logger.info(f"    pema: {r['path']}")


if __name__ == "__main__":
    main()
