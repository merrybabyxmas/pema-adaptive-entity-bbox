"""
PEMA v4 Phase 4 generation script.

Uses EntityStyleAdapter (trained cross-attention branches) for entity identity
and style conditioning, replacing the GLIGEN-phrase injection from Phase 3.

Usage (with trained adapter):
  python scripts/generate_pema_v4.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --out outputs/generations/pema_v4 \
    --adapter-path outputs/runs/phase4_adapter/adapter_best.pt \
    --style-encoder-path outputs/runs/pema_style_encoder/style_encoder_best.pt

Usage (untrained adapter — smoke test, proves architecture works):
  python scripts/generate_pema_v4.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --out outputs/generations/pema_v4_smoke
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
from PIL import Image

from src.memory.entity_encoder import EntityEncoder
from src.memory.memory_bank import EntityMemoryBank
from src.model.style_encoder import StyleEncoder
from src.model.entity_style_adapter import EntityStyleAdapter
from src.generation.pema_v4_pipeline import generate_with_pema_v4
from src.generation.pema_v2_pipeline import _load_memory_gligen
from src.generation.pema_pipeline import bootstrap_entity_image, GLIGEN_MODEL
from src.generation.layout_adapter import plan_to_layout
from src.utils.io import load_json
from src.utils.logging import get_logger

logger = get_logger("pema_v4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout",        required=True)
    parser.add_argument("--out",           required=True)
    parser.add_argument("--refs-dir",      default=None)
    parser.add_argument("--bootstrap-dir", default=None)
    parser.add_argument("--gligen-model",  default=GLIGEN_MODEL)
    parser.add_argument("--gligen-steps",  type=int, default=30)
    parser.add_argument("--guidance-scale",type=float, default=7.5)

    # Phase 4 adapter
    parser.add_argument("--adapter-path",  default=None,
                        help="Phase 4 adapter checkpoint. "
                             "If omitted, uses randomly initialized adapter.")
    parser.add_argument("--gamma-entity",  type=float, default=0.1,
                        help="Initial γ_entity (overridden by checkpoint if loaded)")
    parser.add_argument("--gamma-style",   type=float, default=0.05)

    # Phase 2 style encoder
    parser.add_argument("--style-encoder-path", default=None)
    parser.add_argument("--n-style-tokens",      type=int, default=4)
    parser.add_argument("--style-ref",           default=None,
                        help="Dedicated style reference image. If set, its style "
                             "token is applied uniformly to ALL shots (incl. shot 0), "
                             "instead of bootstrapping style from shot 0's background.")

    # Pipeline options
    parser.add_argument("--no-phase3-grounding", action="store_true",
                        help="Disable GLIGEN phrase grounding (pure Phase 4)")
    parser.add_argument("--no-memory-update",    action="store_true")
    parser.add_argument("--alpha",               type=float, default=0.3)
    args = parser.parse_args()

    base   = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load GLIGEN pipeline ──────────────────────────────────────────────────
    logger.info(f"Loading MemoryGLIGENPipeline: {args.gligen_model}")
    gligen_pipe = _load_memory_gligen(args.gligen_model, device)

    # ── EntityStyleAdapter ────────────────────────────────────────────────────
    unet = gligen_pipe.unet
    if args.adapter_path:
        apath = base / args.adapter_path
        if apath.exists():
            logger.info(f"Loading Phase 4 adapter: {apath.name}")
            adapter = EntityStyleAdapter.load(str(apath), unet, device=device)
        else:
            logger.warning(f"Adapter checkpoint not found: {apath} — random init")
            adapter = EntityStyleAdapter(
                unet, gamma_entity_init=args.gamma_entity,
                gamma_style_init=args.gamma_style,
            )
    else:
        logger.info("No adapter checkpoint — random init (smoke test)")
        adapter = EntityStyleAdapter(
            unet, gamma_entity_init=args.gamma_entity,
            gamma_style_init=args.gamma_style,
        )
    adapter.register_to_unet()
    adapter.processors.to(device)

    counts = adapter.parameter_count()
    logger.info(
        f"Adapter: {counts['total']/1e6:.2f}M params "
        f"({counts['entity_kv']/1e6:.2f}M entity, "
        f"{counts['style_kv']/1e6:.2f}M style)"
    )

    # ── Style encoder ─────────────────────────────────────────────────────────
    style_encoder = None
    if args.n_style_tokens > 0:
        spath = base / args.style_encoder_path if args.style_encoder_path else None
        if spath and spath.exists():
            ckpt = torch.load(str(spath), map_location=device, weights_only=False)
            n_tok = ckpt.get("n_tokens", args.n_style_tokens)
            style_encoder = StyleEncoder(n_tokens=n_tok).to(device)
            style_encoder.load_state_dict(ckpt["model"])
            style_encoder.eval()
            logger.info(f"StyleEncoder loaded (K_g={n_tok})")
        else:
            style_encoder = StyleEncoder(n_tokens=args.n_style_tokens).to(device)
            style_encoder.eval()
            if spath:
                logger.warning(f"StyleEncoder checkpoint not found: {spath} — random init")

    # ── Entity encoder + memory bank ──────────────────────────────────────────
    encoder = EntityEncoder(device=device)
    memory  = EntityMemoryBank(encoder, alpha=args.alpha)

    # Initialize entity memories
    plan_output = load_json(str(base / args.layout))
    out_path    = base / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    bootstrap_dir = (
        base / args.bootstrap_dir if args.bootstrap_dir
        else out_path / "bootstrap"
    )
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    layouts     = plan_to_layout(plan_output)
    all_entities = sorted({e["name"] for l in layouts for e in l["entities"]})
    logger.info(f"Entities: {all_entities}")

    # ── Dedicated global style reference (applied uniformly to ALL shots) ──────
    style_ref_tokens = None
    if args.style_ref and style_encoder is not None:
        sref = base / args.style_ref
        if sref.exists():
            with torch.no_grad():
                sclip = encoder.encode(Image.open(str(sref)).convert("RGB")).to(device)
                style_ref_tokens = style_encoder(sclip.unsqueeze(0)).squeeze(0)  # (K_g,768)
            logger.info(f"Global style reference: {sref.name} → {tuple(style_ref_tokens.shape)}")
        else:
            logger.warning(f"--style-ref not found: {sref}")

    refs_dir = base / args.refs_dir if args.refs_dir else None
    for entity in all_entities:
        ref_path = refs_dir / f"{entity}.png" if refs_dir else None
        if ref_path and ref_path.exists():
            ref_img = Image.open(str(ref_path)).convert("RGB")
        else:
            ref_img = bootstrap_entity_image(
                entity, gligen_pipe, bootstrap_dir,
                steps=args.gligen_steps + 10,
            )
        memory.initialize(entity, ref_img)

    # ── Generate ──────────────────────────────────────────────────────────────
    logger.info("=== PEMA v4 Phase 4 Generation ===")
    results = generate_with_pema_v4(
        plan_output=plan_output,
        memory_bank=memory,
        output_dir=str(out_path),
        adapter=adapter,
        gligen_pipe=gligen_pipe,
        style_encoder=style_encoder,
        entity_encoder=encoder if style_encoder is not None else None,
        device=device,
        update_memory=not args.no_memory_update,
        gligen_steps=args.gligen_steps,
        guidance_scale=args.guidance_scale,
        n_style_tokens=args.n_style_tokens,
        use_phase3_grounding=not args.no_phase3_grounding,
        style_ref_tokens=style_ref_tokens,
    )

    logger.info(f"\nDone. {len(results)} shots → {out_path}")
    for r in results:
        logger.info(
            f"  shot {r['shot_id']:02d}: active={r['active_entities']} "
            f"memory={r['memory_updates']}"
        )
        logger.info(f"    {r['path']}")


if __name__ == "__main__":
    main()
