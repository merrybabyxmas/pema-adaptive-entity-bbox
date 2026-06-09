"""
PEMA v2 Phase 3 generation: text + entity memory + global style tokens.

Usage (text-only baseline):
  python scripts/generate_pema_v2.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --out outputs/generations/pema_v2_baseline

Usage (Phase 1: entity residual conditioner only):
  python scripts/generate_pema_v2.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --out outputs/generations/pema_v2_residual \
    --conditioner-path outputs/runs/pema_conditioner_residual/conditioner_best.pt

Usage (Phase 3: entity residual + global style tokens):
  python scripts/generate_pema_v2.py \
    --layout outputs/eval/user_story_001_boxes.json \
    --out outputs/generations/pema_v2_phase3 \
    --conditioner-path outputs/runs/pema_conditioner_residual/conditioner_best.pt \
    --style-encoder-path outputs/runs/pema_style_encoder/style_encoder_best.pt \
    --n-style-tokens 4 \
    --style-weight 0.2

Usage (ablation: pixel blend enabled):
  python scripts/generate_pema_v2.py ... --use-style-blend
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
from PIL import Image

from src.memory.entity_encoder import EntityEncoder
from src.memory.memory_bank import EntityMemoryBank
from src.model.entity_memory_conditioner import EntityMemoryConditioner
from src.model.residual_entity_conditioner import ResidualEntityConditioner
from src.model.style_encoder import StyleEncoder
from src.generation.pema_v2_pipeline import (
    generate_with_pema_v2, _load_memory_gligen,
)
from src.generation.pema_pipeline import (
    bootstrap_entity_image, GlobalStyleMemory,
    GLIGEN_MODEL,
)
from src.generation.layout_adapter import plan_to_layout
from src.utils.io import load_json
from src.utils.logging import get_logger

logger = get_logger("pema_v2")


def _load_conditioner(cpath: Path, ctype: str, device):
    """Load ResidualEntityConditioner or EntityMemoryConditioner from checkpoint."""
    ckpt = torch.load(str(cpath), map_location=device, weights_only=False)

    if ctype in ("residual", "auto"):
        try:
            model = ResidualEntityConditioner().to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            logger.info(
                f"Loaded ResidualEntityConditioner from {cpath.name} "
                f"(epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss',0):.4f})"
            )
            return model
        except Exception as e:
            if ctype == "residual":
                raise
            logger.warning(f"ResidualEntityConditioner load failed: {e}")

    model = EntityMemoryConditioner().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        f"Loaded EntityMemoryConditioner (text-projection ablation) "
        f"from {cpath.name} (epoch={ckpt.get('epoch','?')})"
    )
    return model


def _load_style_encoder(cpath: Path, n_tokens: int, device):
    """Load StyleEncoder from checkpoint or create randomly initialized."""
    if cpath is not None and cpath.exists():
        ckpt   = torch.load(str(cpath), map_location=device, weights_only=False)
        n_tok  = ckpt.get("n_tokens", n_tokens)
        model  = StyleEncoder(n_tokens=n_tok).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        logger.info(
            f"Loaded StyleEncoder from {cpath.name} "
            f"(epoch={ckpt.get('epoch','?')}, K_g={n_tok})"
        )
        return model
    else:
        # Phase 2 MVP: untrained encoder — CLIP embedding still captures style
        model = StyleEncoder(n_tokens=n_tokens).to(device)
        model.eval()
        if cpath is not None:
            logger.warning(f"StyleEncoder checkpoint not found: {cpath} — using random init")
        logger.info(f"StyleEncoder: random init (K_g={n_tokens})")
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout",           required=True)
    parser.add_argument("--out",              required=True)
    parser.add_argument("--refs-dir",         default=None)
    parser.add_argument("--bootstrap-dir",    default=None)
    parser.add_argument("--gligen-model",     default=GLIGEN_MODEL)
    parser.add_argument("--no-memory-update", action="store_true")
    parser.add_argument("--alpha",            type=float, default=0.3)
    parser.add_argument("--quality-threshold",type=float, default=0.5)
    parser.add_argument("--gligen-steps",     type=int,   default=30)

    # ── Phase 1: Entity residual conditioner ──────────────────────────────
    parser.add_argument("--conditioner-path", default=None,
                        help="ResidualEntityConditioner checkpoint (Phase 1). "
                             "Omit for text-only GLIGEN baseline.")
    parser.add_argument("--conditioner-type", default="auto",
                        choices=["auto", "residual", "text-projection"])
    parser.add_argument("--text-weight",      type=float, default=1.0)
    parser.add_argument("--memory-weight",    type=float, default=1.0,
                        help="λ_m: entity residual scale")

    # ── Phase 2: Style token encoder ──────────────────────────────────────
    parser.add_argument("--style-encoder-path", default=None,
                        help="StyleEncoder checkpoint. If omitted but --n-style-tokens>0, "
                             "uses randomly initialized encoder (MVP mode).")
    parser.add_argument("--n-style-tokens",     type=int,   default=0,
                        help="K_g: number of style tokens. 0 = no style conditioning.")
    parser.add_argument("--style-weight",       type=float, default=0.2,
                        help="γ_style: style token scale relative to entity grounding")

    # ── Ablation: pixel blend (disabled by default) ───────────────────────
    parser.add_argument("--use-style-blend",  action="store_true",
                        help="Enable GlobalStyle pixel blend (ablation — content leakage).")
    parser.add_argument("--style-blend-alpha",type=float, default=0.45)

    args = parser.parse_args()

    base   = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    plan_output = load_json(str(base / args.layout))
    out_path    = base / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    bootstrap_dir = (
        base / args.bootstrap_dir if args.bootstrap_dir
        else out_path / "bootstrap"
    )
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    # ── Load MemoryGLIGENPipeline ─────────────────────────────────────────
    logger.info(f"Loading MemoryGLIGENPipeline: {args.gligen_model}")
    gligen_pipe = _load_memory_gligen(args.gligen_model, device)

    # ── Phase 1: Entity residual conditioner ──────────────────────────────
    conditioner = None
    if args.conditioner_path:
        cpath = base / args.conditioner_path
        if cpath.exists():
            conditioner = _load_conditioner(cpath, args.conditioner_type, device)
        else:
            logger.warning(f"Conditioner checkpoint not found: {cpath}")

    # ── Phase 2: Style encoder ────────────────────────────────────────────
    style_encoder = None
    if args.n_style_tokens > 0:
        spath = base / args.style_encoder_path if args.style_encoder_path else None
        style_encoder = _load_style_encoder(spath, args.n_style_tokens, device)

    # ── Mode string ───────────────────────────────────────────────────────
    parts = []
    if isinstance(conditioner, ResidualEntityConditioner): parts.append("entity-residual")
    elif conditioner is not None:                          parts.append("entity-text-proj")
    if style_encoder is not None:                         parts.append(f"style-K{args.n_style_tokens}")
    if not parts:                                         parts.append("GLIGEN-baseline")
    mode_str = "+".join(parts)
    logger.info(f"Mode: {mode_str}")

    # ── Entity encoder + memory bank ──────────────────────────────────────
    encoder = EntityEncoder(device=device)
    memory  = EntityMemoryBank(
        encoder,
        alpha=args.alpha,
        quality_threshold=args.quality_threshold,
    )

    # ── Initialize entity memories ────────────────────────────────────────
    layouts     = plan_to_layout(plan_output)
    all_entities = sorted({e["name"] for l in layouts for e in l["entities"]})
    logger.info(f"Entities: {all_entities}")

    refs_dir = base / args.refs_dir if args.refs_dir else None
    for entity in all_entities:
        ref_path = refs_dir / f"{entity}.png" if refs_dir else None
        if ref_path and ref_path.exists():
            ref_img = Image.open(str(ref_path)).convert("RGB")
            logger.info(f"  [{entity}] init from user ref")
        else:
            ref_img = bootstrap_entity_image(
                entity, gligen_pipe, bootstrap_dir,
                steps=args.gligen_steps + 10,
            )
        memory.initialize(entity, ref_img)

    # ── GlobalStyle pixel blend (ablation) ───────────────────────────────
    style_memory = None
    if args.use_style_blend:
        style_memory = GlobalStyleMemory(blend_alpha=args.style_blend_alpha)
        logger.warning("Pixel-blend enabled (ablation — content leakage risk)")

    # ── Generate ──────────────────────────────────────────────────────────
    logger.info(f"=== PEMA v2 Phase 3 Generation ({mode_str}) ===")
    results = generate_with_pema_v2(
        plan_output=plan_output,
        memory_bank=memory,
        output_dir=str(out_path),
        gligen_pipe=gligen_pipe,
        conditioner=conditioner,
        style_encoder=style_encoder,
        entity_encoder=encoder if style_encoder is not None else None,
        style_memory=style_memory,
        device=device,
        update_memory=not args.no_memory_update,
        gligen_steps=args.gligen_steps,
        text_weight=args.text_weight,
        memory_weight=args.memory_weight,
        style_weight=args.style_weight,
        use_style_blend=args.use_style_blend,
    )

    logger.info(f"\nDone. {len(results)} shots → {out_path}")
    for r in results:
        logger.info(
            f"  shot {r['shot_id']:02d}: mode={r['mode']} "
            f"style_tokens={r.get('n_style_tokens',0)} "
            f"active={r['active_entities']} memory={r['memory_updates']}"
        )
        logger.info(f"    {r['path']}")


if __name__ == "__main__":
    main()
